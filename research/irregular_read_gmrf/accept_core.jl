using LinearAlgebra, SparseArrays, Serialization, Dates, Random, Statistics
using Bessels: besselk0, besselk1
include(joinpath(ARGS[1],"src/gmrf_precision.jl"))
include(joinpath(ARGS[1],"src/gmrf_vecchia.jl"))
include(joinpath(ARGS[1],"src/gmrf_multiresolution.jl"))
# Independent multiplication of the constructed OU transition operators along
# every shared factor, rather than calling the closed-form covariance helper.
function operator_covariance(g,b,p,ti,tj,theta)
    total=0.0
    for j in g.spatial.basis_ptr[p]:(g.spatial.basis_ptr[p+1]-1)
        factor=g.spatial.basis_node[j]
        a=_brm_multiresolution_state(g.paths,factor,ti)
        z=_brm_multiresolution_state(g.paths,factor,tj)
        product=1.0
        for s in (a+1):z
            product*=exp(-(g.times[g.paths.state_time[s]]-g.times[g.paths.state_time[s-1]])/theta)
        end
        total+=b[j]^2*product
    end
    total
end
function spatial_covariance(g,b,i,j)
    _brm_multiresolution_covariance(g,b,(;position=i,time=1),(;position=j,time=1),1.0)
end
function main()
    support=deserialize(ARGS[2]); results=Any[]
    for field in (:physical,:rain)
        d=getproperty(support,field)
        g=_brm_multiresolution_support(d.east,d.north,d.time; rank=30)
        x,y=g.spatial.conditionals.east,g.spatial.conditionals.north
        n=length(x);rng=MersenneTwister(54181)
        pairs=Tuple{Int,Int}[]
        tree=_BRMVecchiaTree(x,y);_brm_vecchia_insert_all!(tree)
        nearest,distance=zeros(Int32,30),zeros(30)
        for i in unique(vcat(collect(1:23:n),rand(rng,1:n,500)))
            count=_brm_vecchia_nearest!(nearest,distance,tree,x[i],y[i],30)
            append!(pairs,[(i,Int(nearest[j])) for j in 1:count if nearest[j]!=i])
            append!(pairs,[(i,j) for j in rand(rng,1:n,30)])
        end
        rhos=field==:physical ? (20.,94.,453.) : (8.,38.,181.)
        thetas=field==:physical ? (4.7,22.4,107.) : (.6,2.8,13.)
        for rho in rhos
            b=_brm_multiresolution_basis(g.spatial,rho/2).weights
            variances=[sum(abs2,view(b,g.spatial.basis_ptr[p]:(g.spatial.basis_ptr[p+1]-1))) for p in 1:n]
            @assert maximum(abs,variances.-1)<2e-14
            allerrors=Float64[]; bins=[Float64[] for _ in 1:3]
            worst=(error=-1.,positions=(0,0),exact=0.,constructed=0.,distance=0.)
            for (i,j) in pairs
                dist=hypot(x[i]-x[j],y[i]-y[j]);r=sqrt(8.)*dist/rho
                exact=r==0 ? 1.0 : r*besselk1(r)
                constructed=spatial_covariance(g,b,i,j)/sqrt(variances[i]*variances[j])
                err=abs(exact-constructed);push!(allerrors,err)
                if err>worst.error
                    worst=(error=err,positions=(i,j),exact,constructed,distance=dist)
                end
                for (k,scale) in enumerate((.5,1.,2.))
                    abs(dist/rho-scale)<=.05scale && push!(bins[k],err)
                end
            end
            @assert maximum(allerrors)<.03
            @assert all(!isempty,bins)
            i,j=worst.positions
            spatial=(;field,rho,pairs=length(pairs),variance_error=maximum(abs,variances.-1),
                a2_counts=length.(bins),a2_max=maximum.(bins),worst,
                worst_coordinates=((x[i],y[i]),(x[j],y[j])))
            push!(results,spatial);println("SPATIAL ",spatial);flush(stdout)
            # Actual pairs at integer read-hour lags. Noninteger θ cannot be a
            # read-hour pair; report the nearest hour explicitly, not as θ.
            for theta in thetas, lag in unique([1.,round(theta),72.])
                checks=0;error=0.;worstpair=nothing
                for p in unique(round.(Int,range(1,n;length=min(n,220))))
                    times=view(g.reads.state_time,g.reads.state_ptr[p]:(g.reads.state_ptr[p+1]-1))
                    for ti in times
                        tj=searchsortedfirst(g.times,g.times[ti]+lag)
                        tj<=length(g.times) && g.times[tj]==g.times[ti]+lag || continue
                        _brm_multiresolution_state(g.reads,p,tj)!=0 || continue
                        constructed=operator_covariance(g,b,p,ti,tj,theta)/variances[p]
                        err=abs(constructed-exp(-lag/theta));checks+=1
                        if err>error;error=err;worstpair=(x=x[p],y=y[p],from=g.times[ti],to=g.times[tj]);end
                        break
                    end
                end
                @assert checks>0 && error<=1e-6
                temporal=(;field,rho,theta,lag,exact_theta_lag=lag==theta,checks,error,worstpair)
                push!(results,temporal);println("TEMPORAL ",temporal);flush(stdout)
            end
        end
        println("SUPPORT ",(;field,positions=n,read_pairs=length(g.reads.state_time),
            path_states=length(g.paths.state_time),full_grid=n*length(g.times),
            retained_bytes=Base.summarysize(g),basis_entries=length(g.spatial.basis_node),
            max_width=maximum(diff(g.spatial.basis_ptr))))
    end
    # Channel draws on a known and one LKJ(2)-distributed 2x2 factor. For C=2,
    # r=2*Beta(2,2)-1; integer-shape gamma draws use sums of exponentials.
    rng=MersenneTwister(617)
    a,b=sum(randexp(rng,2)),sum(randexp(rng,2))
    for rho in (.4,2a/(a+b)-1)
        L=[1. 0.;rho sqrt(1-rho^2)]
        draws=L*randn(rng,2,100000)
        observed=cor(vec(draws[1,:]),vec(draws[2,:]))
        @assert abs(observed-rho)<.03
        println("CHANNEL ",(;rho,observed,error=abs(observed-rho),draws=size(draws,2)))
    end
    serialize(ARGS[3],results)
    println("actual support operator and spatial acceptance checks passed")
end
main()
