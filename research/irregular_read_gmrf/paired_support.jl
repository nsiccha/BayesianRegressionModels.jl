# Reuse the producer's seed-7 source reads and report concrete before/after
# pairs without constructing an archive covariance matrix.
# julia --project=test paired_support.jl SUPPORTS.jls physical|rain OUTPUT.toml
using LinearAlgebra, SparseArrays, Serialization, Random, Statistics, Dates, SHA, TOML
using Bessels: besselk0, besselk1
const REPO = normpath(joinpath(@__DIR__, "../.."))
include(joinpath(REPO, "src/gmrf_precision.jl"))
include(joinpath(REPO, "src/gmrf_vecchia.jl"))
include(joinpath(REPO, "src/gmrf_multiresolution.jl"))

# These two routines and source selection match delivered probe-footprint.jl.
function convex_hull(points)
    p = sort(unique(points))
    length(p) <= 2 && return p
    cross(o, a, b) = (a[1]-o[1])*(b[2]-o[2]) - (a[2]-o[2])*(b[1]-o[1])
    half(seq) = begin
        h = eltype(p)[]
        for q in seq
            while length(h) >= 2 && cross(h[end-1], h[end], q) <= 0
                pop!(h)
            end
            push!(h, q)
        end
        h
    end
    vcat(half(p)[1:end-1], half(reverse(p))[1:end-1])
end
function hull_distance(hull, x, y)
    best = Inf
    for i in eachindex(hull)
        a, b = hull[i], hull[mod1(i+1,length(hull))]
        dx, dy = b[1]-a[1], b[2]-a[2]
        t = clamp(((x-a[1])*dx+(y-a[2])*dy)/max(dx^2+dy^2,eps()), 0.0, 1.0)
        best = min(best, hypot(x-a[1]-t*dx, y-a[2]-t*dy))
    end
    best
end

function operator_covariance(g, weights, p, ti, tj, theta)
    ti, tj = minmax(ti,tj)
    total = 0.0
    for j in g.spatial.basis_ptr[p]:g.spatial.basis_ptr[p+1]-1
        factor = g.spatial.basis_node[j]
        first = _brm_multiresolution_state(g.paths,factor,ti)
        last = _brm_multiresolution_state(g.paths,factor,tj)
        product = 1.0
        for s in first+1:last
            product *= exp(-(g.times[g.paths.state_time[s]]-g.times[g.paths.state_time[s-1]])/theta)
        end
        total += weights[j]^2 * product
    end
    total
end

plain(x::NamedTuple) = Dict(string(k)=>plain(v) for (k,v) in pairs(x))
plain(x::Tuple) = plain.(collect(x))
plain(x::AbstractArray) = plain.(x)
plain(x::Symbol) = string(x)
plain(x) = x
lagbin(lag,theta) = lag == 0 ? 1 : lag <= 1 ? 2 : lag <= theta ? 3 : lag <= 72 ? 4 : 5
distbin(r,rho) = r == 0 ? 1 : r <= .25rho ? 2 : r <= .75rho ? 3 : r <= 1.5rho ? 4 : r <= 3rho ? 5 : 6

function read_receipt(f,g,mapping,state,time_index)
    p,ti = f.position[state],time_index[state]
    gp = mapping[p]
    times = view(g.reads.state_time,g.reads.state_ptr[gp]:g.reads.state_ptr[gp+1]-1)
    k = searchsortedfirst(times,ti)
    t = f.times[ti]
    (;state,position=Int(p),east=f.east[p],north=f.north[p],time=t,
      position_reads=length(times),read_fraction=length(times)/length(f.times),
      previous_gap=k>1 ? t-g.times[times[k-1]] : -1.,
      next_gap=k<length(times) ? g.times[times[k+1]]-t : -1.,
      reads_at_hour=f.time_ptr[ti+1]-f.time_ptr[ti],
      reads_previous_read_time=ti>1 ? f.time_ptr[ti]-f.time_ptr[ti-1] : 0,
      previous_read_time_gap=ti>1 ? t-f.times[ti-1] : -1.)
end

function main()
    supports_path,field_name,outfile = ARGS
    field = Symbol(field_name)
    field in (:physical,:rain) || error("field must be physical or rain")
    expected_sha = "5d5a162f718b29ae84938c7a31e0a47c7b6153ff7ab05139b623b06a650cc2b0"
    support_sha = open(io -> bytes2hex(sha256(io)),supports_path)
    support_sha == expected_sha || error("support differs from the recovered producer input")
    d = getproperty(deserialize(supports_path),field)
    f = _brm_vecchia_footprint(d.east,d.north,d.time;neighbors=30)
    g = _brm_multiresolution_support(d.east,d.north,d.time;rank=30)
    @assert g.times == f.times
    mapping = [g.spatial.conditionals.position_index[(x,y)] for (x,y) in zip(f.east,f.north)]
    n,np = length(f.position),length(f.east)
    ti = Vector{Int32}(undef,n)
    for k in eachindex(f.times); ti[f.time_ptr[k]:f.time_ptr[k+1]-1] .= k; end
    rhos = field == :physical ? (q025=20.,median=94.,q975=453.) : (q025=8.,median=38.,q975=181.)
    thetas = field == :physical ? (q025=4.7,median=22.4,q975=107.) : (q025=.6,median=2.8,q975=13.)
    nsources = field == :physical ? 60 : 45
    hull = convex_hull(collect(zip(f.east,f.north)))
    hull_of = [hull_distance(hull,f.east[p],f.north[p]) for p in 1:np]
    rng = MersenneTwister(7)
    interior = [s for s in randperm(rng,n)[1:min(n,200_000)] if hull_of[f.position[s]] >= rhos.median]
    boundary = [s for s in randperm(rng,n)[1:min(n,200_000)] if hull_of[f.position[s]] <= 10]
    sources = vcat(interior[1:min(end,2nsources÷3)],boundary[1:min(end,nsources÷3)])
    @assert length(sources) == nsources
    receipt = Dict{String,Any}("field"=>field_name,"input_sha256"=>support_sha,
        "rank"=>30,"neighbors"=>30,"source_seed"=>7,"source_count"=>length(sources),
        "sources"=>plain.([read_receipt(f,g,mapping,s,ti) for s in sources]),
        "cases"=>Any[])
    out,work,other = zeros(n),zeros(n),zeros(n)
    for (rlabel,tlabel) in ((:median,:median),(:q975,:median),(:q025,:median),
                           (:median,:q975),(:median,:q025),(:q975,:q975))
        rho,theta = getproperty(rhos,rlabel),getproperty(thetas,tlabel)
        factors = _brm_vecchia_factors(f,rho/2,theta)
        weights = _brm_multiresolution_basis(g.spatial,rho/2).weights
        variances = [sum(abs2,view(weights,g.spatial.basis_ptr[p]:g.spatial.basis_ptr[p+1]-1)) for p in 1:np]
        @assert maximum(abs,variances.-1) < 2e-14
        # Cache only P entries per sampled position, never an N by N matrix.
        cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}}()
        worst = Dict{String,NamedTuple}()
        counts,oldmax,newmax = zeros(Int,5,6),zeros(5,6),zeros(5,6)
        a4counts,a4old,a4new = zeros(Int,3),zeros(3),zeros(3)
        a2counts,a2old,a2new = zeros(Int,3),zeros(3),zeros(3)
        lags = (1.,Float64(round(theta)),72.)
        function update!(key,err,i,j,old,new,exact,vj)
            if !haskey(worst,key) || err>worst[key].error
                worst[key] = (;error=err,target=i,source=j,old,new,exact,source_variance=vj)
            end
        end
        for j in sources
            _brm_vecchia_covariance_column!(out,work,factors,f,j)
            pj,tj = Int(f.position[j]),ti[j]
            distances,exact_space,new_space = get!(cache,pj) do
                distances = [hypot(f.east[p]-f.east[pj],f.north[p]-f.north[pj]) for p in 1:np]
                exact_space = [_brm_vecchia_matern(sqrt(8.)*r/rho) for r in distances]
                new_space = [_brm_multiresolution_covariance(g,weights,
                    (;position=mapping[p],time=1),(;position=mapping[pj],time=1),1.) for p in 1:np]
                (distances,exact_space,new_space)
            end
            lagtime = abs.(f.times .- f.times[tj])
            temporal = exp.(-lagtime./theta)
            @inbounds for i in 1:n
                pi_,t = f.position[i],ti[i]
                dist,lag = distances[pi_],lagtime[t]
                exact,new = exact_space[pi_]*temporal[t],new_space[pi_]*temporal[t]
                oe,ne = abs(out[i]-exact),abs(new-exact)
                lb,db = lagbin(lag,theta),distbin(dist,rho)
                counts[lb,db]+=1;oldmax[lb,db]=max(oldmax[lb,db],oe);newmax[lb,db]=max(newmax[lb,db],ne)
                if lb==3 && db==2
                    update!("combined_old_worst",oe,i,j,out[i],new,exact,out[j])
                    update!("combined_new_worst",ne,i,j,out[i],new,exact,out[j])
                end
                if lag==0 && hull_of[pi_]>=rho && hull_of[pj]>=rho
                    for (k,multiple) in enumerate((.5,1.,2.))
                        if abs(dist/rho-multiple)<=.05
                            a2counts[k]+=1;a2old[k]=max(a2old[k],oe);a2new[k]=max(a2new[k],ne)
                            update!("a2_$(k)_old_worst",oe,i,j,out[i],new,exact,out[j])
                            update!("a2_$(k)_new_worst",ne,i,j,out[i],new,exact,out[j])
                        end
                    end
                end
                if pi_==pj
                    for k in 1:3
                        if lag==lags[k]
                            oper = operator_covariance(g,weights,mapping[pj],t,tj,theta)
                            ratio = oper/variances[mapping[pj]]
                            olderr,newerr=abs(out[i]/out[j]-temporal[t]),abs(ratio-temporal[t])
                            a4counts[k]+=1;a4old[k]=max(a4old[k],olderr);a4new[k]=max(a4new[k],newerr)
                            update!("a4_$(k)_old_worst",olderr,i,j,out[i],oper,temporal[t],out[j])
                        end
                    end
                end
            end
        end
        # Exact target variance for each reported pair: one additional sparse
        # covariance column per distinct target, not a diagonal approximation.
        diagonal = Dict{Int,Float64}()
        pair_receipts = Any[]
        for (kind,w) in sort!(collect(worst);by=first)
            vi = get!(diagonal,w.target) do
                _brm_vecchia_covariance_column!(other,work,factors,f,w.target)
                @assert isapprox(other[w.source],w.old;atol=2e-12,rtol=2e-10)
                other[w.target]
            end
            nv_i,nv_j = variances[mapping[f.position[w.target]]],variances[mapping[f.position[w.source]]]
            push!(pair_receipts,plain((;kind,w...,target_variance=vi,
                old_source_ratio=w.old/w.source_variance,old_target_ratio=w.old/vi,
                old_correlation=w.old/sqrt(w.source_variance*vi),
                new_source_ratio=w.new/nv_j,new_target_ratio=w.new/nv_i,
                new_correlation=w.new/sqrt(nv_i*nv_j),
                source_read=read_receipt(f,g,mapping,w.source,ti),
                target_read=read_receipt(f,g,mapping,w.target,ti))))
        end
        @assert maximum(a4new)<=1e-6
        @assert maximum(a2new)<=.03
        rlabel == :median && @assert all(>(0),a2counts)
        result = Dict("rho_label"=>string(rlabel),"theta_label"=>string(tlabel),"rho"=>rho,"theta"=>theta,
            "a4_lags"=>collect(lags),"a4_exact_theta_lag"=>collect(lags).==theta,
            "a4_counts"=>a4counts,"a4_old_errors"=>a4old,"a4_new_errors"=>a4new,
            "a2_counts"=>a2counts,"a2_old_errors"=>a2old,"a2_new_errors"=>a2new,
            "variance_error"=>maximum(abs,variances.-1),"pairs"=>pair_receipts,
            "bins"=>[Dict("lag_bin"=>l,"distance_bin"=>d,"count"=>counts[l,d],
                "old_max"=>oldmax[l,d],"new_max"=>newmax[l,d]) for l in 1:5 for d in 1:6 if counts[l,d]>0])
        push!(receipt["cases"],result)
        open(io->TOML.print(io,receipt;sorted=true),outfile,"w")
        println(now()," ",field," ",rlabel,"/",tlabel," A4 before=",a4old," after=",a4new,
            " A2 after=",a2new," pairs=",length(pair_receipts)," peak_rss=",Sys.maxrss());flush(stdout)
        factors = nothing
        empty!(cache)
        GC.gc()
    end
    println("matched producer-source comparison passed; receipt=",outfile)
end
main()
