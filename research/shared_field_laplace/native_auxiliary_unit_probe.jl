# Native likelihood derivatives + local prior conditional on the auxiliary
# core. This validates carried-target unit moments, not a second-window fit.
include("unit_conditional_moments.jl")
include("auxiliary_unit_moments.jl")
using DifferentiationInterface: SecondOrder

function auxiliary_unit_likelihood(m)
    built,plan=m.backend.model,RKEXT._rk_translated(m.backend)
    score=prepare_query_ad(prepare_ad,built,plan,:likelihood,NLME_AD,m.unit)
    second=SecondOrder(NLME_AD,AutoEnzyme(;mode=Enzyme.Forward))
    hvp=prepare_query_ad(prepare_ad_hvp,built,plan,:likelihood,second,(zero(m.unit),),m.unit)
    (;score,hvp,units=m.view.subject_coordinates)
end

# Requires certified quadratic likelihood and conditional unit independence.
# The prior block MUST be the entire unit residual block conditional on eta.
# No full marginal carry precision is used in colored Hessian directions.
function auxiliary_native_units(likelihood,u,eta,priors)
    units=likelihood.units
    length(units)==length(priors) && !isempty(units) || error("unit/prior blocks differ")
    width=maximum(length,units);width<=16 || error("unit block exceeds fixed cap")
    for (indices,b) in zip(units,priors)
        length(indices)==length(b.a) && size(b.B,2)==length(eta) ||
            throw(DimensionMismatch("conditional unit prior"))
    end
    _,g=ad_value_and_gradient(likelihood.score,u)
    hessians=[zeros(length(b),length(b)) for b in units]
    for k in 1:width
        direction=zeros(length(u))
        for indices in units
            k<=length(indices) && (direction[indices[k]]=1)
        end
        _,(hv,)=ad_gradient_and_hvp(likelihood.hvp,(direction,),u)
        for (indices,H) in zip(units,hessians)
            k<=length(indices) && (H[:,k]=hv[indices])
        end
    end
    means=copy(u);covariances=Matrix{Float64}[]
    for (indices,H,b) in zip(units,hessians,priors)
        isapprox(H,H';atol=1e-10,rtol=1e-10) || error("unit likelihood Hessian is not symmetric")
        precision=b.F\Matrix{Float64}(I,length(indices),length(indices))
        F=cholesky(Symmetric(precision-H))
        gradient=g[indices]-b.F\(u[indices]-b.a-b.B*eta)
        means[indices]=u[indices]+F\gradient
        push!(covariances,Matrix(inv(F)))
    end
    (;means,covariances,likelihood_gradient=g,hessians,
        inferred_calls=(;full_likelihood_gradients=1,colored_likelihood_hvps=width))
end

@stestset "native auxiliary conditional units" begin
    base=joinpath(@__DIR__, "results")
    directory=joinpath(base,"native-auxiliary-units-v1")
    ispath(directory) && error("preserve native auxiliary unit evidence")
    input_path=joinpath(base,"unit-conditional-moments-v1/comparison.jls")
    previous=deserialize(input_path)
    fit=deserialize(joinpath(base,"native-fixed-rate-fit-v1/comparison.jls"))
    draw_path=joinpath(base,"native-fixed-rate-fit-v1/draws.jls")
    draws=deserialize(draw_path)
    previous.identity.fit_signature==fit.identity.signature && fit.identity==draws.identity ||
        error("fitted carry identity differs")
    previous.identity.draw_file_sha==bytes2hex(sha256(read(draw_path))) ||
        error("retained draws content changed")
    c=previous.projection.factor
    order=vcat(fit.globals,[fit.beta],fit.units,fit.field)
    n,r=length(c.mu),size(c.B,2)
    n==26 && r==8 || error("bounded native carry fixture changed")
    previous.coordinates==fit.coordinates[order] || error("carried coordinate identity differs")
    keys_=[Symbol(:block_,j) for j in eachindex(c.blocks)]
    core=auxiliary_core(zeros(r),Matrix{Float64}(I,r,r))
    store=Dict(key=>auxiliary_block(c.mu[b],c.B[b,:],Matrix(F.L*F.L'))
        for (key,b,F) in zip(keys_,c.blocks,c.factors))
    selected=draws.draws[:,1:3]
    active_draws=Dict(key=>selected[order[b],:] for (key,b) in zip(keys_,c.blocks))
    eta=auxiliary_lift(Xoshiro(718),core,store,active_draws)
    # Bounded augmented-prior precision oracle, independent of local helper.
    n+r<=40 || error("augmented dense oracle cap")
    Dinv=zeros(n,n)
    for (b,F) in zip(c.blocks,c.factors)
        Dinv[b,b]=F\Matrix{Float64}(I,length(b),length(b))
    end
    P=[Matrix{Float64}(I,r,r)+c.B'*Dinv*c.B -c.B'*Dinv;-Dinv*c.B Dinv]
    files=sort(filter(endswith(".jl"),readdir(@__DIR__;join=true)))
    push!(files,joinpath(@__DIR__,"../../test/Manifest.toml"))
    identity=(;input_sha=bytes2hex(sha256(read(input_path))),previous.identity.fit_signature,
        source_sha=bytes2hex(sha256(reduce(vcat,read.(files)))),
        repository_commit=readchomp(`git -C $(@__DIR__) rev-parse HEAD`),
        core_seed=718,retained_draw_selection=1:3,scope=:native_local_conditional_control)
    mkpath(directory);serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Native correlated carry unit moments") do progress
        snapshot_progress(progress,joinpath(directory,"control.progress")) do
            fixture=with_progress(progress;description="Prepare native likelihood and bounded oracle") do _
                (;rk,m,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
                merge((;rk,m,data),exact_gaussian_reference(rk,data))
            end
            (;rk,m,data,p,view)=fixture
            view.coordinates==fit.coordinates || error("native coordinate identity differs")
            block=affine_gaussian_block(rk,data,p,view)
            likelihood=auxiliary_unit_likelihood(m)
            priors=map(m.view.subject_coordinates) do indices
                j=only(findall(b->order[b]==indices,c.blocks))
                store[keys_[j]]
            end
            results=with_progress(progress,3;description="Compare native conditional units at retained points") do node
                map(1:3) do j
                    u=selected[:,j]
                    exact=gaussian_reference_conditional(block.p3,u[fit.globals])
                    z=u[block.p3.latent]
                    likelihood_gradient=exact.h-exact.H*z+z
                    v=vcat(eta[:,j],u[order]);center=vcat(zeros(r),c.mu)
                    prior_gradient=-P*(v-center)
                    measured=@timed auxiliary_native_units(likelihood,u,eta[:,j],priors)
                    native=measured.value
                    errors=map(zip(m.view.subject_coordinates,native.covariances,native.hessians)) do (indices,C,H)
                        pos=[only(findall(==(i),block.p3.latent)) for i in indices]
                        aug=r.+[only(findall(==(i),order)) for i in indices]
                        Hlike=Matrix{Float64}(I,length(pos),length(pos))-exact.H[pos,pos]
                        Q=P[aug,aug]-Hlike
                        expected=u[indices]+Q\(likelihood_gradient[pos]+prior_gradient[aug])
                        @test native.likelihood_gradient[indices]≈likelihood_gradient[pos] atol=2e-10
                        @test H≈Hlike atol=2e-10
                        @test native.means[indices]≈expected atol=2e-10
                        @test C≈inv(Symmetric(Q)) atol=2e-11
                        (;mean=maximum(abs.(native.means[indices]-expected)),
                            covariance=maximum(abs.(C-inv(Symmetric(Q)))))
                    end
                    update_progress!(node,j)
                    (;j,errors,seconds=measured.time,bytes=measured.bytes,inferred_calls=native.inferred_calls)
                end
            end
            unit_positions=[only(findall(==(i),order)) for i in fit.units]
            marginal_precision=hcat([block_precision(c,Float64.((1:n).==j)) for j in 1:n]...)
            U=marginal_precision[unit_positions,unit_positions]
            marginal_unit_cross=norm(U-Diagonal(diag(U)),Inf)
            @test marginal_unit_cross>1e-3
            serialize(joinpath(directory,"control.jls"),(;identity,results,marginal_unit_cross,
                scope="fitted previous carry with native current likelihood; derivative and projection algebra only, no second-window fit or replay"))
            println((;results,marginal_unit_cross))
        end
    end
end
