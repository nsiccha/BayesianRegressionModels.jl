# One native fixed-budget test of the STL path derivative (Roeder et al2017,
# Eq8/Alg2) and sampled log p-log q objective. No importance weighting, new
# family, larger iteration budget, or reference-informed initialization.
# julia --project=test research/shared_field_laplace/path_derivative_fit.jl 'native path derivative SVI'
include("gaussian_block_fit.jl")

function path_derivative_terms(q,score,epsilon,eta)
    D=length(q.μ)
    D==3 || error("bounded three-global path-derivative control only")
    s=exp.(q.ρ)
    offset=s.*epsilon+q.Β*eta
    F=cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(q))))
    precision_offset=F\offset
    # This is model score MINUS the Gaussian q score. It already supplies
    # the entropy contribution in expectation; do not add it a second time.
    w=score+precision_offset
    gradient=vcat(w,w.*epsilon.*s,vec(w*eta'))
    (;gradient,offset,mahalanobis=dot(offset,precision_offset),entropy=logdet(F)/2)
end

mutable struct PathDerivativeTarget{J,M}
    joint::J
    m::M
    globals::Vector{Int}
    units::Vector{Vector{Int}}
    iterations::Int
    snapshots::Vector{NamedTuple}
end
function PathDerivativeTarget(joint)
    isempty(joint.units) || error("this control requires all units integrated")
    PathDerivativeTarget(joint,joint.m,joint.globals,joint.units,0,NamedTuple[])
end
function BRM._brm_svi_iteration!(rng,t::PathDerivativeTarget,q,opt,batch,lr,beta;whiten=false)
    isempty(batch) || error("no explicit unit batch in the Gaussian block control")
    D=length(t.globals)
    epsilon,eta=randn(rng,D),randn(rng,size(q.Β,2))
    u=copy(t.m.unit)
    u[t.globals]=BRM._brm_svi_draw_global(q,epsilon,eta)
    probes=[ifelse.(rand(rng,Bool,length(t.joint.op.field)),1.,-1.)
        for _ in 1:t.joint.trace_probes]
    state=joint_window_state(rng,t.joint,u,probes)
    terms=path_derivative_terms(q,state.gradient[t.globals],epsilon,eta)
    # This estimates the ELBO up to the same constant omitted by ordinary
    # SVI. Both log p and log q refer to the PRE-update q and the SAME draw.
    sampled_elbo=state.value+terms.entropy+(terms.mahalanobis-D)/2
    x=BRM._brm_svi_global_vector(q)
    whiten ? BRM._brm_svi_whitened_global_step!(opt,q,x,terms.gradient,lr) :
        BRM._brm_svi_step!(opt,x,terms.gradient,lr)
    BRM._brm_svi_set_global!(q,x)
    t.iterations+=1
    if t.iterations==1 || t.iterations%50==0
        push!(t.snapshots,(;iteration=t.iterations,whiten,lr,sampled_elbo,
            diagnostic_phase=:pre_update,parameter_phase=:post_update,
            gradient_norm=norm(terms.gradient),mu=copy(q.μ),rho=copy(q.ρ),B=copy(q.Β),
            covariance=Matrix(BRM._brm_svi_global_covariance(q))))
    end
    # The shared fit loop adds post-update entropy. Cancel that bookkeeping
    # term so it records exactly the pre-update sampled objective above.
    sampled_elbo-BRM._brm_svi_entropy(q)
end
BRM._brm_svi_draws(rng,t::PathDerivativeTarget,q,n;node=nothing)=
    BRM._brm_svi_draws(rng,t.joint,q,n;node)

function path_derivative_quadratic_control()
    q=(;μ=[.2,-.4,.1],ρ=log.([.3,.5,.4]),
        Β=[.5 .1 -.2;.2 .3 .1;.1 -.1 .4],U=Matrix{Float64}[])
    covariance=Matrix(BRM._brm_svi_global_covariance(q))
    precision=inv(Symmetric(covariance))
    P=[1.5 .2 -.1;.2 .8 .15;-.1 .15 1.2];b=[-.3,.2,.4]
    expected=vcat(b-P*q.μ,exp.(2q.ρ).*(diag(precision)-diag(P)),
        vec((precision-P)*q.Β))
    estimate=zeros(length(expected));values=0.
    matched_gradient=0.;matched_values=Float64[]
    basis=Matrix{Float64}(I,6,6)
    for j in 1:6,sign in (-1,1)
        z=sign*sqrt(6)*basis[:,j];epsilon,eta=z[1:3],z[4:6]
        g=BRM._brm_svi_draw_global(q,epsilon,eta)
        terms=path_derivative_terms(q,b-P*g,epsilon,eta)
        estimate.+=terms.gradient/12
        values+=(-dot(g,P*g)/2+dot(b,g)+terms.entropy+(terms.mahalanobis-3)/2)/12
        matched=path_derivative_terms(q,-precision*(g-q.μ),epsilon,eta)
        matched_gradient=max(matched_gradient,norm(matched.gradient,Inf))
        push!(matched_values,-dot(g-q.μ,precision*(g-q.μ))/2+
            matched.entropy+(matched.mahalanobis-3)/2)
    end
    expected_value=-dot(q.μ,P*q.μ)/2+dot(b,q.μ)-tr(P*covariance)/2+logdet(Symmetric(covariance))/2
    (;gradient_error=norm(estimate-expected,Inf),value_error=abs(values-expected_value),
        matched_gradient,matched_value_spread=maximum(matched_values)-minimum(matched_values))
end

@stestset "native path derivative SVI" begin
    algebra=path_derivative_quadratic_control()
    for metric in values(algebra)
        @test metric<2e-12
    end
    all(<(2e-12),values(algebra)) || error("path derivative algebra failed; no native fit")
    base=joinpath(@__DIR__, "results")
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    previous=deserialize(joinpath(base,"native-gaussian-block-fit-v1/comparison.jls"))
    reference.reference_ok && previous.identity.reference_signature==reference.signature ||
        error("native path derivative reference identity failed")
    sources=[joinpath(@__DIR__,name) for name in
        ("gaussian_block_fit.jl","gaussian_block_comparison.jl","stationarity_audit.jl",
         "exact_reference.jl","operator_probe.jl","whole_window_probe.jl",
         "joint_svi_probe.jl","nonlinear_probe.jl","trace_probe.jl","quadrature.jl")]
    append!(sources,[joinpath(@__DIR__,"../../src",name) for name in
        ("nlme_view.jl","nlme_svi.jl","rk_backend.jl")])
    push!(sources,joinpath(@__DIR__,"../../test/Manifest.toml"))
    bytes2hex(sha256(reduce(vcat,read.(sources))))==previous.identity.signature ||
        error("previously validated native target/environment changed")
    all(c->c.value_error<2e-7 && c.mode_error<2e-8 && c.gradient_error<2e-7,previous.checks) ||
        error("previous native target checks failed")
    directory=joinpath(base,"native-path-derivative-fit-v1")
    ispath(directory) && error("preserve prior path derivative fit")
    mkpath(directory)
    identity=(;signature=bytes2hex(sha256(reduce(vcat,read.(vcat([@__FILE__],sources))))),
        native_signature=previous.identity.signature,reference_signature=reference.signature,
        julia=string(VERSION),fit_seed=700,draw_seed=701,max_iterations=5000,ndraws=1000,
        trace_probes=8,logdet_probes=8,estimator=:path_derivative,
        stopping_objective=:sampled_logp_minus_logq)
    serialize(joinpath(directory,"identity.jls"),identity)
    serialize(joinpath(directory,"algebra.jls"),(;identity,algebra))
    with_progress(:state;description="Native path derivative Gaussian-block SVI") do progress
        snapshot_progress(progress,joinpath(directory,"fit.progress")) do
            fixture=with_progress(progress;description="Prepare unchanged native Gaussian-block target") do _
                (;rk,m,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
                merge((;rk,m,data),exact_gaussian_reference(rk,data))
            end
            (;rk,m,data,p,view,field,units)=fixture
            view.coordinates==previous.coordinates==reference.coordinates || error("coordinate identity differs")
            block=affine_gaussian_block(rk,data,p,view)
            joint=native_gaussian_block_target(m,block)
            joint.globals==previous.globals && joint.op.field==previous.latent ||
                error("native structural partition differs")
            t=PathDerivativeTarget(joint)
            start=first(BRM._brm_nlme_prior_median_point(m))
            q=BRM._brm_svi_params(t,start[t.globals],BRM.LowRankGlobal(3);global_scale=.2,unit_scale=.5)
            # The path derivative is unbiased for an exact marginal score;
            # the native mode/solve/quadrature tolerances still apply here.
            record=with_progress(progress;description="Same SVI budget with path derivative") do node
                seconds=@elapsed fit=BRM._brm_svi_fit!(Xoshiro(700),t,q;
                    batch_size=3,max_iterations=5000,lr_global=.01,unit_step=.1,
                    polish_lr=.02,window=50,patience=4,min_windows=8,polish_windows=10,node)
                record=(;identity,fit,last_q=deepcopy(q),snapshots=t.snapshots,seconds,
                    coordinates=view.coordinates,globals=t.globals,latent=joint.op.field,field,units,beta=block.beta)
                serialize(joinpath(directory,"fit.jls"),record)
                record
            end
            fit=record.fit
            @test fit.iterations<=5000 && fit.iterations==t.iterations
            @test all(isfinite,fit.elbo) && all(isfinite,fit.q.μ)
            println((;stage="path derivative fit saved",algebra,fit.converged,fit.iterations,seconds=record.seconds));flush(stdout)
            draw_record=with_progress(progress,1000;description="Unchanged native conditional draw gate") do node
                seconds=@elapsed draws=BRM._brm_svi_draws(Xoshiro(701),t,fit.q,1000;node)
                r=(;identity,draws,seconds);serialize(joinpath(directory,"draws.jls"),r);r
            end
            mean_error=abs.(vec(mean(draw_record.draws;dims=2))-reference.ref_mean)./reference.ref_sd
            sd_ratio=vec(std(draw_record.draws;dims=2))./reference.ref_sd
            accuracy_ok=fit.converged && maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
            digest=bytes2hex(sha256(reinterpret(UInt8,vcat(fit.q.μ,fit.q.ρ,vec(fit.q.Β),fit.elbo,vec(draw_record.draws)))))
            summary=(;identity,algebra,accuracy_ok,fit.converged,fit.iterations,seconds=record.seconds,
                draw_seconds=draw_record.seconds,mean_error,sd_ratio,digest,coordinates=view.coordinates,
                globals=t.globals,latent=joint.op.field,field,units,beta=block.beta,
                scope="one fixed-budget estimator-change test; no independent stationarity, scaling, streaming or reproducibility claim")
            serialize(joinpath(directory,"comparison.jls"),summary)
            println((;stage="path derivative accuracy",accuracy_ok,fit.converged,fit.iterations,
                max_mean_error=maximum(mean_error),original_global_sd_ratio=sd_ratio[p.globals],
                unit_sd_extrema=extrema(sd_ratio[units]),field_sd_extrema=extrema(sd_ratio[field]),digest))
            @test accuracy_ok
        end
    end
end
