# One FASO-inspired fixed-rate stopping/averaging experiment. Same native
# STL target/start/seed/update cap, with fixed Adam lr.01 (no old polishing
# schedule). No RABVI learning-rate bias extrapolation or accuracy guarantee.
# julia --project=test research/shared_field_laplace/fixed_rate_fit.jl 'native fixed rate SVI'
include("path_derivative_fit.jl")
include("iterate_stopping.jl")

function fixed_rate_average(q,history,first,last)
    candidate=deepcopy(q)
    BRM._brm_svi_set_global!(candidate,vec(mean(history[:,first:last];dims=2)))
    candidate
end

function fixed_rate_fit!(rng,t,q;node)
    limit=5000;lr=.01
    raw=zeros(15,limit);distributions=zeros(9,limit);elbo=Float64[]
    checks=NamedTuple[];anchor=nothing;next_precision=0
    opt=BRM._BRMSVIAdam(15,lr)
    for k in 1:limit
        value=BRM._brm_svi_iteration!(rng,t,q,opt,Int[],lr,.1;whiten=false)
        push!(elbo,value+BRM._brm_svi_entropy(q))
        raw[:,k]=BRM._brm_svi_global_vector(q)
        distributions[:,k]=iterate_distribution(q)
        if anchor===nothing && k%200==0
            windows=iterate_windows(k)
            if !isempty(windows)
                scores=map(windows) do W
                    r=iterate_rhat(raw[:,k-W+1:k]);d=iterate_rhat(distributions[:,k-W+1:k])
                    max(r.maximum,d.maximum)
                end
                selected=argmin(scores) # ties select the first/shortest window
                W=windows[selected]
                push!(checks,(;stage=:stability,k,windows,scores,selected_window=W))
                if scores[selected]<=1.1
                    anchor=k-W+1
                    next_precision=k
                end
            end
        end
        if anchor!==nothing && k==next_precision
            candidate=fixed_rate_average(q,raw,anchor,k)
            scale=iterate_scales(candidate)
            r=iterate_precision(raw[:,anchor:k],scale.raw)
            d=iterate_precision(distributions[:,anchor:k],scale.distribution)
            push!(checks,(;stage=:precision,k,anchor,raw=r,distribution=d))
            if r.passed && d.passed
                gap=exact_iterate_covariance_gap(raw[:,anchor:k])
                return (;q=candidate,elbo,iterations=k,iterate_diagnostics_passed=true,
                    budget_exhausted=false,stop_reason=:iterate_diagnostics,
                    averaging_window=anchor:k,checks,raw=raw[:,1:k],
                    distributions=distributions[:,1:k],gap,lr)
            end
            # Same retained start, double the precision window; deterministic
            # across machines. Rhat is rechecked as part of every precision test.
            next_precision=anchor+2*(k-anchor+1)-1
        end
        if k%200==0
            BRM._brm_throw_if_interrupted(node)
            BRM._brm_update_progress!(node,"fixed-rate update $k/$limit; iterate diagnostics pending")
        end
    end
    # Predetermined fallback if the cap is exhausted: last200 iterates, never
    # select the best-looking posterior/ELBO/history window after the result.
    anchor=limit-199
    candidate=fixed_rate_average(q,raw,anchor,limit);scale=iterate_scales(candidate)
    push!(checks,(;stage=:budget_endpoint,k=limit,anchor,
        raw=iterate_precision(raw[:,anchor:limit],scale.raw),
        distribution=iterate_precision(distributions[:,anchor:limit],scale.distribution)))
    (;q=candidate,elbo,iterations=limit,iterate_diagnostics_passed=false,
        budget_exhausted=true,stop_reason=:update_budget,averaging_window=anchor:limit,
        checks,raw,distributions,gap=exact_iterate_covariance_gap(raw[:,anchor:limit]),lr)
end

@stestset "native fixed rate SVI" begin
    base=joinpath(@__DIR__, "results")
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    previous=deserialize(joinpath(base,"native-path-derivative-fit-v1/comparison.jls"))
    reference.reference_ok && previous.identity.reference_signature==reference.signature ||
        error("fixed-rate reference identity differs")
    sources=[joinpath(@__DIR__,name) for name in
        ("path_derivative_fit.jl","gaussian_block_fit.jl","gaussian_block_comparison.jl",
         "stationarity_audit.jl","exact_reference.jl","operator_probe.jl","whole_window_probe.jl",
         "joint_svi_probe.jl","nonlinear_probe.jl","trace_probe.jl","quadrature.jl")]
    append!(sources,[joinpath(@__DIR__,"../../src",name) for name in
        ("nlme_view.jl","nlme_svi.jl","rk_backend.jl")])
    push!(sources,joinpath(@__DIR__,"../../test/Manifest.toml"))
    bytes2hex(sha256(reduce(vcat,read.(sources))))==previous.identity.signature ||
        error("previously validated native STL target/environment changed")
    directory=joinpath(base,"native-fixed-rate-fit-v1")
    ispath(directory) && error("preserve prior fixed-rate experiment")
    mkpath(directory)
    identity=(;signature=bytes2hex(sha256(reduce(vcat,read.(vcat(
            [@__FILE__,joinpath(@__DIR__,"iterate_stopping.jl")],sources))))),
        native_signature=previous.identity.native_signature,stl_signature=previous.identity.signature,
        reference_signature=reference.signature,julia=string(VERSION),fit_seed=700,draw_seed=701,
        max_iterations=5000,ndraws=1000,lr=.01,trace_probes=8,logdet_probes=8,
        minimum_window=200,stability_check_period=200,window_candidates=5,
        rhat_threshold=1.1,maximum_normalized_mcse=.1,minimum_ess=50,
        precision_recheck=:double_window,budget_return=:last_200,
        method=:faso_inspired_fixed_rate)
    serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Native fixed-rate iterate stopping experiment") do progress
        snapshot_progress(progress,joinpath(directory,"fit.progress")) do
            fixture=with_progress(progress;description="Prepare unchanged native Gaussian target") do _
                (;rk,m,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
                merge((;rk,m,data),exact_gaussian_reference(rk,data))
            end
            (;rk,m,data,p,view,field,units)=fixture
            view.coordinates==previous.coordinates==reference.coordinates || error("coordinate identity differs")
            block=affine_gaussian_block(rk,data,p,view)
            joint=native_gaussian_block_target(m,block)
            joint.globals==previous.globals && joint.op.field==previous.latent || error("partition differs")
            t=PathDerivativeTarget(joint)
            start=first(BRM._brm_nlme_prior_median_point(m))
            q=BRM._brm_svi_params(t,start[t.globals],BRM.LowRankGlobal(3);global_scale=.2,unit_scale=.5)
            record=with_progress(progress;description="Fixed-rate updates, stability and precision checks") do node
                seconds=@elapsed fit=fixed_rate_fit!(Xoshiro(700),t,q;node)
                r=(;identity,fit,last_q=deepcopy(q),snapshots=t.snapshots,seconds,
                    coordinates=view.coordinates,globals=t.globals,latent=joint.op.field,field,units,beta=block.beta)
                serialize(joinpath(directory,"fit.jls"),r);r
            end
            fit=record.fit
            @test fit.iterations<=5000 && fit.iterations==t.iterations
            @test all(isfinite,fit.elbo) && all(isfinite,fit.q.μ)
            @test minimum(eigvals(Symmetric(fit.gap.gap)))>=-1e-12
            println((;stage="fixed-rate fit saved",fit.iterate_diagnostics_passed,fit.budget_exhausted,
                fit.iterations,fit.averaging_window,gap_fraction=fit.gap.fraction,seconds=record.seconds));flush(stdout)
            draw_record=with_progress(progress,1000;description="Unchanged native conditional draws") do node
                seconds=@elapsed draws=BRM._brm_svi_draws(Xoshiro(701),t,fit.q,1000;node)
                r=(;identity,draws,seconds);serialize(joinpath(directory,"draws.jls"),r);r
            end
            mean_error=abs.(vec(mean(draw_record.draws;dims=2))-reference.ref_mean)./reference.ref_sd
            sd_ratio=vec(std(draw_record.draws;dims=2))./reference.ref_sd
            draw_accuracy_ok=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
            accuracy_ok=fit.iterate_diagnostics_passed && draw_accuracy_ok
            digest=bytes2hex(sha256(reinterpret(UInt8,vcat(fit.q.μ,fit.q.ρ,vec(fit.q.Β),fit.elbo,vec(draw_record.draws)))))
            summary=(;identity,accuracy_ok,draw_accuracy_ok,fit.iterate_diagnostics_passed,fit.budget_exhausted,
                fit.iterations,seconds=record.seconds,draw_seconds=draw_record.seconds,mean_error,sd_ratio,digest,
                coordinates=view.coordinates,globals=t.globals,latent=joint.op.field,field,units,beta=block.beta,
                scope="one bounded fixed-rate stopping/averaging change; fixed-step iterate diagnostics differ from ELBO stationarity and posterior accuracy; no streaming/scaling/reproducibility claim")
            serialize(joinpath(directory,"comparison.jls"),summary)
            println((;stage="fixed-rate accuracy",accuracy_ok,draw_accuracy_ok,fit.iterate_diagnostics_passed,
                fit.budget_exhausted,fit.iterations,max_mean_error=maximum(mean_error),
                original_global_sd_ratio=sd_ratio[p.globals],unit_sd_extrema=extrema(sd_ratio[units]),
                field_sd_extrema=extrema(sd_ratio[field]),digest))
            @test accuracy_ok
        end
    end
end
