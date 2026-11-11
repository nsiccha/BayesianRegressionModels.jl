# A posterior-accuracy measurement, separate from the short integration gate.
# Run under one compute token:
# julia --project=test research/shared_field_laplace/posterior_probe.jl 'shared-field posterior comparison'
# Reference chains and data-only fits are retained in the task scratchpad.
include("joint_svi_probe.jl")
using WarmupHMC, MCMCDiagnosticTools, Serialization, Statistics

@stestset "shared-field posterior comparison" begin
    directory = get(ENV,"KB_LAPLACE_REFERENCE_DIR",
        joinpath(@__DIR__, "results", "posterior-v2-accept99"))
    mkpath(directory)
    signature = bytes2hex(sha256(vcat(read(@__FILE__),read(joinpath(@__DIR__,"operator_probe.jl")),
        read(joinpath(@__DIR__,"../../test/Manifest.toml")))))
    identity_file = joinpath(directory,"identity.txt")
    if isfile(identity_file)
        readchomp(identity_file) == signature || error(
            "reference directory belongs to different code/environment; use a fresh directory")
    else
        write(identity_file,signature*"\n")
    end
    (;rk,m) = shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2)
    t = joint_window_target(m;trace_probes=8,logdet_probes=4)
    start = first(BRM._brm_nlme_prior_median_point(m))
    chains = map(1:4) do chain
        saved = joinpath(directory,"reference-$chain.jls")
        isfile(saved) && return deserialize(saved)
        problem = rk_logdensity_problem(rk;ad_backend=NLME_AD)
        checkpoint_dir = joinpath(directory,"chain-$chain")
        resume = isfile(joinpath(checkpoint_dir,"cp_latest.jls"))
        println((;stage="reference",chain,resume,target_acceptance_rate=0.99)); flush(stdout)
        callback = (state,stage) -> begin
            println((;stage="reference boundary",chain,boundary=stage)); flush(stdout)
            false
        end
        seconds = @elapsed result = adaptive_warmup_mcmc(Xoshiro(500+chain),problem;
            init=(;position=start+0.05randn(Xoshiro(600+chain),length(start)),
                squared_scale=ones(length(start))),
            n_draws=2000,target_acceptance_rate=0.99,checkpoint_dir,resume,callback)
        record = (;position=Matrix{Float64}(result.posterior_position),
            divergences=result.n_divergent_samples,seconds,signature)
        serialize(saved,record)
        println((;stage="reference completed",chain,draws=size(record.position,2),
            record.divergences,seconds)); flush(stdout)
        record
    end
    count = minimum(size(c.position,2) for c in chains)
    samples = permutedims(cat((c.position[:,end-count+1:end]' for c in chains)...;
        dims=3),(1,3,2))
    reference_diagnostics = (;rhat=maximum(MCMCDiagnosticTools.rhat(samples)),
        bulk=minimum(MCMCDiagnosticTools.ess(samples;kind=:bulk)),
        tail=minimum(MCMCDiagnosticTools.ess(samples;kind=:tail)),
        divergences=sum(c.divergences for c in chains),
        draws=sum(size(c.position,2) for c in chains))
    println((;stage="reference diagnostics",reference_diagnostics...)); flush(stdout)
    reference_ok = reference_diagnostics.rhat < 1.01 &&
        min(reference_diagnostics.bulk,reference_diagnostics.tail) >= 400 &&
        reference_diagnostics.divergences == 0
    # Retain a failed reference as evidence; do not silently rerun until green.
    serialize(joinpath(directory,"reference-diagnostics.jls"),reference_diagnostics)
    @test reference_ok
    reference_ok || error("reference diagnostics are inadequate for accuracy acceptance")
    reference = reduce(hcat,(c.position for c in chains))
    q = BRM._brm_svi_params(t,start[t.globals],BRM.LowRankGlobal(4);
        global_scale=0.2,unit_scale=0.5)
    println((;stage="joint-window SVI fit")); flush(stdout)
    seconds = @elapsed fit = BRM._brm_svi_fit!(Xoshiro(700),t,q;batch_size=3,
        max_iterations=5000,lr_global=0.01,unit_step=0.1,polish_lr=0.02,
        window=50,patience=4,min_windows=8,polish_windows=10)
    # Save before the draw phase so a failure retains the fitted factor.
    serialize(joinpath(directory,"fit.jls"),(;fit,seconds,signature))
    draw_seconds = @elapsed draws = BRM._brm_svi_draws(Xoshiro(701),t,fit.q,1000)
    serialize(joinpath(directory,"draws.jls"),(;draws,draw_seconds,signature))
    ref_mean, ref_sd = vec(mean(reference;dims=2)),vec(std(reference;dims=2))
    approx_mean, approx_sd = vec(mean(draws;dims=2)),vec(std(draws;dims=2))
    mean_error = abs.(approx_mean-ref_mean)./ref_sd
    sd_ratio = approx_sd./ref_sd
    # Predetermined accuracy target, reported even when missed. It is not
    # weakened to the broader known ordinary-SVI underdispersion envelope.
    accuracy_ok = fit.converged && maximum(mean_error) <= 0.3 &&
        all(x -> 0.8 <= x <= 1.2,sd_ratio)
    summary = (;reference_diagnostics,fit.converged,fit.iterations,seconds,draw_seconds,
        coordinates=m.view.coordinates,field=t.op.field,globals=t.globals,
        mean_error,sd_ratio,accuracy_ok,signature,
        scope="full-current-window accuracy; no streaming or locality acceptance")
    serialize(joinpath(directory,"comparison.jls"),summary)
    println((;stage="accuracy",summary...)); flush(stdout)
    @test accuracy_ok
end
