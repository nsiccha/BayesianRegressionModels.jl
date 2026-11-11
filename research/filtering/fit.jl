# Run from the repository root in the pinned test environment:
# julia --project=test research/filtering/fit.jl /path/to/checkpoints
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, Statistics, Serialization, Treebars, TreeArrays, StatsBase, SHA
import MCMCDiagnosticTools
include("model.jl")
include("provenance.jl")

const RESULTS = joinpath(@__DIR__, "results")
const PARAMETERS = (:level, :rho, :tau)
const PARTICLES = 1024
const TRAINING_DAYS = 20
const MIDDLE_DAY = 40

function write_table(name, rows)
    isempty(rows) && error("No rows for $name")
    mkpath(RESULTS)
    open(joinpath(RESULTS, name * ".tsv"), "w") do io
        println(io, join(string.(keys(first(rows))), '\t'))
        for row in rows
            println(io, join((v isa AbstractFloat ? string(round(v; sigdigits=6)) : string(v) for v in values(row)), '\t'))
        end
    end
end

function cached(f, directory, name)
    path = joinpath(directory, name * ".jls")
    isfile(path) && return deserialize(path)
    result = f()
    serialize(path * ".tmp", result)
    mv(path * ".tmp", path)
    result
end

function phase(f, parent, label)
    println("START ", label); flush(stdout)
    result = with_progress(f, parent; description=label)
    println("DONE ", label); flush(stdout)
    result
end

function record_failure(f, directory)
    try
        f()
    catch err
        # Julia's fully specialized MethodError can be enormous; retain it
        # while keeping the progress log readable.
        path = joinpath(directory, "failure.txt")
        open(path, "w") do io
            showerror(io, err, catch_backtrace())
        end
        println(stderr, "Filtering run failed: ", nameof(typeof(err)), "; diagnostic: ", path)
        exit(1)
    end
end

function tuning_options()
    SMCOptions(; moves=3, max_tree_depth=7, history_limit=128,
        nthreads=Threads.nthreads())
end
production_options(schedule) = SMCOptions(; tuning=schedule,
    history_limit=128, nthreads=Threads.nthreads())

function tune_batch(rk, seed, progress)
    d = length(coordinate_names(rk.model.layout))
    brm_smc_tune(rk; rng=Xoshiro(seed), init=zeros(d),
        pilot=:adaptive, reference=:pilot, pilot_steps=64,
        grid_start=:independent, options=tuning_options(), progress)
end

function weighted_summary(values, mass)
    w = Weights(mass)
    qs = quantile(values, w, [0.05, 0.25, 0.5, 0.75, 0.95])
    m = sum(values .* mass) / sum(mass)
    s = sqrt(sum(mass .* (values .- m).^2) / sum(mass))
    (; mean=m, sd=s, q05=qs[1], q25=qs[2], q50=qs[3], q75=qs[4], q95=qs[5])
end

function posterior_rows(rk, positions, mass, method)
    post = rk_draw_tree(rk, positions, PARAMETERS)
    [(; method, parameter=string(p),
        weighted_summary(vec(collect(TreeActualArray(getproperty(post, p)))), mass)...)
        for p in PARAMETERS]
end

function learning_rows(filter, method)
    parameters = [smc_parameters(filter.calibration.posterior, collect(q))
        for q in eachcol(filter.state.positions)]
    [(; method, parameter=string(p),
        weighted_summary([getproperty(q, p) for q in parameters], filter.state.parameter_mass)...)
        for p in PARAMETERS]
end

function selective_example(progress)
    initial_data = (; U=2, unit=[1, 2], y=[6, 14],
        level_location=2.0, level_scale=1.0, offset_scale=0.5)
    initial = brm_smc_fit(growing_model(initial_data); rng=Xoshiro(701),
        particles=1024, options=tuning_options(), progress)
    carried = brm_smc_carry(initial; basis=:correlation)
    next_data = (; initial_data..., U=3, unit=[3], y=[22])
    window = brm_smc_update(carried, growing_model(next_data);
        coordinates=:auto, rng=Xoshiro(702), particles=1024,
        options=tuning_options(), progress)
    @assert window.state.temperature == 1
    @assert size(window.state.positions, 1) == 2
    recovered = brm_smc_recover(window, window.state.positions[:, 1]; rng=Xoshiro(703))
    @assert length(recovered) == 4
    [(; quantity="growing_active_coordinates", value=size(window.state.positions, 1)),
        (; quantity="growing_full_coordinates", value=length(recovered))]
end

function main(checkpoint_dir, root; check_growing=true)
    mkpath(checkpoint_dir)
    fixture = filtering_fixture()
    data, truth = fixture.data, fixture.truth
    receipt = filtering_receipt(data, checkpoint_dir)
    write_table("receipt", receipt)
    started_this_invocation = time()
    training = filtering_window(data, 1, TRAINING_DAYS)
    middle = filtering_window(data, 1, MIDDLE_DAY)
    full = filtering_window(data, 1, data.T)
    rk = phase(root, "Compile count model") do _
        RKBRMI(filtering_model(training))
    end
    summary = NamedTuple[]
    function measure(name, f)
        elapsed = @elapsed value = f()
        push!(summary, (; quantity=name * "_seconds_this_invocation", value=elapsed))
        value
    end
    schedule = phase(root, "Tune training bridge") do p
        cached(checkpoint_dir, "training-schedule") do
            tune_batch(rk, 1201, p)
        end
    end
    fit = phase(root, "Fit training window") do p
        cp = cached(checkpoint_dir, "training-fit") do
            fitted = measure("training_fit", () -> brm_smc_fit(rk;
                rng=Xoshiro(201), particles=PARTICLES, options=production_options(schedule), progress=p))
            brm_smc_checkpoint(fitted)
        end
        brm_smc_resume(rk, cp; nthreads=Threads.nthreads())
    end
    @assert fit.state.temperature == 1
    push!(summary, (; quantity="training_stages", value=length(fit.state.history)))

    records = [(; observation=(; y=[ys[t] for ys in data.y]),
        inputs=(; exposure=ones(data.U))) for t in 1:data.T]
    state_rows = NamedTuple[]
    filter_scores = NamedTuple[]
    filter_result = phase(root, "Filter new observations") do p
      cached(checkpoint_dir, "filters") do
        ordinary = brm_filter(fit; inputs=(:exposure,),
            groups=(; i=1:data.U), particles=64, rng=Xoshiro(401))
        warm = brm_filter_stream(ordinary, records[1:TRAINING_DAYS];
            condition_state_only=true, progress=p)
        @assert warm.state.log_normalizer == 0
        current = warm
        for t in (TRAINING_DAYS + 1):data.T
            current = brm_filter_step(current, records[t].observation; inputs=records[t].inputs)
            for i in 1:data.U
                values = [getproperty(carry, Symbol("i.state"))[end, i] for carry in current.state.carries]
                push!(state_rows, (; day=t, unit=i, observed=data.y[i][t],
                    truth=truth.state[i, t], weighted_summary(values, current.state.weights)...))
            end
            push!(filter_scores, (; day=t, score=current.state.predictive_logscore,
                cumulative_score=current.state.log_normalizer,
                parameter_ess=inv(sum(abs2, current.state.parameter_mass))))
        end
        restored = brm_filter_resume(rk, brm_filter_checkpoint(current))
        @assert restored.state.weights == current.state.weights
        calibration = brm_smc_carry(fit; parameters=:filter,
            inputs=(:exposure,), basis=:correlation)
        learning = brm_filter(calibration, filtering_model(training); boundary=warm,
            groups=(; i=1:data.U),
            members=256, particles=128, window_length=10,
            rng=Xoshiro(402), options=tuning_options())
        learning_history = NamedTuple[]
        for t in (TRAINING_DAYS + 1):data.T
            learning = brm_filter_step(learning, records[t].observation;
                inputs=records[t].inputs)
            if t % 10 == 0 || t == data.T
                append!(learning_history, [(; day=t, row...)
                    for row in learning_rows(learning, "Learning filter")])
            end
        end
        (; state_rows, filter_scores, learning_history,
            marginals=learning_rows(learning, "Learning filter"),
            retained_records=length(learning.state.records))
      end
    end
    write_table("filtered_state", filter_result.state_rows)
    write_table("filter_scores", filter_result.filter_scores)
    write_table("learning", filter_result.learning_history)
    marginals = collect(filter_result.marginals)
    push!(summary, (; quantity="learning_retained_records", value=filter_result.retained_records))


    cumulative = fit
    previous_data = training
    for (label, next_data, seed) in (("middle", middle, 202), ("full", full, 203))
        cumulative = phase(root, "Cumulative update to " * label) do p
            bridge = cached(checkpoint_dir, label * "-update-schedule") do
                brm_smc_tune(cumulative, next_data; rng=Xoshiro(seed + 1000),
                    init=zeros(size(cumulative.state.positions, 1)), pilot=:adaptive,
                    pilot_steps=64, grid_start=:independent, options=tuning_options(), progress=p)
            end
            cp = cached(checkpoint_dir, label * "-update-fit") do
                partial = brm_smc_update(cumulative, next_data;
                    options=production_options(bridge), stages=1, progress=p)
                # Resume the partial update before finishing it: this is the
                # same route used after a process restart.
                restored = brm_smc_resume(rk, brm_smc_checkpoint(partial);
                    data=next_data, previous_data, nthreads=Threads.nthreads())
                @assert restored.state.positions == partial.state.positions
                @assert restored.state.weights == partial.state.weights
                brm_smc_checkpoint(brm_smc_fit(restored; progress=p))
            end
            brm_smc_resume(rk, cp; data=next_data, previous_data,
                nthreads=Threads.nthreads())
        end
        previous_data = next_data
    end

    append!(marginals, posterior_rows(cumulative.backend, cumulative.state.positions,
        cumulative.state.weights, "Cumulative SMC"))

    phase(root, "Carried windows") do p
        previous = brm_smc_carry(fit; basis=:fit, max_reflections=16, progress=p)
        for (lo, hi, seed) in ((21, 40, 301), (41, 60, 302))
            window = filtering_window(data, lo, hi)
            cp = cached(checkpoint_dir, "window-$hi") do
                partial = brm_smc_update(previous, filtering_model(window);
                    coordinates=:auto, rng=Xoshiro(seed), particles=PARTICLES,
                    options=tuning_options(), stages=0)
                bridge = brm_smc_tune(previous, filtering_model(window);
                    coordinates=:auto, rng=Xoshiro(seed + 1000),
                    init=zeros(size(partial.state.positions, 1)), pilot=:adaptive,
                    reference=:pilot, pilot_steps=64, grid_start=:independent,
                    options=tuning_options(), progress=p)
                completed = brm_smc_update(previous, filtering_model(window);
                    coordinates=:auto, rng=Xoshiro(seed), particles=PARTICLES,
                    options=production_options(bridge), progress=p)
                brm_smc_checkpoint(completed)
            end
            window_fit = brm_smc_resume(filtering_model(window), cp;
                nthreads=Threads.nthreads())
            mapping = brm_smc_coordinates(window_fit)
            push!(summary, (; quantity="window_$(hi)_active_coordinates", value=size(window_fit.state.positions, 1)))
            previous = brm_smc_carry(window_fit; basis=:fit, max_reflections=16, progress=p)
            if hi == data.T
                rng = Xoshiro(309)
                recovered = reduce(hcat, (brm_smc_recover(window_fit, collect(q); rng)
                    for q in eachcol(window_fit.state.positions)))
                append!(marginals, posterior_rows(RKBRMI(filtering_model(full)),
                    recovered, window_fit.state.weights, "Carried windows"))
            end
        end
    end

    rk_full = RKBRMI(filtering_model(full))
    phase(root, "Independent full-data SMC runs") do p
        full_schedule = cached(checkpoint_dir, "full-batch-schedule") do
            tune_batch(rk_full, 1500, p)
        end
        result = cached(checkpoint_dir, "independent-runs") do
            runs = brm_smc_runs(rk_full; seeds=(501, 502, 503), particles=PARTICLES,
                options=production_options(full_schedule), progress=p)
            (; comparison=runs.comparison, checkpoints=brm_smc_checkpoint.(runs.fits))
        end
        comparison = result.comparison
        write_table("comparison", [(; coordinate=string(comparison.names[i]),
            mean_difference=comparison.mean_difference[i],
            sd_ratio_min=comparison.sd_ratio_min[i], sd_ratio_max=comparison.sd_ratio_max[i])
            for i in eachindex(comparison.names)])
        for (run, cp) in enumerate(result.checkpoints)
            fitted = brm_smc_resume(rk_full, cp)
            append!(marginals, posterior_rows(rk_full, fitted.state.positions,
                fitted.state.weights, "Batch SMC $run"))
        end
    end

    phase(root, "MCMC reference on all observations") do p
        result = cached(checkpoint_dir, "mcmc") do
            problem = rk_logdensity_problem(rk_full; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            d = length(coordinate_names(rk_full.model.layout))
            out = completion_warmup_mcmc(Xoshiro.([601, 602, 603, 604]), problem;
                min_completed=3, n_draws=1000,
                init=(; position=zeros(d), squared_scale=ones(d)), progress=p)
            (; positions=[Matrix(r.posterior_position) for r in out.results],
                divergences=sum(r.n_divergent_samples for r in out.results),
                completion=out.completion)
        end
        n = minimum(size(q, 2) for q in result.positions)
        samples = cat((permutedims(q[:, 1:n]) for q in result.positions)...; dims=3)
        samples = permutedims(samples, (1, 3, 2)) # draws × chains × coordinates
        write_table("diagnostics", [(; max_rhat=maximum(MCMCDiagnosticTools.rhat(samples)),
            min_bulk_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:bulk)),
            min_tail_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:tail)),
            divergences=result.divergences,
            divergence_percent=100result.divergences / sum(size(q, 2) for q in result.positions),
            chains=length(result.positions), draws=sum(size(q, 2) for q in result.positions))])
        write_table("completion", [(; quantity=string(k), value=string(v))
            for (k, v) in pairs(result.completion) if v isa Union{Number,Symbol,Bool}])
        append!(receipt, [(; quantity="omitted_chains", value=string(result.completion.n_omitted)),
            (; quantity="mcmc_seconds", value=string(result.completion.elapsed_seconds))])
        post = rk_draw_tree(rk_full, result.positions, PARAMETERS)
        for parameter in PARAMETERS
            values = vec(collect(TreeActualArray(getproperty(post, parameter); dims=:draw, align=:truncate_min)))
            append!(marginals, [(; method="MCMC", parameter=string(parameter),
                weighted_summary(values, fill(inv(length(values)), length(values)))...)])
        end
    end

    append!(summary, [(; quantity="particles", value=PARTICLES),
        (; quantity="units", value=data.U), (; quantity="days", value=data.T),
        (; quantity="training_days", value=TRAINING_DAYS),
        (; quantity="batch_coordinates", value=size(fit.state.positions, 1))])
    write_table("marginals", marginals)
    write_table("summary", summary)
    write_table("observations", [(; day=t, unit=i, count=data.y[i][t], state=truth.state[i, t])
        for t in 1:data.T for i in 1:data.U])
    write_table("generating", [(; parameter=string(p), value=getproperty(truth, p)) for p in PARAMETERS])
    diagnostic = split(readlines(joinpath(RESULTS, "diagnostics.tsv"))[2], '\t')
    append!(receipt, [(; quantity="seconds_this_invocation", value=string(time() - started_this_invocation)),
        (; quantity="dimension", value=string(size(fit.state.positions, 1))),
        (; quantity="divergent", value=diagnostic[4]),
        (; quantity="retained_chains", value=diagnostic[6]),
        (; quantity="draws", value=diagnostic[7]),
        (; quantity="model_sha256", value=bytes2hex(sha256(read(joinpath(@__DIR__, "model.jl"))))),
        (; quantity="fit_sha256", value=bytes2hex(sha256(read(@__FILE__))))])
    write_table("receipt", receipt)
    write_table("run_info", [(; quantity="julia_version", value=string(VERSION)),
        (; quantity="brm_version", value=string(Base.pkgversion(BayesianRegressionModels))),
        (; quantity="rkppl_version", value=string(Base.pkgversion(ReactiveKernelsPPL))),
        (; quantity="warmuphmc_version", value=string(Base.pkgversion(WarmupHMC))),
        (; quantity="model_sha256", value=bytes2hex(sha256(read(joinpath(@__DIR__, "model.jl"))))),
        (; quantity="fit_sha256", value=bytes2hex(sha256(read(@__FILE__))))])
    if check_growing
        append!(summary, phase(root, "Selective growing coordinates") do p
            cached(checkpoint_dir, "selective-example") do
                selective_example(p)
            end
        end)
        write_table("summary", summary)
    end
    return (; receipt, summary)

end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Pass an empty directory for trusted fit checkpoints.")
    mkpath(ARGS[1])
    record_failure(ARGS[1]) do
        with_progress(:state; description="SMC and filtering documentation") do root
            snapshot_progress(root, get(ENV, "KB_PROGRESS_FILE", joinpath(ARGS[1], "progress.txt"))) do
                main(ARGS[1], root)
            end
        end
    end
end
