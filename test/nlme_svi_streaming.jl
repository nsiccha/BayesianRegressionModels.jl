include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_streaming_fixture.jl"))
using LinearAlgebra, Serialization, WarmupHMC

@stestset "streaming SVI matches unit identities and replaces priors" begin
    data = svi_stream_data(Xoshiro(31), ["a", "b", "c"], 3)
    initial = brm_svi_fit(svi_stream_model(data); rng=Xoshiro(32), SVI_STREAM_QUICK...,
        centering=nothing)
    # Set a recognizable, nonstandard carried density. This checks the exact
    # streaming target, not just the optimizer's response to a warm start.
    q = initial.diagnostics.q
    q.μ .= 0.6
    q.ρ .= log(0.4)
    for i in eachindex(q.M)
        q.M[i] .= 0.2i
        q.Λ[i] .= 2.0 + i
        q.U[i] .= sqrt(2.0 + i)
    end
    state = brm_svi_checkpoint(initial)
    before = deepcopy(state)
    next_data = svi_stream_data(Xoshiro(33), ["c", "a", "new"], 2)
    update = brm_svi_fit(state, svi_stream_model(next_data);
        rng=Xoshiro(34), SVI_STREAM_QUICK...)
    @test update.diagnostics.centering === nothing
    m = update.diagnostics.model
    carried = brm_svi_checkpoint(update)
    @test svi_stream_same_factors(state, before)
    @test carried.q == state.q
    for label in ("a", "b", "c")
        @test carried.units[label].M == state.units[label].M
        @test carried.units[label].Λ == state.units[label].Λ
    end
    @test carried.units["new"].M == [0.0]
    @test carried.units["new"].Λ == ones(1, 1)
    @test carried.updates == 1
    t0 = BRM._brm_svi_target(m)
    t, warm = BRM._brm_svi_initialize(t0, state, state.q.family;
        start=nothing, inits=nothing, global_scale=0.1, unit_scale=1.0)
    g = fill(0.9, length(t.globals))
    value, grad = BRM._brm_svi_global_term(t, g)
    @test value ≈ sum(logpdf.(Normal(0.6, 0.4), g))
    @test grad ≈ -(g .- 0.6) ./ 0.4^2
    for (i, label) in enumerate(m.view.levels)
        z = [0.7]
        gg0, gz0 = zeros(length(g)), zeros(1)
        gg, gz = similar(gg0), similar(gz0)
        oldvalue = BRM._brm_svi_unit_term!(gg0, gz0, t0, i, g, z)
        newvalue = BRM._brm_svi_unit_term!(gg, gz, t, i, g, z)
        factor = carried.units[label]
        μ, precision = only(factor.M), only(factor.Λ)
        @test newvalue - oldvalue ≈ logpdf(Normal(μ, inv(sqrt(precision))), only(z)) -
            logpdf(Normal(), only(z))
        @test gg ≈ gg0
        @test gz - gz0 ≈ -precision .* (z .- μ) .+ z
    end
    returned = brm_svi_fit(carried, svi_stream_model(data); SVI_STREAM_QUICK...)
    @test brm_svi_checkpoint(returned).units["b"].M == state.units["b"].M
    raw_update = brm_svi_fit(initial, next_data; SVI_STREAM_QUICK...)
    @test svi_stream_same_factors(brm_svi_checkpoint(raw_update), carried)
    @test_throws "model source changed" brm_svi_fit(state, svi_stream_scaled_model(next_data);
        SVI_STREAM_QUICK...)
    @test_throws "family and rank" brm_svi_fit(state, svi_stream_model(next_data);
        family=LowRankGlobal(1), SVI_STREAM_QUICK...)
end

@stestset "streaming SVI freezes preprocessing and refuses refitting it" begin
    data = svi_stream_data(Xoshiro(41), ["a", "b", "c"], 5)
    first_fit = brm_svi_fit(svi_stream_scaled_model(data); SVI_STREAM_QUICK...)
    state = brm_svi_checkpoint(first_fit)
    next_data = svi_stream_data(Xoshiro(42), ["a", "new"], 3; offset=5.0)
    frozen = brm_svi_fit(state, svi_stream_scaled_model(next_data); SVI_STREAM_QUICK...)
    @test brm_svi_checkpoint(frozen).preprocessing.constants == state.preprocessing.constants
    bound = BRM._rk_frozen_values(frozen.diagnostics.model.backend, [:xbar]).values
    @test bound[:xbar] == state.preprocessing.constants[:xbar]
    @test bound[:xbar] != mean(next_data.x)
    @test brm_fitted_state(frozen.diagnostics.model.backend).constants == state.preprocessing.constants
    fresh = RKBRMI(svi_stream_scaled_model(next_data))
    @test_throws "preprocessing changed" brm_svi_fit(state, fresh; SVI_STREAM_QUICK...)
    thawed = reprocess(first_fit.diagnostics.model.backend, next_data;
        resample_groups=(:subject,), freeze_constants=false)
    @test_throws "preprocessing changed" brm_svi_fit(state, thawed; SVI_STREAM_QUICK...)
end

@stestset "streaming SVI fresh-process resume" begin
    for dense_limit in (64, 0)
    data = svi_stream_data(Xoshiro(51), ["a", "b", "c"], 4)
    first_fit = brm_svi_fit(svi_stream_model(data); rng=Xoshiro(52),
        max_iterations=25, ndraws=0, unit_dense_limit=dense_limit, unit_rank=1)
    state = brm_svi_checkpoint(first_fit)
    next_data = svi_stream_data(Xoshiro(53), ["b", "c", "new"], 4)
    in_process = brm_svi_fit(state, svi_stream_model(next_data); rng=Xoshiro(54),
        max_iterations=25, ndraws=0)
    mktempdir() do dir
        input, output = joinpath(dir, "input.jls"), joinpath(dir, "output.jls")
        serialize(input, (; state, data=next_data))
        script = joinpath(@__DIR__, "nlme_svi_streaming_process.jl")
        run(`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $script $input $output`)
        resumed = deserialize(output)
        @test svi_stream_same_factors(resumed, brm_svi_checkpoint(in_process))
    end
    end
end

@stestset "streaming SVI panel factors and constant window work" begin
    labels = ["s$i" for i in 1:8]
    windows = [svi_stream_data(Xoshiro(70 + i), labels, 12) for i in 1:6]
    # This oracle compares packed unit means and precisions across windows, of the
    # independent factorization (the carried coupling is checked against NUTS below).
    controls = (; centering=nothing, coupling=nothing, max_iterations=4000, window=50,
        min_windows=20, patience=8, polish_windows=15, batch_size=8, ndraws=0)
    first_fit = brm_svi_fit(svi_stream_model(first(windows)); rng=Xoshiro(80), controls...)
    state = brm_svi_checkpoint(first_fit)
    times, work, sizes = Float64[], Int[], Int[]
    # Warm the streaming specialization before timing. This branch is
    # discarded; the actual stream still absorbs each observation once.
    brm_svi_fit(state, svi_stream_model(windows[2]); rng=Xoshiro(180),
        max_iterations=1, ndraws=0)
    for i in 2:length(windows)
        cost = @elapsed probe = brm_svi_fit(state, svi_stream_model(windows[i]);
            rng=Xoshiro(180 + i), max_iterations=200, min_windows=1000,
            window=50, batch_size=8, ndraws=0)
        push!(times, cost)
        push!(work, probe.diagnostics.unit_gradient_calls)
        elapsed = @elapsed fit = brm_svi_fit(state, svi_stream_model(windows[i]);
            rng=Xoshiro(80 + i), controls...)
        state = brm_svi_checkpoint(fit)
        io = IOBuffer()
        serialize(io, state)
        push!(sizes, position(io))
        println("SVI_STREAM_WINDOW window=$i accumulated_rows=$(96i) new_rows=96 " *
            "fixed_200_step_seconds=$cost fixed_unit_gradient_calls=$(last(work)) " *
            "total_seconds=$elapsed optimization_seconds=$(fit.diagnostics.seconds) " *
            "iterations=$(fit.diagnostics.iterations) " *
            "unit_gradient_calls=$(fit.diagnostics.unit_gradient_calls) checkpoint_bytes=$(last(sizes))")
    end
    all_data = (; subject=reduce(vcat, [w.subject for w in windows]),
        x=reduce(vcat, [w.x for w in windows]), y=reduce(vcat, [w.y for w in windows]))
    # The confounded intercept/scale batch has a much slower mean drift than
    # the warm streaming windows. Its early stochastic ELBO plateau is not
    # an accurate reference: a bounded exact-expectation control put its
    # largest mean error at 0.964 after 2050 iterations, versus 0.014 after
    # 25000. Give the batch reference a longer exploration and polish.
    batch = brm_svi_fit(svi_stream_model(all_data); centering=nothing, coupling=nothing,
        rng=Xoshiro(90), batch_size=8, ndraws=0, max_iterations=30000, window=100, min_windows=200,
        patience=20, polish_windows=50)
    batch_state = brm_svi_checkpoint(batch)
    global_error = maximum(abs, state.q.μ - batch_state.q.μ)
    unit_error = maximum(maximum(abs, state.units[l].M - batch_state.units[l].M) for l in labels)
    global_sd_ratio = exp.(state.q.ρ - batch_state.q.ρ)
    unit_sd_ratios = [sqrt(only(batch_state.units[l].Λ) / only(state.units[l].Λ)) for l in labels]
    println("SVI_STREAM_BATCH global_mean_error=$global_error unit_mean_error=$unit_error " *
        "global_sd_ratio=$global_sd_ratio unit_sd_ratios=$unit_sd_ratios " *
        "reference_iterations=$(batch.diagnostics.iterations)")
    @test global_error < 0.2
    @test unit_error < 0.3
    @test all(0.8 .< global_sd_ratio .< 1.2)
    @test all(0.8 .< unit_sd_ratios .< 1.2)
    @test work == fill(3200, 5)
    @test maximum(sizes) == minimum(sizes)
end

# A coupled window's units carry their regression on the globals they were coupled
# to (src/nlme_svi_coupling.jl), so the next window's prior keeps the global/unit
# dependence. The panel above, all windows' draws against four-chain NUTS on every
# window's rows, without centering so that the subjects' effects are coupled.
# Measured (seeds below, BRM 0.2.283): the intercept's sd 0.71 of NUTS's streamed
# coupled, 0.70 for a coupled batch fit, 0.13 streamed and batch without coupling;
# the group sd's log scale 0.38 streamed coupled, 0.50 batch coupled, 0.26 streamed
# without coupling.
@stestset "streaming SVI carries the coupling: streamed intervals against NUTS" begin
    labels = ["s$i" for i in 1:8]
    windows = [svi_stream_data(Xoshiro(70 + i), labels, 12) for i in 1:6]
    all_data = (; subject=reduce(vcat, [w.subject for w in windows]),
        x=reduce(vcat, [w.x for w in windows]), y=reduce(vcat, [w.y for w in windows]))
    rk = RKBRMI(svi_stream_model(all_data))
    problems = [rk_logdensity_problem(rk; ad_backend=NLME_AD) for _ in 1:4]
    init = brm_start(rk; chains=4, rng=Xoshiro(3))
    reference = completion_warmup_mcmc([Xoshiro(10 + c) for c in 1:4], problems;
        min_completed=4, n_draws=1000, blas_threads=1, init)
    @test reference.completion.n_completed == 4
    R = reduce(hcat, [Matrix{Float64}(r.posterior_position) for r in reference.results])
    function streamed(coupling)
        controls = (; centering=nothing, coupling, max_iterations=4000, window=50,
            min_windows=20, patience=8, polish_windows=15, batch_size=8, ndraws=4000)
        state = brm_svi_checkpoint(brm_svi_fit(svi_stream_model(first(windows));
            rng=Xoshiro(80), controls...))
        fit = nothing
        for i in 2:length(windows)
            fit = brm_svi_fit(state, svi_stream_model(windows[i]); rng=Xoshiro(80 + i),
                controls...)
            state = brm_svi_checkpoint(fit)
        end
        fit, state
    end
    coupled, coupled_state = streamed(LinearCoupling())
    independent, independent_state = streamed(nothing)
    @test all(u -> hasproperty(u, :coupling), values(coupled_state.units))
    @test !any(u -> hasproperty(u, :coupling), values(independent_state.units))
    globals = coupled.diagnostics.globals
    intercept = findfirst(==(:a), coupled.coordinates[globals])
    ratio(f) = [std(f.draws[j, :]) / std(R[j, :]) for j in globals]
    offset(f) = [abs(mean(f.draws[j, :]) - mean(R[j, :])) / std(R[j, :]) for j in globals]
    rc, ri = ratio(coupled), ratio(independent)
    println("SVI_STREAM_COUPLING globals=$(coupled.coordinates[globals]) coupled_sd_ratio=$rc " *
        "independent_sd_ratio=$ri coupled_mean_error=$(offset(coupled))")
    @test all(rc .> ri)
    @test rc[intercept] > 0.5 > 2ri[intercept]
    @test all(offset(coupled) .< 1)
end
