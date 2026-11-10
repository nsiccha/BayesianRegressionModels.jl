# Synthetic station field for docs/src/space-time-fields.md.
# julia --startup-file=no --project=test research/space_time_fields/fields.jl
# Plot the committed tables with figures.jl in research/adaptive_centering/plots.
using BayesianRegressionModels, Random, Statistics, LinearAlgebra

# Two smooth features move across a square. This is an analytic test surface,
# not a realization from the fitted Matérn prior.
function true_field(east, north, time)
    positive = 1.8exp(-((east - 25 - 3time)^2 + (north - 62)^2) / (2 * 19^2))
    negative = 1.3exp(-((east - 73)^2 + (north - 25 - 2time)^2) / (2 * 17^2))
    positive - negative
end

function station_data(; seed=20261110)
    rng = Xoshiro(seed)
    positions = [(x + 5rand(rng), y + 5rand(rng))
                 for y in range(10, 85; length=4) for x in range(10, 85; length=6)]
    rows = [(; east=x, north=y, time=t, station=s)
            for (s, (x, y)) in enumerate(positions) for t in (1.0, 3.0, 6.0) for _ in 1:2]
    east, north, time = ([r[k] for r in rows] for k in (:east, :north, :time))
    truth = true_field.(east, north, time)
    noise_sd = 0.25
    y = truth + noise_sd * randn(rng, length(rows))
    (; east, north, time, y, station=[r.station for r in rows], truth, noise_sd,
       neighbors=8, amplitude_scale=1.0, spatial_prior=25.0, temporal_prior=4.0,
       length_prior_sd=0.5, noise_prior=0.5)
end

function station_field_model(data=station_data())
    @brm data begin
        sigma ~ Exponential(noise_prior)
        field ~ 0 + gmrf((east, north), time; neighbors=neighbors)
        sd(field, gmrf((east, north), time)) ~ Exponential(amplitude_scale)
        length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(spatial_prior), length_prior_sd)
        length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(temporal_prior), length_prior_sd)
        y ~ Normal(field, sigma)
    end
end

# Everything above this line also runs when Documenter displays the model.
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, DifferentiationInterface
using WarmupHMC, TreeArrays, Treebars, JSON, MCMCDiagnosticTools
import LogDensityProblems

function fit_station_field(data; n_draws=600, seed=37, progress=nothing,
        checkpoint_dir=nothing, resume=false)
    rk = RKBRMI(station_field_model(data))
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    problems = [rk_logdensity_problem(rk; ad_backend=backend) for _ in 1:4]
    init = [brm_start(rk, (;); base=:prior_median) for _ in problems]
    out = completion_warmup_mcmc([Xoshiro(seed + i) for i in 1:4], problems;
        init, min_completed=3, n_draws, progress, checkpoint_dir, resume)
    (; rk, out)
end

function map_field(rk, draws; nside=21, seed=721)
    axis = collect(range(0.0, 100.0; length=nside))
    grid = [(; east=x, north=y, time=t) for t in (1.0, 3.0, 6.0) for y in axis for x in axis]
    east, north, time = ([r[k] for r in grid] for k in (:east, :north, :time))
    # rng adds the conditional uncertainty at locations the fit did not read.
    field_draws = brm_gmrf_predict(rk, draws, :field; east, north, time, rng=Xoshiro(seed))
    field = TreeData(field_draws, :point, :draw)
    bands = quantile(field, (; q025=0.025, q25=0.25, q50=0.5, q75=0.75, q975=0.975); dims=:draw)
    values = map(k -> collect(TreeActualArray(getproperty(bands, k))), (:q025, :q25, :q50, :q75, :q975))
    uncertainty = collect(TreeActualArray(std(field; dims=:draw)))
    [(; grid[i]..., truth=true_field(east[i], north[i], time[i]),
        q025=values[1][i], q25=values[2][i], q50=values[3][i],
        q75=values[4][i], q975=values[5][i], sd=uncertainty[i]) for i in eachindex(grid)]
end

function write_table(path, rows)
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), '\t'))
        for row in rows
            println(io, join(values(row), '\t'))
        end
    end
end

function diagnostics(out)
    chains = [Matrix(r.posterior_position) for r in out.results]
    n = minimum(size(c, 2) for c in chains)
    # Equal lengths are needed only for the cross-chain diagnostic array.
    draws = cat([permutedims(c[:, 1:n]) for c in chains]...; dims=3)
    draws = permutedims(draws, (1, 3, 2)) # iteration × chain × coordinate
    bulk, rhat = MCMCDiagnosticTools.ess_rhat(draws)
    tail = MCMCDiagnosticTools.ess(draws; kind=:tail)
    divergences = sum(r.n_divergent_samples for r in out.results)
    retained = sum(size(c, 2) for c in chains)
    (; max_rhat=maximum(rhat), min_bulk_ess=minimum(bulk), min_tail_ess=minimum(tail),
       divergences, divergence_percent=100divergences / retained, retained,
       chains=length(chains), omitted_chains=out.completion.omitted_chain_indices,
       diagnostic_draws_per_chain=n)
end

function main()
    BLAS.set_num_threads(1)
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    progress = Treebars.initialize_progress!(:state; description="Space-time field example")
    Treebars.snapshot_progress(progress, get(ENV, "KB_PROGRESS_FILE", joinpath(outdir, "fit.progress"))) do
        @progress progress begin
            @progress "Fit station observations"
            data = station_data()
            checkpoint_dir = get(ENV, "FIELD_CHECKPOINT_DIR", nothing)
            resume = get(ENV, "FIELD_RESUME", "0") == "1"
            seconds = @elapsed fit = fit_station_field(data; progress=__progress__, checkpoint_dir, resume)
            diag = diagnostics(fit.out)
            println("FIT ", JSON.json(diag)); flush(stdout)

            @progress "Map posterior field"
            # The same number of evenly spaced draws from each retained chain.
            per_chain = 100
            selected = [r.posterior_position[:, round.(Int, range(1, size(r.posterior_position, 2); length=per_chain))]
                        for r in fit.out.results]
            draws = reduce(hcat, selected)
            grid = map_field(fit.rk, draws)
            sigma = rk_draw_tree(fit.rk, draws, (:sigma,)).sigma
            noise = parent(quantile(sigma, (; q025=0.025, q50=0.5, q975=0.975); dims=:draw))
            run = (; stations=length(unique(data.station)), observations=length(data.y),
                states=length(unique(zip(data.east, data.north, data.time))),
                times=sort(unique(data.time)), domain_km=100, neighbors=data.neighbors,
                grid_side=21, map_draws=size(draws, 2), noise_truth=data.noise_sd,
                noise_posterior=noise, fit_seconds=seconds, diagnostics=diag,
                grid_rmse=sqrt(mean((r.q50 - r.truth)^2 for r in grid)),
                grid_pointwise_coverage=mean(r.q025 <= r.truth <= r.q975 for r in grid),
                julia=string(VERSION), brm=string(pkgversion(BayesianRegressionModels)),
                warmuphmc=string(pkgversion(WarmupHMC)),
                reactivekernels=string(pkgversion(ReactiveKernels)),
                reactivekernelsppl=string(pkgversion(ReactiveKernelsPPL)),
                enzyme=string(pkgversion(Enzyme)), data_seed=20261110,
                chain_seeds=collect(38:41), prediction_seed=721)
            @progress "Write results"
            write_table(joinpath(outdir, "field_grid.tsv"), grid)
            observations = [(; station=data.station[i], east=data.east[i], north=data.north[i],
                time=data.time[i], y=data.y[i], truth=data.truth[i]) for i in eachindex(data.y)]
            write_table(joinpath(outdir, "observations.tsv"), observations)
            open(joinpath(outdir, "run.json"), "w") do io
                JSON.print(io, run, 2)
            end
            println("RESULTS ", outdir)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch error
        if error isa WarmupHMC.CompletionQuorumError
            for chain in error.outcome.completion.chains
                println(stderr, "Chain ", chain.chain_index, ": ", chain.status)
                isnothing(chain.error) && continue
                showerror(stderr, chain.error.ex)
                println(stderr)
            end
        end
        rethrow()
    end
end
