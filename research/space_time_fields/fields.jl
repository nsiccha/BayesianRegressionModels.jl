# Synthetic station field for docs/src/spatial-fields.md.
# julia --startup-file=no --project=test research/space_time_fields/fields.jl
# The docs build renders the small summary tables through figures.jl.
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
using WarmupHMC, TreeArrays, Treebars, MCMCDiagnosticTools, DynamicObjects, Dates, Serialization
using StanBlocks

const FIELD_HARNESS_VERSION = 1
const FIELD_SEED = 37
const FIELD_SETTINGS = (; n_chains=4, min_completed=3, n_draws=600,
    map_draws_per_chain=100, grid_side=21, prediction_seed=721)

function fit_station_field(data; n_draws=600, n_chains=4, min_completed=3,
        seed=37, progress=nothing, checkpoint_dir=nothing, resume=false)
    rk = RKBRMI(station_field_model(data))
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    problems = [rk_logdensity_problem(rk; ad_backend=backend) for _ in 1:n_chains]
    init = [brm_start(rk, (;); base=:prior_median) for _ in problems]
    seconds = @elapsed out = completion_warmup_mcmc(
        [Xoshiro(seed + i) for i in 1:n_chains], problems;
        init, min_completed, n_draws, progress, checkpoint_dir, resume)
    chains = [Matrix(r.posterior_position) for r in out.results]
    chain_lengths = [size(c, 2) for c in chains]
    n = minimum(chain_lengths)
    # DocsFits' draw contract: coordinate × draw × chain. Store equal prefixes;
    # record the original lengths so truncation and omitted chains stay visible.
    draws = cat([c[:, 1:n] for c in chains]...; dims=3)
    (; draws, seconds, divergent=sum(r.n_divergent_samples for r in out.results),
       omitted_chains=out.completion.omitted_chain_indices, dimension=size(draws, 1),
       chain_lengths, retained_draws=sum(chain_lengths), stored_draws_per_chain=n,
       retained_chains=length(chains))
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

# Transitional small TSV summaries until DocsFits is published. These tables
# already follow its summary contract; the page must never sample to obtain them.
function write_table(path, rows)
    cell(x::AbstractFloat) = string(round(x; sigdigits=5))
    cell(x) = replace(string(x), '\t' => ' ', '\n' => ' ')
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), '\t'))
        for row in rows
            println(io, join(cell.(values(row)), '\t'))
        end
    end
    filesize(path) < 100_000 || error("Summary exceeds the 100 KB table budget: $path")
end

function diagnostics(draws)
    samples = permutedims(draws, (2, 3, 1)) # iteration × chain × coordinate
    bulk, rhat = MCMCDiagnosticTools.ess_rhat(samples)
    tail = MCMCDiagnosticTools.ess(samples; kind=:tail)
    (; max_rhat=maximum(rhat), min_bulk_ess=minimum(bulk), min_tail_ess=minimum(tail))
end

function summarize_station_field(data, draws, info; settings=FIELD_SETTINGS)
    rk = RKBRMI(station_field_model(data))
    n = size(draws, 2)
    keep = unique(round.(Int, range(1, n; length=min(n, settings.map_draws_per_chain))))
    mapped_draws = reshape(draws[:, keep, :], size(draws, 1), :)
    grid = map_field(rk, mapped_draws; nside=settings.grid_side, seed=settings.prediction_seed)
    sigma = hcat([vec(collect(TreeActualArray(rk_draw_tree(rk, draws[:, :, c], (:sigma,)).sigma)))
        for c in axes(draws, 3)]...)
    noise_samples = reshape(sigma, size(sigma, 1), size(sigma, 2), 1)
    bulk, rhat = MCMCDiagnosticTools.ess_rhat(noise_samples)
    tail = MCMCDiagnosticTools.ess(noise_samples; kind=:tail)
    q025, q50, q975 = quantile(vec(sigma), [0.025, 0.5, 0.975])
    noise = [(; quantity="Observation noise SD", unit="response units",
        description="Independent Gaussian measurement noise", median=q50, q025, q975,
        rhat=only(rhat), ess_bulk=only(bulk), ess_tail=only(tail))]
    observations = [(; station=data.station[i], east=data.east[i], north=data.north[i],
        time=data.time[i], y=data.y[i], truth=data.truth[i]) for i in eachindex(data.y)]
    data_table = [(;quantity=string(k),value=string(v)) for (k,v) in (
        "Square side (km)" => 100, "Stations" => length(unique(data.station)),
        "Observation times" => join(Int.(sort(unique(data.time))), ", "),
        "Observations" => length(data.y),
        "Position–time states" => length(unique(zip(data.east,data.north,data.time))),
        "Neighbors" => data.neighbors, "Generating noise SD" => data.noise_sd)]
    diag = diagnostics(draws)
    diagnostic_table = [(; quantity=string(k),value=string(v)) for (k,v) in (
        "Retained chains" => info.retained_chains,
        "Omitted chain indices" => (isempty(info.omitted_chains) ? "none" : join(info.omitted_chains, ", ")),
        "Retained draws before equal-length truncation" => info.retained_draws,
        "Draws per chain used in summaries" => info.stored_draws_per_chain,
        "Maximum R-hat, all sampler coordinates" => round(diag.max_rhat; sigdigits=4),
        "Minimum bulk ESS" => round(diag.min_bulk_ess; sigdigits=4),
        "Minimum tail ESS" => round(diag.min_tail_ess; sigdigits=4),
        "Divergences in all retained draws" => info.divergent)]
    checks = [(; quantity=string(k),value=string(v)) for (k,v) in (
        "Grid root mean squared error" => round(sqrt(mean((r.q50-r.truth)^2 for r in grid)); sigdigits=4),
        "Grid points inside pointwise 95% intervals (%)" => round(100mean(r.q025 <= r.truth <= r.q975 for r in grid); sigdigits=4),
        "Draws used for mapping" => size(mapped_draws, 2))]
    (; field_grid=grid, observations, data=data_table, diagnostics=diagnostic_table,
       noise, map_checks=checks)
end

function run_receipt(key, settings; started=now())
    root = normpath(joinpath(@__DIR__, "..", ".."))
    git(args...) = readchomp(`git -C $root $args`)
    rows = [(; key=string(k),value=string(v)) for (k,v) in (
        "key" => key, "commit" => git("rev-parse","HEAD"),
        "uncommitted_changes" => (isempty(git("status","--porcelain")) ? "none" : "yes"),
        "started" => started, "host" => "a Linux fleet host, $(Threads.nthreads()) Julia threads",
        "julia" => VERSION, "harness" => "Station field RK/Enzyme/WarmupHMC v$(FIELD_HARNESS_VERSION)",
        "settings" => settings, "seed" => FIELD_SEED)]
    for package in (BayesianRegressionModels, WarmupHMC, ReactiveKernels, ReactiveKernelsPPL, Enzyme)
        push!(rows, (;key=string(nameof(package)),value=string(pkgversion(package))))
    end
    rows
end

function main()
    BLAS.set_num_threads(1)
    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    data = station_data()
    settings = FIELD_SETTINGS
    # Same content-key tuple as DocsFits 0.1; data includes simulation settings.
    program = BayesianRegressionModels.stan_code(SBBRMI(station_field_model(data)))
    key = DynamicObjects.canonical_hash((program, data, settings, FIELD_SEED, FIELD_HARNESS_VERSION))
    receipt_path = joinpath(outdir, "receipt.tsv")
    if isfile(receipt_path)
        previous = Dict(k => v for (k,v) in (split(line, '\t'; limit=2) for line in readlines(receipt_path)[2:end]))
        if get(previous,"key","") == string(key) && get(previous,"status","") == "complete"
            for name in ("field_grid", "observations", "data", "diagnostics", "noise", "map_checks")
                isfile(joinpath(outdir, name*".tsv")) || error("Cached fit lacks $name.tsv; restore its summaries")
            end
            println("CACHED ", key)
            return
        end
    end
    receipt = run_receipt(key, settings)
    write_table(receipt_path, [receipt; (;key="status",value="started")])
    progress = Treebars.initialize_progress!(:state; description="Space-time field example")
    Treebars.snapshot_progress(progress, get(ENV, "KB_PROGRESS_FILE", joinpath(outdir, "fit.progress"))) do
        @progress progress begin
            @progress "Fit station observations"
            local_draws = joinpath(outdir, "fit.jls") # git-ignored; never a docs input
            cached = isfile(local_draws) ? deserialize(local_draws) : nothing
            if !isnothing(cached) && cached.key == key
                sampled, info, receipt = cached.sampled, cached.info, cached.receipt
            else
                sampled = fit_station_field(data; n_draws=settings.n_draws, n_chains=settings.n_chains,
                    min_completed=settings.min_completed, seed=FIELD_SEED, progress=__progress__,
                    checkpoint_dir=get(ENV,"FIELD_CHECKPOINT_DIR",nothing), resume=get(ENV,"FIELD_RESUME","0") == "1")
                info = Base.structdiff(sampled, (;draws=nothing))
                serialize(local_draws, (;key, sampled, info, receipt))
            end
            @progress "Summarize posterior field"
            tables = summarize_station_field(data, sampled.draws, info; settings)
            @progress "Write summaries"
            for (name, rows) in pairs(tables)
                write_table(joinpath(outdir, string(name)*".tsv"), rows)
            end
            for (name,value) in pairs(info)
                push!(receipt,(;key=string(name),value=string(value)))
            end
            write_table(receipt_path, [receipt; (;key="status",value="complete")])
            println("RESULTS ", outdir, " key=", key)
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
