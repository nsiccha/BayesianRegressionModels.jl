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
using WarmupHMC, TreeArrays, Tables, Treebars, MCMCDiagnosticTools, DynamicObjects, Dates, Serialization, Printf
using StanBlocks, Distributions

const FIELD_HARNESS_VERSION = 2
const FIELD_SEED = 37
const FIELD_SETTINGS = (; n_chains=4, min_completed=3, n_draws=4000,
    map_draws_per_chain=100, grid_side=21, prediction_seed=721)

function fit_station_field(data; n_draws=4000, n_chains=4, min_completed=3,
        seed=37, progress=nothing, checkpoint_dir=nothing, resume=false)
    # One RK log density per chain, prior-median starts, a completion quorum.
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

# Two significant digits of the 95% interval's width: the last digit shown is
# one the posterior resolves.
function resolved(x, width)
    digits = 1 - floor(Int, log10(width))
    digits > 0 ? round(x; digits) : round(Int, round(x; digits))
end
three_decimals(x) = Printf.@sprintf("%.3f", x)

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
    # Field values on the response scale, to the hundredth: the smallest
    # pointwise posterior SD on this grid is several hundredths.
    r2(x) = round(x; digits=2)
    [(; grid[i]..., truth=r2(true_field(east[i], north[i], time[i])),
        q025=r2(values[1][i]), q25=r2(values[2][i]), q50=r2(values[3][i]),
        q75=r2(values[4][i]), q975=r2(values[5][i]), sd=round(uncertainty[i]; digits=3))
     for i in eachindex(grid)]
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

"One scalar per draw and chain (draw × chain) for each named model quantity."
function scalar_draws(rk, draws)
    nd, nc = size(draws, 2), size(draws, 3)
    sigma = reduce(hcat, [vec(collect(TreeActualArray(rk_draw_tree(rk, draws[:, :, c], (:sigma,)).sigma)))
                          for c in 1:nc])
    constrained = [ReactiveKernelsPPL.constrain(rk.model.layout, draws[:, j, c]) for j in 1:nd, c in 1:nc]
    term = only(k for k in keys(first(constrained)) if startswith(String(k), "gmrf"))
    field(name) = [getproperty(c[term], name) for c in constrained]
    (; sigma, amplitude=field(:sigma), spatial=field(:spatial_length), temporal=field(:temporal_length))
end

"The DocsFits quantity row of one scalar: median, 95% interval, split R-hat, bulk and tail ESS."
function quantity_row(quantity, unit, description, x::Matrix{Float64})
    tree = TreeData([TreeData(x[:, c], :draw) for c in axes(x, 2)], :chain)
    q = only(Tables.rows(TreeTable(quantile(tree, (; q025=0.025, q50=0.5, q975=0.975); dims=(:draw, :chain)))))
    samples = reshape(x, size(x, 1), size(x, 2), 1)
    bulk, rhat = MCMCDiagnosticTools.ess_rhat(samples)
    tail = MCMCDiagnosticTools.ess(samples; kind=:tail)
    width = q.q975 - q.q025
    (; quantity, unit, description, median=resolved(q.q50, width), q025=resolved(q.q025, width),
       q975=resolved(q.q975, width), rhat=three_decimals(only(rhat)),
       ess_bulk=round(Int, only(bulk)), ess_tail=round(Int, only(tail)))
end

# The prior's median and central 95% range, to two significant digits.
function prior_text(d)
    two(x) = (y = round(x; sigdigits=2); isinteger(y) ? string(Int(y)) : string(y))
    lo, mid, hi = quantile.(d, (0.025, 0.5, 0.975))
    "prior median $(two(mid)), 95% $(two(lo))–$(two(hi))"
end

function summarize_station_field(data, draws, info; settings=FIELD_SETTINGS)
    rk = RKBRMI(station_field_model(data))
    n = size(draws, 2)
    keep = unique(round.(Int, range(1, n; length=min(n, settings.map_draws_per_chain))))
    mapped_draws = reshape(draws[:, keep, :], size(draws, 1), :)
    grid = map_field(rk, mapped_draws; nside=settings.grid_side, seed=settings.prediction_seed)
    s = scalar_draws(rk, draws)
    quantities = [
        quantity_row("Observation noise SD", "response units",
            "Generated with $(data.noise_sd); " * prior_text(Exponential(data.noise_prior)), s.sigma),
        quantity_row("Field SD", "response units",
            uppercasefirst(prior_text(Exponential(data.amplitude_scale))), s.amplitude),
        quantity_row("Spatial length", "km",
            uppercasefirst(prior_text(LogNormal(log(data.spatial_prior), data.length_prior_sd))), s.spatial),
        quantity_row("Temporal length", "time units",
            uppercasefirst(prior_text(LogNormal(log(data.temporal_prior), data.length_prior_sd))), s.temporal)]
    observations = [(; station=data.station[i], east=round(data.east[i]; digits=1),
        north=round(data.north[i]; digits=1), time=data.time[i], y=round(data.y[i]; digits=2),
        truth=round(data.truth[i]; digits=2)) for i in eachindex(data.y)]
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
        "Maximum R-hat, all sampler coordinates" => three_decimals(diag.max_rhat),
        "Minimum bulk ESS" => round(Int, diag.min_bulk_ess),
        "Minimum tail ESS" => round(Int, diag.min_tail_ess),
        "Divergences in all retained draws" => info.divergent,
        "Sampling time (s)" => round(Int, info.seconds))]
    # Distance from each grid point to its nearest station.
    sites = unique(zip(data.east, data.north))
    nearest(r) = minimum(hypot(r.east - x, r.north - y) for (x, y) in sites)
    near = [r.sd for r in grid if nearest(r) < 5]
    far = [r.sd for r in grid if nearest(r) > 12]
    checks = [(; quantity=string(k),value=string(v)) for (k,v) in (
        "Grid root mean squared error" => round(sqrt(mean((r.q50-r.truth)^2 for r in grid)); digits=2),
        "Grid points inside pointwise 95% intervals (%)" => round(Int, 100mean(r.q025 <= r.truth <= r.q975 for r in grid)),
        "Mean field SD, grid points within 5 km of a station" => round(mean(near); digits=2),
        "Mean field SD, grid points over 12 km from every station" => round(mean(far); digits=2),
        "Draws used for mapping" => size(mapped_draws, 2))]
    (; field_grid=grid, observations, data=data_table, diagnostics=diagnostic_table,
       quantities, map_checks=checks)
end

function run_receipt(key, settings; started=now())
    root = normpath(joinpath(@__DIR__, "..", ".."))
    git(args...) = readchomp(`git -C $root $args`)
    rows = [(; key=string(k),value=string(v)) for (k,v) in (
        "key" => key, "commit" => git("rev-parse","HEAD"),
        "uncommitted_changes" => (isempty(git("status","--porcelain")) ? "none" : "yes"),
        "started" => Dates.format(started, "yyyy-mm-ddTHH:MM:SS"),
        "host" => "a Linux fleet host, $(Threads.nthreads()) Julia threads",
        "julia" => VERSION, "harness" => "Station field RK/Enzyme/WarmupHMC v$(FIELD_HARNESS_VERSION)",
        "sampler" => "WarmupHMC completion_warmup_mcmc on the RKBRMI log density, AutoEnzyme reverse mode",
        "settings" => settings, "seed" => FIELD_SEED)]
    for package in (BayesianRegressionModels, WarmupHMC, ReactiveKernels, ReactiveKernelsPPL, Enzyme, TreeArrays)
        push!(rows, (;key=string(nameof(package)),value=string(pkgversion(package))))
    end
    rows
end

read_receipt(path) = Dict(String(k) => String(v)
    for (k, v) in (split(line, '\t'; limit=2) for line in readlines(path)[2:end]))

# A docs fit runs once and is rerun only when its model, data or settings
# change, at most about once a month. A changed key within 30 days of the last
# start needs BRM_DOCS_FIT_FORCE=1.
function guard_rerun(receipt_path, key)
    isfile(receipt_path) || return
    previous = read_receipt(receipt_path)
    get(previous, "key", "") == string(key) && return
    started = tryparse(DateTime, first(get(previous, "started", ""), 19))
    if !isnothing(started) && now() - started < Day(30) && get(ENV, "BRM_DOCS_FIT_FORCE", "0") != "1"
        error("$receipt_path records a fit started $(previous["started"]) under another key; " *
              "a docs fit is rerun at most about once a month. Set BRM_DOCS_FIT_FORCE=1 to rerun it now.")
    end
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
    guard_rerun(receipt_path, key)
    # Draws stay outside git, where they were computed; only the summaries are committed.
    cache = joinpath(get(ENV, "BRM_DOCS_CACHE", joinpath(tempdir(), "brm-docs-examples")),
                     "space_time_fields", string(key))
    mkpath(cache)
    local_draws = joinpath(cache, "draws.jls")
    cached = isfile(local_draws) ? deserialize(local_draws) : nothing
    receipt = isnothing(cached) ? run_receipt(key, settings) : cached.receipt
    write_table(receipt_path, [receipt; (;key="status",value="started")])
    progress = Treebars.initialize_progress!(:state; description="Space-time field example")
    Treebars.snapshot_progress(progress, get(ENV, "KB_PROGRESS_FILE", joinpath(cache, "fit.progress"))) do
        @progress progress begin
            @progress "Fit station observations"
            if !isnothing(cached) && cached.key == key
                sampled, info = cached.sampled, cached.info
                println("CACHED ", local_draws)
            else
                sampled = fit_station_field(data; n_draws=settings.n_draws, n_chains=settings.n_chains,
                    min_completed=settings.min_completed, seed=FIELD_SEED, progress=__progress__,
                    checkpoint_dir=get(ENV,"FIELD_CHECKPOINT_DIR",nothing), resume=get(ENV,"FIELD_RESUME","0") == "1")
                info = Base.structdiff(sampled, (;draws=nothing))
                serialize(local_draws, (;key, sampled, info, receipt))
                println("DRAWS_CACHE ", local_draws)
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
    catch err
        if err isa WarmupHMC.CompletionQuorumError
            for chain in err.outcome.completion.chains
                println(stderr, "Chain ", chain.chain_index, ": ", chain.status)
                isnothing(chain.error) && continue
                showerror(stderr, chain.error.ex)
                println(stderr)
            end
        end
        rethrow()
    end
end
