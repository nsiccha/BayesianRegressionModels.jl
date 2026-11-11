using Dates
import DynamicObjects

const FILTERING_SEED = 20261110
const FILTERING_SETTINGS = (
    particles=1024, training_days=20, middle_day=40,
    moves=3, max_tree_depth=7, pilot_steps=64,
    reference=:pilot, grid_start=:independent, max_reflections=16,
    training_seed=201, training_pilot_seed=1201,
    cumulative_pilot_seeds=(1202, 1203), window_seeds=(301, 302),
    window_pilot_seeds=(1301, 1302), recovery_seed=309,
    filter_seed=401, filter_particles=64,
    learning_seed=402, learning_members=256, learning_particles=128, window_length=10,
    batch_seeds=(501, 502, 503), batch_pilot_seed=1500,
    mcmc_seeds=(601, 602, 603, 604), min_completed=3, n_draws=1000,
    selective_seeds=(701, 702, 703), harness_version=1,
)

function filtering_receipt(data, directory)
    # The authoring program is the input to the ReactiveKernels compiler used
    # here; its data, all numerical settings and seed also enter the key.
    key = DynamicObjects.canonical_hash((read(joinpath(@__DIR__, "model.jl"), String),
        data, FILTERING_SETTINGS, FILTERING_SEED, 1))
    identity = joinpath(directory, "content-key.txt")
    isfile(identity) && strip(read(identity, String)) != key && error(
        "Filtering checkpoint content changed. Use a new checkpoint directory.")
    write(identity, key * "\n")
    origin = joinpath(directory, "started.txt")
    isfile(origin) || write(origin, string(now(UTC)) * " UTC\n")
    git(args...) = readchomp(`git -C $(@__DIR__) $args`)
    rows = [
        (; quantity="key", value=key),
        (; quantity="commit", value=git("rev-parse", "HEAD")),
        (; quantity="uncommitted_changes", value=isempty(git("status", "--porcelain")) ? "none" : "yes"),
        (; quantity="started", value=strip(read(origin, String))),
        (; quantity="host", value="a Linux fleet host; $(Threads.nthreads()) Julia threads"),
        (; quantity="julia", value=string(VERSION)),
        (; quantity="harness", value="SMC, state filtering and completion MCMC; version 1"),
        (; quantity="settings", value=string(FILTERING_SETTINGS)),
        (; quantity="seed", value=string(FILTERING_SEED)),
    ]
    for pkg in (BayesianRegressionModels, ReactiveKernelsPPL, SequentialMonteCarlo, WarmupHMC)
        push!(rows, (; quantity=string(nameof(pkg)), value=string(Base.pkgversion(pkg))))
    end
    rows
end
