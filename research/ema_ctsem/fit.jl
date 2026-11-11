using BayesianRegressionModels, StanBlocks
include("docs_model.jl")
include("../docs_example_fit.jl")
using .DocsExampleFit

function fit_state_space(; n_draws=1000, n_chains=4)
    data = state_space_data()
    r = fit_example(SBBRMI(state_space_model(data); mod=@__MODULE__), data,
        "state-space", joinpath(@__DIR__, "results", "docs"); n_draws, n_chains,
        seed=20261114, adaptive_centering=false, target_acceptance_rate=0.99,
        sources="research/ema_ctsem/docs_model.jl; seed 144; 40 irregular times")
    population_table(r, [
        (; name="mean", logical=:mu, unit="response", description="stationary mean", kind=:scalar, transform=identity),
        (; name="rate", logical=:rate, unit="1/day", description="mean-reversion rate", kind=:scalar, transform=identity),
        (; name="stationary SD", logical=:stationary_sd, unit="response", description="latent process SD", kind=:scalar, transform=identity),
        (; name="observation SD", logical=:sigma, unit="response", description="measurement SD", kind=:scalar, transform=identity),
    ])
    trajectory_table(r, :state; time=data.time, observed=data.y, truth=data.truth)
    predictive_table(r, :y; time=data.time, observed=data.y)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fit_state_space()
end
