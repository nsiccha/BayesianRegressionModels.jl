include("renewal.jl")
include("../docs_example_fit.jl")
using .DocsExampleFit

function fit_renewal(; n_draws=1000, n_chains=4)
    data = renewal_single_data()
    r = fit_example(SBBRMI(renewal_single_model(data); mod=@__MODULE__), data,
        "renewal", joinpath(@__DIR__, "results", "docs"); n_draws, n_chains,
        seed=20261112, sources="research/epi_renewal/renewal.jl; simulate_single(seed=132); 56 days",
        target_acceptance_rate=0.99, adaptive_centering=false)
    population_table(r, [
        (; name="initial infections", logical=:log_I0, unit="people/day", description="incidence at the seeded origin", kind=:scalar, transform=exp),
        (; name="initial R", logical=:log_R, unit="", description="initial reproduction number", kind=:intercept, transform=exp),
        (; name="cluster", logical=:cluster, unit="", description="negative-binomial overdispersion scale", kind=:scalar, transform=identity),
    ])
    truth = simulate_single()
    trajectory_table(r, :Y; time=data.time, observed=data.cases, truth=truth.Y_t)
    predictive_table(r, :cases; time=data.time, observed=data.cases)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fit_renewal()
end
