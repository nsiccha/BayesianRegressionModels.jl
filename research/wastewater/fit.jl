include("cdc_ww_inference.jl")
include("data.jl")
include("../docs_example_fit.jl")
using .DocsExampleFit

function fit_wastewater(; n_draws=1000, n_chains=4)
    data = wastewater_public_data()
    r = fit_example(SBBRMI(wastewater_public_model(data); mod=@__MODULE__), data,
        "wastewater", joinpath(@__DIR__, "results"); n_draws, n_chains,
        seed=20261111, target_acceptance_rate=0.95,
        sources="CDCgov/ww-inference-model 3e92fea29a0b96212e1690dbdcaeb8abfe19cc4f; public simulated site 1 and state admissions; 2023-09-01 to 2023-10-05")
    population_table(r, [
        (; name="initial reference R", logical=:log_ru_week, unit="", description="unadjusted reference reproduction number at the seeded origin", kind=:intercept, transform=exp),
        (; name="feedback", logical=:gamma, unit="", description="infection feedback coefficient", kind=:scalar, transform=identity),
        (; name="subpopulation SD", logical=:sigma_delta, unit="log R", description="daily AR innovation scale", kind=:scalar, transform=identity),
        (; name="shedding peak", logical=:t_peak, unit="days", description="time of peak shedding after infection", kind=:scalar, transform=identity),
    ])
    trajectory_table(r, :log_ru; time=collect((1 - data.uot):(data.nt + data.ht)), transform=exp)
    predictive_table(r, :ww_log_conc; time=data.ww_time .- data.uot, observed=data.ww_log_conc)
end

if abspath(PROGRAM_FILE) == @__FILE__
    fit_wastewater()
end
