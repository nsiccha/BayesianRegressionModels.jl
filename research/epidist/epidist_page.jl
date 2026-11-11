# Executable delay example linked from docs/src/renewal.md.
#
# The data are epidist's own: its "approximate inference" article's simulated
# outbreak (lognormal delay, meanlog 1.8, sdlog 0.5, observed on day 25, 200
# cases), prepared by epidist 1.0.0 into its marginal model's rows and handed to
# Stan. `epidist_reference.R` wrote them to results/n200/; the comparison with
# epidist's own program and the fits are in epidist_parity.jl.
using BayesianRegressionModels, JSON

# epidist's marginal rows: identical windows merged, with their count `n`.
function epidist_rows(dir = joinpath(@__DIR__, "results", "n200"))
    standata = JSON.parsefile(joinpath(dir, "epidist_marginal.data.json"))
    column(key) = Float64.(standata[key])
    (; delay = column("Y"),                 # start of the secondary window (delay_lwr)
       swindow = column("vreal3"),          # its width (delay_upr - delay_lwr)
       pwindow = column("vreal2"),          # width of the primary window
       obs_time = column("vreal1"),         # observation time, from the primary window start
       n = Int.(standata["weights"]))       # rows merged into this one
end

function epidist_delay_model(data = epidist_rows())
    @brm data begin
        mu ~ 1
        log(sigma) ~ 1
        effect(mu, Intercept) ~ Normal(1.0, 1.0)
        effect(sigma, Intercept) ~ Normal(-0.7, 0.4)
        delay ~ weighted(double_interval_censored(LogNormal(mu, sigma);
            primary_event = Uniform(0, pwindow), upper = obs_time, interval = swindow),
            fweights(n))
    end
end
