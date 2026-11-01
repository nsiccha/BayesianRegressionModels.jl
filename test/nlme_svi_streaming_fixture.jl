using BayesianRegressionModels, Random, Statistics

# A public synthetic long-format panel with an estimated random-effect scale.
# The prior-replacement checks below are exact even though this likelihood
# is not jointly Gaussian in the scale and standardized unit draws.
function svi_stream_model(data)
    @brm data begin
        a ~ Normal(0, 2)
        mu ~ 0 + (1 | subject)
        sd(mu, subject) ~ LogNormal(log(0.7), 0.4)
        mean_y = a + mu
        y ~ Normal(mean_y, 0.5)
    end
end

function svi_stream_scaled_model(data)
    @brm data begin
        a ~ Normal(0, 2)
        mu ~ 0 + standardize(x) + (1 | subject)
        sd(mu, subject) ~ LogNormal(log(0.7), 0.4)
        xbar = constant(mean(x))
        mean_y = a + mu + xbar
        y ~ Normal(mean_y, 0.5)
    end
end

function svi_stream_data(rng, labels, n; offset=0.0)
    subject = repeat(collect(labels), inner=n)
    x = randn(rng, length(subject)) .+ offset
    effects = Dict(label => 0.4sin(sum(Int, codeunits(label))) for label in labels)
    y = [0.8 + effects[label] + 0.5randn(rng) for label in subject]
    (; subject, x, y)
end

const SVI_STREAM_QUICK = (; max_iterations=0, ndraws=0)

function svi_stream_same_factors(a, b)
    a.q == b.q && a.globals == b.globals && a.units == b.units &&
        a.source_id == b.source_id && a.updates == b.updates
end
