# Structure first, priors last at each lexical scope. Compose all structural
# builders before the prior builders; bind data only after composition.
module StructureFirstExamples
using BayesianRegressionModels

const predictor = @brm begin
    mu ~ 1 + (1 | subject)
end
const observations = @brm begin
    @plate for i in eachindex(y)
        y[i] ~ Normal(mu[i], sigma)
    end
end
const population_priors = @brm begin
    effect(mu, Intercept) ~ Normal(0, 1)
    sd(mu, subject) ~ Exponential(1)
end
const observation_priors = @brm begin
    sigma ~ Exponential(1)
end
const joint = merge(predictor, observations, population_priors, observation_priors)

# Correlation priors occupy the same final block as coefficient and scale priors.
const correlated_model = @brm begin
    mu ~ 1 + x + (1 + x | p | subject)
    y ~ Normal(mu, sigma)
    sigma ~ Exponential(1)
    effect(mu, Intercept) ~ Normal(0, 1)
    effect(mu, x) ~ Normal(0, 0.5)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky(2, 2)
end

# A vector's declaration and its scalar hyperpriors belong to the prior block.
const vector_model = @brm begin
    @plate for i in eachindex(y)
        y[i] ~ Normal(x[i], sigma)
    end
    x[1:n] .~ Normal.(0, tau)
    sigma ~ Exponential(1)
    tau ~ Exponential(1)
end

# A local prior stays in its cell: moving theta outside would make one shared
# parameter instead of an independent parameter in each cell.
const local_model = @brm begin
    @plate for i in eachindex(y)
        y[i] ~ Normal(theta, 0.5)
        theta ~ Normal(0, 1)
    end
end
end
