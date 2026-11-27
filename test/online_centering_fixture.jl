# Shared with sampler integration tests. All coordinates come from the fitted backend.
module OnlineCenteringFixture

using BayesianRegressionModels, BridgeStan, CategoricalArrays, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, WarmupHMC
using DifferentiationInterface: AutoEnzyme

const AD = AutoEnzyme(; mode=Enzyme.Reverse)

function scalar_model(; swapped=false)
    group = categorical(["weak", "strong", "prior"]; levels=["weak", "strong", "prior", "unused"])
    se = swapped ? [0.2, 5.0, 1.0] : [5.0, 0.2, 1.0]
    data = (; group, se, loading=[1.0, 1.0, 0.0], y=zeros(3))
    @brm data begin
        mu ~ 1 + (1 | block | group)
        effect(mu, Intercept) ~ Normal(0, 1)
        sd(:, block) ~ LogNormal(0, 0.4)
        y ~ Normal(loading .* mu, se)
    end
end

function correlated_model()
    data = (; group=fill("a", 3), site=["b", "c", "c"], x=[-0.5, 0.8, 1.4],
        y=[0.1, -0.8, 1.2], se=[0.8, 1.1, 0.4], y2=[0.6, -0.2, 0.4])
    @brm data begin
        mu ~ 1 + x + (1 + x | pair | group)
        effect(mu, :) ~ Normal(0, 1)
        sd(:, pair) ~ LogNormal(0, 0.4)
        cor(:, pair) ~ LKJCholesky(2, 2)
        y ~ Normal(mu, se)
        aux ~ 0 + (1 | scalar | site)
        sd(:, scalar) ~ LogNormal(0, 0.4)
        y2 ~ Normal(aux, 1)
    end
end

function prepare(::Type{SBBRMI}, brmi)
    model = SBBRMI(brmi; total_groups=(), mod=@__MODULE__)
    target = stan_instantiate(model)
    schema = adaptive_centering_cells(model, BridgeStan.param_unc_names(target.model))
    (; model, target, schema)
end

function prepare(::Type{RKBRMI}, brmi)
    model = RKBRMI(brmi)
    target = rk_logdensity_problem(model; ad_backend=AD)
    (; model, target, schema=adaptive_centering_cells(model))
end

wrap(fit, controls) = adaptive_centering_problem(fit.model, fit.target, AD; centeredness=controls)

end
