module BRMTermsReference

using BayesianRegressionModels, Distributions, LinearAlgebra, Markdown

# One small synthetic fixture exercises the declarations, never a fit.
const DATA = (;
    x=collect(range(-1, 1; length=25)),
    z=repeat(collect(-2.:2.); inner=5),
    t=collect(1.:25.),
    g=repeat(["a", "b", "c", "d", "e"]; inner=5),
    other=repeat(["e", "d", "c", "b", "a"]; inner=5),
    level=repeat(1:5; inner=5),
    observed=collect(range(0.2, 2; length=25)),
    lo=fill(0.2, 25), hi=fill(3., 25),
    y=collect(range(-0.5, 0.5; length=25)),
    positive=collect(range(0.3, 1.5; length=25)),
    proportion=collect(range(0.05, 0.95; length=25)),
    binary=repeat([0, 1, 1, 0, 1]; inner=5),
    count=repeat([0, 1, 2, 3, 4]; inner=5),
    trials=fill(10, 25),
    from=["a", "b", "c", "d"], to=["b", "c", "d", "e"],
    C=Matrix{Float64}(I, 5, 5),
)

const EXAMPLES = Dict(
    :columns => [
        "Offset" => "mu ~ 1 + offset(x); y ~ Normal(mu, 1)",
        "Raw expression" => "mu ~ 1 + protect(x^2); y ~ Normal(mu, 1)",
        "Frozen scaling" => "mu ~ 1 + zscale(x); y ~ Normal(mu, 1)",
        "Factor contrasts" => "mu ~ 1 + factor(level); y ~ Normal(mu, 1)",
        "Cell means" => "mu ~ 0 + factor(level); y ~ Normal(mu, 1)",
    ],
    :smooths => [
        "Thin-plate spline" => "mu ~ 1 + s(x); y ~ Normal(mu, 1); sd(mu, s(x)) ~ Exponential(2)",
        "Tensor spline" => "mu ~ 1 + t2(x, z); y ~ Normal(mu, 1); sd(mu, t2(x, z), rr) ~ Exponential(2)",
        "Exact GP" => "mu ~ 1 + gp(x); y ~ Normal(mu, 1); sd(mu, gp(x)) ~ LogNormal(0, 0.5)",
        "HSGP" => "mu ~ 1 + hsgp(x; k=5); y ~ Normal(mu, 1); sd(mu, hsgp(x)) ~ Exponential(1)",
        "Grouped hyper-predictor" => "mu ~ 1 + hsgp(x; k=4, by=g, cor=C); log(sd(hsgp(x))) ~ 1 + (1 | g); y ~ Normal(mu, 1)",
        "Periodic HSGP" => "mu ~ 1 + hsgp(x; k=4, cov=:periodic, period=3); y ~ Normal(mu, 1)",
    ],
    :latent => [
        "Monotonic" => "mu ~ 1 + mo(level); y ~ Normal(mu, 1); simplex(mu, mo(level)) ~ Dirichlet(2)",
        "Fixed monotonic contrast" => "mu ~ 1 + mo1(level); y ~ Normal(mu, 1); simplex(mu, mo1(level)) ~ Dirichlet(2)",
        "Measurement error" => "mu ~ 1 + me(observed, 0.2); y ~ Normal(mu, 1); latent(mu, me(observed)) ~ Normal(1, 1)",
        "Censored covariate" => "mu ~ 1 + interval_censored(observed; upper=lo); y ~ Normal(mu, 1); latent(mu, interval_censored(observed)) ~ Normal(1, 1)",
    ],
    :trajectories => [
        "AR(1)" => "mu ~ 1 + ar(t; p=1); y ~ Normal(mu, 1)",
        "Differenced AR(1)" => "mu ~ 1 + dar(t); y ~ Normal(mu, 1); ar(mu, dar(t)) ~ Beta(2, 2)",
        "Random walk" => "mu ~ 1 + rw(t); y ~ Normal(mu, 1); sd(mu, rw(t)) ~ Exponential(0.2)",
        "Stationary grouped AR(1)" => "mu ~ 1 + cdar(t; by=g, cor=C); y ~ Normal(mu, 1)",
        "Areal BYM2" => "mu ~ 1 + car(g; edges=(from, to), type=:bym2); y ~ Normal(mu, 1); mixing(mu, car(g)) ~ Beta(2, 2)",
    ],
    :groups => [
        "Correlated margins" => "mu ~ 1 + x + (1 + x | p | g); y ~ Normal(mu, 1); sd(:, p) ~ Exponential(0.5); cor(:, p) ~ LKJCholesky(2, 2)",
        "Correlated levels" => "mu ~ 1 + (1 | gr(g; cor=C)); y ~ Normal(mu, 1)",
        "Multiple membership" => "mu ~ 1 + (1 | mm(g, other)); y ~ Normal(mu, 1)",
    ],
    :parameters => [
        "Scalar" => "mu ~ Normal(0, 1); y ~ Normal(mu, 1)",
        "Scalar horseshoe" => "beta ~ Horseshoe(local_scale=0.5, global_scale=0.1); mu = beta * x; y ~ Normal(mu, 1)",
        "Restricted support" => "mu ~ restricted(Normal(0, 1); lower=0); positive ~ Normal(mu, 1)",
        "Truncated law" => "mu ~ truncated(Normal(0, 1); lower=0); positive ~ Normal(mu, 1)",
        "Simplex" => "share ~ Dirichlet(2, 2.0); y ~ MixtureModel([Normal(-1, 1), Normal(1, 1)], share)",
        "Joint Gaussian vector" => "beta ~ MvNormal([0., 0.], [1. 0.2; 0.2 1.]); mu = beta[1] .+ beta[2] .* x; y ~ Normal(mu, 1)",
        "Independent vector" => "scales[1:2] .~ truncated.(Normal.(0, 1), 0, Inf); y ~ Normal(0, scales[1])",
    ],
    :shrinkage => [
        "Predictor budget" => "mu ~ 1 + x + (1 | g); y ~ Normal(mu, 1); effect(mu, :) ~ r2d2(R2=Beta(2, 2), tau_bsv=1)",
        "Group budget" => "mu ~ 1 + x + (1 + x | p | g); y ~ Normal(mu, 1); sd(:, p) ~ r2d2(reference_scale=1, R2=Beta(2, 2))",
        "Joint budget" => "mu ~ 1 + x + factor(level) + (1 | p | g); y ~ Normal(mu, 1); sd(:, p) ~ r2d2(reference_scale=1, include=(:population, :contrasts))",
    ],
    :responses => [
        "Continuous" => "mu ~ Normal(0, 1); y ~ Normal(mu, 1)",
        "Positive" => "mu ~ LogNormal(0, 1); positive ~ InverseGaussian(mu, 2)",
        "Probability" => "shape ~ Exponential(1); proportion ~ Beta(shape, 2)",
        "Binary" => "eta ~ Normal(0, 1); binary ~ BernoulliLogit(eta)",
        "Counts" => "mu ~ LogNormal(0, 1); count ~ NegativeBinomial2(mu, 2)",
        "Trials" => "mu ~ Beta(2, 2); count ~ BetaBinomial2(trials, mu, 3)",
        "Categorical" => "eta ~ 1 + x; level ~ CategoricalLogit(eta, eta, eta, eta)",
        "Ordinal" => "eta ~ 0 + x; level ~ Ordinal(StoppingRatio(), LogitLink(), eta)",
        "Circular" => "mu ~ Normal(0, 1); y ~ CircularVonMises(mu, 2)",
        "Joint Gaussian" => "L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2); [y, positive] ~ MvNormalCholesky([x, x], L)",
    ],
    :observation => [
        "Truncation" => "mu ~ Normal(0, 1); positive ~ truncated(Normal(mu, 1); lower=0)",
        "Censoring" => "mu ~ Normal(0, 1); observed ~ censored(Normal(mu, 1); lower=lo)",
        "Interval censoring" => "mu ~ Normal(0, 1); observed ~ interval_censored(Normal(mu, 1); upper=hi)",
        "Mixture" => "share ~ Dirichlet(2, 1.0); y ~ MixtureModel([Normal(-1, 1), Cauchy(1, 1)], share)",
        "Precision weights" => "mu ~ Normal(0, 1); y ~ weighted(Normal(mu, 1), aweights(positive))",
    ],
)

codecell(s) = "`" * replace(strip(s), "|" => "\\|", "\n" => "; ", "`" => "'") * "`"

"""Build each row and display the actual BRMI and emitted density statements."""
function table(group)
    lines = ["| Surface | Bound BRMI | Emitted density statements |",
        "| --- | --- | --- |"]
    for (label, declaration) in EXAMPLES[group]
        model = brm(declaration; mod=@__MODULE__)(DATA)
        sb = SBBRMI(model; mod=@__MODULE__, total_groups=())
        emitted = stan_code(sb)
        block = match(r"(?s)\bmodel\s*\{(.*?)(?:\n\}\s*(?:generated quantities|$))", emitted)
        isnothing(block) && error("No model block in emitted Stan for $label")
        statements = filter(split(block.captures[1], ';')) do statement
            occursin(r"~|target\s*\+=", statement)
        end
        isempty(statements) && error("No density statements for $label")
        density = join((replace(strip(s), r"\s+" => " ") * ";" for s in statements), " ")
        push!(lines, "| $label | $(codecell(string(model))) | $(codecell(density)) |")
    end
    Markdown.parse(join(lines, "\n"))
end

end
