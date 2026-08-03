# Emission survey: emit Stan for representative SBBRMI model classes.
using BayesianRegressionModels, Distributions, StanBlocks, Random
using LogExpFunctions: logit
const BRM = BayesianRegressionModels
const OUT = ENV["SURVEY_OUT"]

rng = MersenneTwister(2)
const N = 12
d = (;
    x  = randn(rng, N), z = randn(rng, N), w = randn(rng, N),
    g1 = repeat(1:3, inner=4), g2 = repeat(1:2, outer=6), g6 = repeat(1:6, inner=2),
    y  = randn(rng, N), y2 = randn(rng, N), ycens = clamp.(randn(rng, N), -2.0, 2.0),
    yb = rand(rng, 0:1, N), yc = rand(rng, 0:5, N),
    yo = rand(rng, 1:3, N), ycat = rand(rng, 1:3, N),
    trials = fill(10, N), ybb = rand(rng, 0:10, N),
)

models = Pair{String,Any}[]
push_model(name, f) = push!(models, name => f)

push_model("A1_gauss_intercept",  () -> @brm d begin sigma ~ Exponential(1); mu ~ 1; y ~ Normal(mu, sigma) end)
push_model("A2_gauss_glm3",       () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + z + w; y ~ Normal(mu, sigma) end)
push_model("A3_gauss_interaction",() -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x*z; y ~ Normal(mu, sigma) end)
push_model("B1_bernoulli_logit",  () -> @brm d begin logit(p) ~ 1 + x + z; yb ~ Bernoulli(p) end)
push_model("B2_poisson_log",      () -> @brm d begin log(lam) ~ 1 + x + z; yc ~ Poisson(lam) end)
push_model("B3_negbin2",          () -> @brm d begin log(mu) ~ 1 + x; log(phi) ~ 1; yc ~ NegativeBinomial2(mu, phi) end)
push_model("B4_binomial_logit",   () -> @brm d begin logit(p) ~ 1 + x; ybb ~ Binomial(trials, p) end)
push_model("C1_dpar_scalar",      () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x; y ~ Normal(mu, sigma) end)
push_model("C2_dpar_vector",      () -> @brm d begin log(sigma) ~ 1 + z; mu ~ 1 + x; y ~ Normal(mu, sigma) end)
push_model("D1_varying_intercept",() -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + (1 | g1); y ~ Normal(mu, sigma) end)
push_model("D2_varying_slope_indep",()-> @brm d begin sigma ~ Exponential(1); mu ~ 0 + x + (0 + x | g1); y ~ Normal(mu, sigma) end)
push_model("D3_corr_K1",          () -> @brm d begin sigma ~ Exponential(1); mu ~ 0 + x + (0 + x | p | g1); sd(:, p) ~ Exponential(1); y ~ Normal(mu, sigma) end)
push_model("D4_corr_K2",          () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + (1 + x | p | g1); sd(:, p) ~ Exponential(1); y ~ Normal(mu, sigma) end)
push_model("D5_corr_K3",          () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + z + (1 + x + z | p | g1); sd(:, p) ~ Exponential(1); y ~ Normal(mu, sigma) end)
push_model("E1_two_grouping",     () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + (1 | g1) + (1 | g2); y ~ Normal(mu, sigma) end)
push_model("E2_crossed_slopes",   () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + (1 + x | p | g1) + (1 | q | g2); sd(:, p) ~ Exponential(1); sd(:, q) ~ Exponential(1); y ~ Normal(mu, sigma) end)
push_model("F1_multi_response",   () -> @brm d begin s1 ~ Exponential(1); s2 ~ Exponential(1); mu ~ 1 + x; nu ~ 1 + z; y ~ Normal(mu, s1); y2 ~ Normal(nu, s2) end)
push_model("G1_ordinal",          () -> @brm d begin eta ~ 1 + x; yo ~ OrderedLogistic(eta) end)
push_model("G2_categorical",      () -> @brm d begin e2 ~ 1 + x; e3 ~ 1 + z; ycat ~ CategoricalLogit(e2, e3) end)
push_model("G3_student_t",        () -> @brm d begin log(s) ~ 1; mu ~ 1 + x; y ~ LocationScale(mu, s, TDist(4.0)) end)
push_model("H1_truncated",        () -> @brm d begin s ~ Exponential(1); mu ~ 1 + x; y ~ truncated(Normal(mu, s); lower=-3.0, upper=3.0) end)
push_model("H2_censored",         () -> @brm d begin s ~ Exponential(1); mu ~ 1 + x; ycens ~ censored(Normal(mu, s); lower=-2.0, upper=2.0) end)
push_model("D6_corr_K5",          () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + z + w + x*z + (1 + x + z + w + x*z | p | g1); sd(:, p) ~ Exponential(1); y ~ Normal(mu, sigma) end)
push_model("D7_corr_K2_g6",       () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + (1 + x | p | g6); sd(:, p) ~ Exponential(1); y ~ Normal(mu, sigma) end)
push_model("E3_three_grouping",   () -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + (1 | p | g1) + (1 | q | g2) + (1 | r | g6); sd(:, p) ~ Exponential(1); sd(:, q) ~ Exponential(1); sd(:, r) ~ Exponential(1); y ~ Normal(mu, sigma) end)
push_model("D8_varying_slope_noid",()-> @brm d begin sigma ~ Exponential(1); mu ~ 0 + x + (0 + x | g1); y ~ Normal(mu, sigma) end)
push_model("E4_two_grouping_noid",() -> @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + (1 | g1) + (1 | g2); y ~ Normal(mu, sigma) end)

ok = 0; failed = String[]
for (name, f) in models
    try
        m = f()
        sb = SBBRMI(m; mod=@__MODULE__)
        code = StanBlocks.stan_code(sb.model)
        write(joinpath(OUT, name * ".stan"), code)
        open(joinpath(OUT, name * ".meta"), "w") do io
            for (k, v) in pairs(sb.data)
                println(io, k, "\t", typeof(v), "\t", v isa AbstractArray ? "size=$(size(v))" : string(v))
            end
        end
        global ok += 1
        println("OK   ", name, "  (", count(==('\n'), code)+1, " lines)")
    catch e
        push!(failed, name)
        msg = sprint(showerror, e)
        write(joinpath(OUT, name * ".ERROR"), msg)
        println("FAIL ", name, "  ", first(replace(msg, '\n'=>' '), 200))
    end
end
println("\n== emitted $ok / $(length(models)); failed: ", isempty(failed) ? "none" : join(failed, ", "))
