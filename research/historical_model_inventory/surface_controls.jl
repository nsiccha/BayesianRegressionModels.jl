#!/usr/bin/env julia

# Executable controls for the current BRM surface audit. These are representative
# feature receipts only; no result is fanned to a historical row unless that
# row's normalized program and data schema are identical.

using BayesianRegressionModels
using StanBlocks
using Distributions
using CategoricalArrays
using LogExpFunctions: logit

const OUTPUT = joinpath(@__DIR__, "surface_controls.tsv")
const RESULTS = NamedTuple[]

function runprobe(name, f; expect=:pass)
    try
        brmi = f()
        sb = SBBRMI(brmi; mod=@__MODULE__)
        trans = StanBlocks.stan.transpiles(sb.model)
        stanc = trans && StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
        outcome = stanc ? :pass : :fail
        push!(RESULTS, (; feature=name, expected=String(expect), outcome=String(outcome),
                         built="true", transpiles=string(trans), stanc=string(stanc), error=""))
        println(name, '\t', expect, '\t', "built=true;transpiles=$trans;stanc=$stanc")
    catch e
        msg = replace(sprint(showerror, e), '\n'=>' ')
        compact = first(msg, min(length(msg), 500))
        push!(RESULTS, (; feature=name, expected=String(expect), outcome="raise",
                         built="false", transpiles="false", stanc="false", error=compact))
        println(name, '\t', expect, '\t', "raised=$(typeof(e));message=$(first(msg, min(length(msg), 240)))")
    end
end

n = 12
base = (;
    x=collect(range(-1, 1; length=n)), z=collect(range(0.2, 1.3; length=n)),
    w=collect(range(0.1, 0.4; length=n)), t=collect(1.0:n),
    y=collect(range(0.4, 1.6; length=n)), y2=collect(range(1.0, 2.2; length=n)),
    yi=repeat(1:3, 4), count=collect(mod.(0:n-1, 4)),
    g=repeat(1:4, inner=3), h=repeat([1,2], inner=6),
    cat=repeat(1:3, 4),
    se=fill(0.2, n), exposure=collect(range(1.0, 2.0; length=n)),
)

runprobe("linked_distributional_multiresp", () -> @brm base begin
    mu ~ 1 + x
    log(sigma) ~ 1 + z
    y ~ Normal(mu, sigma)
    y2 ~ Normal(mu, 2 * sigma)
end)

runprobe("meta_known_se_rewrite", () -> @brm base begin
    mu ~ 1 + x
    y ~ Normal(mu, se)
end)

runprobe("measurement_error_me", () -> @brm base begin
    mu ~ 1 + me(x, 0.1)
    y ~ Normal(mu, 1)
end)

runprobe("spline_s", () -> @brm base begin
    mu ~ 1 + s(x)
    y ~ Normal(mu, 1)
end)

runprobe("hsgp_gp", () -> @brm base begin
    mu ~ 1 + gp(x; k=8, c=1.5)
    y ~ Normal(mu, 1)
end)

runprobe("grouped_hsgp", () -> @brm base begin
    mu ~ 1 + gp(x; by=g, k=8, c=1.5)
    y ~ Normal(mu, 1)
end)

runprobe("ar1", () -> @brm base begin
    mu ~ 1 + ar(t; p=1)
    y ~ Normal(mu, 1)
end)

runprobe("interaction_cont_cat", () -> @brm base begin
    mu ~ 1 + x + cat + x&cat
    y ~ Normal(mu, 1)
end)

runprobe("gr_by", () -> @brm base begin
    mu ~ 1 + x + (1 + x | gr(g; by=h))
    y ~ Normal(mu, 1)
end)

runprobe("response_mi_normal", () -> begin
    ym = Union{Missing,Float64}[base.y...]; ym[[3,8]] .= missing
    df = merge(base, (;ym))
    @brm df begin
        mu ~ 1 + x
        mi(ym) ~ Normal(mu, 1)
    end
end)

runprobe("zip", () -> @brm base begin
    log(lambda) ~ 1 + x
    logit(zi) ~ 1 + z
    count ~ ZeroInflatedPoisson(lambda, zi)
end)

runprobe("negative_binomial_2", () -> @brm base begin
    log(mu) ~ 1 + x
    log(phi) ~ 1
    count ~ NegativeBinomial2(mu, phi)
end)

runprobe("ordinal", () -> @brm base begin
    eta ~ 1 + x
    yi ~ OrderedLogistic(eta)
end)

runprobe("student_t", () -> @brm base begin
    mu ~ 1 + x
    log(scale) ~ 1
    y ~ LocationScale(mu, scale, TDist(4))
end)

runprobe("offset_semantic_rewrite", () -> @brm base begin
    log_rate ~ 1 + x
    count ~ Poisson(exp(log_rate + log(exposure)))
end)

derived = merge(base, (; xyz=base.x .* base.z .* base.w,
                        gh=(base.g .- 1) .* 10 .+ base.h,
                        ylog=log.(base.y)))
runprobe("three_way_derived_rewrite", () -> @brm derived begin
    mu ~ 1 + x + z + w + xyz
    y ~ Normal(mu, 1)
end)
runprobe("composite_group_derived_rewrite", () -> @brm derived begin
    mu ~ 1 + x + (1 | gh)
    y ~ Normal(mu, 1)
end)
runprobe("response_expression_derived_rewrite", () -> @brm derived begin
    mu ~ 1 + x
    ylog ~ Normal(mu, 1)
end)

runprobe("multi_membership_mm", () -> @brm base begin
    mu ~ 1 + (1 | mm(g,h))
    y ~ Normal(mu, 1)
end; expect=:raise)
runprobe("hsgp_multi_axis", () -> @brm base begin
    mu ~ 1 + gp(x,z; k=8)
    y ~ Normal(mu, 1)
end; expect=:raise)
runprobe("ar2", () -> @brm base begin
    mu ~ 1 + ar(t; p=2)
    y ~ Normal(mu, 1)
end; expect=:raise)
runprobe("offset_wrapper", () -> @brm base begin
    mu ~ 1 + x + offset(log(exposure))
    count ~ Poisson(exp(mu))
end; expect=:raise)
runprobe("predictor_mi", () -> begin
    xm = Union{Missing,Float64}[base.x...]; xm[[2,7]] .= missing
    df=merge(base,(;xm))
    @brm df begin
        mu ~ 1 + mi(xm)
        y ~ Normal(mu, 1)
    end
end; expect=:raise)
runprobe("three_way_native", () -> @brm base begin
    mu ~ 1 + x&z&w
    y ~ Normal(mu, 1)
end; expect=:raise)
runprobe("categorical_likelihood", () -> @brm base begin
    eta ~ 1 + x
    yi ~ Categorical([0.2,0.3,0.5])
end; expect=:raise)

open(OUTPUT, "w") do io
    println(io, "feature\texpected\toutcome\tbuilt\ttranspiles\tstanc\terror")
    for row in RESULTS
        values = replace.([row.feature, row.expected, row.outcome, row.built,
                           row.transpiles, row.stanc, row.error], '\t'=>' ', '\n'=>' ')
        println(io, join(values, '\t'))
    end
end

bad = filter(row -> (row.expected == "pass") != (row.outcome == "pass"), RESULTS)
isempty(bad) || error("surface control expectation mismatch: $(getproperty.(bad, :feature))")
println("surface_controls=$(length(RESULTS))")
println("output=$(abspath(OUTPUT))")
