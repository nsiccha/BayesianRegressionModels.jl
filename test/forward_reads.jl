# test/forward_reads.jl — a statement may read a name declared below it.
#
# Run: julia --project=test test/forward_reads.jl
#
# The builder body runs in source order, so a read above the name's declaration
# used to hold a placeholder (`NamedColumn(name, MissingColumn())`). Likelihood
# arguments resolved it by name, but a formula term or a grouping needs the
# declaration, so `mu ~ 1 + c` above `c = ...` failed on both backends
# (SBBRMI: "expected data-backed NamedColumn for `c`, got MissingColumn";
# RKBRMI: "term `c` is not a data column"), whether `c` read a parameter, an
# `mi()` completion or data alone (snag formula-covariat-5673c470). Each model
# below, with the declaration moved below its readers, is the same model as in
# source order: the same Stan program and the same RK density and gradient.
# A read of a column that a later `mi(y) ~ ...` completes reads the completion,
# as a read below it does; before, it read the raw column with its missing
# entries and failed ("data column `c` contains `missing` values").
#
# `merge(::BRMI, ::BRMI)` composes fragments that read each other in either
# order, and a fragment's binding of a data column it only reads no longer
# replaces another fragment's likelihood of that column (before,
# `merge(likelihood, reader)` dropped the likelihood from the model).
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module ForwardReads
using BayesianRegressionModels, Distributions

const DATA = (;
    x=[0.1, 0.4, 0.2, 0.9, 0.5, 0.3, 0.7, 0.6],
    z=[0.3, 0.1, 0.8, 0.2, 0.6, 0.4, 0.5, 0.9],
    c0=0.3,
    g=[1, 1, 2, 2, 3, 3, 1, 2],
    y1=Union{Missing,Float64}[1.2, 0.8, missing, 1.5, 0.9, missing, 1.1, 1.3],
    yy=[0.2, -0.1, 0.5, 0.3, -0.4, 0.1, 0.0, 0.6])

# (label, statements in source order, the order with declarations below their readers)
const SHAPES = [
    # A covariate computed from a parameter and data, with a prior on its
    # coefficient: the ArbeitsplatzAtlas PLZ key.
    (:parameter_covariate, quote
        sigma ~ Exponential(1)
        log_w ~ Normal(0, 2)
        c_key = log(1 + x + exp(log_w) * z) - c0
        mu ~ 1 + c_key + (1 | k | g)
        effect(mu, c_key) ~ Normal(1, 0.25)
        mi(y1) ~ LogNormal(mu, sigma)
    end, [1, 4, 5, 6, 2, 3]),
    (:completion_covariate, quote
        sigma ~ Exponential(1)
        sx ~ Exponential(1)
        mi(y1) ~ LogNormal(0.0, sx)
        c_y = log(y1) - c0
        mu ~ 1 + c_y + (1 | g)
        effect(mu, c_y) ~ Normal(1, 0.25)
        yy ~ Normal(mu, sigma)
    end, [1, 2, 3, 5, 6, 7, 4]),
    (:data_covariate, quote
        sigma ~ Exponential(1)
        c_d = log(1 + x) - c0
        mu ~ 1 + c_d + standardize(c_d) + (1 | g)
        effect(mu, c_d) ~ Normal(1, 0.25)
        yy ~ Normal(mu, sigma)
    end, [1, 3, 4, 5, 2]),
    (:data_grouping, quote
        sigma ~ Exponential(1)
        k2 = g + 0
        mu ~ 1 + x + (1 | k2)
        yy ~ Normal(mu, sigma)
    end, [1, 3, 4, 2]),
    (:assignment_chain, quote
        sigma ~ Exponential(1)
        b ~ Normal(0, 1)
        c1 = log(1 + x)
        c2 = c1 * exp(b)
        mu ~ 1 + c2
        yy ~ Normal(mu, sigma)
    end, [1, 2, 5, 6, 4, 3]),
    # The completed column itself read above its `mi(...)` statement.
    (:completed_column, quote
        sigma ~ Exponential(1)
        sx ~ Exponential(1)
        mi(y1) ~ LogNormal(0.0, sx)
        c_y = log(y1) - c0
        mu ~ 1 + c_y
        yy ~ Normal(mu, sigma)
    end, [1, 2, 4, 5, 6, 3]),
]

statements(q) = filter(a -> !(a isa LineNumberNode), q.args)
reordered(q, order) = Expr(:block, statements(q)[order]...)
model(q) = brm(q; mod=@__MODULE__)(DATA)

# The reporter's model written as two fragments, the formula first.
const FORMULA = quote
    sigma ~ Exponential(1)
    mu ~ 1 + c_key + (1 | k | g)
    effect(mu, c_key) ~ Normal(1, 0.25)
    mi(y1) ~ LogNormal(mu, sigma)
end
const KEY = quote
    log_w ~ Normal(0, 2)
    c_key = log(1 + x + exp(log_w) * z) - c0
end

# A likelihood of `yy`, and a fragment that reads `yy` as a covariate.
const LIKELIHOOD = quote
    sigma ~ Exponential(1)
    mu ~ 1 + x
    yy ~ Normal(mu, sigma)
end
const READER = quote
    s2 ~ Exponential(1)
    m2 ~ 1 + yy
    z ~ Normal(m2, s2)
end
end

const FR = ForwardReads
stan(brmi) = BRM.stan_code(SBBRMI(brmi; mod=FR))
probe_points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

function rk_equal(a, b)
    @test coordinate_names(b.model.layout) == coordinate_names(a.model.layout)
    pa = rk_logdensity_problem(a; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    pb = rk_logdensity_problem(b; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in probe_points(LogDensityProblems.dimension(pa))
        va, ga = LogDensityProblems.logdensity_and_gradient(pa, u)
        vb, gb = LogDensityProblems.logdensity_and_gradient(pb, u)
        @test isfinite(va)
        @test vb ≈ va atol=1e-12 rtol=1e-12
        @test gb ≈ ga atol=1e-12 rtol=1e-12
    end
end

@stestset "a declaration below its readers gives the same model" begin
    for (label, q, order) in FR.SHAPES
        @testset "$label" begin
            natural = FR.model(q)
            forward = FR.model(FR.reordered(q, order))
            @test stan(forward) == stan(natural)
            rk_equal(RKBRMI(natural), RKBRMI(forward))
        end
    end
end

@stestset "the covariate's coefficient is addressable from above" begin
    natural = FR.model(FR.SHAPES[1][2])
    forward = FR.model(FR.reordered(FR.SHAPES[1][2], FR.SHAPES[1][3]))
    @test popcoefnames(forward, :mu) == popcoefnames(natural, :mu) == [:Intercept, :c_key]
    sb = SBBRMI(forward; mod=FR)
    @test StanBlocks.stan.compiles(sb.model)
end

@stestset "fragments that read each other compose in either order" begin
    # Each composition is the one block with the fragments' statements in the
    # same order (parameters are declared in that order).
    block(qs...) = FR.model(Expr(:block, (s for q in qs for s in FR.statements(q))...))
    formula_first, key_first = stan(block(FR.FORMULA, FR.KEY)), stan(block(FR.KEY, FR.FORMULA))
    formula, key = (brm(q; mod=FR) for q in (FR.FORMULA, FR.KEY))
    # Builders compose their sources; built models compose their statements.
    @test stan(merge(formula, key)(FR.DATA)) == formula_first
    @test stan(merge(formula(FR.DATA), key(FR.DATA))) == formula_first
    @test stan(merge(key(FR.DATA), formula(FR.DATA))) == key_first
    rk_equal(RKBRMI(block(FR.FORMULA, FR.KEY)), RKBRMI(merge(formula(FR.DATA), key(FR.DATA))))
end

@stestset "a fragment's data binding never replaces a likelihood" begin
    likelihood, reader = (FR.model(q) for q in (FR.LIKELIHOOD, FR.READER))
    codes = map((merge(likelihood, reader), merge(reader, likelihood))) do merged
        @test parent(merged.operations.yy) isa BRM.ExprColumn{typeof(~)}
        stan(merged)
    end
    @test all(code -> occursin("yy ~ normal_id_glm(", code), codes)
    # The same statements, declared in the two fragments' orders.
    @test sort(split(codes[1], "\n")) == sort(split(codes[2], "\n"))
end
