# test/rk_unread_assignments.jl — a model-scope `=` assignment no response
# reads is an RKBRMI graph value of its authored name (user decision `0kpugvc`;
# snag `rk-unread-assign-95f83dc1`), on the value route and on the structural
# route alike, when everything it reads is data, another such assignment, or a
# value the fitted program already computes.
#
# Each model is compared with the same model without its unread line: the
# coordinates, the density and its gradient are the same, and the sampler never
# runs the unread value. Oracles recompute each value from the draws' own
# predictor and parameter values. Public synthetic models only.
#
# Run: julia --startup-file=no --project=test test/rk_unread_assignments.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module UnreadModels
using BayesianRegressionModels, Distributions
const CALLS = Ref(0)
counted(v) = (CALLS[] += 1; 2 .* v)
# Value route: an assignment reading a formula predictor.
decayed(d) = @brm d begin
    mu ~ 1 + x
    q = exp(mu - decay * elapsed)
    y ~ Normal(mu, 0.4)
end
chained(d) = @brm d begin
    mu ~ 1 + x
    a = exp(mu)
    q = a .* 2
    y ~ Normal(mu, 0.4)
end
mixed(d) = @brm d begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    q = mu .* sigma
    y ~ Normal(mu, sigma)
end
called(d) = @brm d begin
    mu ~ 1 + x
    q = counted(mu)
    y ~ Normal(mu, 0.4)
end
plain(d) = @brm d begin
    mu ~ 1 + x
    y ~ Normal(mu, 0.4)
end
# Structural route: scalar assignments over a sampled parameter.
scalars(d) = @brm d begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    variance = sigma^2
    two = 2.0
    scale = sigma
    doubled = two * sigma
    spread = sigma * mean(elapsed)
    y ~ Normal(mu, sigma)
end
plain_scale(d) = @brm d begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
# Stays out: a value of a declaration no response reaches (a parameter, a
# formula predictor's coefficients). Reading a predictor puts this model on the
# value route.
unreached(d) = @brm d begin
    mu ~ 1 + x
    z ~ Normal(0, 1)
    q = exp(z)
    other ~ 1 + x
    p = exp(other)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
# Stays out: a structural-route assignment reading a data column outside a
# reduction, which that route also refuses when a response reads it.
column_scale(d) = @brm d begin
    mu ~ 1 + x
    sigma ~ Exponential(1)
    w = sigma .* elapsed
    y ~ Normal(mu, sigma)
end
end
const UM = UnreadModels

unread_data() = (; x=[-0.4, 0.1, 0.7, 1.2], decay=0.02, elapsed=[1.0, 2.0, 3.0, 4.0],
    y=[0.1, -0.2, 0.3, 0.5])
draws(rk) = hcat(collect(range(-0.6, 0.8; length=rk.model.layout.total)),
    zeros(rk.model.layout.total), fill(0.3, rk.model.layout.total))
problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
# The coordinates, density and gradient of `rk` are those of `reference`.
function same_density(rk, reference)
    @test coordinate_names(rk.model.layout) == coordinate_names(reference.model.layout)
    for u in eachcol(draws(rk))
        @test LogDensityProblems.logdensity(problem(rk), u) ==
              LogDensityProblems.logdensity(problem(reference), u)
        l, g = LogDensityProblems.logdensity_and_gradient(problem(rk), u)
        l0, g0 = LogDensityProblems.logdensity_and_gradient(problem(reference), u)
        @test l == l0
        @test g == g0
    end
end

@stestset "value route: an unread assignment reading a predictor is a graph value" begin
    data = unread_data()
    reference = RKBRMI(UM.plain(data))
    for (model, expected) in (
            (UM.decayed, v -> exp.(v.mu .- data.decay .* data.elapsed)),
            (UM.chained, v -> 2 .* exp.(v.mu)),
            (UM.mixed, v -> v.mu .* v.sigma))
        rk = check_rk_source_roundtrip(RKBRMI(model(data)))
        @test rk.plan isa BRM._RKValuePlan
        model === UM.mixed || same_density(rk, reference)
        U = draws(rk)
        names = model === UM.mixed ? (:q, :mu, :sigma) : (:q, :mu)
        values = rk_draw_values(rk, U, names)
        for k in axes(U, 2)
            @test values.q[k] == expected(map(v -> v[k], values))
        end
    end
    # `mixed` reads `sigma`, which the response also reads.
    same_density(RKBRMI(UM.mixed(data)), RKBRMI(UM.plain_scale(data)))
    # A frozen replay of other rows answers for its own rows.
    rk = RKBRMI(UM.decayed(data))
    new = (; x=[0.3, -0.2, 0.9], decay=0.02, elapsed=[5.0, 6.0, 7.0], y=[0.0, 0.1, 0.2])
    replay = reprocess(rk, new)
    U = draws(replay)
    values = rk_draw_values(replay, U, (:q, :mu))
    for k in axes(U, 2)
        @test length(values.q[k]) == 3
        @test values.q[k] == exp.(values.mu[k] .- new.decay .* new.elapsed)
    end
end

@stestset "value route: the sampler never runs an unread value" begin
    data = unread_data()
    rk = RKBRMI(UM.called(data))
    same_density(rk, RKBRMI(UM.plain(data)))
    p = problem(rk)
    u = draws(rk)[:, 1]
    LogDensityProblems.logdensity_and_gradient(p, u)
    UM.CALLS[] = 0
    for v in eachcol(draws(rk))
        LogDensityProblems.logdensity(p, v)
        LogDensityProblems.logdensity_and_gradient(p, v)
    end
    @test UM.CALLS[] == 0
    values = rk_draw_values(rk, reshape(u, :, 1), (:q, :mu))
    @test UM.CALLS[] > 0
    @test only(values.q) == 2 .* only(values.mu)
end

@stestset "structural route: unread scalars, literals and aliases are graph values" begin
    data = unread_data()
    rk = check_rk_source_roundtrip(RKBRMI(UM.scalars(data)))
    @test rk.plan isa BRM._RKStructuralPlan
    same_density(rk, RKBRMI(UM.plain_scale(data)))
    U = draws(rk)
    v = rk_draw_values(rk, U, (:variance, :two, :scale, :doubled, :spread, :sigma))
    for k in axes(U, 2)
        sigma = v.sigma[k]
        @test v.variance[k] == sigma^2
        @test v.two[k] == 2.0
        @test v.scale[k] == sigma
        @test v.doubled[k] == 2.0 * sigma
        @test v.spread[k] == sigma * (sum(data.elapsed) / length(data.elapsed))
    end
end

@stestset "a value needing a declaration no response reaches stays out" begin
    data = unread_data()
    rk = RKBRMI(UM.unreached(data))
    # No `z` or `other` coordinate: the fitted program samples neither.
    same_density(rk, RKBRMI(UM.plain_scale(data)))
    @test rk.plan isa BRM._RKValuePlan
    U = draws(rk)
    for name in (:q, :p)
        @test_throws "kernel has no port :$name" rk_draw_values(rk, U, (name,))
    end
    rk = RKBRMI(UM.column_scale(data))
    @test rk.plan isa BRM._RKStructuralPlan
    same_density(rk, RKBRMI(UM.plain_scale(data)))
    @test_throws "kernel has no port :w" rk_draw_values(rk, draws(rk), (:w,))
end
