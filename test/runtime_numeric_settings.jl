# test/runtime_numeric_settings.jl — sizes and numeric settings are inputs of
# the emitted programs, not numbers they spell (user principle
# `ODLPlaus:precompile/decisions/2026-11-06T01-09-46-212-0ikeidi`; snag
# `runtime-numeric-f1388e0e`).
#
# `gmrf(...; neighbors=m)`, `gmrf(...; multiresolution=m)` and
# `hsgp(...; k, c)` take a literal or a scalar (per-axis) data field holding
# it. Oracles:
#   * one declaration, many values: models that differ only in such a value
#     emit the identical Stan program (SBBRMI) and the identical RK program
#     (RKBRMI), literal or data-field spelled;
#   * a data-field spelling is the literal twin: the same Stan program and Stan
#     data, and the same RK density at a point;
#   * a frozen replay keeps the fitted value and refuses a design holding
#     another one; a fresh replay takes the new value.
#
# Run: julia --startup-file=no --project=test test/runtime_numeric_settings.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

module RuntimeSettingsModels
using BayesianRegressionModels, Distributions
const FIELD_PRIORS = quote
    sd(field, gmrf((east, north), time)) ~ Exponential(0.7)
    length_scale(field, gmrf((east, north), time), space) ~ LogNormal(log(3.0), 0.4)
    length_scale(field, gmrf((east, north), time), time) ~ LogNormal(log(1.7), 0.4)
    y ~ Normal(field, 0.2)
end
footprint(m) = brm(:(begin
    field ~ 0 + gmrf((east, north), time; neighbors=$m)
    $(FIELD_PRIORS.args...)
end); mod=@__MODULE__)
multiresolution(m) = brm(:(begin
    field ~ 0 + gmrf((east, north), time; multiresolution=$m)
    $(FIELD_PRIORS.args...)
end); mod=@__MODULE__)
hsgp_model(k, c) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(t; k=$k, c=$c)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
tensor_model(k) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(t, s; k=$k, iso=false)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
lkj_model(K, eta) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) ~ Exponential(1)
    cor(:, p) ~ LKJCholesky($K, $eta)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
domain_model(k, domain) = brm(:(begin
    sigma ~ Exponential(1)
    mu ~ 0 + hsgp(t; k=$k, domain=$domain)
    length_scale(:, hsgp(t)) ~ truncated(LogNormal(0, 1); lower=length_scale_floor, upper=24.0)
    y ~ Normal(mu, sigma)
end); mod=@__MODULE__)
end
const RSM = RuntimeSettingsModels

function field_data(; seed=21)
    rng = MersenneTwister(seed)
    npositions, times = 7, [0.0, 1.0, 2.0, 80.0]
    x, y = 10 .* rand(rng, npositions), 10 .* rand(rng, npositions)
    east, north = repeat(x; outer=length(times)), repeat(y; outer=length(times))
    time = repeat(times; inner=npositions)
    (; east, north, time, y=0.3 .* randn(rng, length(time)))
end
const LDATA = (; subject=[1, 1, 2, 2, 3, 3], x=[-1.0, 0.5, 0.2, 1.1, -0.4, 0.9],
    y=[0.3, 0.1, -0.2, 0.6, 0.0, 0.4])
const HDATA = let rng = MersenneTwister(3), t = collect(range(0.0, 5.0; length=24))
    (; t, s=cos.(t), y=sin.(t) .+ 0.1 .* randn(rng, length(t)))
end

stan_program(brmi) = StanBlocks.stan_code(SBBRMI(brmi; mod=RSM).model)
stan_inputs(brmi) = BRM.stan_data(SBBRMI(brmi; mod=RSM))
rk_program(brmi) = BRM._rk_program_statements(BRM._rk_emit_ast(RKBRMI(brmi).plan))
rk_point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
rk_density(rk, u) = LogDensityProblems.logdensity(
    rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)

# The models of `values`, built from `build(value)` over `data(value)`, emit
# one Stan program and one RK program.
function check_one_program(label, build, values; data=_ -> HDATA)
    @testset "$label" begin
        models = [build(v)(data(v)) for v in values]
        stan = map(stan_program, models)
        @test all(==(first(stan)), stan)
        rk = map(rk_program, models)
        @test all(==(first(rk)), rk)
    end
end

@stestset "one program for every count and setting" begin
    D = field_data()
    check_one_program("gmrf neighbors literal", RSM.footprint, (3, 5); data=_ -> D)
    check_one_program("gmrf neighbors field", _ -> RSM.footprint(:nb), (3, 5);
        data=m -> merge(D, (; nb=m)))
    check_one_program("gmrf multiresolution field", _ -> RSM.multiresolution(:m), (2, 4);
        data=m -> merge(D, (; m)))
    check_one_program("hsgp k literal", k -> RSM.hsgp_model(k, 1.5), (6, 9))
    check_one_program("hsgp k and c fields", _ -> RSM.hsgp_model(:kk, :cc),
        ((6, 1.5), (9, 2.0)); data=v -> merge(HDATA, (; kk=v[1], cc=v[2])))
    check_one_program("hsgp per-axis k field", _ -> RSM.tensor_model(:kk),
        ([3, 4], [5, 2]); data=k -> merge(HDATA, (; kk=k)))
    check_one_program("hsgp fixed domain", d -> RSM.domain_model(7, d),
        ((-1.0, 6.0), (-2.0, 8.0)))
end

@stestset "a data-field setting is its literal twin" begin
    D = field_data()
    for (field, literal) in (
            (RSM.footprint(:nb)(merge(D, (; nb=3))), RSM.footprint(3)(D)),
            (RSM.multiresolution(:m)(merge(D, (; m=2))), RSM.multiresolution(2)(D)),
            (RSM.hsgp_model(:kk, :cc)(merge(HDATA, (; kk=7, cc=1.8))),
                RSM.hsgp_model(7, 1.8)(HDATA)),
            (RSM.tensor_model(:kk)(merge(HDATA, (; kk=[3, 4]))),
                RSM.tensor_model((3, 4))(HDATA)),
            # A random-effect block's correlation prior reads K and eta from the
            # data (ODLPlaus co-report on this snag).
            (RSM.lkj_model(:K, :eta)(merge(LDATA, (; K=2, eta=1.7))),
                RSM.lkj_model(2, 1.7)(LDATA)))
        @test stan_program(field) == stan_program(literal)
        stan_field, stan_literal = stan_inputs(field), stan_inputs(literal)
        for key in keys(stan_literal)
            @test isequal(stan_field[key], stan_literal[key])
        end
        rk_field, rk_literal = RKBRMI(field), RKBRMI(literal)
        @test coordinate_names(rk_field.model.layout) == coordinate_names(rk_literal.model.layout)
        u = rk_point(rk_literal)
        @test rk_density(rk_field, u) == rk_density(rk_literal, u)
    end
end

@stestset "different values are different laws" begin
    # Positive control for the identity tests: one program, but the bound
    # setting changes the basis size and the density.
    small, large = RKBRMI(RSM.hsgp_model(6, 1.5)(HDATA)), RKBRMI(RSM.hsgp_model(9, 1.5)(HDATA))
    @test small.model.layout.total + 3 == large.model.layout.total
    wide = RKBRMI(RSM.hsgp_model(6, 2.5)(HDATA))
    u = rk_point(small)
    @test rk_density(small, u) != rk_density(wide, u)
end

@stestset "a data-field LKJ dimension is checked against the block" begin
    @test_throws "does not match the addressed random-effect block width 2" RKBRMI(
        RSM.lkj_model(:K, :eta)(merge(LDATA, (; K=3, eta=1.7))))
    @test_throws "dimension K must be an integer" RKBRMI(
        RSM.lkj_model(:K, :eta)(merge(LDATA, (; K=2.5, eta=1.7))))
end

@stestset "a frozen replay keeps the fitted setting" begin
    D = field_data()
    build = RSM.hsgp_model(:kk, :cc)
    train = merge(HDATA, (; kk=6, cc=1.5))
    rk = RKBRMI(build(train))
    @test_throws "frozen replay keeps the fitted hsgp" reprocess(rk, merge(train, (; kk=9)))
    @test_throws "frozen replay keeps the fitted hsgp" reprocess(rk, merge(train, (; cc=2.0)))
    # SBBRMI's replay keeps the fitted basis in either mode and refuses too; a
    # new build from that data compiles the same Stan program.
    sb = SBBRMI(build(train); mod=RSM)
    @test_throws "replay keeps the fitted hsgp" reprocess(sb, merge(train, (; kk=9)))
    @test_throws "replay keeps the fitted hsgp" reprocess(sb, merge(train, (; kk=9));
        freeze_constants=false)
    @test stan_program(build(merge(train, (; kk=9)))) == stan_program(build(train))
    @test reprocess(sb, train) isa SBBRMI
    same = reprocess(rk, train)
    @test same.model.layout.total == rk.model.layout.total
    fresh = reprocess(rk, merge(train, (; kk=9)); freeze_constants=false)
    @test fresh.model.layout.total == rk.model.layout.total + 3
    field = RSM.footprint(:nb)
    fit = RKBRMI(field(merge(D, (; nb=3))))
    @test_throws "frozen replay keeps the fitted gmrf" reprocess(fit, merge(D, (; nb=5)))
    sb_field = SBBRMI(field(merge(D, (; nb=3))); mod=RSM)
    @test_throws "replay keeps the fitted gmrf" reprocess(sb_field, merge(D, (; nb=5)))
    # A fresh SBBRMI replay builds the support from the new data's count.
    fresh_sb = reprocess(sb_field, merge(D, (; nb=5)); freeze_constants=false)
    rebuilt = BRM.stan_data(SBBRMI(field(merge(D, (; nb=5))); mod=RSM))
    replayed = BRM.stan_data(fresh_sb)
    support = [k for k in keys(rebuilt) if startswith(string(k), "gmrf_")]
    @test !isempty(support)
    @test all(k -> isequal(replayed[k], rebuilt[k]), support)
    @test !all(k -> isequal(BRM.stan_data(sb_field)[k], rebuilt[k]), support)
end
