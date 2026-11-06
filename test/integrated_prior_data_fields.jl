# test/integrated_prior_data_fields.jl — a population prior that reads data under
# exact totals and sum-to-zero (S2Z), snag `gaussian-data-pr-863d5653`.
#
# These blocks integrate the population coefficient they absorb, so they need its
# prior's moments as numbers at build time. They read those numbers only from
# literals: `effect(mu, Intercept) ~ Normal(pop_mean, pop_sd)` with `pop_mean` and
# `pop_sd` in the data was refused by an explicit `total_groups` or `s2z_groups`
# request and silently left `total_groups = :auto` conventional, while the same
# prior with the numbers spelled out integrated. The moments now resolve from the
# data, and enter the program as the data vectors they always were, so the Stan
# source never spells the value.
#
# Each model is ONE declaration built twice, its priors as numbers (the reference
# law) and as data fields. The twins must agree on representation, coordinates,
# log density, gradient and generated quantities, at the default values and at
# changed ones, and models differing only in a value must share one source.
#
# Run: julia --startup-file=no --project=test test/integrated_prior_data_fields.jl
# Set BRM_INTEGRATED_RUNTIME=0 to skip the compiled Stan comparisons.

using Test, Random
using BayesianRegressionModels
using Distributions
using StanBlocks
import BridgeStan
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels
const RUNTIME = get(ENV, "BRM_INTEGRATED_RUNTIME", "1") != "0"

const STAN_DIR = mktempdir()

# Log density, gradient and generated quantities at fixed unconstrained points,
# with the coordinate names: equal results mean the same law under the same
# parameterisation. Programs with one source share one compiled library.
function stan_points(sb)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, "m" * string(hash(code); base=16) * ".stan"))
    n = BridgeStan.param_unc_num(stan.model)
    points = (zeros(n), fill(0.3, n), collect(range(-0.7, 0.9; length=n)))
    (; names=BridgeStan.param_unc_names(stan.model),
       values=map(points) do u
           lp, gradient = BridgeStan.log_density_gradient(stan.model, u; propto=false)
           constrained = BridgeStan.param_constrain(stan.model, u;
               include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(stan.model, 20261106))
           (; lp, gradient, constrained)
       end)
end

function same_law(a, b; atol=1e-10)
    a.names == b.names || return false
    all(zip(a.values, b.values)) do (x, y)
        isapprox(x.lp, y.lp; atol) && isapprox(x.gradient, y.gradient; atol) &&
            isapprox(x.constrained, y.constrained; atol)
    end
end

const blocks_of = sb -> (total=[(b.predictor, b.group, b.columns, b.population_columns) for b in total_effect_blocks(sb)],
                         s2z=[(b.predictor, b.group, b.columns, b.population_columns) for b in s2z_effect_blocks(sb)])

const TEAM_ROWS = (; y=[1.4, 1.6, 1.9, 2.0], team=["A", "A", "B", "B"])
const GROUP_ROWS = (; y=[0.4, 1.1, 2.2, 2.9, 1.7, 2.4, 0.2, 0.9],
                      x=[-1.0, 0.5, -0.5, 1.0, 0.0, 1.5, -1.5, 0.25],
                      g=[1, 1, 2, 2, 3, 3, 4, 4])
function plate_rows(ngroups, ndays)
    day = repeat(vcat(0:div(ndays, 2)-1, div(ndays, 2)+1:ndays), ngroups)
    g = repeat(["g$i" for i in 1:ngroups]; inner=ndays)
    c1, s1 = cos.(0.4 .* day), sin.(0.4 .* day)
    trend = (day .- ndays / 2) ./ 10
    (; g, day, c1, s1, trend,
       y=[2.0 + 0.3i + 0.4c1[r] - 0.2s1[r] + 0.1trend[r] + 0.2sin(1.3r)
          for (r, i) in enumerate(repeat(1:ngroups; inner=ndays))])
end
function site_rows()
    n = 12
    a = collect(range(-0.8, 0.8; length=n))
    (; c1=cos.(a), s1=sin.(a), trend=a, site=repeat(string.(1:4); inner=3),
       team=repeat(["A", "B", "A"], 4), y_low=[0.0; fill(11.5, n - 1)], y_high=fill(12.5, n))
end

# One declaration per model, `p` holding either the values or the field names.
const CASES = (
    # One block of two groups, under S2Z beside `total_groups = :auto`.
    (; name="s2z_two_groups",
       build=p -> brm(quote
           mu ~ 1 + (1 | team)
           effect(mu, Intercept) ~ Normal($(p.m), $(p.s))
           sd(mu, team) ~ Exponential(0.2)
           y ~ Normal(mu, 0.5)
       end; mod=@__MODULE__),
       rows=TEAM_ROWS, values=(; m=1.7, s=0.8), changed=(; m=2.2, s=0.45), same_source=true,
       options=(; s2z_groups=[:team], s2z_rho=0.0, total_groups=:auto),
       expect=sb -> !isempty(s2z_effect_blocks(sb))),
    # An interval-censored response with a centred block beside an S2Z one, every
    # population and scale prior reading data.
    (; name="s2z_beside_centered",
       build=p -> brm(quote
           mu ~ 1 + c1 + s1 + trend + (1 | a | site) + (1 | b | team)
           log(sigma) ~ 1 + (1 | c | team)
           effect(mu, Intercept) ~ Normal($(p.level_m), $(p.level_s))
           effect(mu, c1) ~ Normal($(p.cycle_m), $(p.cycle_s))
           effect(mu, s1) ~ Normal($(p.cycle_m), $(p.cycle_s))
           effect(mu, trend) ~ Normal($(p.trend_m), $(p.trend_s))
           effect(sigma, Intercept) ~ Normal($(p.noise_m), $(p.noise_s))
           sd(:, a) ~ Exponential($(p.site_rate))
           sd(:, b) ~ Exponential($(p.team_rate))
           sd(:, c) ~ Exponential($(p.noise_rate))
           y_low ~ interval_censored(LogNormal(mu, sigma); upper = y_high)
       end; mod=@__MODULE__),
       rows=site_rows(),
       values=(; level_m=2.5, level_s=1.0, cycle_m=0.0, cycle_s=0.2, trend_m=0.0, trend_s=0.1,
                 noise_m=-2.3, noise_s=1.0, site_rate=1/3, team_rate=1/20, noise_rate=1/3),
       changed=(; level_m=2.1, level_s=0.7, cycle_m=0.1, cycle_s=0.3, trend_m=-0.1, trend_s=0.2,
                  noise_m=-1.9, noise_s=0.8, site_rate=1/2, team_rate=1/10, noise_rate=1/4),
       same_source=false,
       options=(; centered_groups=Set([:site]), s2z_groups=[:team], s2z_rho=0.0, total_groups=:auto),
       expect=sb -> length(s2z_effect_blocks(sb)) == 2),
    # Two predictors on explicit totals and a third left conventional, in a plate
    # with every number of the program, the bounds of `phi` included, in the data.
    (; name="explicit_totals_in_a_plate",
       build=p -> brm(quote
           mu ~ 1 + c1 + s1 + (1 | lv | g) + (0 + c1 | cc | g) + (0 + s1 | ss | g)
           tr ~ 0 + trend + (0 + trend | tt | g)
           log(sigma) ~ 1 + (1 | ns | g)
           effect(mu, Intercept) ~ Normal($(p.level_m), $(p.level_s))
           effect(mu, c1) ~ Normal($(p.cycle_m), $(p.cycle_s))
           effect(mu, s1) ~ Normal($(p.cycle_m), $(p.cycle_s))
           effect(tr, trend) ~ Normal($(p.trend_m), $(p.trend_s))
           effect(sigma, Intercept) ~ Normal($(p.noise_m), $(p.noise_s))
           sd(:, lv) ~ Exponential($(p.level_rate))
           sd(:, cc) ~ Exponential($(p.cycle_rate))
           sd(:, ss) ~ Exponential($(p.cycle_rate))
           sd(:, tt) ~ Exponential($(p.trend_rate))
           sd(:, ns) ~ Exponential($(p.noise_rate))
           phi ~ Uniform($(p.phi_lo), $(p.phi_hi))
           @plate for i in groups(g; order = day)
               m = mu[i] .+ tr[i]
               y[i] ~ AR1Normal(m, sigma[i], phi; time = day[i])
           end
       end; mod=@__MODULE__),
       rows=plate_rows(3, 12),
       values=(; level_m=2.0, level_s=3.0, cycle_m=0.0, cycle_s=1.5, trend_m=0.0, trend_s=2.0,
                 noise_m=log(0.5), noise_s=1.0, level_rate=1.5, cycle_rate=0.5, trend_rate=0.5,
                 noise_rate=0.4, phi_lo=-1.0, phi_hi=1.0),
       changed=(; level_m=2.5, level_s=2.0, cycle_m=0.2, cycle_s=1.1, trend_m=-0.3, trend_s=1.2,
                  noise_m=log(0.7), noise_s=0.7, level_rate=1.1, cycle_rate=0.4, trend_rate=0.7,
                  noise_rate=0.3, phi_lo=-0.9, phi_hi=0.95),
       same_source=false,
       options=(; total_groups=[:g => [:mu, :sigma]]),
       expect=sb -> length(total_effect_blocks(sb)) == 2),
    # `:auto` integrates what its literal twin integrates.
    (; name="auto_totals",
       build=p -> brm(quote
           mu ~ 1 + x + (1 + x || g)
           effect(mu, Intercept) ~ Normal($(p.m), $(p.s))
           effect(mu, x) ~ Normal($(p.xm), $(p.xs))
           y ~ Normal(mu, 1.0)
       end; mod=@__MODULE__),
       rows=GROUP_ROWS, values=(; m=1.2, s=3.0, xm=-0.4, xs=0.9), changed=(; m=0.2, s=1.5, xm=0.4, xs=2.0),
       same_source=true, options=(; total_groups=:auto),
       expect=sb -> length(total_effect_blocks(sb)) == 1),
    # The Student-t forms, which integrate through a Gaussian scale mixture.
    (; name="student_t_totals",
       build=p -> brm(quote
           mu ~ 1 + x + (1 + x || g)
           effect(mu, Intercept) ~ LocationScale($(p.m), $(p.s), TDist($(p.nu)))
           effect(mu, x) ~ TDist($(p.nux))
           y ~ Normal(mu, 1.0)
       end; mod=@__MODULE__),
       rows=GROUP_ROWS, values=(; m=1.2, s=3.0, nu=4.0, nux=7.0), changed=(; m=0.2, s=1.5, nu=3.0, nux=9.0),
       same_source=true, options=(; total_groups=[:g]),
       expect=sb -> !isempty(only(total_effect_blocks(sb)).mixture)),
    # The other S2Z shape: a flat coefficient beside one reading data.
    (; name="s2z_flat_beside_data",
       build=p -> brm(quote
           mu ~ 1 + x + (1 + x || g)
           effect(mu, Intercept) ~ Normal($(p.m), $(p.s))
           effect(mu, x) ~ Flat()
           y ~ Normal(mu, 1.0)
       end; mod=@__MODULE__),
       rows=GROUP_ROWS, values=(; m=1.2, s=3.0), changed=(; m=0.2, s=1.5), same_source=true,
       options=(; s2z_groups=[:g], s2z_rho=0.4, total_groups=()),
       expect=sb -> !isempty(s2z_effect_blocks(sb))),
)

@stestset "a Normal or Student-t prior reading data is integrated like its literal twin" begin
    for case in CASES
        @testset "$(case.name)" begin
            names = NamedTuple{keys(case.values)}(keys(case.values))
            literal(values) = SBBRMI(case.build(values)(case.rows); mod=@__MODULE__, case.options...)
            fielded(values) = SBBRMI(case.build(names)(merge(case.rows, values)); mod=@__MODULE__, case.options...)
            for values in (case.values, case.changed)
                reference, sb = literal(values), fielded(values)
                @test case.expect(sb)
                @test blocks_of(sb) == blocks_of(reference)
                for (a, b) in zip(total_effect_blocks(sb), total_effect_blocks(reference))
                    @test a.location == b.location && a.precision == b.precision && a.mixture == b.mixture
                end
                for (a, b) in zip(s2z_effect_blocks(sb), s2z_effect_blocks(reference))
                    @test a.location == b.location && a.precision == b.precision
                end
                case.same_source && @test BRM.stan_code(sb) == BRM.stan_code(reference)
                @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
                RUNTIME && @test same_law(stan_points(sb), stan_points(reference))
            end
            # One program across the values, whatever a literal twin spells.
            @test BRM.stan_code(fielded(case.values)) == BRM.stan_code(fielded(case.changed))
            # The value is read: changing it changes the law, here and on the twin.
            RUNTIME && @test !same_law(stan_points(fielded(case.values)), stan_points(fielded(case.changed)))
        end
    end
end

@stestset "every data-only spelling of a hyperparameter" begin
    rows = GROUP_ROWS
    reference = SBBRMI(brm(quote
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(log(2.0), 2 * 1.5)
        effect(mu, x) ~ Normal(0.25, 0.75)
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)(rows); mod=@__MODULE__, total_groups=[:g])
    arithmetic = SBBRMI(brm(quote
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(log(two), 2 * s)
        effect(mu, x) ~ Normal(x_m, x_s)
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)(merge(rows, (; two=2.0, s=1.5, x_m=0.25, x_s=0.75)));
        mod=@__MODULE__, total_groups=[:g])
    constants = SBBRMI(brm(quote
        s = constant(1.5 * 2)
        x_s = constant(0.75)
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(log(2.0), s)
        effect(mu, x) ~ Normal(0.25, x_s)
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)(rows); mod=@__MODULE__, total_groups=[:g])
    elements = SBBRMI(brm(quote
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(ms[1], ss[1])
        effect(mu, x) ~ Normal(ms[2], ss[2])
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)(merge(rows, (; ms=[log(2.0), 0.25], ss=[3.0, 0.75])));
        mod=@__MODULE__, total_groups=[:g])
    expected = only(total_effect_blocks(reference))
    for sb in (arithmetic, constants, elements)
        block = only(total_effect_blocks(sb))
        @test block.location == expected.location && block.precision == expected.precision
        @test BRM.stan_code(sb) == BRM.stan_code(reference)
        RUNTIME && @test same_law(stan_points(sb), stan_points(reference))
    end
    # A data-only assignment, chained, is kept as a program value on both twins.
    assigned(base, half) = brm(quote
        double = 2 * $base
        s = double + $half
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(log($base), s)
        effect(mu, x) ~ Normal(0.25, $half)
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)
    for options in ((; total_groups=[:g]), (; s2z_groups=[:g], s2z_rho=0.4, total_groups=()))
        twin = SBBRMI(assigned(2.0, 0.75)(rows); mod=@__MODULE__, options...)
        sb = SBBRMI(assigned(:two, :half)(merge(rows, (; two=2.0, half=0.75))); mod=@__MODULE__, options...)
        blocks = [total_effect_blocks(sb); s2z_effect_blocks(sb)]
        @test only(blocks).location == [log(2.0), 0.25]
        @test only(blocks).precision == [inv(4.75^2), inv(0.75^2)]
        RUNTIME && @test same_law(stan_points(sb), stan_points(twin))
    end
    # An assignment of a model value is a model value.
    sampled = brm(quote
        center ~ Normal(0.0, 1.0)
        shifted = center + 1
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(shifted, 0.4)
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)(rows)
    @test_throws "read only numbers and data" SBBRMI(sampled; mod=@__MODULE__, total_groups=[:g])
    # A data-only call that cannot be evaluated says which call.
    broken = brm(quote
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(log(m), 1.0)
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)(merge(rows, (; m=-1.0)))
    @test_throws "evaluating the data-only call" SBBRMI(broken; mod=@__MODULE__, total_groups=[:g])
end

@stestset "what cannot be integrated stays refused, saying why" begin
    plain(prior) = brm(quote
        mu ~ 1 + x + (1 | g)
        effect(mu, Intercept) ~ $prior
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)
    sampled = brm(quote
        mu ~ 1 + x + (1 | g)
        center ~ Normal(0.0, 1.0)
        effect(mu, Intercept) ~ Normal(center, 0.4)
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)
    refused = (
        ("a sampled location", sampled(GROUP_ROWS)),
        ("a zero scale from the data", plain(:(Normal(m, s)))(merge(GROUP_ROWS, (; m=0.0, s=0.0)))),
        ("a negative scale from the data", plain(:(Normal(m, s)))(merge(GROUP_ROWS, (; m=0.0, s=-1.0)))),
        ("a non-finite location from the data", plain(:(Normal(m, s)))(merge(GROUP_ROWS, (; m=NaN, s=1.0)))),
        ("another family", plain(:(Cauchy(m, s)))(merge(GROUP_ROWS, (; m=0.0, s=1.0)))),
        ("a restricted prior", plain(:(restricted(Normal(m, s), 0.0, 10.0)))(merge(GROUP_ROWS, (; m=0.0, s=1.0)))),
    )
    for (what, model) in refused
        @testset "$what" begin
            @test_throws "read only numbers and data" SBBRMI(model; mod=@__MODULE__, total_groups=[:g])
            @test_throws "read only numbers and data" SBBRMI(model; mod=@__MODULE__,
                s2z_groups=[:g], s2z_rho=0.0, total_groups=())
            # `:auto` keeps the whole model conventional, as it does for any
            # block that stays conventional.
            @test isempty(total_effect_blocks(SBBRMI(model; mod=@__MODULE__, total_groups=:auto)))
        end
    end
    # An array-valued field is no scalar hyperparameter on any path: the conventional
    # emission `:auto` falls back to refuses it too, naming the elementwise spelling.
    array = plain(:(Normal(m, 1.0)))(merge(GROUP_ROWS, (; m=[0.0, 1.0])))
    @test_throws "read only numbers and data" SBBRMI(array; mod=@__MODULE__, total_groups=[:g])
    @test_throws "read only numbers and data" SBBRMI(array; mod=@__MODULE__,
        s2z_groups=[:g], s2z_rho=0.0, total_groups=())
    @test_throws "needs scalar location and scale" SBBRMI(array; mod=@__MODULE__, total_groups=:auto)
    # Non-positive degrees of freedom from the data are no Student-t prior.
    student = plain(:(TDist(nu)))(merge(GROUP_ROWS, (; nu=-1.0)))
    @test_throws "read only numbers and data" SBBRMI(student; mod=@__MODULE__, total_groups=[:g])
    # S2Z has no Student-t mixture yet, whichever way the degrees of freedom are spelled.
    for model in (plain(:(TDist(4.0)))(GROUP_ROWS), plain(:(TDist(nu)))(merge(GROUP_ROWS, (; nu=4.0))))
        @test_throws "Student-t population prior" SBBRMI(model; mod=@__MODULE__,
            s2z_groups=[:g], s2z_rho=0.0, total_groups=())
    end
end

@stestset "replay and rebuild keep the fitted prior" begin
    build = @brm begin
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal(m, s)
        effect(mu, x) ~ Normal(0.1, s)
        y ~ Normal(mu, 1.0)
    end
    fields = (; m=1.2, s=3.0)
    fit_data = merge(GROUP_ROWS, fields)
    forms = ((; name="total", location=:total_location_mu, precision=:total_precision_mu,
                options=(; total_groups=[:g])),
             (; name="s2z", location=:s2z_location_mu, precision=:s2z_precision_mu,
                options=(; s2z_groups=[:g], s2z_rho=0.0, total_groups=())))
    for form in forms
        @testset "$(form.name)" begin
            sb = SBBRMI(build(fit_data); mod=@__MODULE__, form.options...)
            data = BRM.stan_data(sb)
            fitted = (data[form.location], data[form.precision])
            @test fitted == ([1.2, 0.1], fill(inv(9.0), 2))
            # A frozen replay reads the rows; these are the fit's own scalars,
            # whether the new data repeats, omits or changes them.
            for (what, new) in (("repeats", fit_data), ("omits", GROUP_ROWS),
                                ("changes", merge(GROUP_ROWS, (; m=9.0, s=0.5))))
                replay = reprocess(sb, new)
                @test (BRM.stan_data(replay)[form.location], BRM.stan_data(replay)[form.precision]) == fitted
                @test BRM.stan_code(replay) == BRM.stan_code(sb)
                RUNTIME && @test same_law(stan_points(replay), stan_points(sb))
            end
            # These blocks refuse a fresh replay, literal or not.
            @test_throws "freeze_constants=true" reprocess(sb, fit_data; freeze_constants=false)
        end
    end
    # The conventional form keeps a scalar field the same way on a frozen replay.
    conventional = SBBRMI(build(fit_data); mod=@__MODULE__, total_groups=())
    @test BRM.stan_data(reprocess(conventional, merge(GROUP_ROWS, (; m=9.0, s=0.5))))[:m] == 1.2

    # A rebuild for new groups keeps exactly the fitted blocks, and transporting
    # draws onto a changed prior is refused, as for a literal prior.
    plan = generative_plan(build, fit_data; mod=@__MODULE__, total_groups=[:g])
    future = (; y=[0.5, 1.0, 1.5, 2.0], x=[-1.0, 1.0, -0.5, 0.5], g=[5, 5, 6, 6])
    same = generative_plan(plan, merge(future, fields))
    @test only(total_effect_blocks(same)).location == [1.2, 0.1]
    changed = generative_plan(plan, merge(future, (; m=9.0, s=0.5)))
    @test only(total_effect_blocks(changed)).location == [9.0, 0.1]
    if RUNTIME
        names = stan_points(plan.model).names
        draws = zeros(4, length(names))
        target_names = stan_points(same.model).names
        @test size(transport_draws(plan, same, draws, names, target_names; rng=Xoshiro(1)), 1) == 4
        @test_throws "changes the fitted design or prior" transport_draws(plan, changed, draws, names,
            stan_points(changed.model).names; rng=Xoshiro(1))
    end
end

@stestset "readers of these blocks see the same model" begin
    build = p -> brm(quote
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ Normal($(p.m), $(p.s))
        effect(mu, x) ~ Normal($(p.xm), $(p.xs))
        y ~ Normal(mu, 1.0)
    end; mod=@__MODULE__)
    values = (; m=1.2, s=3.0, xm=-0.4, xs=0.9)
    names = NamedTuple{keys(values)}(keys(values))
    for options in ((; total_groups=[:g]), (; s2z_groups=[:g], s2z_rho=0.0, total_groups=()))
        reference = SBBRMI(build(values)(GROUP_ROWS); mod=@__MODULE__, options...)
        sb = SBBRMI(build(names)(merge(GROUP_ROWS, values)); mod=@__MODULE__, options...)
        # The descriptor and the description name the same outputs and the same
        # laws; the description spells a data field by the field's name.
        @test [o.name for o in brm_descriptor(sb).outputs] == [o.name for o in brm_descriptor(reference).outputs]
        report, expected = brm_description(sb), brm_description(reference)
        @test report.complete == expected.complete
        @test [p.id for p in report.priors] == [p.id for p in expected.priors]
        @test [p.support for p in report.priors] == [p.support for p in expected.priors]
        if RUNTIME
            problem = BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, "start" * string(hash(BRM.stan_code(sb)); base=16) * ".stan"))
            inits = (; mu=(; Intercept=0.5, x=0.2))
            @test brm_start(sb, problem.model, inits) ≈ brm_start(reference, problem.model, inits)
        end
    end
end
