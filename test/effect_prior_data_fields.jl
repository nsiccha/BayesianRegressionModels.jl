# test/effect_prior_data_fields.jl — a scalar data field or `name = constant(expr)`
# as an argument of an addressed `effect(...)` prior whose family or bounds take
# SBBRMI's generic coefficient-prior path (snag `brm-data-field-i-62ec2369`).
#
# A plain `Normal(loc, scale)` coefficient prior is a pair of kwargs evaluated
# in the model scope, so a data field already worked there. Any other prior --
# a `lower`/`upper` bound, another family, per-level priors of differing
# families -- is merged into a BRM-owned StanBlocks base, which resolved the
# names its arguments read in BRM's module: a data field failed to trace
# (`Could not find wscale in model`), and one named `scale` silently became
# Distributions' `scale` function (`cat_w_g_beta ~ normal_scale(0.0)`). Those
# names now arrive as the base's `brm_prior_input_<name>` inputs, as a term or
# `sd(...)` prior's already did, so each model's density equals its literal
# twin's. The exact-total and sum-to-zero SD priors now pass those inputs too.
# One prior still covers every coefficient its address reaches, so an
# array-valued field is refused there exactly as on the Normal path.
#
# Run: julia --startup-file=no --project=test test/effect_prior_data_fields.jl
# Set BRM_EFFECT_RUNTIME=0 to skip the compiled Stan densities.

using Test
using BayesianRegressionModels
using StanBlocks
import BridgeStan
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels
const RUNTIME = get(ENV, "BRM_EFFECT_RUNTIME", "1") != "0"

const LEVELS = (; y=[0.4, 1.3, -0.2, 2.1, 0.9, 1.7],
                  g=["a", "b", "c", "d", "a", "c"])
const PANEL = (; y=[-1.1, 0.3, 0.8, 1.9, 0.2, -0.4],
                 x=[-1.0, -0.4, 0.1, 0.6, 1.2, -0.7])

sb(brmi) = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
code(brmi) = BRM.stan_code(sb(brmi))
compiles(brmi) = StanBlocks.stanc_check(code(brmi); warn_pedantic=false).ok

# The compiled program's full log density at fixed unconstrained points, with
# its coordinate names: equal results mean the same density under the same
# parameterisation.
density_points(brmi, name) = density_points(sb(brmi), name)
function density_points(model::SBBRMI, name)
    folder = joinpath(tempdir(), "brm-effect-prior-data-fields")
    mkpath(folder)
    stan = BRM.stan_instantiate(model; path=joinpath(folder, name * ".stan"))
    n = BridgeStan.param_unc_num(stan.model)
    points = (zeros(n), fill(0.3, n), collect(range(-0.7, 0.9; length=n)))
    BridgeStan.param_unc_names(stan.model),
        [BridgeStan.log_density(stan.model, u; propto=false) for u in points]
end

# Each case: a builder reading data fields, its data, and the same model with
# those fields written as literals.
const CASES = (
    (; name="cell_means_bounded",
       fields=(; wscale=2.0),
       model=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g) ~ Normal(0.0, wscale; lower=0.0)
           y ~ Normal(w, 1.0)
       end),
       literal=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g) ~ Normal(0.0, 2.0; lower=0.0)
           y ~ Normal(w, 1.0)
       end),
       data=LEVELS),
    # A field whose name BRM's module also binds to a function.
    (; name="field_named_scale",
       fields=(; scale=2.0),
       model=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g) ~ Normal(0.0, scale; lower=0.0)
           y ~ Normal(w, 1.0)
       end),
       literal=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g) ~ Normal(0.0, 2.0; lower=0.0)
           y ~ Normal(w, 1.0)
       end),
       data=LEVELS),
    (; name="treatment_contrasts_cauchy",
       fields=(; wloc=0.5, wscale=1.5),
       model=@brm(begin
           w ~ 1 + factor(g)
           effect(w, g) ~ Cauchy(wloc, wscale)
           y ~ Normal(w, 1.0)
       end),
       literal=@brm(begin
           w ~ 1 + factor(g)
           effect(w, g) ~ Cauchy(0.5, 1.5)
           y ~ Normal(w, 1.0)
       end),
       data=LEVELS),
    # Per-level addresses sharing one bounded family: one native statement
    # over a vector of the levels' scales.
    (; name="per_level_bounded",
       fields=(; s1=0.5, s2=3.0),
       model=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g_lvl_1) ~ Normal(0.0, s1; lower=0.0)
           effect(w, g_lvl_2) ~ Normal(0.0, s2; lower=0.0)
           y ~ Normal(w, 1.0)
       end),
       literal=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g_lvl_1) ~ Normal(0.0, 0.5; lower=0.0)
           effect(w, g_lvl_2) ~ Normal(0.0, 3.0; lower=0.0)
           y ~ Normal(w, 1.0)
       end),
       data=LEVELS),
    # Per-level addresses of differing families: the generated family.
    (; name="per_level_mixed_families",
       fields=(; s1=0.5, rate=3.0),
       model=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g_lvl_1) ~ Normal(0.0, s1; lower=0.0)
           effect(w, g_lvl_2) ~ Exponential(rate)
           y ~ Normal(w, 1.0)
       end),
       literal=@brm(begin
           w ~ 0 + factor(g)
           effect(w, g_lvl_1) ~ Normal(0.0, 0.5; lower=0.0)
           effect(w, g_lvl_2) ~ Exponential(3.0)
           y ~ Normal(w, 1.0)
       end),
       data=LEVELS),
    (; name="population_bounded",
       fields=(; m=0.5, s=2.0),
       model=@brm(begin
           mu ~ 1 + x
           effect(mu, x) ~ Normal(m, s; lower=0.0)
           y ~ Normal(mu, 1.0)
       end),
       literal=@brm(begin
           mu ~ 1 + x
           effect(mu, x) ~ Normal(0.5, 2.0; lower=0.0)
           y ~ Normal(mu, 1.0)
       end),
       data=PANEL),
    (; name="population_student_t",
       fields=(; s=2.0),
       model=@brm(begin
           mu ~ 1 + x
           effect(mu, x) ~ LocationScale(0.0, s, TDist(3.0))
           y ~ Normal(mu, 1.0)
       end),
       literal=@brm(begin
           mu ~ 1 + x
           effect(mu, x) ~ LocationScale(0.0, 2.0, TDist(3.0))
           y ~ Normal(mu, 1.0)
       end),
       data=PANEL),
)

@stestset "SBBRMI: a data field in a generic effect prior is that number" begin
    for case in CASES
        brmi = case.model(merge(case.data, case.fields))
        program = code(brmi)
        @test compiles(brmi)
        @test !occursin("normal_scale", program)
        for (name, value) in pairs(case.fields)
            @test BRM.stan_data(sb(brmi))[name] == value
        end
        RUNTIME || continue
        @test density_points(brmi, case.name) ==
              density_points(case.literal(case.data), case.name * "_literal")
    end
end

# The exact-total and sum-to-zero emissions sample a block's SD vector in a
# BRM-owned scale base too, and dropped the prior's inputs at the call: a
# field failed to trace (`Could not find brm_prior_input_rate`).
const GROUPED = (; x=[0.0, 1.0, 0.0, 1.0, 2.0, 3.0], g=[1, 1, 2, 2, 3, 3],
                   y=[0.0, 1.0, 2.0, 3.0, 1.0, 2.0])

@stestset "SBBRMI: a data field in an exact-total or sum-to-zero SD prior" begin
    model = @brm begin
        mu ~ 1 + x + (1 + x || g)
        sd(mu, g) ~ Exponential(rate)
        y ~ Normal(mu, 1.0)
    end
    literal = @brm begin
        mu ~ 1 + x + (1 + x || g)
        sd(mu, g) ~ Exponential(0.5)
        y ~ Normal(mu, 1.0)
    end
    forms = ((; name="total", kwargs=(; total_groups=[:g]), blocks=total_effect_blocks),
             (; name="s2z", kwargs=(; s2z_groups=[:g], s2z_rho=0.4, total_groups=()),
                blocks=s2z_effect_blocks))
    for form in forms
        build(brmi) = SBBRMI(brmi; mod=@__MODULE__, form.kwargs...)
        fielded = build(model(merge(GROUPED, (; rate=0.5))))
        @test !isempty(form.blocks(fielded))
        @test StanBlocks.stanc_check(BRM.stan_code(fielded); warn_pedantic=false).ok
        @test BRM.stan_data(fielded)[:rate] == 0.5
        RUNTIME || continue
        @test density_points(fielded, "sd_" * form.name) ==
              density_points(build(literal(GROUPED)), "sd_" * form.name * "_literal")
    end
end

@stestset "SBBRMI: a constant in a generic effect prior is the field it names" begin
    constant = @brm begin
        wscale = constant(2.0 * maximum(abs.(y)))
        w ~ 0 + factor(g)
        effect(w, g) ~ Normal(0.0, wscale; lower=0.0)
        y ~ Normal(w, 1.0)
    end
    supplied = first(c for c in CASES if c.name == "cell_means_bounded").model
    value = 2.0 * maximum(abs.(LEVELS.y))
    @test code(constant(LEVELS)) == code(supplied(merge(LEVELS, (; wscale=value))))
    @test BRM.stan_data(sb(constant(LEVELS)))[:wscale] == value
end

@stestset "SBBRMI: an array-valued field in a generic effect prior" begin
    per_level = @brm begin
        w ~ 0 + factor(g)
        effect(w, g) ~ Normal(0.0, wscale; lower=0.0)
        y ~ Normal(w, 1.0)
    end
    data = merge(LEVELS, (; wscale=[1.0, 2.0, 3.0, 4.0]))
    # One prior covers every coefficient its address reaches, so this is
    # refused, as on the plain Normal path, rather than vectorised silently.
    # One value per level is a capability gap whose spelling awaits the
    # user's design decision (snag `brm-data-field-i-62ec2369`), not a wrong
    # shape.
    @test_broken (sb(per_level(data)); true)
end
