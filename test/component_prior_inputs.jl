include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# A component's formula priors read its inputs, model values included, as its
# scalar priors do: one definition, called with different prior locations and
# scales. Snag sbbrmi-refuses-a-b2c36677: SBBRMI refused
# `effect(rise, Intercept) ~ Normal(center, 0.4)` with "model-level prior
# references `center`, a extension" while it accepted `a ~ Normal(center, 0.2)`.
module ComponentPriorInputs
using BayesianRegressionModels, Distributions

@brm function located(x, center, slope_scale)
    value ~ 1 + x
    effect(value, Intercept) ~ Normal(center, 0.4)
    effect(value, x) ~ Normal(0, slope_scale)
    a ~ Normal(center, 0.2)
    return value + a
end

@brm function nested_located(x, center, slope_scale)
    inner ~ located(x, center, slope_scale)
    return inner
end

const located_model = @brm begin
    left ~ located(x_left, center, slope_scale)
    right ~ nested_located(x_right, center, 0.5)
    y_left ~ Normal(left, 0.7)
    y_right ~ Normal(right, 0.7)
    center ~ Normal(0.3, 0.4)
    slope_scale ~ Exponential(1)
end

# The reported shape, with the SD prior of the component-local `|ID|` block
# reading a parent parameter in one call and a number in the other.
@brm function scaled_rates(unit, center, spread)
    log(rise) ~ 1 + (1 | rates | unit)
    log(fall) ~ 1 + (1 | rates | unit)
    effect(rise, Intercept) ~ Normal(center, 0.4)
    sd(rise, rates) ~ Exponential(spread)
    a ~ Normal(center, 0.2)
    return rise + fall + a
end

const rates_model = @brm begin
    left ~ scaled_rates(unit_l, center, spread)
    right ~ scaled_rates(unit_r, center, 0.5)
    y_l ~ Normal(left, sigma)
    y_r ~ Normal(right, sigma)
    center ~ Normal(0.3, 0.4)
    spread ~ Exponential(1)
    sigma ~ Exponential(1)
end

# An SD prior reading a model-valued input builds on both backends.
@brm function unit_field(unit, scale)
    value ~ 1 + (1 | unit)
    effect(value, Intercept) ~ Normal(0, 1)
    sd(value, unit) ~ Exponential(scale)
    return value
end

const sd_model = @brm begin
    left ~ unit_field(unit_l, scale)
    right ~ unit_field(unit_r, 0.5)
    y_l ~ Normal(left, 0.7)
    y_r ~ Normal(right, 0.7)
    scale ~ Exponential(1)
end
end

const CPI = ComponentPriorInputs
const located_data = (; x_left=[-0.4, 0.2, 0.7], x_right=[0.3, -0.2],
    y_left=[0.1, 0.4, 0.8], y_right=[0.3, -0.1])
const rates_data = (; unit_l=["a", "a", "b", "c"], unit_r=["p", "q", "q"],
    y_l=[2.3, 2.4, 2.2, 2.5], y_r=[2.1, 1.9, 2.6])

stan_index(names, name) = coordinate(names, String(name))
function stan_value(stan, u)
    gradient = similar(u)
    value, _ = BridgeStan.log_density_gradient!(stan.model, u, gradient;
        propto=false, jacobian=true)
    value, gradient
end
# Moving one coordinate changes the log density by exactly the change of the
# terms that read it; `part(u)` is the sum of those terms.
function check_coordinate(density, u, index, part; steps=(-0.7, 0.4, 1.1))
    base, gradient = density(u)
    for step in steps
        moved = copy(u)
        moved[index] += step
        @test first(density(moved)) - base ≈ part(moved) - part(u) atol=1e-10 rtol=1e-10
    end
    h = 1e-5
    plus, minus = copy(u), copy(u)
    plus[index] += h
    minus[index] -= h
    @test gradient[index] ≈ (part(plus) - part(minus)) / (2h) atol=1e-7 rtol=1e-7
end

@stestset "component formula priors read model-valued inputs (SBBRMI)" begin
    data = located_data
    saved = deepcopy(data)
    model = CPI.located_model(data)
    stan = consumer_stan(model, "component-prior-inputs-located"; mod=CPI)
    names = BridgeStan.param_unc_names(stan.model)
    # No component declares the input it reads: `center` and `slope_scale`
    # are the parent's parameters.
    @test sort(names) == sort([
        "center", "slope_scale",
        "left_pop_value_beta_pop.1", "left_pop_value_beta_pop.2", "left_a",
        "right_inner_pop_value_beta_pop.1", "right_inner_pop_value_beta_pop.2",
        "right_inner_a"])
    ic, is = stan_index(names, :center), stan_index(names, :slope_scale)
    l0, l1 = stan_index(names, "left_pop_value_beta_pop.1"),
        stan_index(names, "left_pop_value_beta_pop.2")
    r0, r1 = stan_index(names, "right_inner_pop_value_beta_pop.1"),
        stan_index(names, "right_inner_pop_value_beta_pop.2")
    la, ra = stan_index(names, :left_a), stan_index(names, :right_inner_a)
    function oracle(u)
        center, scale = u[ic], exp(u[is])
        logpdf(Normal(0.3, 0.4), center) +
        logpdf(Exponential(1), scale) + u[is] +
        logpdf(Normal(center, 0.4), u[l0]) + logpdf(Normal(0, scale), u[l1]) +
        logpdf(Normal(center, 0.2), u[la]) +
        logpdf(Normal(center, 0.4), u[r0]) + logpdf(Normal(0, 0.5), u[r1]) +
        logpdf(Normal(center, 0.2), u[ra]) +
        sum(logpdf.(Normal.(u[l0] .+ u[l1] .* data.x_left .+ u[la], 0.7), data.y_left)) +
        sum(logpdf.(Normal.(u[r0] .+ u[r1] .* data.x_right .+ u[ra], 0.7), data.y_right))
    end
    n = length(names)
    for u in (zeros(n), collect(range(-0.3, 0.4; length=n)), fill(0.2, n))
        value, gradient = stan_value(stan, u)
        @test value ≈ oracle(u) atol=2e-11 rtol=2e-11
        h = 1e-5
        reference = map(1:n) do j
            plus, minus = copy(u), copy(u)
            plus[j] += h
            minus[j] -= h
            (oracle(plus) - oracle(minus)) / (2h)
        end
        @test gradient ≈ reference atol=2e-8 rtol=2e-8
    end
    @test isequal(data, saved)
end

@stestset "reported component with a local |ID| block (SBBRMI)" begin
    data = rates_data
    model = CPI.rates_model(data)
    stan = consumer_stan(model, "component-prior-inputs-rates"; mod=CPI)
    names = BridgeStan.param_unc_names(stan.model)
    @test count(==("center"), names) == 1 && count(==("spread"), names) == 1
    @test !any(name -> occursin(r"_(center|spread)$", name), names)
    ic, isp = stan_index(names, :center), stan_index(names, :spread)
    rises = [stan_index(names, "$(call)_pop_log_rise_beta_pop.1") for call in (:left, :right)]
    as = [stan_index(names, "$(call)_a") for call in (:left, :right)]
    # The `rise` margin's SD comes first in each call's `rates` block.
    left_tau = stan_index(names, "left_b_rates_unit_tau.1")
    right_tau = stan_index(names, "right_b_rates_unit_tau.1")
    center_part(u) = logpdf(Normal(0.3, 0.4), u[ic]) +
        sum(logpdf(Normal(u[ic], 0.4), u[i]) for i in rises) +
        sum(logpdf(Normal(u[ic], 0.2), u[i]) for i in as)
    # Only `left`'s SD prior reads `spread`; `right`'s reads its literal 0.5,
    # so a change along `spread` carries no term of `right_tau`.
    spread_part(u) = logpdf(Exponential(1), exp(u[isp])) + u[isp] +
        logpdf(Exponential(exp(u[isp])), exp(u[left_tau]))
    density(u) = stan_value(stan, u)
    n = length(names)
    for u in (zeros(n), collect(range(-0.4, 0.3; length=n)))
        check_coordinate(density, u, ic, center_part)
        check_coordinate(density, u, isp, spread_part)
    end
    # With every standardized draw at zero, `right`'s SD reaches the density
    # only through its own prior, Exponential(0.5), and its log Jacobian.
    zs = [i for (i, name) in enumerate(names) if occursin("_z_flat", name)]
    u = collect(range(-0.4, 0.3; length=n))
    u[zs] .= 0
    right_part(v) = logpdf(Exponential(0.5), exp(v[right_tau])) + v[right_tau]
    check_coordinate(density, u, right_tau, right_part)
end

@stestset "component SD prior reads a model-valued input on both backends" begin
    data = rates_data
    model = CPI.sd_model(data)
    rk, problem = consumer_problem(model)
    stan = consumer_stan(model, "component-prior-inputs-sd"; mod=CPI)
    rk_names = coordinate_names(rk.model.layout)
    names = BridgeStan.param_unc_names(stan.model)
    rk_scale = coordinate(rk_names, :scale)
    rk_tau = coordinate(rk_names, Symbol("left.b_value__unit_unit.tau.1"))
    stan_scale = stan_index(names, :scale)
    stan_tau = stan_index(names, "left_b_value__unit_unit_tau.1")
    part(scale, tau) = u -> logpdf(Exponential(1), exp(u[scale])) + u[scale] +
        logpdf(Exponential(exp(u[scale])), exp(u[tau]))
    rk_density(u) = LogDensityProblems.logdensity_and_gradient(problem, u)
    for u in (zeros(length(rk_names)), collect(range(-0.4, 0.3; length=length(rk_names))))
        check_coordinate(rk_density, u, rk_scale, part(rk_scale, rk_tau))
    end
    density(u) = stan_value(stan, u)
    for u in (zeros(length(names)), collect(range(-0.4, 0.3; length=length(names))))
        check_coordinate(density, u, stan_scale, part(stan_scale, stan_tau))
    end
end
