# Core Turing-facing types and semantic validation. This file deliberately has
# no dependency on Turing, DynamicPPL, StanBlocks, SBBRMI, or emitted SLIC. The
# package extension supplies the executable model after Turing is loaded.

struct _TuringGaussianPlan{C<:_BRMBackendContext,D<:_BRMPopulationDesign,
                           Y<:AbstractVector,B<:AbstractVector}
    context::C
    design::D
    response::Y
    beta_location::B
    beta_scale::B
    sigma_scale::Float64
end

"""
    TuringBRMI

A [`BRMI`](@ref) lowered to the Turing backend. `plan` is the strict,
Stan-independent semantic plan; `model` is the concrete DynamicPPL model
provided by `BayesianRegressionModelsTuringExt` when Turing is loaded.
"""
struct TuringBRMI{P<:BRMI,PL,M}
    parent::P
    plan::PL
    model::M
end

Base.parent(x::TuringBRMI) = x.parent
Base.show(io::IO, x::TuringBRMI) = print(
    io, "TuringBRMI with ", size(x.plan.design.matrix, 2),
    " population coefficients and ", length(x.plan.response), " observations")

_brm_numeric_constant(x::Real) = Float64(x)
function _brm_numeric_constant(x::ExprColumn)
    isempty(getkwargs(x)) || return nothing
    values = map(_brm_numeric_constant, getargs(x))
    any(isnothing, values) && return nothing
    f = getf(x)
    f in (+, -, *, /, ^) || return nothing
    value = try
        f(values...)
    catch
        return nothing
    end
    value isa Real ? Float64(value) : nothing
end
_brm_numeric_constant(_x) = nothing

function _turing_direct_observation(brmi::BRMI)
    found = Any[]
    for (key, op_nc) in pairs(brmi.operations)
        op_nc isa NamedColumn || continue
        op = parent(op_nc)
        op isa ExprColumn{typeof(~)} || continue
        lhs, rhs = getargs(op, 2)
        isnothing(_brm_observation_name(lhs)) && continue
        push!(found, (; key, lhs, rhs))
    end
    length(found) == 1 || error(
        "Turing backend: the initial executable slice requires exactly one " *
        "observed likelihood; found $(length(found))")
    only(found)
end

function _turing_named_reference(x, role::AbstractString)
    x isa NamedColumn || error(
        "Turing backend: Gaussian $role must be a direct named predictor; " *
        "got $(typeof(x))")
    name(x)
end

function _turing_scalar_exponential_prior(brmi::BRMI, target::Symbol)
    op = linear_predictor_op(brmi, target)
    isnothing(op) && error(
        "Turing backend: Gaussian scale `$target` needs an explicit " *
        "`$target ~ Exponential(scale)` prior")
    lhs, rhs = getargs(op, 2)
    lhs isa NamedColumn && name(lhs) === target || error(
        "Turing backend: Gaussian scale prior must bind bare `$target`")
    rhs isa ExprColumn && getf(rhs) === Exponential || error(
        "Turing backend: `$target` currently requires `Exponential(scale)`; " *
        "got $(rhs isa ExprColumn ? getf(rhs) : typeof(rhs))")
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Exponential(scale)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 1 || error(
        "Turing backend: `Exponential(scale)` requires one scale argument")
    scale = _brm_numeric_constant(only(args))
    isnothing(scale) && error(
        "Turing backend: `Exponential(scale)` scale must be a numeric constant")
    isfinite(scale) && scale > 0 || error(
        "Turing backend: `Exponential(scale)` requires a finite positive scale")
    scale
end

function _brm_turing_gaussian_plan(brmi::BRMI)
    observation = _turing_direct_observation(brmi)
    observation.key === :y || error(
        "Turing backend: the initial Gaussian slice requires response name `y`; " *
        "got `$(observation.key)`")
    observation.lhs isa NamedColumn || error(
        "Turing backend: response decorators and response links are not yet supported")
    rhs = observation.rhs
    rhs isa ExprColumn && getf(rhs) === Normal || error(
        "Turing backend: the initial executable family is `Normal(mu, sigma)`")
    isempty(getkwargs(rhs)) || error(
        "Turing backend: `Normal(mu, sigma)` accepts no formula keywords")
    args = getargs(rhs)
    length(args) == 2 || error(
        "Turing backend: `Normal(mu, sigma)` requires exactly two arguments")
    predictor = _turing_named_reference(args[1], "location")
    scale = _turing_named_reference(args[2], "scale")
    predictor === :mu || error(
        "Turing backend: the initial Gaussian slice requires location predictor " *
        "name `mu`; got `$predictor`")
    scale === :sigma || error(
        "Turing backend: the initial Gaussian slice requires scale name `sigma`; " *
        "got `$scale`")

    lp = linear_predictor_op(brmi, predictor)
    isnothing(lp) && error(
        "Turing backend: location `$predictor` has no population formula")
    lp_lhs, lp_rhs = getargs(lp, 2)
    lp_lhs isa NamedColumn && name(lp_lhs) === predictor || error(
        "Turing backend: linked/distributional predictor LHSs are not yet supported")

    isempty(effect_priors(brmi)) || error(
        "Turing backend: population `effect(...)` prior overrides are not yet supported")
    isempty(ranef_effect_priors(brmi)) || error(
        "Turing backend: random-effect prior overrides are not yet supported")
    isempty(r2d2_priors(brmi)) || error(
        "Turing backend: R2D2 priors are not yet supported")
    isempty(term_priors(brmi)) || error(
        "Turing backend: term-parameter priors are not yet supported")

    context = _brm_backend_context(brmi)
    # Raw data columns are represented in `brmi.operations` too. They are
    # inputs to the backend plan, not additional model operations.
    allowed = union(Set((:y, :mu, :sigma)), Set(keys(context.data)))
    extras = setdiff(Set(keys(brmi.operations)), allowed)
    isempty(extras) || error(
        "Turing backend: unsupported top-level operation(s): " *
        join(sort!(collect(extras)), ", "))

    design = _brm_simple_population_design(
        predictor, lp_rhs, context.data, get(context.target_obs, predictor, nothing);
        required=true)
    response = context.data[:y]
    response isa AbstractVector{<:Real} || error(
        "Turing backend: response `y` must be a real vector")
    length(response) == size(design.matrix, 1) || error(
        "Turing backend: response and population design have different row counts")
    sigma_scale = _turing_scalar_exponential_prior(brmi, scale)
    k = size(design.matrix, 2)
    beta_location = zeros(Float64, k)
    beta_scale = ones(Float64, k)
    _TuringGaussianPlan(context, design, collect(Float64, response),
                        beta_location, beta_scale, sigma_scale)
end
