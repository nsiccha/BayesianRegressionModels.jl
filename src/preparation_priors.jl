# Common composite-prior preparation. Concrete backends own declarations and
# parameterization translations; the prepared prior retains the Julia call.

_brm_prior_constant(value) = value
function _brm_prior_constant(value::ExprColumn)
    args = map(_brm_prior_constant, getargs(value))
    kwargs = map(_brm_prior_constant, getkwargs(value))
    if all(x -> x isa Real, args) && all(x -> x isa Real, values(kwargs))
        result = try getf(value)(args...; kwargs...) catch; nothing end
        result isa Real && return result
    end
    value
end

function _brm_horseshoe_scale(target, key, value; prefix="BRM preparation")
    resolved = _brm_prior_constant(value)
    resolved isa Bool && error(
        "$prefix: `$target ~ Horseshoe($key=...)` requires a numeric formula constant or model value")
    if resolved isa Real
        isfinite(resolved) && resolved > 0 || error(
            "$prefix: `$target ~ Horseshoe($key=...)` must be finite and strictly positive")
        return Float64(resolved)
    end
    resolved
end

function _brm_horseshoe_spec(target, args, kwargs; prefix="BRM preparation")
    isempty(args) || error(
        "$prefix: `$target ~ Horseshoe(...)` accepts no positional arguments; " *
        "use `local_scale=` and/or `global_scale=`")
    unknown = setdiff(keys(kwargs), (:local_scale, :global_scale))
    isempty(unknown) || error(
        "$prefix: `$target ~ Horseshoe(...)` accepts only `local_scale` and `global_scale` keywords, got $unknown")
    (; local_scale=_brm_horseshoe_scale(target, :local_scale,
                                      get(kwargs, :local_scale, 1.0); prefix),
       global_scale=_brm_horseshoe_scale(target, :global_scale,
                                       get(kwargs, :global_scale, 1.0); prefix))
end

# A structured Horseshoe's scale for a backend that plans its coefficients: the
# validated number, a literal's or a data field's, and that field's name, which a
# program reads instead of the number (user principle `0ikeidi`). `nothing` when it
# holds neither, as a sampled value does.
function _brm_horseshoe_structured_scale(target, key, scale; prefix)
    number = _brm_axis_option_value(scale)
    number isa Real || return nothing
    (; number=_brm_horseshoe_scale(target, key, number; prefix),
       source=scale isa NamedColumn ? name(scale) : nothing)
end

# Shape/support traits describe distributions and composition, rather than a
# catalogue of constructors accepted by one backend.
_brm_distribution_shape(::Type{D}, _args) where {D<:Distribution} =
    (Distributions.variate_form(D), Distributions.value_support(D))
function _brm_distribution_shape(::Type{<:MixtureModel}, args)
    length(args) >= 1 || error(
        "BRM preparation: MixtureModel needs at least one component")
    components = first(args)
    components isa AbstractVector && !isempty(components) || error(
        "BRM preparation: MixtureModel components must be a nonempty vector")
    shapes = map(_brm_mixture_component_shape, components)
    all(==(first(shapes)), shapes) || error(
        "BRM preparation: MixtureModel components must share one variate shape")
    first(shapes)
end
function _brm_distribution_shape(::Type{<:LocationScale}, args)
    length(args) == 3 || error("BRM preparation: LocationScale expects three arguments")
    _brm_distribution_shape(last(args))
end
_brm_distribution_shape(expression::ExprColumn) =
    _brm_distribution_shape(getf(expression), getargs(expression))
function _brm_distribution_shape(constructor, args)
    type = brm_distribution_type(constructor)
    isnothing(type) && return nothing
    type isa Type && type <: Distribution || error(
        "BRM preparation: brm_distribution_type($constructor) must return a Distribution type or nothing")
    _brm_distribution_shape(type, args)
end

# Ordinary custom components already register their observation shape and
# support in StanBlocks. Query typed arguments without evaluating live model
# values, so bounds remain symbolic and no duplicate BRM shape trait is needed.
function _brm_mixture_family_support(component::ExprColumn)
    typed_args = map(enumerate(getargs(component))) do (j, arg)
        arg isa Real && !(arg isa Bool) ? Float64(arg) :
            Expr(:(::), Symbol(:arg, j), :real)
    end
    StanBlocks.family_support(getf(component), typed_args...)
end

function _brm_mixture_component_shape(component::ExprColumn)
    getf(component) isa Function || return _brm_distribution_shape(component)
    support = _brm_mixture_family_support(component)
    isempty(support.size) || return nothing
    support.kind === :continuous && support.type === :real &&
        return (Distributions.Univariate, Distributions.Continuous)
    support.kind === :discrete && support.type === :int &&
        return (Distributions.Univariate, Distributions.Discrete)
    nothing
end

function _brm_mixture_full_real(component::ExprColumn)
    constructor = getf(component)
    if constructor isa Function
        support = _brm_mixture_family_support(component)
        return support.kind === :continuous && support.type === :real &&
            isempty(support.size) && isempty(support.support)
    end
    _brm_distribution_shape(component) ===
        (Distributions.Univariate, Distributions.Continuous) || return false
    constructor isa Type || return false
    if constructor <: LocationScale
        return _brm_mixture_full_real(last(getargs(component)))
    end
    type = brm_distribution_type(constructor)
    isnothing(type) && return false
    try
        minimum(type) == -Inf && maximum(type) == Inf
    catch err
        err isa MethodError || rethrow()
        false
    end
end

function _brm_check_heterogeneous_mixture(components; prefix)
    constructors = map(getf, components)
    allequal(constructors) && all(constructor -> constructor isa Type, constructors) &&
        return nothing
    all(_brm_mixture_full_real, components) || error(
        "$prefix: heterogeneous or custom MixtureModel components need known full-real " *
        "scalar support; mixed, bounded and parameter-dependent supports are not supported")
    nothing
end

function _brm_r2d2_prior(value, target)
    prior = isnothing(value) ? ExprColumn(Beta, 1.0, 1.0) :
            _as_expr_column(value)
    isnothing(prior) && error(
        "BRM preparation: R2 prior for `$target` must be a distribution call")
    _brm_r2d2_prior_normalize(getf(prior), prior)
end
_brm_r2d2_prior_normalize(_constructor, prior) = prior
function _brm_r2d2_prior_normalize(::Type{<:Beta}, prior)
    # Preserve the established constant Beta representation. Symbolic shapes
    # remain model references and are ordered with the other sampled values.
    args = map(x -> x isa Real ? Float64(x) : x, getargs(prior))
    if all(x -> x isa Real, args)
        Beta(args...)
    end
    ExprColumn(getf(prior), args...; getkwargs(prior)...)
end

struct _BRMR2D2Plan{P,A,T}
    prior::P
    share_indices::Vector{Int}
    alpha::A
    total_scale::T
end

Base.@nospecializeinfer function _brm_whole_predictor_r2d2(@nospecialize(brmi::BRMI), design, coefficient_priors;
                                    prefix="BRM preparation",
                                    available_predictors=(design.target,))
    any(spec -> isnothing(spec.predictor), r2d2_priors(brmi)) &&
        length(available_predictors) != 1 && error(
            "$prefix: `effect(:, :) ~ r2d2(...)` is ambiguous; name a predictor")
    specs = [spec for spec in r2d2_priors(brmi)
             if isnothing(spec.predictor) || spec.predictor === design.target]
    isempty(specs) && return nothing
    length(specs) == 1 || error("$prefix: duplicate `r2d2` statement for `$(design.target)`")
    spec = only(specs)
    getf(spec.expression) === r2d2 || error("$prefix: malformed R2D2 prior")
    isempty(spec.arguments) || error("$prefix: `r2d2(...)` takes keyword arguments only")
    all(k -> k in (:R2, :tau_bsv, :alpha), keys(spec.keywords)) || error(
        "$prefix: unknown `r2d2` keyword")
    prior = _brm_r2d2_prior(get(spec.keywords, :R2, nothing), design.target)
    alpha = _brm_numeric_constant(get(spec.keywords, :alpha, 1.0))
    isnothing(alpha) && error("$prefix: R2D2 `alpha` must be a numeric constant")
    isfinite(alpha) && alpha > 0 || error("$prefix: R2D2 `alpha` must be positive")
    total = get(spec.keywords, :tau_bsv, nothing)
    if !isnothing(total)
        total = _brm_numeric_constant(total)
        isnothing(total) && error("$prefix: R2D2 `tau_bsv` must be a numeric constant")
        isfinite(total) && total > 0 || error("$prefix: R2D2 `tau_bsv` must be positive")
    end
    shares = zeros(Int, length(design.columns))
    next = 0
    for i in eachindex(shares)
        design.columns[i].label === :Intercept && continue
        !isnothing(coefficient_priors[i]) && continue
        next += 1
        shares[i] = next
    end
    _BRMR2D2Plan(prior, shares, Float64(alpha), total)
end

function _brm_lkj_covariance_factor_spec(target::Symbol, op::ExprColumn;
                                          prefix="BRM preparation")
    args = getargs(op)
    length(args) == 1 || error(
        "$prefix: `$target ~ LKJCovarianceFactor(K; ...)` needs exactly one " *
        "dimension argument, got $(length(args))")
    K = only(args)
    K isa Integer && !(K isa Bool) && K >= 1 || error(
        "$prefix: `$target ~ LKJCovarianceFactor(K; ...)` needs an integer " *
        "dimension >= 1, got $(repr(K))")

    kw = getkwargs(op)
    unknown = Symbol[k for k in keys(kw) if !(k in (:scale_prior, :shape))]
    isempty(unknown) || error(
        "$prefix: `$target ~ LKJCovarianceFactor(...)` accepts only " *
        "`scale_prior` and `shape`, got $unknown")

    scale_prior = get(kw, :scale_prior, ExprColumn(Exponential, 1.0))
    scale_prior isa ExprColumn || error(
        "$prefix: `scale_prior` for `$target` must be a positive-support " *
        "distribution call, got $(typeof(scale_prior))")
    # Keep the callable distribution expression; positive declaration support
    # is a property of the covariance scale, not a restriction to one family.
    # Each backend preserves the ordinary prior kernel on that support.

    shape = _brm_prior_constant(get(kw, :shape, 1.0))
    if shape isa Real
        !(shape isa Bool) && isfinite(shape) && shape > 0 || error(
            "$prefix: LKJ `shape` for `$target` must be finite and strictly " *
            "positive, got $(repr(shape))")
        shape = Float64(shape)
    end
    (; K=Int(K), scale_prior, shape)
end

# `Omega ~ LKJCholesky(K, eta)` at model scope: the Cholesky factor of a K x K
# correlation matrix under the LKJ law, read by name wherever a term correlates
# the levels of a grouping factor (`gr(g; cor=Omega)`, `hsgp(x; by=g, cor=Omega)`).
# Distributions.jl's third positional argument (`uplo`) is a storage choice BRM
# does not offer: every backend uses the lower factor.
function _brm_lkj_cholesky_spec(target::Symbol, op::ExprColumn;
                                prefix="BRM preparation")
    args, kw = getargs(op), getkwargs(op)
    isempty(kw) || error(
        "$prefix: `$target ~ LKJCholesky(K, eta)` takes no keywords, got " *
        "$(collect(keys(kw)))")
    length(args) == 2 || error(
        "$prefix: `$target ~ LKJCholesky(K, eta)` needs a dimension and a " *
        "shape, got $(length(args)) arguments")
    state = _BRMDataAssignments(prefix)
    bound(arg) = _brm_data_assignment_value(arg, state, Set{Symbol}())
    K = bound(args[1])
    eta = _brm_prior_constant(args[2])
    values = _brm_lkj_cholesky_values(target, K, bound(eta); prefix)
    # Preserve a named concentration in Stan source; the resolved dimension
    # is also used by BRM's grouping/coordinate checks.
    (; K=values.K, eta=eta isa Real ? values.eta : eta)
end

function _brm_lkj_cholesky_values(target, K, eta; prefix="BRM preparation")
    K isa Integer && !(K isa Bool) && K >= 1 || error(
        "$prefix: `$target ~ LKJCholesky(K, eta)` needs an integer dimension " *
        ">= 1, got $(repr(K))")
    eta isa Real && !(eta isa Bool) && isfinite(eta) && eta > 0 || error(
        "$prefix: LKJ shape `eta` for `$target` must be a finite, strictly " *
        "positive number, got $(repr(eta))")
    (; K=Int(K), eta=Float64(eta))
end
