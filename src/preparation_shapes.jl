# Reference indexing follows the distribution's mathematical value shape.
# A simplex/vector or covariance factor is a whole model value at each row.
_brm_parameter_reference_axis(prior::ExprColumn) =
    _brm_parameter_reference_axis(getf(prior), prior)
function _brm_parameter_reference_axis(_constructor, prior)
    shape = _brm_distribution_shape(prior)
    isnothing(shape) || first(shape) === Distributions.Univariate ? :scalar : :whole
end
_brm_parameter_reference_axis(::typeof(LKJCovarianceFactor), _prior) = :whole

_brm_distribution_shape(::typeof(truncated), args) =
    _brm_distribution_shape(first(args))
_brm_distribution_shape(::typeof(censored), args) =
    _brm_distribution_shape(first(args))
_brm_distribution_shape(::typeof(weighted), args) =
    _brm_distribution_shape(first(args))

_brm_prior_constructor(::typeof(truncated)) = true
_brm_prior_constructor(::typeof(censored)) = true
# A `weighted` observation whose response is omitted is a sampled declaration,
# exactly as its unweighted base would be; a backend draws it from the law its
# fitted predictive draw uses.
_brm_prior_constructor(::typeof(weighted)) = true

# A declared vector, `name[1:n] .~ law`, is a parameter whose value is the whole
# vector of `n` elements of `law`'s support.
_brm_prior_constructor(::typeof(_brm_vector_prior)) = true
function _brm_distribution_shape(::typeof(_brm_vector_prior), args)
    shape = _brm_distribution_shape(last(args))
    isnothing(shape) || first(shape) === Distributions.Univariate || error(
        "@brm: a declared vector `name[1:n] .~ law` needs a scalar law; got " *
        "`$(last(args))`. A joint vector law is `name ~ MvNormal(...)`.")
    (Distributions.Multivariate,
     isnothing(shape) ? Distributions.Continuous : last(shape))
end

# Evaluate a formula argument in Julia when it is data-only (literals, data columns, and
# calls over them); `nothing` when it involves a model quantity (a sampled declaration, a
# linear predictor, an assignment), which a backend then emits as an expression instead.
_brm_data_value(x::Real) = x
_brm_data_value(x::AbstractArray{<:Real}) = x
_brm_data_value(x::UniformScaling) = x
_brm_data_value(::Nothing) = nothing
_brm_data_value(_) = nothing
function _brm_data_value(x::NamedColumn)
    backing = parent(x)
    backing isa DataColumn ? parent(backing) : nothing
end
function _brm_data_value(x::ExprColumn)
    args = map(_brm_data_value, getargs(x))
    any(isnothing, args) && return nothing
    kwargs = map(_brm_data_value, getkwargs(x))
    any(isnothing, values(kwargs)) && return nothing
    getf(x)(args...; kwargs...)
end

# `x ~ MvNormal(...)` on a non-data LHS declares a VECTOR PARAMETER (decision `187g4va`,
# 2026-09-16: the faithful Distributions.jl surface). The constructor shapes are
# Distributions.jl's, nothing invented on top of them; every backend reads them here:
#
#   MvNormal(mu, Σ::AbstractMatrix)   covariance
#   MvNormal(mu, Diagonal(v))         variances
#   MvNormal(mu, λ * I)               covariance λ·I
#   MvNormal(Σ)                       zero mean, covariance
#   MvNormal(mu, σ::Real)             standard deviation σ
#   MvNormal(mu, σ::AbstractVector)   standard deviations
#   MvNormal(n::Int, σ::Real)         zero mean, standard deviation σ
#
# (The three `σ` forms are deprecated constructors in Distributions.jl but keep their
# Distributions meaning here — a standard deviation — because `eps ~ MvNormal(zeros(T - 1), 1.0)`
# is the spelling a random walk wants.) The dimension `n` is fixed from a DATA-valued mean —
# a numeric vector, a data column, or a data-only formula expression such as
# `zeros(length(time) - 1)` evaluated in Julia at build time — from a data-valued vector /
# matrix scale, or from the integer form; a parameter-bearing mean with a scalar scale has no
# data-determinable size and is refused loudly. `mean` is `:zero`, a data vector, or `nothing`
# (parameter-bearing: `mean_arg` as written); `scale` is the data value, or `nothing` for a
# sampled scalar standard deviation (`scale_arg` as written).
function _brm_mvnormal_parts(target, op; prefix)
    args, kwargs = getargs(op), getkwargs(op)
    isempty(kwargs) || error(
        "$prefix: `$target ~ MvNormal(...)` takes no keywords, got ",
        "$(collect(keys(kwargs))). Bounds on a multivariate normal parameter are not supported.")
    length(args) in (1, 2) || error(
        "$prefix: `$target ~ MvNormal(...)` takes `MvNormal(mu, scale)`, `MvNormal(Σ)` or ",
        "`MvNormal(n, σ)`; got $(length(args)) positional arguments.")
    mu_arg, scale_arg = length(args) == 2 ? (args[1], args[2]) : (nothing, only(args))
    mu_val = _brm_data_value(mu_arg)
    scale_val = _brm_data_value(scale_arg)

    n = nothing
    if mu_arg === nothing || mu_val isa Integer
        mu_val isa Integer && (n = Int(mu_val))
        mu_val = :zero
    elseif mu_val isa AbstractVector{<:Real}
        n = length(mu_val)
    elseif mu_val !== nothing
        error("$prefix: `$target ~ MvNormal(...)` mean must be a vector, a data column, or the ",
              "integer dimension `MvNormal(n, σ)`; got $(typeof(mu_val)).")
    end
    scale_n = scale_val isa AbstractVector{<:Real} ? length(scale_val) :
              scale_val isa AbstractMatrix{<:Real} ? size(scale_val, 1) : nothing
    if isnothing(n)
        isnothing(scale_n) && error(
            "$prefix: the dimension of `$target ~ MvNormal(...)` must be determinable from data: ",
            "give a data-valued mean (a numeric vector, a data column, or a data-only expression ",
            "such as `zeros(length(t) - 1)`), a vector/matrix scale, or the integer form ",
            "`MvNormal(n, σ)`. A parameter-bearing mean with a scalar scale has no size.")
        n = scale_n
    elseif !isnothing(scale_n) && scale_n != n
        error("$prefix: `$target ~ MvNormal(...)` mean has length $n but the scale has size $scale_n.")
    end
    n >= 1 || error("$prefix: `$target ~ MvNormal(...)` dimension must be positive, got $n.")

    if scale_val isa Diagonal
        all(>(0), diag(scale_val)) || error(
            "$prefix: `$target ~ MvNormal(mu, Diagonal(v))` needs positive variances.")
    elseif scale_val isa AbstractMatrix{<:Real}
        size(scale_val, 1) == size(scale_val, 2) || error(
            "$prefix: `$target ~ MvNormal(mu, Σ)` needs a square covariance, got $(size(scale_val)).")
    elseif scale_val isa UniformScaling
        scale_val.λ > 0 || error("$prefix: `$target ~ MvNormal(mu, λ * I)` needs λ > 0.")
    elseif scale_val isa AbstractVector{<:Real}
        all(>(0), scale_val) || error(
            "$prefix: `$target ~ MvNormal(mu, σ)` needs positive standard deviations.")
    elseif scale_val isa Real
        scale_val > 0 || error("$prefix: `$target ~ MvNormal(mu, σ)` needs σ > 0, got $scale_val.")
    elseif scale_val !== nothing
        error("$prefix: `$target ~ MvNormal(...)` scale of type $(typeof(scale_val)) is not ",
              "supported; use a real or vector of standard deviations, `Diagonal(variances)`, ",
              "`λ * I`, or a covariance matrix.")
    end
    (; n, mean=mu_val, mean_arg=mu_arg, scale=scale_val, scale_arg)
end

# This is a backend extension seam; the common program retains the marker and
# its complete prior call until the concrete backend chooses its geometry.
function _brm_turing_covariance_prior end
function _brm_turing_horseshoe_prior end
