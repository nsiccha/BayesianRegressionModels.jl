# Large local factors share the global diagonal-plus-low-rank covariance
# algebra. See Ong, Nott & Smith (2018), doi:10.1080/10618600.2017.1390472.
# Optimization uses the stopped-q path derivative (Roeder et al., 2017,
# arxiv:1703.09194), already used by the conditional Gaussian SVI estimator.
# Dense natural-gradient factors remain available only below a fixed cap.
struct _BRMSVILowRankUnit
    ρ::Vector{Float64}
    Β::Matrix{Float64}
end

# These operators preserve the meanings of diagnostics.q.Λ and q.U without
# storing d × d arrays. U U' = Λ, although this U is not triangular.
struct _BRMSVIUnitPrecision <: AbstractMatrix{Float64}
    factor::_BRMSVILowRankUnit
end
struct _BRMSVIUnitRoot <: AbstractMatrix{Float64}
    factor::_BRMSVILowRankUnit
end
const _BRMSVIUnitOperator = Union{_BRMSVIUnitPrecision,_BRMSVIUnitRoot}
Base.size(a::_BRMSVIUnitOperator) = (length(a.factor.ρ), length(a.factor.ρ))

function _brm_svi_unit_options(dense_limit, rank)
    dense_limit isa Integer && 0 <= dense_limit <= 256 || throw(ArgumentError(
        "brm_svi_fit: unit_dense_limit must be an integer in 0:256"))
    rank isa Integer && 0 <= rank <= 32 || throw(ArgumentError(
        "brm_svi_fit: unit_rank must be an integer in 0:32"))
    nothing
end

function _brm_svi_unit_mul(a::_BRMSVIUnitPrecision, x)
    _brm_svi_global_precision(_brm_svi_global_geometry(a.factor), x)
end

# Inverse square root of I + V V', with only an r × r eigendecomposition.
function _brm_svi_unit_inverse_root(f)
    h = _brm_svi_global_geometry(f)
    E = eigen(Symmetric(h.V' * h.V))
    W = h.V * E.vectors
    root = sqrt.(1 .+ max.(E.values, 0))
    (; h.s, W, b=-1 ./ (root .* (root .+ 1)))
end
function _brm_svi_unit_mul(a::_BRMSVIUnitRoot, x)
    h = _brm_svi_unit_inverse_root(a.factor)
    (x + h.W * (h.b .* (h.W' * x))) ./ h.s
end
Base.:*(a::_BRMSVIUnitOperator, x::AbstractVector) = _brm_svi_unit_mul(a, x)
Base.:*(a::_BRMSVIUnitOperator, x::AbstractMatrix) = _brm_svi_unit_mul(a, x)
function _brm_svi_unit_root_transpose_mul(a, x)
    h = _brm_svi_unit_inverse_root(a.factor)
    y = x ./ h.s
    y + h.W * (h.b .* (h.W' * y))
end
Base.:*(a::Adjoint{Float64,_BRMSVIUnitRoot}, x::AbstractVector) =
    _brm_svi_unit_root_transpose_mul(parent(a), x)
Base.:*(a::Adjoint{Float64,_BRMSVIUnitRoot}, x::AbstractMatrix) =
    _brm_svi_unit_root_transpose_mul(parent(a), x)
Base.:*(a::Adjoint{Float64,_BRMSVIUnitPrecision}, x::AbstractVector) = parent(a) * x
Base.:*(a::Adjoint{Float64,_BRMSVIUnitPrecision}, x::AbstractMatrix) = parent(a) * x

function LinearAlgebra.diag(a::_BRMSVIUnitPrecision)
    h = _brm_svi_global_geometry(a.factor)
    (1 .- vec(sum(abs2, h.V / h.K.U; dims=2))) ./ h.s.^2
end
function LinearAlgebra.diag(a::_BRMSVIUnitRoot)
    h = _brm_svi_unit_inverse_root(a.factor)
    (1 .+ vec(sum(h.W.^2 .* h.b'; dims=2))) ./ h.s
end
function Base.getindex(a::_BRMSVIUnitOperator, i::Int, j::Int)
    checkbounds(a, i, j)
    e = zeros(size(a, 2)); e[j] = 1
    (a * e)[i]
end

_brm_svi_unit_logdet_root(U::AbstractMatrix) = sum(log, diag(U); init=0.0)
_brm_svi_unit_logdet_root(U::_BRMSVIUnitRoot) =
    -_brm_svi_global_logdet(_brm_svi_global_geometry(U.factor)) / 2

function _brm_svi_unit_factors(d, scale, dense_limit, rank)
    if d <= dense_limit
        eye = Matrix{Float64}(I, d, d)
        return eye ./ scale^2, eye ./ scale, nothing
    end
    f = _BRMSVILowRankUnit(fill(log(scale), d), zeros(d, min(rank, d)))
    _BRMSVIUnitPrecision(f), _BRMSVIUnitRoot(f), _BRMSVIAdam(d * (2 + size(f.Β, 2)), 0.1)
end

_brm_svi_unit_noise(rng, U::AbstractMatrix) = randn(rng, size(U, 1))
_brm_svi_unit_noise(rng, U::_BRMSVIUnitRoot) =
    randn(rng, size(U, 1) + size(U.factor.Β, 2))

function _brm_svi_unit_offset(U::_BRMSVIUnitRoot, ε)
    f, d = U.factor, size(U, 1)
    exp.(f.ρ) .* view(ε, 1:d) + f.Β * view(ε, d+1:length(ε))
end

function _brm_svi_lowrank_gradient(f, gz, gd, ε)
    d = length(f.ρ)
    e, η = view(ε, 1:d), view(ε, d+1:length(ε))
    h = _brm_svi_global_geometry(f)
    δ = h.s .* e + f.Β * η
    score = gd + _brm_svi_global_precision(h, δ)
    vcat(gz, score .* h.s .* e, vec(score * η'))
end

function _brm_svi_lowrank_step!(mean, f, opt, gz, gd, ε, β)
    q = (; μ=mean, f.ρ, f.Β)
    gradient = _brm_svi_lowrank_gradient(f, gz, gd, ε)
    all(isfinite, gradient) || error("brm_svi_fit: nonfinite low-rank unit gradient")
    old = _brm_svi_global_vector(q)
    x = copy(old)
    # A unit's own Adam steps only when the unit is drawn, so its bound needs no √w.
    _brm_svi_whitened_global_step!(opt, q, x, gradient, β; clip=_BRM_SVI_GLOBAL_CLIP)
    Δ = x - old
    d = length(mean)
    h = _brm_svi_global_geometry(f)
    # Bound mean displacement in old posterior sds and factor/scale changes
    # relative to the old diagonal. One scalar shortens the whole Adam step.
    δμ, δρ = view(Δ, 1:d), view(Δ, d+1:2d)
    δΒ = reshape(view(Δ, 2d+1:length(Δ)), size(f.Β))
    distance = sqrt(max(0.0, dot(δμ, _brm_svi_global_precision(h, δμ))))
    relative = max(maximum(abs, δρ; init=0.0),
        maximum(abs, δΒ ./ h.s; init=0.0))
    fraction = min(1.0, 1 / distance, 0.5 / relative)
    x .= old .+ fraction .* Δ
    all(isfinite, x) || error("brm_svi_fit: nonfinite low-rank unit parameters")
    _brm_svi_set_global!(q, x)
    nothing
end

function _brm_svi_unit_step!(Λ::_BRMSVIUnitPrecision, q, i, gz, gd, ε, β)
    _brm_svi_lowrank_step!(q.M[i], Λ.factor, q.unit_opt[i], gz, gd, ε, β)
    q
end

function _brm_svi_unit_average!(a::_BRMSVIUnitPrecision, x::_BRMSVIUnitPrecision, weight)
    @. a.factor.ρ += weight * (x.factor.ρ - a.factor.ρ)
    @. a.factor.Β += weight * (x.factor.Β - a.factor.Β)
    a
end

# Polishing averages a dense factor's covariance, not its precision. A visit
# changes the precision by up to 2.5-fold along one drawn direction (the trust
# region of `_brm_svi_unit_step!`), so the precision iterates are skewed
# upwards: their mean lies far above a typical iterate, and its inverse is
# narrow however long polishing runs. In covariance the iterates are close to
# unbiased. Measured on exact Gaussian units under the default schedule with the
# unit in every step: the precision average gave sds of 0.86, 0.76 and 0.76 times
# the truth at 13, 40 and 66 coordinates; the covariance average 0.99-1.02, as
# the raw iterate does. While polishing, a dense unit's `Λ` slot of the running
# average therefore holds a covariance; `_brm_svi_unit_from_average!` turns it
# back into a precision and its Cholesky factor.
_brm_svi_unit_covariance(Λ::AbstractMatrix) =
    isempty(Λ) ? copy(Λ) : inv(cholesky(Symmetric(Λ)))
function _brm_svi_unit_average!(a::AbstractMatrix, x::AbstractMatrix, weight)
    a .+= weight .* (_brm_svi_unit_covariance(x) .- a)
    a
end
_brm_svi_unit_average_start!(a::_BRMSVIUnitPrecision) = a
_brm_svi_unit_average_start!(a::AbstractMatrix) = (a .= _brm_svi_unit_covariance(a); a)

_brm_svi_unit_from_average!(U::_BRMSVIUnitRoot, Λ::_BRMSVIUnitPrecision) = U
function _brm_svi_unit_from_average!(U::AbstractMatrix, Λ::AbstractMatrix)
    isempty(Λ) && return U
    Λ .= _brm_svi_unit_covariance(Λ)
    U .= cholesky(Symmetric(Λ)).L
end

_brm_svi_unit_description(U::AbstractMatrix) = (; family=:dense, dimension=size(U, 1), rank=size(U, 1))
_brm_svi_unit_description(U::_BRMSVIUnitRoot) =
    (; family=:diagonal_lowrank, dimension=size(U, 1), rank=size(U.factor.Β, 2))

# Unit factors as plain data, so a snapshot loads without BRM (`_brm_svi_snapshot_run`):
# a dense unit's matrix, or a structured unit's log scales `ρ` and loadings `Β`,
# whose covariance is `Diagonal(exp.(2ρ)) + Β * Β'`, as `brm_svi_checkpoint` stores
# it. The element type names no BRM type either, even when every unit is dense.
const _BRMSVIPlainUnit = Union{Matrix{Float64},
    @NamedTuple{family::Symbol,ρ::Vector{Float64},Β::Matrix{Float64}}}
_brm_svi_plain_unit(x::Matrix{Float64}) = copy(x)
_brm_svi_plain_unit(x::_BRMSVIUnitOperator) =
    (; family=:diagonal_lowrank, ρ=copy(x.factor.ρ), Β=copy(x.factor.Β))
_brm_svi_plain_units(xs) = _BRMSVIPlainUnit[_brm_svi_plain_unit(x) for x in xs]
