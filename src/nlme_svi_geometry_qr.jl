# With A = [V; I], A'A = I + V'V even when forming the Gram matrix would
# round away the identity. Keep the Householder reflectors, O(D r) storage.
struct _BRMSVIQRGeometry{F}
    s::Vector{Float64}
    V::Matrix{Float64}
    rows::Vector{Int}
    factor::F
end

function _brm_svi_qr_geometry(s, V)
    r = size(V, 2)
    # Put the largest scaled rows first for rowwise accuracy. Errors in these
    # rows are magnified again when a precision or gradient divides by s.
    rows = sortperm(vec(maximum(abs, V; dims=2)); rev=true)
    A = [V[rows, :]; Matrix{Float64}(I, r, r)]
    _BRMSVIQRGeometry(s, V, rows, qr!(A, ColumnNorm()))
end

_brm_svi_global_logdet(h::_BRMSVIQRGeometry) =
    2sum(log, h.s) + 2sum(log ∘ abs, diag(h.factor.R))

# The leading D × D block of the orthogonal projector onto the complement
# of A is (I + VV')⁻¹. Apply it with Q so a narrow row never subtracts two
# nearly equal large numbers in the Woodbury formula.
function _brm_svi_global_precision(h::_BRMSVIQRGeometry, x)
    D, r = size(h.V)
    z = zeros(D + r, Base.tail(size(x))...)
    top = selectdim(z, 1, 1:D)
    scales = h.s[h.rows]
    top .= selectdim(x, 1, h.rows) ./ scales
    lmul!(h.factor.Q', z)
    fill!(selectdim(z, 1, 1:r), 0)
    lmul!(h.factor.Q, z)
    result = similar(z, size(x))
    selectdim(result, 1, h.rows) .= top ./ scales
    result
end

function _brm_svi_entropy_diagonal(h::_BRMSVIQRGeometry, Q=Matrix(h.factor.Q))
    D, r = size(h.V)
    diagonal = Vector{Float64}(undef, D)
    for (i, row) in pairs(h.rows)
        residual = 1 - sum(abs2, view(Q, i, :))
        if residual < 0.5
            # A leverage near one loses the relative accuracy of 1 - leverage.
            # At most O(r) rows need their residual measured directly.
            e = zeros(D + r)
            e[i] = 1
            lmul!(h.factor.Q', e)
            residual = sum(abs2, view(e, r+1:D+r))
        end
        diagonal[row] = residual
    end
    diagonal
end

function _brm_svi_entropy_gradients(h::_BRMSVIQRGeometry, Β)
    Q = Matrix(h.factor.Q) # thin (D + r) × r
    # A[:, p] = Q R. The leading rows of Q / R' give V (I + V'V)⁻¹
    # in pivot order, without cancelling the two terms in Σ⁻¹ Β.
    loading = similar(Β)
    projection = view(Q, axes(Β, 1), :) / UpperTriangular(h.factor.R)'
    loading[h.rows, h.factor.p] = projection ./ h.s[h.rows]
    _brm_svi_entropy_diagonal(h, Q), loading
end

function _brm_svi_sqrt_axes(h::_BRMSVIQRGeometry)
    # The square-root operators also need the small singular directions
    # which eigen(V'V) loses at these scales.
    E = svd(h.V[h.rows, :])
    W = (E.U .* E.S')[invperm(h.rows), :]
    W, hypot.(1, E.S)
end

function _brm_svi_global_sqrt(h::_BRMSVIQRGeometry)
    W, root = _brm_svi_sqrt_axes(h)
    (; s=h.s, W, a=1 ./ (root .+ 1))
end

# The shared kernels compose their updates from the Cholesky form
# (`_svi_structured_gradient`, `_svi_structured_adam`, `_svi_structured_trust`).
# These are the same updates on this geometry.
function _brm_svi_narrow_gradient(f, gz, gd, e, η)
    h = _brm_svi_global_geometry(f)
    score = gd + _brm_svi_global_precision(h, _svi_structured_offset(f.ρ, f.Β, e, η))
    vcat(gz, score .* h.s .* e, vec(score * η'))
end

function _brm_svi_narrow_whitened_step!(opt, q, x, grad, lr; clip, rank)
    D = length(q.ρ)
    C = _brm_svi_global_sqrt(_brm_svi_global_geometry(q))
    scale = vcat(ones(D), rank .* repeat(exp.(q.ρ), size(q.Β, 2)))
    rest = D+1:length(x)
    g = vcat(_brm_svi_global_sqrt_tmul(C, view(grad, 1:D)), view(grad, rest) .* scale)
    step = zeros(length(x))
    _brm_svi_step!(opt, step, g, lr; clip)
    x[1:D] .+= _brm_svi_global_sqrt_mul(C, view(step, 1:D))
    x[rest] .+= scale .* view(step, rest)
    x
end

# One scalar shortens the whole move: the mean by at most one old posterior
# standard deviation, log scales and scaled loadings by at most one half.
function _brm_svi_narrow_trust(f, old, proposed)
    d = length(f.ρ)
    h = _brm_svi_global_geometry(f)
    Δ = proposed .- old
    δμ, δρ = view(Δ, 1:d), view(Δ, d+1:2d)
    δΒ = reshape(view(Δ, 2d+1:length(Δ)), size(f.Β))
    distance = sqrt(max(0.0, dot(δμ, _brm_svi_global_precision(h, δμ))))
    relative = max(maximum(abs, δρ; init=0.0), maximum(abs, δΒ ./ h.s; init=0.0))
    old .+ min(1.0, 1 / distance, 0.5 / relative) .* Δ
end
