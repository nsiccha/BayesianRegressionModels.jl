# Statistical preparation and algebra owned by BRM. The emitted model spells
# every parameter and prior; these ordinary Julia functions supply fitted
# data bases and differentiable covariance/spectral calculations.
# A basis is fitted to the fit-role columns (`x_fit`: the column itself when
# the model is built, the training column on a frozen `reprocess`) and
# evaluated at the bound rows.
brm_tps_basis(x, x_fit, k) = brm_tps_apply(x, brm_tps_fit(x_fit, k), k)
brm_tps_basis(x, k) = brm_tps_basis(x, x, k)
# A thin-plate basis fitted to its fit-role axis, as one numeric vector (the
# shape of data a program binds): the shift, the shifted training values (the
# knots) and the penalty-whitened range projection. The emitted program
# evaluates the basis at the rows from it with `brm_tps_apply`; the
# three-argument `brm_tps_basis(x, x_fit, k)` above, which fits from the raw
# axis, stays for sources printed before the fit was split from its basis.
function brm_tps_fit(x_fit, k)
    fit = _brm_fit_spline(x_fit; k)
    vcat(fit.shift, fit.centers, vec(fit.range_projection))
end
brm_tps_apply(x, fit::AbstractVector{<:Real}, k) =
    _brm_apply_spline(_brm_unpack_tps_fit(fit, k), x)

function _brm_unpack_tps_fit(packed, k)
    n, extra = divrem(length(packed) - 1, k - 1)
    extra == 0 && n >= k || throw(ArgumentError(
        "a packed thin-plate fit of rank $k holds 1 + n * $(k - 1) values for " *
        "n >= $k knots, not $(length(packed))"))
    (; shift=packed[1], centers=packed[2:n+1],
        range_projection=reshape(packed[n+2:end], n, k - 2), k)
end

brm_t2_basis(x, z, x_fit, z_fit, k) = _brm_apply_t2(_brm_fit_t2(x_fit, z_fit; k), x, z)
brm_t2_basis(x, z, k) = brm_t2_basis(x, z, x, z, k)
# A `t2` basis fitted to its fit-role axes, as one numeric vector (the shape of
# data a program binds): each margin's shift, scale, null-space constants,
# knots, second-derivative map and range projection, then the fixed block's
# center. The emitted program evaluates the basis at the rows from it.
function brm_t2_fit(x_fit, z_fit, k)
    fit = _brm_fit_t2(x_fit, z_fit; k)
    vcat((_brm_pack_cr_fit(margin) for margin in fit.margins)..., fit.fixed_center)
end
brm_t2_basis(x, z, fit::AbstractVector{<:Real}, k) =
    _brm_apply_t2(_brm_unpack_t2_fit(fit, k), x, z)

_brm_pack_cr_fit(fit) = vcat(fit.shift, fit.scale, fit.null_const_scale, fit.slope_norm,
    fit.knots, vec(fit.F), vec(fit.range_projection))
function _brm_unpack_t2_fit(packed, k)
    offset = Ref(0)
    take(n) = (values = packed[offset[]+1:offset[]+n]; offset[] += n; values)
    margins = map(k) do m
        shift, scale, null_const_scale, slope_norm = take(4)
        knots = take(m)
        F = reshape(take(m^2), m, m)
        range_projection = reshape(take(m * (m - 2)), m, m - 2)
        (; shift, scale, knots, F, range_projection, null_const_scale, slope_norm, k=m)
    end
    (; margins, fixed_center=packed[offset[]+1:end], k)
end

_brm_covariate_var(values) = var(values)

function brm_hsgp_basis(axes, k, c, iso)
    K = k isa Tuple ? k : (k,)
    C = c isa Tuple ? c : (c,)
    state = _brm_hsgp_basis_state(axes, K, :exp_quad, iso, nothing;
        fits=_brm_fit_hsgp(axes, K, C))
    state.PHI, state.omega2, state.rho_lower
end

brm_hsgp_periodic_basis(x, k, period) = _brm_rk_periodic_basis(x, k, period, nothing, nothing)
# An anchored (`anchor::Real`) and/or row-masked (`mask`, the bound 0/1 rows)
# periodic basis (decision 1fk9b29).
brm_hsgp_periodic_basis(x, k, period, anchor::Real) =
    _brm_rk_periodic_basis(x, k, period, (anchor,), nothing)
brm_hsgp_periodic_basis(x, k, period, mask::AbstractVector) =
    _brm_rk_periodic_basis(x, k, period, nothing, mask .!= 0)
brm_hsgp_periodic_basis(x, k, period, anchor::Real, mask::AbstractVector) =
    _brm_rk_periodic_basis(x, k, period, (anchor,), mask .!= 0)

function _brm_rk_periodic_basis(x, k, period, anchor, mask)
    state = _brm_hsgp_basis_state((x,), (k,), :periodic, true, period; anchor, mask)
    state.PHI, state.harmonics, state.rho_lower
end

# `hsgp(x...; by=g, cor=multigroup(Omega, tau))` (src/level_correlation.jl):
# basis column `b` of the levels × basis standardized weights `z` is correlated
# across levels by the multigroup coherence
# `C_b = exp(-(sum_a rho_a^2 omega2[b, a]) / 2 * D2)`, `D2 = 2 tau (1 - Omega)`,
# with `Omega = L L'`. The returned weights are `chol(C_b + 1e-6 I) * z[:, b]`;
# the spectral scaling stays with the caller. The unit diagonal is set exactly,
# so rounding in `L L'` never perturbs it. StanBlocks' twin is the
# `@stanonly` deffun of the same name in src/sbimpl.jl; keep both in lockstep.
function brm_hsgp_multigroup_weights(z, omega2, rho, L, tau)
    G, M = size(z, 1), size(z, 2)
    T = promote_type(eltype(z), eltype(L), typeof(tau), eltype(rho))
    Omega = L * transpose(L)
    out = similar(z, T, G, M)
    for b in 1:M
        e = _brm_hsgp_frequency_exponent(omega2, rho, b)
        C = Matrix{T}(undef, G, G)
        for j in 1:G, i in 1:G
            C[i, j] = i == j ? one(T) + T(1e-6) : exp(-e * tau * (1 - Omega[i, j]))
        end
        chol = _brm_lower_cholesky(C)
        for i in 1:G
            acc = zero(T)
            for k in 1:i
                acc += chol[i, k] * z[k, b]
            end
            out[i, b] = acc
        end
    end
    out
end

# `hsgp(x...; by=g, cor=multigroup(coordinates, lambda; k, method))`
# (src/level_correlation.jl): basis frequency `b` correlates the levels by
# `exp(-d^2 / (2 r_b^2))`, `d` the distance between their coordinates and
# `r_b = lambda / sqrt(sum_a rho_a^2 omega2[b, a])`. A frequency whose range
# is below the approximation's resolution gets a well-defined but inexact
# coherence; with `lambda >= 2 sqrt(log(100)) r_min` its spectral amplitude is
# below 1/100 (src/level_correlation.jl).
# Per axis, each level gets a unit feature vector of `k` entries, and level
# `i`'s weight is `w_i' z[:, b]` with `w_i` the Kronecker product of its axes'
# features, so the levels' correlation `W W'` has a unit diagonal and no
# `K x K` factor exists. `method` 2 (Hilbert space, the formula default) takes
# the sine basis of a box of half-width `sqrt(scale_a^2 + (3.2 r_b)^2)` around
# `center`, weighted by the square root of the spectral density. `method` 1
# (inducing points) takes `L_a \ k_a(s_ia)` for a `k`-point grid with spacing
# `sqrt(scale_a^2 + (r_b / 1.5)^2)`, `L_a` the Cholesky factor of its kernel:
# the unit-diagonal predictive process. Both grow smoothly with the range, so
# the weights stay differentiable in it. `z` is `k^D x M` standard normals, axis 1 fastest; the
# spectral scaling of the temporal basis stays with the caller. StanBlocks'
# twin is the `@stanonly` deffun of the same name in src/sbimpl.jl; keep both
# in lockstep.
brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, coordinates, center, scale,
                                       lambda, method::Integer) =
    _brm_multigroup_coordinate_weights(_brm_multigroup_approximation(method), z, omega2,
                                       rho, coordinates, center, scale, lambda)
function _brm_multigroup_coordinate_weights(approximation, z, omega2, rho, coordinates,
                                            center, scale, lambda)
    G, D = size(coordinates, 1), size(coordinates, 2)
    P, M = size(z, 1), size(z, 2)
    k = round(Int, P^(1 / D))
    k^D == P || error("brm_hsgp_multigroup_coordinate_weights: the spatial " *
                      "weights have $P rows, not k^$D for a whole k")
    T = promote_type(eltype(z), eltype(rho), typeof(lambda))
    out = similar(z, T, G, M)
    frame = Array{T}(undef, k, k, D)
    span = Vector{T}(undef, D)
    a = Matrix{T}(undef, k, D)
    v = Vector{T}(undef, P ÷ k)
    for b in 1:M
        e = _brm_hsgp_frequency_exponent(omega2, rho, b)
        e > 0 || error("brm_hsgp_multigroup_coordinate_weights: basis " *
                       "frequency $b is zero; its spatial range is unbounded")
        r = lambda / sqrt(e)
        for axis in 1:D
            span[axis] = _brm_multigroup_axis_frame!(approximation, frame, axis,
                                                     scale[axis], r, k)
        end
        for i in 1:G
            for axis in 1:D
                _brm_multigroup_axis_features!(approximation, a, axis, frame,
                    span[axis], coordinates[i, axis] - center[axis], r, k)
                norm2 = zero(T)
                for p in 1:k
                    norm2 += a[p, axis]^2
                end
                norm = sqrt(norm2)
                for p in 1:k
                    a[p, axis] /= norm
                end
            end
            # Contract `z[:, b]` with the axes' features one axis at a time,
            # axis 1 (the fastest index) first.
            n = P ÷ k
            for q in 1:n
                t = zero(T)
                for p in 1:k
                    t += a[p, 1] * z[p + k * (q - 1), b]
                end
                v[q] = t
            end
            for axis in 2:D
                n ÷= k
                for q in 1:n
                    t = zero(T)
                    for p in 1:k
                        t += a[p, axis] * v[p + k * (q - 1)]
                    end
                    v[q] = t
                end
            end
            out[i, b] = v[1]
        end
    end
    out
end

# The grid as one value, `brm_multigroup_coordinate_grid`'s tuple: what an RK
# graph passes on without unpacking it; the weights do not read its resolution.
brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, coordinates, grid::Tuple,
                                       lambda, method::Integer) =
    brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, coordinates, grid[1], grid[2],
                                           lambda, method)

"""
    brm_hsgp_multigroup_inducing_weights(z, omega2, rho, coordinates, positions, lambda)

The coordinate coherence of
`multigroup((x, y), lambda; method = :inducing, inducing = (ux, uy))`: the
predictive process of free inducing `positions` (points × axes). Per frequency
`b`, with range `r_b = lambda / sqrt(sum_a rho_a^2 omega2[b, a])`, level `i`'s
weight is `a_i' z[:, b] / |a_i|`, `a_i = L \\ k(s_i)`, with `L` the Cholesky
factor of the positions' kernel (jitter `1e-8`) and `k(s_i)` the kernel between
the level and the positions, so the levels' correlation has a unit diagonal.
`z` is points × frequencies standard normals. Each frequency costs a Cholesky
factor of the points and `levels × points²` for the solves. StanBlocks' twin is
the `@stanonly` deffun of the same name in src/sbimpl.jl; keep both in
lockstep.
"""
function brm_hsgp_multigroup_inducing_weights(z, omega2, rho, coordinates, positions, lambda)
    G, D = size(coordinates, 1), size(coordinates, 2)
    P, M = size(z, 1), size(z, 2)
    size(positions) == (P, D) || error(
        "brm_hsgp_multigroup_inducing_weights: the weights have $P rows, so the " *
        "positions must be $P x $D, got $(size(positions, 1)) x $(size(positions, 2))")
    T = promote_type(eltype(z), eltype(rho), eltype(positions), typeof(lambda))
    out = similar(z, T, G, M)
    chol = Matrix{T}(undef, P, P)
    a = Vector{T}(undef, P)
    d2 = Vector{T}(undef, P)
    for b in 1:M
        e = _brm_hsgp_frequency_exponent(omega2, rho, b)
        e > 0 || error("brm_hsgp_multigroup_inducing_weights: basis frequency $b " *
                       "is zero; its spatial range is unbounded")
        r2 = lambda^2 / e
        for q in 1:P, p in 1:P
            t = zero(T)
            for axis in 1:D
                t += (positions[p, axis] - positions[q, axis])^2
            end
            chol[p, q] = p == q ? 1.00000001 : exp(-t / (2r2))
        end
        _brm_lower_cholesky!(chol)
        for i in 1:G
            for p in 1:P
                t = zero(T)
                for axis in 1:D
                    t += (coordinates[i, axis] - positions[p, axis])^2
                end
                d2[p] = t
            end
            # Relative to the nearest position, a factor the normalization
            # cancels: finite at any range.
            shift = minimum(d2)
            norm2 = zero(T)
            for p in 1:P
                t = exp(-(d2[p] - shift) / (2r2))
                for q in 1:(p - 1)
                    t -= chol[p, q] * a[q]
                end
                a[p] = t / chol[p, p]
                norm2 += a[p]^2
            end
            w = zero(T)
            for p in 1:P
                w += a[p] * z[p, b]
            end
            out[i, b] = w / sqrt(norm2)
        end
    end
    out
end

"""
    brm_multigroup_inducing_resolution(coordinates, n_points) -> r_min

The finest spatial range `n_points` free inducing positions are taken to
resolve: that of a grid of as many points over the levels' extent, 1.25
spacings `extent / (n_points^(1/D) - 1)` of the longest axis.
"""
brm_multigroup_inducing_resolution(coordinates::AbstractMatrix, n_points::Integer) =
    1.25 * maximum(maximum(c) - minimum(c) for c in eachcol(coordinates)) /
        (n_points^(1 / size(coordinates, 2)) - 1)

_brm_multigroup_approximation(method::Integer) =
    method == 1 ? Val(:inducing) : method == 2 ? Val(:hsgp) : error(
        "brm_hsgp_multigroup_coordinate_weights: `method` is 1 (inducing " *
        "points) or 2 (Hilbert space), got $method")

# Per frequency and axis, the approximation's frame for range `r`: the grid
# spacing and its kernel's Cholesky factor (in `frame`), or the box half-width.
function _brm_multigroup_axis_frame!(::Val{:inducing}, frame, axis, scale, r, k)
    h = sqrt(scale^2 + (r / 1.5)^2)
    for j in 1:k, i in 1:k
        frame[i, j, axis] = i == j ? 1.000000001 : exp(-(h * (i - j))^2 / (2r^2))
    end
    _brm_lower_cholesky!(view(frame, :, :, axis))
    h
end
_brm_multigroup_axis_frame!(::Val{:hsgp}, frame, axis, scale, r, k) = sqrt(scale^2 + (3.2r)^2)

# A level's (unnormalized) features on one axis at `offset` from the centre.
# The inducing kernel is taken relative to its value at the nearest grid
# point, a constant factor the normalization cancels exactly; at a range far
# below the spacing it would otherwise underflow to zero at every point.
function _brm_multigroup_axis_features!(::Val{:inducing}, a, axis, frame, h, offset, r, k)
    shift = (offset - h * (round(offset / h + (k + 1) / 2) - (k + 1) / 2))^2
    for p in 1:k
        t = exp(-((offset - h * (p - (k + 1) / 2))^2 - shift) / (2r^2))
        for q in 1:(p - 1)
            t -= frame[p, q, axis] * a[q, axis]
        end
        a[p, axis] = t / frame[p, p, axis]
    end
    a
end
function _brm_multigroup_axis_features!(::Val{:hsgp}, a, axis, frame, L, offset, r, k)
    nu1 = pi / (2L)
    for l in 1:k
        nu = l * nu1
        a[l, axis] = sin(nu * (offset + L)) * exp(-r^2 / 4 * (nu^2 - nu1^2))
    end
    a
end

"""
    brm_multigroup_coordinate_grid(coordinates, k, method, c=1.5) -> (center, scale, r_min)

The data-derived geometry of `multigroup(coordinates, lambda; k, method, c)`:
per axis the midpoint of the coordinates and the approximation's base scale,
and the resolution `r_min` every frequency's spatial range enters with: the
range is `√(r² + r_min²)` for the exact kernel's range `r`. Hilbert space
(`method = 2`, the formula default): the base box half-width `c · extent / 2`
and the HSGP length-scale floor of that box. Inducing points (`method = 1`):
the base spacing `extent / (k - 1)` and 1.25 times the largest of them.
"""
function brm_multigroup_coordinate_grid(coordinates::AbstractMatrix, k::Integer,
                                        method::Integer, c::Real=1.5)
    k >= 2 || error("multigroup(coordinates, lambda; k): `k` must be at least 2, got $k")
    D = size(coordinates, 2)
    lower = [minimum(view(coordinates, :, a)) for a in 1:D]
    upper = [maximum(view(coordinates, :, a)) for a in 1:D]
    center = (lower .+ upper) ./ 2
    _brm_multigroup_coordinate_grid(_brm_multigroup_approximation(method), center,
                                    upper .- lower, k, c)
end
_brm_multigroup_coordinate_grid(approximation::Val{:inducing}, center, extent, k, c) =
    _brm_multigroup_grid_frame(approximation, center, extent ./ (k - 1), k)
_brm_multigroup_coordinate_grid(approximation::Val{:hsgp}, center, extent, k, c) =
    _brm_multigroup_grid_frame(approximation, center, c .* extent ./ 2, k)
_brm_multigroup_grid_frame(approximation, center, scale, k) =
    (center, scale, _brm_multigroup_resolution(approximation, center, scale, k))
# The finest range the approximation resolves: 1.25 base grid spacings, or the
# HSGP length-scale floor of the base box.
_brm_multigroup_resolution(::Val{:inducing}, center, h0, k) = 1.25maximum(h0)
_brm_multigroup_resolution(::Val{:hsgp}, center, L0, k) =
    maximum(_brm_hsgp_rho_lower((center[a], L0[a]), k) for a in eachindex(L0))

# `sum_a rho_a^2 omega2[b, a]`: a one-axis basis carries one frequency per mode.
_brm_hsgp_frequency_exponent(omega2::AbstractVector, rho::Real, b) = rho^2 * omega2[b]
_brm_hsgp_frequency_exponent(omega2::AbstractVector, rho, b) = rho[1]^2 * omega2[b]
function _brm_hsgp_frequency_exponent(omega2::AbstractMatrix, rho::Real, b)
    e = zero(rho)
    for a in axes(omega2, 2)
        e += rho^2 * omega2[b, a]
    end
    e
end
function _brm_hsgp_frequency_exponent(omega2::AbstractMatrix, rho, b)
    e = zero(eltype(rho))
    for a in axes(omega2, 2)
        e += rho[a]^2 * omega2[b, a]
    end
    e
end

# Plain-loop lower Cholesky factor of a symmetric positive-definite matrix, so
# Enzyme differentiates it without a LAPACK rule.
_brm_lower_cholesky(C::AbstractMatrix{T}) where {T} =
    _brm_lower_cholesky_into!(zeros(T, size(C)...), C)
# In place on a freshly allocated, exclusively owned matrix: its lower
# triangle becomes the factor and its strict upper triangle zero.
function _brm_lower_cholesky!(C::AbstractMatrix)
    _brm_lower_cholesky_into!(C, C)
    for j in axes(C, 2), i in 1:(j - 1)
        C[i, j] = 0
    end
    C
end
function _brm_lower_cholesky_into!(chol::AbstractMatrix, C::AbstractMatrix)
    n = size(C, 1)
    for j in 1:n
        d = C[j, j]
        for k in 1:(j - 1)
            d -= chol[j, k]^2
        end
        d > 0 || error("brm_hsgp_multigroup_weights: the multigroup coherence " *
                       "is not positive definite (pivot $j is $d)")
        chol[j, j] = sqrt(d)
        for i in (j + 1):n
            t = C[i, j]
            for k in 1:(j - 1)
                t -= chol[i, k] * chol[j, k]
            end
            chol[i, j] = t / chol[j, j]
        end
    end
    chol
end

function brm_hsgp_sqrt_spd(omega2, sigma, rho::Real)
    scale = sigma * (rho * sqrt(2pi))^(size(omega2, 2) / 2)
    result = Vector{promote_type(typeof(sigma), typeof(rho))}(undef, size(omega2, 1))
    for b in axes(omega2, 1)
        exponent = zero(rho)
        for j in axes(omega2, 2)
            exponent += rho^2 * omega2[b, j]
        end
        result[b] = scale * exp(-0.25exponent)
    end
    result
end

function brm_hsgp_sqrt_spd(omega2, sigma, rhos)
    scale = sigma
    for rho in rhos
        scale *= sqrt(rho * sqrt(2pi))
    end
    result = Vector{promote_type(typeof(sigma), eltype(rhos))}(undef, size(omega2, 1))
    for b in axes(omega2, 1)
        exponent = zero(sigma)
        for j in axes(omega2, 2)
            exponent += rhos[j]^2 * omega2[b, j]
        end
        result[b] = scale * exp(-0.25exponent)
    end
    result
end

function brm_hsgp_periodic_sqrt_spd(harmonics, sigma, rho)
    result = Vector{promote_type(typeof(sigma), typeof(rho))}(undef, length(harmonics))
    for j in eachindex(harmonics)
        result[j] = exp(_brm_hsgp_periodic_log_scale(harmonics[j], sigma, rho))
    end
    result
end

# Legacy emitter bindings delegate to the adopted construction family.
# All backends importing these names use one statistical implementation.
brm_gp_covariance(x, sigma, rho, period, jitter) = iszero(period) ?
    StatisticalPreparation.gp_exp_quad_cov(x, sigma, rho, jitter) :
    StatisticalPreparation.gp_periodic_cov(x, sigma, rho, period, jitter)

brm_gp_latent(covariance, z) = StatisticalPreparation.gp_chol_latent(covariance, z)

brm_level_indices(labels, source) = _rk_value_level_indices(labels, source)
# A grouping's fitted levels: reading them as a grouping column gives the same
# levels back (`_rk_grouping_levels`), and a categorical column keeps its
# level order. The emitted program reads them in place of the fit-role column.
brm_fit_levels(source::CA.CategoricalArray) = CA.categorical(CA.levels(source);
    levels=CA.levels(source), ordered=CA.isordered(source))
brm_fit_levels(source::AbstractVector) = collect(CA.levels(source))
# No longer emitted: the gather is written inline as `draws[indices, margin]`.
# Kept for RK sources printed before that change, which import it by name.
brm_ranef_column(draws, indices, margin) = draws[indices, margin]
brm_dummy(values, level) = _rk_value_dummy(values, level)

# A rectangular panel's ordinary Julia slice, preserving subject order.
brm_panel_slice(column, timepoints, subject) =
    column[((subject - 1) * timepoints + 1):(subject * timepoints)]

# Per-subject cells concatenated in subject order with one allocation of their
# total length. `reduce(vcat, cells; init=…)` would fold one pairwise `vcat`
# per cell, reallocating the accumulator each time: quadratic in the number of
# cells, on every evaluation and reverse pass. Base's specialized
# `reduce(vcat, cells)` refuses zero cells, so that case is explicit. Scalar
# cells already hold one value per subject.
brm_flatten_cells(cells::AbstractVector{<:AbstractVector}) =
    isempty(cells) ? eltype(eltype(cells))[] : reduce(vcat, cells)
brm_flatten_cells(cells::AbstractVector{<:Number}) = collect(cells)
brm_flatten_cells(cells::AbstractVector) = collect(Iterators.flatten(cells))
