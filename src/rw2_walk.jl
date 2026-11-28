# The second-order random walk's basis transform, as plain Julia.
#
# D takes second differences. If AA' = DD' and CC' = A'A + sigma² I,
# r = A(C' \ z) transforms the packed curvature to standardized differences
# (the derivation is beside the Stan definitions in sbimpl.jl).
#
# Both backends use these three functions. The RK program calls them by name
# (`rk_ast.jl`, `_RK_SOURCE_IMPORTS`), and the Stan backend's `@deffun` block
# in sbimpl.jl extends them with their Stan definitions, which StanBlocks keeps
# Stan-only for a function that already exists. Until snag
# `stanblocks-and-b-0e9cf921` their only Julia methods were the ones StanBlocks
# generated from that block (`@juliacompat`), so an RK `rw(…; order=2)` term
# needed StanBlocks to run. The bodies follow the Stan definitions line for
# line, and each is a `@_brm_traceable` definition (ragged_rows.jl), so
# `jax_source` translates the RK program's calls from this source.

# Columns 1:3 and 4:6 hold the diagonal and first two subdiagonals of the
# lower-triangular factors A and C, indexed by row. A is known exactly; C comes
# from the QR of [A; sigma*I], with rows inserted in order of their first
# nonzero and a 2-by-2 upper triangle carried to the next column. Orthogonal
# rotations avoid subtracting nearly equal squares in the final pivots of a
# long walk with a small innovation scale.
@_brm_traceable function brm_rw2_factors(sigma::Real, m::Integer)
    bands = zeros(m, 6)
    for i in 1:m
        denom = (1.0 * i) * (i + 1.0)
        bands[i, 1] = sqrt((i + 2.0) * (i + 3.0) / denom)
        if i > 1
            bands[i, 2] = -2.0 * sqrt((i - 1.0) * (i + 2.0) / denom)
        end
        if i > 2
            bands[i, 3] = 1.0 / bands[i - 2, 1]
        end
    end
    carry_diag = bands[1, 1]
    carry_off = 0.0
    carry_next = 0.0
    if m > 1
        carry_diag = hypot(bands[1, 1], bands[2, 2])
        carry_off = (bands[2, 2] / carry_diag) * bands[2, 1]
        carry_next = (bands[1, 1] / carry_diag) * bands[2, 1]
    end
    for i in 1:m
        pivot = hypot(carry_diag, sigma)
        bands[i, 4] = pivot
        if i < m
            super1 = (carry_diag / pivot) * carry_off
            next_diag = hypot(carry_next, (sigma / pivot) * carry_off)
            if i + 1 < m
                # The new row of A spans columns i through i+2.
                pivot2 = hypot(pivot, bands[i + 2, 3])
                cosine = pivot / pivot2
                sine = bands[i + 2, 3] / pivot2
                bands[i, 4] = pivot2
                bands[i + 1, 5] = cosine * super1 + sine * bands[i + 2, 2]
                bands[i + 2, 6] = sine * bands[i + 2, 1]
                rem1 = cosine * bands[i + 2, 2] - sine * super1
                rem2 = cosine * bands[i + 2, 1]
                carry_diag = hypot(next_diag, rem1)
                carry_off = (rem1 / carry_diag) * rem2
                carry_next = (next_diag / carry_diag) * rem2
            else
                bands[i + 1, 5] = super1
                carry_diag = next_diag
            end
        end
    end
    bands
end

# The basis's first two entries unchanged, then r = A(C' \ z) for the packed
# curvature z = basis[3:n], then log|A| - log|C| in entry n + 1.
@_brm_traceable function brm_rw2_transform(sigma::Real, basis::AbstractVector)
    n = length(basis)
    m = n - 2
    bands = brm_rw2_factors(sigma, m)
    work = zeros(m)
    out = zeros(n + 1)
    out[1] = basis[1]
    out[2] = basis[2]
    for k in 1:m
        i = m + 1 - k
        rhs = basis[i + 2]
        if i < m
            rhs -= bands[i + 1, 5] * work[i + 1]
        end
        if i + 1 < m
            rhs -= bands[i + 2, 6] * work[i + 2]
        end
        work[i] = rhs / bands[i, 4]
    end
    for i in 1:m
        r = bands[i, 1] * work[i]
        if i > 1
            r += bands[i, 2] * work[i - 1]
        end
        if i > 2
            r += bands[i, 3] * work[i - 2]
        end
        out[i + 2] = r
        out[n + 1] += log(bands[i, 1]) - log(bands[i, 4])
    end
    out
end

# The field path of a basis: integrate the standardized second differences
# twice, remove the level and linear trend, scale by sigma, and add back the
# basis's level (entry 1, scaled) and trend (entry 2).
@_brm_traceable function brm_rw2_basis_path(sigma::Real, basis::AbstractVector)
    n = length(basis)
    x = zeros(n)
    increment = 0.0
    for t in 3:n
        increment += basis[t]
        x[t] = x[t - 1] + increment
    end
    level = sum(x) / n
    moment = 0.0
    mid = (n + 1.0) / 2.0
    for t in 1:n
        moment += (t - mid) * x[t]
    end
    slope = 12.0 * moment / (n * (n - 1.0) * (n + 1.0))
    for t in 1:n
        x[t] = sigma * ((x[t] - level) - (t - mid) * slope + 0.001 * basis[1]) +
            (t - mid) * basis[2]
    end
    x
end
