# test/level_correlation_multigroup_coordinates.jl — the multigroup coherence
# of a grouped HSGP over known level coordinates,
# `hsgp(t; by=g, cor=multigroup((x, y), lambda; k))` (src/level_correlation.jl):
# basis frequency `omega` correlates two levels at distance `d` by
# `exp(-d^2 / (2 r^2))`, `r = lambda / (rho omega)`, approximated with `k`
# functions per axis by a Hilbert-space sine basis (`method = :hsgp`, the
# default) or inducing points on a grid (`method = :inducing`, for
# approximate-inference loops); with `lambda` at or above 2√(ln 100) times the
# approximation's resolution, frequencies it cannot resolve are negligible.
# Design: decisions `1ug0391` (spelling, range law), `0u9tpd6` with
# the user's feedback on it and on brief `dg6hem` (both approximations, HSGP the
# default), `114k5fg` and `100gkpq` (no clamp; a lower bound on lambda's prior),
# `0yrmmdq` (free inducing positions).
#
#     julia --project=test test/level_correlation_multigroup_coordinates.jl
#
# Set `BRM_LEVEL_CORRELATION_RUNTIME=0` to skip the compiled-model sets.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LinearAlgebra
using Random
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
import BridgeStan
import JSON

const BRM = BayesianRegressionModels
const BS = BridgeStan
const RUNTIME = get(ENV, "BRM_LEVEL_CORRELATION_RUNTIME", "1") != "0"
const CACHE = joinpath(tempdir(), "brm-level-correlation-multigroup-coordinates")
const G = 12           # levels
const T = 8            # time points per level
const KG = 6           # inducing points per axis

function coordinate_data(; seed=3)
    rng = Random.MersenneTwister(seed)
    sx, sy = 100 .* rand(rng, G), 150 .* rand(rng, G)
    (; t=repeat(collect(range(0.0, 10.0; length=T)), G), g=repeat(1:G, inner=T),
       y=randn(rng, G * T), sx, sy, u=randn(rng, G * T),
       # 16 free inducing positions inside the levels' bounding box
       ux=minimum(sx) .+ (maximum(sx) - minimum(sx)) .* rand(rng, 16),
       uy=minimum(sy) .+ (maximum(sy) - minimum(sy)) .* rand(rng, 16))
end

const mgc = @brm begin
    lambda ~ LogNormal(5.7, 0.3; lower=230.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), lambda; k=6))
    y ~ Normal(mu, 1.0)
end
# A constant spatial scale and the default grid.
const fixed = @brm begin
    mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 80.0))
    y ~ Normal(mu, 1.0)
end
# Inducing points.
const mgi = @brm begin
    lambda ~ LogNormal(5.7, 0.3; lower=230.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), lambda; k=6, method=:inducing))
    y ~ Normal(mu, 1.0)
end
# The term's own spatial scale (decision `100gkpq`): `lambda` left out.
const mgd = @brm begin
    mu ~ 1 + t + hsgp(t; k=5, by=g, cor=multigroup((sx, sy); k=6))
    y ~ Normal(mu, 1.0)
end
# Free inducing positions (decision `0yrmmdq`): data, and declared vector
# parameters an approximate-inference loop would optimize.
const mgp = @brm begin
    lambda ~ LogNormal(5.8, 0.3; lower=260.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g,
                      cor=multigroup((sx, sy), lambda; method=:inducing, inducing=(ux, uy)))
    y ~ Normal(mu, 1.0)
end
const mgq = @brm begin
    lambda ~ LogNormal(5.8, 0.3; lower=260.0)
    px[1:16] .~ Normal.(50.0, 30.0)
    py[1:16] .~ Normal.(75.0, 40.0)
    mu ~ 1 + t + hsgp(t; k=5, by=g,
                      cor=multigroup((sx, sy), lambda; method=:inducing, inducing=(px, py)))
    y ~ Normal(mu, 1.0)
end

stan(brmi) = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__))
# Stan source with each call's arguments on one line: a long call wraps.
flat(code) = replace(code, r"\(\s+" => "(", r",\s+" => ", ", r"\s+\)" => ")")

# The K x P feature matrix of one frequency, read off unit standardized
# weights: column p is the weights of z = e_p.
function features(omega2, rho, s, center, scale, lambda, k, b; method=1)
    P, M = k^size(s, 2), size(omega2, 1)
    hcat((brm_hsgp_multigroup_coordinate_weights(
        [p == q ? 1.0 : 0.0 for q in 1:P, _ in 1:M], omega2, rho, s, center, scale,
        lambda, method)[:, b] for p in 1:P)...)
end

# An independent dense reference: the unit-diagonal predictive process over
# the whole `k^2` grid at once, with no Kronecker factorization.
function dense_coherence(s, center, h0, r, k)
    hs = [sqrt(h0[a]^2 + (r / 1.5)^2) for a in 1:2]
    grid = [[center[1] + hs[1] * (p - (k + 1) / 2), center[2] + hs[2] * (q - (k + 1) / 2)]
            for p in 1:k for q in 1:k]
    kern(a, b) = exp(-sum(abs2, a .- b) / (2r^2))
    # The same 1e-9 jitter per axis, as a Kronecker sum of the axis kernels.
    Kx = [p == q ? 1.000000001 : exp(-(hs[1] * (p - q))^2 / (2r^2)) for p in 1:k, q in 1:k]
    Ky = [p == q ? 1.000000001 : exp(-(hs[2] * (p - q))^2 / (2r^2)) for p in 1:k, q in 1:k]
    Kuu = kron(Kx, Ky)
    Kxu = [kern(s[i, :], u) for i in axes(s, 1), u in grid]
    C = Kxu * (Kuu \ Kxu')
    d = sqrt.(diag(C))
    C ./ (d * d')
end

# An independent dense Hilbert-space reference: the whole `k^2` tensor basis
# over the box of half-width sqrt(L0^2 + (3.2 r)^2) and its spectral weights.
function dense_hsgp(s, center, L0, r, k)
    L = sqrt.(L0 .^ 2 .+ (3.2r)^2)
    modes = [(a, b) for a in 1:k for b in 1:k]
    Psi = [sin(pi * a / (2L[1]) * (s[i, 1] - center[1] + L[1])) *
           sin(pi * b / (2L[2]) * (s[i, 2] - center[2] + L[2])) for i in axes(s, 1), (a, b) in modes]
    nu2 = [(pi * a / (2L[1]))^2 + (pi * b / (2L[2]))^2 for (a, b) in modes]
    B = Psi * Diagonal(exp.(-(r^2 / 2) .* (nu2 .- minimum(nu2)))) * Psi'
    d = sqrt.(diag(B))
    B ./ (d * d')
end

@testset "the coordinate coherence" begin
    rng = Random.MersenneTwister(5)
    s = hcat(100 .* rand(rng, 30), 150 .* rand(rng, 30))
    k = 10
    center, h0, r_min = brm_multigroup_coordinate_grid(s, k, 1)
    @test center ≈ [(minimum(s[:, a]) + maximum(s[:, a])) / 2 for a in 1:2]
    @test h0 ≈ [(maximum(s[:, a]) - minimum(s[:, a])) / (k - 1) for a in 1:2]
    @test r_min == 1.25maximum(h0)
    D2 = [sum(abs2, s[i, :] .- s[j, :]) for i in axes(s, 1), j in axes(s, 1)]
    rho, lambda = 1.3, 60.0
    omega2 = [0.01, 0.04, 0.2, 1.5, 9.0]
    for b in eachindex(omega2)
        r = lambda / (rho * sqrt(omega2[b]))
        F = features(omega2, rho, s, center, h0, lambda, k, b)
        C = F * F'
        @test all(diag(C) .≈ 1)
        # the Kronecker form equals the dense predictive process
        @test C ≈ dense_coherence(s, center, h0, r, k) atol=1e-8
        # and approximates the exact kernel at every range it resolves
        r >= r_min && @test maximum(abs.(C .- exp.(-D2 ./ (2r^2)))) <= 0.02
    end
    # A range far below the resolution stays a finite, unit-diagonal
    # coherence (its frequency is negligible by the formula's bound on λ), and
    # the weights move with λ there too: no plateau.
    @test lambda / (rho * 3) < r_min
    z = randn(rng, k^2, length(omega2))
    for method in (1, 2)
        scale = method == 1 ? h0 : brm_multigroup_coordinate_grid(s, k, 2, 1.5)[2]
        F = features([1.0], 1.0, s, center, scale, r_min / 200, k, 1; method)
        @test all(isfinite, F)
        @test all(diag(F * F') .≈ 1)
        Wl(l) = brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, s, center, scale,
                                                       l, method)[:, 5]
        @test maximum(abs.(Wl(lambda * 1.001) .- Wl(lambda * 0.999))) > 1e-7
    end
    # Weights are linear in z.
    W = brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, s, center, h0, lambda, 1)
    @test W[:, 2] ≈ features(omega2, rho, s, center, h0, lambda, k, 2) * z[:, 2]
    # One length scale per time axis: the exponent is the sum over axes.
    omega2_2 = [0.01 0.02; 0.3 0.1]
    W2 = brm_hsgp_multigroup_coordinate_weights(z[:, 1:2], omega2_2, [1.3, 0.4], s,
                                                center, h0, lambda, 1)
    W1 = brm_hsgp_multigroup_coordinate_weights(z[:, 1:2],
        [1.3^2 * 0.01 + 0.4^2 * 0.02, 1.3^2 * 0.3 + 0.4^2 * 0.1], 1.0, s, center, h0,
        lambda, 1)
    @test W2 ≈ W1 atol=1e-12
    # The Hilbert-space approximation: box half-width c · half-extent, the
    # HSGP length-scale floor of that box, and the dense tensor basis.
    center2, L0, r_min2 = brm_multigroup_coordinate_grid(s, k, 2, 1.5)
    @test center2 == center
    @test L0 ≈ 1.5 .* [(maximum(s[:, a]) - minimum(s[:, a])) / 2 for a in 1:2]
    @test r_min2 ≈ maximum((4 * L / pi) * sqrt(log(100) / (k^2 - 1)) for L in L0)
    for b in eachindex(omega2)
        r = lambda / (rho * sqrt(omega2[b]))
        F = features(omega2, rho, s, center, L0, lambda, k, b; method=2)
        C = F * F'
        @test all(diag(C) .≈ 1)
        @test C ≈ dense_hsgp(s, center, L0, r, k) atol=1e-10
        r >= r_min2 && @test maximum(abs.(C .- exp.(-D2 ./ (2r^2)))) <= 0.03
    end
    @test_throws "`method` is 1" brm_hsgp_multigroup_coordinate_weights(z, omega2, rho, s,
        center, h0, lambda, 3)
end

@testset "the approximation at 1,514 levels" begin
    # The probe table of decision `100gkpq`: 1,514 uniform points in a
    # 640 x 870 km box, k = 17 functions per axis, the error at ranges from the
    # resolution up: inducing points 0.016 at theirs (1.25 grid spacings),
    # falling with the range; the Hilbert-space basis 1e-3 throughout.
    rng = Random.MersenneTwister(1)
    s = hcat(640 .* rand(rng, 1514), 870 .* rand(rng, 1514))
    k = 17
    center, h0, r_min = brm_multigroup_coordinate_grid(s, k, 1)
    D2 = [sum(abs2, s[i, :] .- s[j, :]) for i in axes(s, 1), j in axes(s, 1)]
    for (r, tolerance) in ((r_min, 0.016), (1.5r_min, 0.005), (200.0, 1e-3), (1000.0, 1e-4), (1e4, 1e-4))
        F = features([1.0], 1.0, s, center, h0, r, k, 1)
        @test maximum(abs.(F * F' .- exp.(-D2 ./ (2r^2)))) <= tolerance
    end
    center, L0, r_min = brm_multigroup_coordinate_grid(s, k, 2, 1.5)
    for r in (r_min, 200.0, 400.0, 1000.0, 1e4)
        F = features([1.0], 1.0, s, center, L0, r, k, 1; method=2)
        @test maximum(abs.(F * F' .- exp.(-D2 ./ (2r^2)))) <= 1e-3
    end
end

@testset "free inducing positions" begin
    rng = Random.MersenneTwister(9)
    s = hcat(100 .* rand(rng, 30), 150 .* rand(rng, 30))
    u = hcat(100 .* rand(rng, 12), 150 .* rand(rng, 12))
    P = size(u, 1)
    omega2, rho, lambda = [0.01, 0.2, 9.0], 1.3, 60.0
    unit = [[p == q ? 1.0 : 0.0 for q in 1:P, _ in eachindex(omega2)] for p in 1:P]
    for b in eachindex(omega2)
        r = lambda / (rho * sqrt(omega2[b]))
        F = hcat((brm_hsgp_multigroup_inducing_weights(e, omega2, rho, s, u, lambda)[:, b]
                  for e in unit)...)
        C = F * F'
        @test all(diag(C) .≈ 1)
        # the dense, unit-diagonal predictive process with the same jitter
        kern(a, c) = exp(-sum(abs2, a .- c) / (2r^2))
        Kuu = [kern(u[p, :], u[q, :]) + (p == q ? 1e-8 : 0.0) for p in 1:P, q in 1:P]
        Kxu = [kern(s[i, :], u[p, :]) for i in axes(s, 1), p in 1:P]
        Q = Kxu * (Symmetric(Kuu) \ Kxu')
        @test C ≈ Q ./ (sqrt.(diag(Q)) * sqrt.(diag(Q))') atol=1e-6
    end
    # finite far below the positions' spacing
    @test all(isfinite, brm_hsgp_multigroup_inducing_weights(randn(rng, P, 1), [1.0], 1.0,
                                                             s, u, 0.01))
    @test brm_multigroup_inducing_resolution(s, 16) ≈
        1.25 * maximum(maximum(c) - minimum(c) for c in eachcol(s)) / 3
    @test_throws "positions must be 11 x 2" brm_hsgp_multigroup_inducing_weights(
        randn(rng, 11, 1), [1.0], 1.0, s, u, 60.0)
end

@testset "SBBRMI emission" begin
    df = coordinate_data()
    sb = SBBRMI(mgc(df); mod=@__MODULE__)
    code = flat(BRM.stan_code(sb))
    @test occursin("real<lower=230.0> lambda;", code)
    @test occursin("brm_hsgp_multigroup_coordinate_weights(b_hsgpw_t_g, omega2_hsgp_t_by_g, " *
                   "hsgp_t_by_g_rho, multigroup_sx_sy_coordinates, multigroup_sx_sy_hsgp6_c1p5_center, " *
                   "multigroup_sx_sy_hsgp6_c1p5_scale, lambda, 6, 2)", code)
    @test occursin("hsgp_t_by_g_beta_scaled = diag_post_multiply(hsgp_t_by_g_beta_level, hsgp_t_by_g_sqrt_spd);", code)
    # The standardized weights live on the 6 x 6 grid, not one row per level.
    @test sb.data[:n_rows_hsgpw_t_g] == KG^2
    @test sb.data[:multigroup_sx_sy_coordinates] == hcat(df.sx, df.sy)
    center, L0, _ = brm_multigroup_coordinate_grid(hcat(df.sx, df.sy), KG, 2, 1.5)
    @test sb.data[:multigroup_sx_sy_hsgp6_c1p5_center] == center
    @test sb.data[:multigroup_sx_sy_hsgp6_c1p5_scale] == L0
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # A constant spatial scale is inline; the default basis has 20 functions per axis.
    code = flat(stan(fixed(df)))
    @test occursin("multigroup_sx_sy_hsgp20_c1p5_scale, 80.0, 20, 2)", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # Inducing points: their own grid geometry and method code 1.
    sb = SBBRMI(mgi(df); mod=@__MODULE__)
    code = flat(BRM.stan_code(sb))
    @test occursin("multigroup_sx_sy_inducing6_scale, lambda, 6, 1)", code)
    @test sb.data[:multigroup_sx_sy_inducing6_scale] ==
        brm_multigroup_coordinate_grid(hcat(df.sx, df.sy), KG, 1)[2]
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # Free positions: one weight row per position, read from a data matrix or
    # from the declared vectors side by side.
    sb = SBBRMI(mgp(df); mod=@__MODULE__)
    code = flat(BRM.stan_code(sb))
    @test occursin("multigroup_sx_sy_coordinates, multigroup_sx_sy_inducing_ux_uy, lambda)", code)
    @test sb.data[:multigroup_sx_sy_inducing_ux_uy] == hcat(df.ux, df.uy)
    @test sb.data[:n_rows_hsgpw_t_g] == 16
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    code = flat(stan(mgq(df)))
    @test occursin("brm_hsgp_multigroup_inducing_weights(", code)
    @test occursin("append_col(px, py)", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # The term's own spatial scale: log-normal around the largest extent,
    # bounded below at 2√(ln 100) times the resolution.
    sb = SBBRMI(mgd(df); mod=@__MODULE__)
    code = flat(BRM.stan_code(sb))
    @test occursin("hsgp_t_by_g_lambda", code)
    s = hcat(df.sx, df.sy)
    @test sb.data[:multigroup_sx_sy_lambda_location] ≈
        log(maximum(maximum(c) - minimum(c) for c in eachcol(s)))
    @test sb.data[:multigroup_sx_sy_hsgp6_c1p5_lambda_lower] ≈
        2sqrt(log(100)) * brm_multigroup_coordinate_grid(s, KG, 2, 1.5)[3]
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "descriptor and description" begin
    df = coordinate_data()
    d = brm_descriptor(SBBRMI(mgc(df); mod=@__MODULE__))
    @test brm_output(d, :lambda).role === :parameter
    description = BRM.brm_description(d)
    # `cor = multigroup(...)` is described by the HSGP's level prose: no gap.
    @test description.complete
    md = BRM.brm_description_markdown(description)
    @test occursin("multigroup coherence over the groups' coordinates `sx`, `sy` with the " *
                   "spatial scale `lambda`", md)
    @test occursin("Hilbert-space basis of 6 sine functions per axis", md)
    de = BRM.brm_description_markdown(BRM.brm_description(d; texts=BRM.description_texts(:de)))
    @test occursin("Multigroup-Kohärenz über die Koordinaten `sx`, `sy`", de)
    @test occursin("Hilbertraum-Basis aus 6 Sinusfunktionen je Achse", de)
    md = BRM.brm_description_markdown(BRM.brm_description(brm_descriptor(
        SBBRMI(mgi(df); mod=@__MODULE__))))
    @test occursin("grid of 6 points per axis", md)
    description = BRM.brm_description(brm_descriptor(SBBRMI(mgd(df); mod=@__MODULE__)))
    @test description.complete
    @test occursin("The spatial scale λ is the term's own parameter",
                   BRM.brm_description_markdown(description))
    description = BRM.brm_description(brm_descriptor(SBBRMI(mgq(df); mod=@__MODULE__)))
    @test description.complete
    @test occursin("inducing points at the positions `px`, `py`",
                   BRM.brm_description_markdown(description))
end

@testset "wrong shapes are refused" begin
    df = coordinate_data()
    refused(model, message) = (@test_throws message SBBRMI(model(df); mod=@__MODULE__))
    # refused: a random-effect block has no frequencies for the coherence to
    # depend on (decision `0w54lkq`, scope)
    refused(@brm(begin
        mu ~ 1 + (1 | gr(g; cor=multigroup((sx, sy), 400.0)))
        y ~ Normal(mu, 1.0)
    end), "only a grouped `hsgp(...; by=..., cor=multigroup(...))` has")
    # refused: one coordinate per level (decision `1ug0391`, spelling)
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((t, u), 40.0))
        y ~ Normal(mu, 1.0)
    end), "has 96 coordinates per axis, but grouping `g` has 12 levels")
    # refused: free positions need `method = :inducing`, replace `k`, give one
    # field per axis of one length, each a data vector or a declared vector
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; inducing=(ux, uy)))
        y ~ Normal(mu, 1.0)
    end), "takes `inducing` positions only with `method = :inducing`")
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g,
                      cor=multigroup((sx, sy), 400.0; k=4, method=:inducing, inducing=(ux, uy)))
        y ~ Normal(mu, 1.0)
    end), "takes either `k`")
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; method=:inducing, inducing=(ux,)))
        y ~ Normal(mu, 1.0)
    end), "one field per coordinate axis (2)")
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; method=:inducing, inducing=(ux, sx)))
        y ~ Normal(mu, 1.0)
    end), "positions have different lengths (16, 12)")
    refused(@brm(begin
        px ~ Normal(50.0, 30.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; method=:inducing, inducing=(px, uy)))
        y ~ Normal(mu, 1.0)
    end), "neither a data vector nor a declared vector parameter")
    # refused: the range `lambda / (rho omega)` needs lambda > 0
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), -1.0))
        y ~ Normal(mu, 1.0)
    end), "needs a positive finite spatial scale `lambda`")
    # refused: the grid needs at least two points per axis
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; k=1))
        y ~ Normal(mu, 1.0)
    end), "integer `k >= 2`")
    # refused: the inducing grid spans the coordinates, so it has no box factor
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; method=:inducing, c=1.5))
        y ~ Normal(mu, 1.0)
    end), "takes the box factor `c` only with `method = :hsgp`")
    # refused: the box must contain the coordinates with a margin
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; c=0.9))
        y ~ Normal(mu, 1.0)
    end), "needs a box factor `c > 1`")
    # refused: the two approximations of decision `0u9tpd6` only
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; method=:fourier))
        y ~ Normal(mu, 1.0)
    end), "takes `method = :inducing` or `method = :hsgp`")
    # refused: no maximum spatial range (decision `1ug0391`, range)
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy), 400.0; range=100.0))
        y ~ Normal(mu, 1.0)
    end), "takes the keywords `k` (functions per axis)")
    # refused: at most three coordinate axes (decision `1ug0391`, scope)
    refused(@brm(begin
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup((sx, sy, sx, sy), 40.0; k=2))
        y ~ Normal(mu, 1.0)
    end), "one to three coordinate axes")
end

# A raw-Stan coordinate multigroup HSGP written by hand: per frequency, each
# level's weight is the unit-normalized Kronecker product of its two axes'
# features, applied to that frequency's standardized weights. The features are
# the range-scaled inducing grid's whitened kernel (method 1) or the
# range-scaled box's spectrally weighted sine basis (method 2).
const MGC_REFERENCE = """
data { int N; int G; int M; int k; int method; vector[N] t; vector[N] y; array[N] int g;
       matrix[N, M] PHI; vector[M] omega2; real rho_lower;
       matrix[G, 2] s; vector[2] center; vector[2] scale; }
parameters {
  real<lower=230> lambda;
  vector[2] beta;
  vector[M * k * k] zw;
  real<lower=rho_lower> rho;
  real<lower=0> sf;
}
model {
  matrix[k * k, M] Z = to_matrix(zw, M, k * k)';
  matrix[G, M] W;
  for (j in 1:M) {
    real r = lambda / sqrt(square(rho) * omega2[j]);
    array[2] matrix[G, k] A;
    for (a in 1:2) {
      if (method == 1) {
        real h = sqrt(square(scale[a]) + square(r / 1.5));
        matrix[k, k] Kuu;
        for (p in 1:k) for (q in 1:k)
          Kuu[p, q] = p == q ? 1.000000001 : exp(-square(h * (p - q)) / (2 * square(r)));
        matrix[k, k] Lu = cholesky_decompose(Kuu);
        for (i in 1:G) {
          vector[k] kv;
          real o = s[i, a] - center[a];
          real near = square(o - h * (round(o / h + (k + 1) / 2.0) - (k + 1) / 2.0));
          for (p in 1:k)
            kv[p] = exp(-(square(o - h * (p - (k + 1) / 2.0)) - near) / (2 * square(r)));
          vector[k] av = mdivide_left_tri_low(Lu, kv);
          A[a][i] = (av / norm2(av))';
        }
      } else {
        real L = sqrt(square(scale[a]) + square(3.2 * r));
        for (i in 1:G) {
          vector[k] av;
          for (p in 1:k) {
            real nu = p * pi() / (2 * L);
            av[p] = sin(nu * (s[i, a] - center[a] + L)) * exp(-square(r) / 4 * (square(nu) - square(pi() / (2 * L))));
          }
          A[a][i] = (av / norm2(av))';
        }
      }
    }
    for (i in 1:G) W[i, j] = dot_product(to_vector(A[1][i]' * A[2][i]), Z[:, j]);
  }
  vector[M] spd = sf * sqrt(rho * sqrt(2 * pi())) * exp(-0.25 * square(rho) * omega2);
  vector[N] f;
  for (n in 1:N) f[n] = dot_product(PHI[n], W[g[n]]' .* spd);
  lambda ~ lognormal(5.7, 0.3); // constant bound: no truncation normalizer, as BRM emits
  beta ~ std_normal();
  zw ~ std_normal();
  rho ~ lognormal(0, 1);
  sf ~ lognormal(0, 1);
  y ~ normal(beta[1] + beta[2] * t + f, 1);
}
"""

function compiled(sb, tag)
    mkpath(CACHE)
    code = BRM.stan_code(sb)
    BRM.stan_instantiate(sb; path=joinpath(CACHE, tag * "_" * string(hash(code)) * ".stan"))
end

function reference_name(n)
    n == "lambda" && return n
    n == "hsgp_t_by_g_rho_iso" && return "rho"
    n == "hsgp_t_by_g_sigma" && return "sf"
    for (a, r) in ("pop_mu_beta_pop." => "beta.", "zflat_hsgpw_t_g." => "zw.")
        startswith(n, a) && return r * n[length(a)+1:end]
    end
    error("unmapped BRM coordinate $n")
end

@testset "SBBRMI density equals a hand-written coordinate multigroup program" begin
    if RUNTIME
    df = coordinate_data()
    for (tag, model, method) in (("mgc", mgc, 2), ("mgi", mgi, 1))
        sb = SBBRMI(model(df); mod=@__MODULE__)
        problem = compiled(sb, tag)
        names_brm = String.(BS.param_unc_names(problem.model))
        PHI = sb.data[:PHI_hsgp_t_by_g]
        M = size(PHI, 2)
        mkpath(CACHE)
        path = joinpath(CACHE, "mgc_reference_" * string(hash(MGC_REFERENCE)) * ".stan")
        isfile(path) || write(path, MGC_REFERENCE)
        s = hcat(df.sx, df.sy)
        center, scale, _ = brm_multigroup_coordinate_grid(s, KG, method, 1.5)
        reference = BS.StanModel(path, JSON.json(Dict("N"=>length(df.t), "G"=>G, "M"=>M,
            "k"=>KG, "method"=>method, "t"=>df.t, "y"=>df.y, "g"=>df.g,
            "PHI"=>[PHI[i, :] for i in axes(PHI, 1)],
            "omega2"=>vec(sb.data[:omega2_hsgp_t_by_g]),
            "rho_lower"=>sb.data[:rho_lower_hsgp_t_by_g],
            "s"=>[s[i, :] for i in axes(s, 1)], "center"=>center, "scale"=>scale)))
        names_ref = String.(BS.param_unc_names(reference))
        @test length(names_ref) == length(names_brm)
        perm = [only(findall(==(reference_name(n)), names_ref)) for n in names_brm]
        rng = Random.MersenneTwister(7)
        for _ in 1:4
            u = 0.7 .* randn(rng, length(names_brm))
            v = zeros(length(names_ref)); v[perm] .= u
            gu, gv = similar(u), similar(v)
            lu, _ = BS.log_density_gradient!(problem.model, u, gu; propto=false, jacobian=true)
            lv, _ = BS.log_density_gradient!(reference, v, gv; propto=false, jacobian=true)
            @test lu ≈ lv atol=1e-10 rtol=1e-12
            @test gu ≈ gv[perm] atol=1e-10 rtol=1e-10
        end
    end
    else
        @info "Skipping the compiled coordinate reference (BRM_LEVEL_CORRELATION_RUNTIME=0)"
        @test true
    end
end

# The free-position program by hand: per frequency, the positions' jittered
# kernel, each level's whitened kernel to them, normalized.
const MGP_REFERENCE = """
data { int N; int G; int M; int P; vector[N] t; vector[N] y; array[N] int g;
       matrix[N, M] PHI; vector[M] omega2; real rho_lower; matrix[G, 2] s; matrix[P, 2] u; }
parameters {
  real<lower=260> lambda;
  vector[2] beta;
  vector[M * P] zw;
  real<lower=rho_lower> rho;
  real<lower=0> sf;
}
model {
  matrix[P, M] Z = to_matrix(zw, M, P)';
  matrix[G, M] W;
  for (j in 1:M) {
    real r2 = square(lambda) / (square(rho) * omega2[j]);
    matrix[P, P] Kuu;
    for (p in 1:P) for (q in 1:P)
      Kuu[p, q] = p == q ? 1.00000001 : exp(-squared_distance(u[p], u[q]) / (2 * r2));
    matrix[P, P] Lu = cholesky_decompose(Kuu);
    for (i in 1:G) {
      vector[P] d2;
      for (p in 1:P) d2[p] = squared_distance(s[i], u[p]);
      vector[P] av = mdivide_left_tri_low(Lu, exp(-(d2 - min(d2)) / (2 * r2)));
      W[i, j] = dot_product(av, Z[:, j]) / norm2(av);
    }
  }
  vector[M] spd = sf * sqrt(rho * sqrt(2 * pi())) * exp(-0.25 * square(rho) * omega2);
  vector[N] f;
  for (n in 1:N) f[n] = dot_product(PHI[n], W[g[n]]' .* spd);
  lambda ~ lognormal(5.8, 0.3); // constant bound: no truncation normalizer, as BRM emits
  beta ~ std_normal();
  zw ~ std_normal();
  rho ~ lognormal(0, 1);
  sf ~ lognormal(0, 1);
  y ~ normal(beta[1] + beta[2] * t + f, 1);
}
"""

@testset "SBBRMI density equals a hand-written free-position program" begin
    if RUNTIME
        df = coordinate_data()
        sb = SBBRMI(mgp(df); mod=@__MODULE__)
        problem = compiled(sb, "mgp")
        names_brm = String.(BS.param_unc_names(problem.model))
        PHI = sb.data[:PHI_hsgp_t_by_g]
        path = joinpath(CACHE, "mgp_reference_" * string(hash(MGP_REFERENCE)) * ".stan")
        isfile(path) || write(path, MGP_REFERENCE)
        s, u = hcat(df.sx, df.sy), hcat(df.ux, df.uy)
        reference = BS.StanModel(path, JSON.json(Dict("N"=>length(df.t), "G"=>G,
            "M"=>size(PHI, 2), "P"=>16, "t"=>df.t, "y"=>df.y, "g"=>df.g,
            "PHI"=>[PHI[i, :] for i in axes(PHI, 1)],
            "omega2"=>vec(sb.data[:omega2_hsgp_t_by_g]),
            "rho_lower"=>sb.data[:rho_lower_hsgp_t_by_g],
            "s"=>[s[i, :] for i in axes(s, 1)], "u"=>[u[i, :] for i in axes(u, 1)])))
        names_ref = String.(BS.param_unc_names(reference))
        @test length(names_ref) == length(names_brm)
        perm = [only(findall(==(reference_name(n)), names_ref)) for n in names_brm]
        rng = Random.MersenneTwister(8)
        for _ in 1:4
            x = 0.7 .* randn(rng, length(names_brm))
            v = zeros(length(names_ref)); v[perm] .= x
            gx, gv = similar(x), similar(v)
            lx, _ = BS.log_density_gradient!(problem.model, x, gx; propto=false, jacobian=true)
            lv, _ = BS.log_density_gradient!(reference, v, gv; propto=false, jacobian=true)
            @test lx ≈ lv atol=1e-10 rtol=1e-12
            @test gx ≈ gv[perm] atol=1e-10 rtol=1e-10
        end
    else
        @test true
    end
end

rk_problem(rk) = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))

points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])

@testset "RKBRMI equals SBBRMI" begin
    if RUNTIME
        df = coordinate_data()
        for (tag, model) in ("mgc" => mgc, "fixed" => fixed, "mgi" => mgi, "mgd" => mgd,
                             "mgp" => mgp, "mgq" => mgq)
            m = model(df)
            rk = RKBRMI(m)
            sb = SBBRMI(m; mod=@__MODULE__)
            stan = compiled(sb, tag)
            rk_names = String.(coordinate_names(rk.model.layout))
            stan_names = BS.param_unc_names(stan.model)
            M = size(sb.data[:PHI_hsgp_t_by_g], 2)
            function stan_name(n)
                n == "lambda" && return n
                n == "hsgp_t.lambda" && return "hsgp_t_by_g_lambda"
                n == "hsgp_t.rho" && return "hsgp_t_by_g_rho_iso"
                n == "hsgp_t.sigma" && return "hsgp_t_by_g_sigma"
                n == "mu_Intercept" && return "pop_mu_beta_pop.1"
                occursin(r"^p[xy]\.\d+$", n) && return n
                x = match(r"^hsgp_t\.z\.(\d+)\.(\d+)$", n)
                x === nothing || return "zflat_hsgpw_t_g.$((parse(Int, x[1]) - 1) * M + parse(Int, x[2]))"
                x = match(r"^pop_mu\.beta_pop\.(\d+)$", n)
                x === nothing || return "pop_mu_beta_pop.$(x[1])"
                error("unmapped RK coordinate $n")
            end
            perm = [only(findall(==(stan_name(n)), stan_names)) for n in rk_names]
            @test sort(perm) == collect(eachindex(stan_names))
            problem = rk_problem(rk)
            # Declared positions sit at the data positions, offset by the point:
            # near-coincident positions make `K_uu` nearly singular, where any two
            # implementations agree only to its condition number times eps.
            at_positions(u) = [(x = match(r"^p([xy])\.(\d+)$", n); x === nothing ? u[i] :
                getproperty(df, Symbol(:u, x[1]))[parse(Int, x[2])] + u[i])
                for (i, n) in enumerate(rk_names)]
            for u in at_positions.(points(length(rk_names)))
                stan_u = zeros(length(u)); stan_u[perm] .= u
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                g = similar(stan_u)
                stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, g;
                                                        propto=false, jacobian=true)
                @test value ≈ stan_value atol=1e-10 rtol=1e-12
                @test gradient ≈ g[perm] atol=1e-10 rtol=1e-10
            end
        end
    else
        @info "Skipping RK/SB parity (BRM_LEVEL_CORRELATION_RUNTIME=0)"
        @test true
    end
end

@testset "readers: coherence by distance and the fitted term" begin
    if RUNTIME
      df = coordinate_data()
      # the grid basis, and free positions read per draw from the declared vectors
      for (tag, model) in ("mgc" => mgc, "mgd" => mgd, "mgq" => mgq)
        sb = SBBRMI(model(df); mod=@__MODULE__)
        stan = compiled(sb, tag)
        names = BS.param_names(stan.model; include_tp=true)
        rng = Random.MersenneTwister(4)
        unc = [0.6 .* randn(rng, BS.param_unc_num(stan.model)) for _ in 1:3]
        # declared positions spread like the data positions (see the parity test)
        for (j, n) in enumerate(BS.param_unc_names(stan.model))
            x = match(r"^p([xy])\.(\d+)$", n)
            x === nothing || foreach(u -> u[j] += getproperty(df, Symbol(:u, x[1]))[parse(Int, x[2])], unc)
        end
        draws = reduce(vcat, permutedims(BS.param_constrain(stan.model, u; include_tp=true))
                       for u in unc)
        d = brm_descriptor(sb)
        column(name) = draws[:, only(findall(==(name), names))]
        omega2 = vec(sb.data[:omega2_hsgp_t_by_g])
        r_min = tag == "mgq" ? brm_multigroup_inducing_resolution(hcat(df.sx, df.sy), 16) :
            brm_multigroup_coordinate_grid(hcat(df.sx, df.sy), KG, 2, 1.5)[3]
        rho, sigma = column("hsgp_t_by_g_rho_iso"), column("hsgp_t_by_g_sigma")
        lambda = column(tag == "mgd" ? "hsgp_t_by_g_lambda" : "lambda")
        c = brm_multigroup_coherence(d, draws, names; distances=[0.0, 10.0, 50.0])
        @test c.predictor === :mu
        @test c.period ≈ 2pi ./ sqrt.(omega2)
        @test c.range ≈ lambda ./ (rho .* sqrt.(omega2'))
        @test c.resolution ≈ r_min
        @test c.resolved == (c.range .>= c.resolution)
        @test c.coherence[:, :, 1] ≈ ones(3, length(omega2))
        @test c.coherence[:, 2, 3] ≈ exp.(-50.0^2 ./ (2 .* c.range[:, 2] .^ 2))
        # At the levels' own coordinates and the fitted times, the term is the
        # emitted weights times the spectrally scaled basis.
        s = hcat(df.sx, df.sy)
        times = unique(df.t)
        f = brm_multigroup_field(d, draws, names; times, coordinates=s)
        @test size(f) == (3, T, G)
        # One vector per axis, as the term takes its coordinates, reads the same
        # locations as the matrix.
        @test brm_multigroup_field(d, draws, names; times,
                                   coordinates=(df.sx, df.sy)) == f
        PHI = sb.data[:PHI_hsgp_t_by_g]
        M = length(omega2)
        for i in 1:3
            beta = [column("hsgp_t_by_g_beta_level.$l.$b")[i] for l in 1:G, b in 1:M]
            scale = brm_hsgp_sqrt_spd(reshape(omega2, :, 1), sigma[i], rho[i])
            fitted = [sum(PHI[n, b] * scale[b] * beta[df.g[n], b] for b in 1:M)
                      for n in eachindex(df.t)]
            # rows run over times within each level: f[i, :, :] is times × levels
            @test vec(f[i, :, :]) ≈ fitted atol=1e-10
        end
        # A held-out location between two levels gets the term's prediction there.
        mid = (s[1, :] .+ s[2, :]) ./ 2
        between = brm_multigroup_field(d, draws, names; times, coordinates=permutedims(mid))
        @test size(between) == (3, T, 1)
        @test all(isfinite, between)
        @test_throws "basis interval" brm_multigroup_field(d, draws, names;
            times=[1e6], coordinates=s)
        # refused: a location outside the region the approximation covers is
        # an extrapolation (decision `0yrmmdq`, coverage)
        @test_throws "outside the region" brm_multigroup_field(d, draws, names;
            times, coordinates=[1e5 1e5])
      end
    else
        @test true
    end
end

@testset "readers: field locations given as the term takes its coordinates" begin
    df = coordinate_data()
    term = BRM._brm_multigroup_coordinate_term(brm_descriptor(SBBRMI(mgc(df); mod=@__MODULE__)),
                                               nothing)
    s = hcat(df.sx, df.sy)
    locations(c) = BRM._brm_multigroup_locations(term, c)
    @test locations((df.sx, df.sy)) == s
    @test locations(s) == s
    @test locations((df.sx[2:3], df.sy[2:3])) == s[2:3, :]
    @test locations((Float32.(df.sx), df.sy)) ≈ s
    # refused: axes of different lengths name no set of locations
    @test_throws "different lengths" locations((df.sx, df.sy[1:3]))
    # refused: the term has two axes
    @test_throws "per coordinate axis (2), got 1" locations((df.sx,))
    @test_throws "per coordinate axis (2), got 3" locations((df.sx, df.sy, df.sx))
    # refused: a vector of locations is neither one vector per axis nor a matrix
    @test_throws "one vector per coordinate axis" locations([s[1, :], s[2, :]])
end

@testset "replay keeps the fitted levels and their coordinates" begin
    df = coordinate_data()
    sb = SBBRMI(mgc(df); mod=@__MODULE__)
    @test BRM.stan_code(reprocess(sb, df)) == BRM.stan_code(sb)
    subset = (; t=df.t[1:16], g=df.g[1:16], u=df.u[1:16], y=df.y[1:16], df.sx, df.sy)
    replayed = reprocess(sb, subset)
    @test occursin("brm_hsgp_multigroup_coordinate_weights(", BRM.stan_code(replayed))
    @test replayed.data[:multigroup_sx_sy_coordinates] == hcat(df.sx, df.sy)
    @test reprocess(RKBRMI(mgc(df)), df) isa RKBRMI
    # free positions given as data stay the training positions
    sb = SBBRMI(mgp(df); mod=@__MODULE__)
    replayed = reprocess(sb, merge(subset, (; df.ux, df.uy)))
    @test replayed.data[:multigroup_sx_sy_inducing_ux_uy] == hcat(df.ux, df.uy)
end
