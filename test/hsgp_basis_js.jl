# `StatisticalPreparation.hsgp_basis` is type-stable on every path and
# translates through `ReactiveKernels.js_source` (RK 0.1.274+), so a browser
# picture can recompute an HSGP basis when a slider moves its domain (todo
# 0ggll1f, from ReactiveKernels snag js-source-all-pr-b15faf4f).
#
#     julia --project=test test/hsgp_basis_js.jl [filter...]
using Test, BayesianRegressionModels, ReactiveKernels, Random
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels
const Prep = BRM.StatisticalPreparation

# The independent oracle: SB's per-axis preparation (`_brm_apply_hsgp`) at the
# same center and half-width, tensored in `CartesianIndices(k)` order by the same
# axis-by-axis products, so the comparison is exact.
function tensor_oracle(axes_, K, fits)
    per = [BRM._brm_apply_hsgp(fits[j], axes_[j], K[j]) for j in eachindex(axes_)]
    idx = vec(collect(CartesianIndices(Tuple(K))))
    PHI = Matrix{Float64}(undef, length(first(axes_)), length(idx))
    lambda = Matrix{Float64}(undef, length(idx), length(axes_))
    for (b, I) in enumerate(idx)
        col = copy(per[1][1][:, I[1]])
        for j in 2:length(axes_)
            col .*= per[j][1][:, I[j]]
        end
        PHI[:, b] = col
        for j in eachindex(axes_)
            lambda[b, j] = per[j][2][I[j]]
        end
    end
    PHI, lambda
end

domain_fit(lo, hi) = ((lo + hi) / 2, (hi - lo) / 2)

rng = Xoshiro(17)
x = 4 .* rand(rng, 30) .- 2
y = 3 .* rand(rng, 30) .- 1
z = randn(rng, 30)
xi = rand(rng, -5:5, 30)
# One case per argument shape: axes, keywords, the oracle's `K` and fits.
const SHAPES = [
    ("one axis, fixed domain", (x,), (; k = 4, domain = (-2.5, 2.5)), [4], [domain_fit(-2.5, 2.5)]),
    ("one axis, data domain", (x,), (; k = 5, c = 1.5), [5], [BRM._brm_fit_hsgp(x, 1, 1.5)]),
    ("integer axis, data domain", (xi,), (; k = 3), [3], [BRM._brm_fit_hsgp(xi, 1, 1.5)]),
    ("two axes, k vector, fixed domain", (x, y), (; k = [3, 4], domain = ((-2.0, 2.0), (-1.0, 2.0))),
     [3, 4], [domain_fit(-2.0, 2.0), domain_fit(-1.0, 2.0)]),
    ("two axes, k tuple, c per axis", (x, y), (; k = (3, 4), c = (1.5, 2.0)),
     [3, 4], [BRM._brm_fit_hsgp(x, 1, 1.5), BRM._brm_fit_hsgp(y, 1, 2.0)]),
    ("three axes, one k, fixed domain", (x, y, z), (; k = 2, domain = [(-2.0, 2.0), (-1.0, 2.0), (-5.0, 5.0)]),
     [2, 2, 2], [domain_fit(-2.0, 2.0), domain_fit(-1.0, 2.0), domain_fit(-5.0, 5.0)]),
]

@stestset "hsgp_basis equals the per-axis oracle bit for bit, every shape" begin
    for (label, axes_, kw, K, fits) in SHAPES
        PHI, lambda = Prep.hsgp_basis(axes_...; kw...)
        oPHI, olambda = tensor_oracle(axes_, K, fits)
        @test size(PHI) == (length(x), prod(K))
        @test PHI == oPHI
        @test lambda == olambda
    end
end

# Every slot and SSA value of the keyword body is concrete, a constant, a type,
# `Union{}` (a throw) or a union of concrete types (a loop's `iterate`).
function stable_type(t)
    t = Core.Compiler.widenconst(t)
    t === Union{} || isconcretetype(t) || t <: Type ||
        (t isa Union && all(isconcretetype, Base.uniontypes(t)))
end
function unstable_types(axes_, kw)
    m = which(Prep.hsgp_basis, Tuple{map(typeof, axes_)...})
    body = Base.bodyfunction(m)
    full = merge((; k = 20, c = 1.5, domain = nothing, by = nothing), kw)
    args = Tuple{map(typeof, values(full))..., typeof(Prep.hsgp_basis), map(typeof, axes_)...}
    ci, = only(code_typed(body, args; optimize = false))
    [t for t in vcat(ci.slottypes, ci.ssavaluetypes) if !stable_type(t)]
end

@stestset "hsgp_basis is type-stable on the domain, data and by paths" begin
    g = repeat(["a", "b", "c"], 10)
    for (label, axes_, kw, _, _) in SHAPES
        @test (@inferred Prep.hsgp_basis(axes_...; kw...)) isa Tuple{Matrix{Float64},Matrix{Float64}}
        @test isempty(unstable_types(axes_, kw))
        bykw = merge(kw, (; by = g))
        @test (@inferred Prep.hsgp_basis(axes_...; bykw...)) isa Tuple{Matrix{Float64},Matrix{Float64}}
        @test isempty(unstable_types(axes_, bykw))
    end
    # The grouped layout: column (m - 1) * G + g is basis function m on level g.
    PHI, _ = Prep.hsgp_basis(x, y; k = [2, 3], domain = ((-2.0, 2.0), (-1.0, 2.0)))
    GPHI, _ = Prep.hsgp_basis(x, y; k = [2, 3], domain = ((-2.0, 2.0), (-1.0, 2.0)), by = g)
    codes = [findfirst(==(v), ["a", "b", "c"]) for v in g]
    @test size(GPHI) == (length(x), 3 * 6)
    @test all(GPHI[i, (m - 1) * 3 + codes[i]] == PHI[i, m] for i in eachindex(x), m in 1:6)
    @test count(!iszero, GPHI) == count(!iszero, PHI)
end

@stestset "hsgp_basis argument errors name the axis and the count" begin
    @test_throws r"axis 1 has 30 values, axis 2 has 29" Prep.hsgp_basis(x, y[1:29]; k = 3)
    @test_throws r"`k` takes one value or one per axis \(2\), got 3 values" Prep.hsgp_basis(x, y; k = [3, 3, 3])
    @test_throws r"`k` must be positive, got 0 for axis 2" Prep.hsgp_basis(x, y; k = (3, 0))
    @test_throws r"`c` must exceed 1, got 0.5 for axis 1" Prep.hsgp_basis(x; c = 0.5)
    @test_throws r"axis 1 has values outside its domain \(-1.0, 1.0\)" Prep.hsgp_basis(x; domain = (-1.0, 1.0))
    @test_throws r"domain 2 must be finite with lo < hi, got \(2.0, 1.0\)" Prep.hsgp_basis(x, y;
        domain = ((-2.0, 2.0), (2.0, 1.0)))
    # A bare (lo, hi) is one axis's domain; two axes need two pairs.
    @test_throws r"one \(lo, hi\) pair per axis \(2\), got one pair" Prep.hsgp_basis(x, y; domain = (-2.0, 2.0))
    @test_throws r"one \(lo, hi\) pair per axis \(2\), got 3" Prep.hsgp_basis(x, y;
        domain = ((-2.0, 2.0), (-1.0, 2.0), (0.0, 1.0)))
    @test_throws ArgumentError Prep.hsgp_basis(; k = 3)
end

# --- js_source: the browser picture's kernels, against native `prepare` ------

js_basis_2d(x, y, lo1, hi1, lo2, hi2) =
    Prep.hsgp_basis(x, y; k = [3, 4], domain = ((lo1, hi1), (lo2, hi2)))
js_basis_1d(x, lo, hi) = Prep.hsgp_basis(x; k = 5, domain = (lo, hi))

@kernel js_hsgp_field(x, y, lo1, hi1, lo2, hi2, sigma, rho) = begin
    basis = js_basis_2d(x, y, lo1, hi1, lo2, hi2)
    PHI = basis[1]
    lambda = basis[2]
    w = Prep.hsgp_sqrt_spd(lambda, sigma, rho)
    floors = Prep.hsgp_rho_floors(lambda)
    field = PHI * w
    return field
end

@kernel js_hsgp_curve(x, lo, hi, sigma, rho) = begin
    basis = js_basis_1d(x, lo, hi)
    PHI = basis[1]
    lambda = basis[2]
    w = Prep.hsgp_sqrt_spd(lambda, sigma, rho)
    curve = PHI * w
    return curve
end

const NODE = ReactiveKernels._js_default_node() !== nothing
const V = Vector{Float64}
const JS_CASES = [
    (:js_hsgp_field, js_hsgp_field, (:x, :y, :lo1, :hi1, :lo2, :hi2, :sigma, :rho),
     (:PHI, :lambda, :w, :floors, :field),
     (; x = V, y = V, lo1 = Float64, hi1 = Float64, lo2 = Float64, hi2 = Float64,
        sigma = Float64, rho = Float64),
     [(x, y, -2.0, 2.0, -1.0, 2.0, 0.8, 0.6),
      (x[1:7], y[1:7], -3.0, 3.0, -2.0, 2.5, 1.3, 1.1),   # a slider's other slice
      (x, y, -1.0, 1.0, -1.0, 2.0, 0.8, 0.6),             # x outside its domain
      (x, y[1:29], -2.0, 2.0, -1.0, 2.0, 0.8, 0.6),       # axes of different lengths
      (x, y, -2.0, 2.0, 2.0, -1.0, 0.8, 0.6)]),           # lo > hi
    (:js_hsgp_curve, js_hsgp_curve, (:x, :lo, :hi, :sigma, :rho), (:PHI, :lambda, :w, :curve),
     (; x = V, lo = Float64, hi = Float64, sigma = Float64, rho = Float64),
     [(x, -2.5, 2.5, 1.0, 0.5), (x, -2.5, NaN, 1.0, 0.5), (x, -1.0, 1.0, 1.0, 0.5)]),
]

agree(a::Real, b::Real) = (isnan(a) && isnan(b)) ||
    (iszero(a) ? iszero(b) : isapprox(a, b; rtol = 1e-12, atol = 1e-15))
agree(a::AbstractArray, b::AbstractArray) = size(a) == size(b) && all(agree.(a, b))
agree(a, b) = false

@stestset "js_source of a kernel computing an HSGP basis agrees with native" begin
    @test pkgversion(ReactiveKernels) >= v"0.1.274"
    for (name, k, have, want, types, cases) in JS_CASES, vectors in (:array, :float64array)
        source = ReactiveKernels.js_source(k; have, want, types, name)
        @test occursin("export function $name(", source)
        if !NODE
            @test_skip "no node on the PATH"
            continue
        end
        native = prepare(k; have, want)
        outcomes = ReactiveKernels.js_run(source, name, cases; vectors)
        for (c, r) in zip(cases, outcomes)
            expected = try
                native(c...)
            catch e
                e
            end
            @test !r.inputs_mutated
            if expected isa Exception
                @test r.julia_error && r.error == string(nameof(typeof(expected)))
                # The page shows the same message native raises.
                @test r.message == sprint(showerror, expected; context = :limit => true) ||
                      r.message == expected.msg
            else
                @test r.error === nothing
                r.error === nothing || continue
                @test all(agree(expected[j], r.value[j]) for j in eachindex(want))
            end
        end
    end
end
