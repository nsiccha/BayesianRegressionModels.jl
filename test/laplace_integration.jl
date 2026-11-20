# brm_laplace_integration (src/laplace_integration.jl): deterministic
# integration over the population coordinates of an embedded-Laplace target,
# rung 5c of brm_ladder_fit (design `0r49iwd`). Checked:
#   - the CCD's fractions are resolution V and its weights reproduce a
#     standard Gaussian's first two moments exactly; the grid reproduces them
#     up to its truncation;
#   - on Gaussian models whose Laplace marginal is exact, the weighted design,
#     the marginal tables, the grid's log evidence and the effect draws agree
#     with brute-force quadrature of the exact marginal (grid at d = 2, CCD
#     at d = 3);
#   - the refusals name their reason.
# RUN: julia --project=test test/laplace_integration.jl
using WarmupHMC   # loads Optim, which brm_laplace_mode needs
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

@stestset "CCD fractions are resolution V" begin
    sizes = Dict(5 => 16, 6 => 32, 7 => 64, 8 => 64, 9 => 128, 10 => 128)
    for d in 2:10
        F = BRM._brm_ccd_fraction(d)
        @test size(F, 1) == d
        @test all(x -> x == 1 || x == -1, F)
        d >= 5 && @test size(F, 2) == sizes[d]
        d <= 4 && @test size(F, 2) == 2^d
        # main effects and every two-factor interaction are mutually orthogonal
        X = vcat(F, (F[i:i, :] .* F[j:j, :] for i in 1:d for j in (i + 1):d)...)
        @test X * X' == size(F, 2) * I
        @test allunique(eachcol(F))
    end
end

@stestset "design weights reproduce a standard Gaussian's moments" begin
    gaussian(z) = -sum(abs2, z) / 2
    for d in 1:8
        out = BRM._brm_lint_ccd(gaussian, 0.0, d, 1.1)
        lw = out.log_weight .+ out.log_density
        w = exp.(lw .- maximum(lw)); w ./= sum(w)
        @test sum(exp.(out.log_weight)) ≈ 1
        @test out.z * w ≈ zeros(d) atol=1e-12
        @test out.z .* w' * out.z' ≈ Matrix(I, d, d) atol=1e-12
        @test size(out.z, 2) == 1 + 2d + (d == 1 ? 0 : size(BRM._brm_ccd_fraction(d), 2))
    end
    for d in 1:2
        out = BRM._brm_lint_grid(gaussian, 0.0, d, 0.75, 6.0)
        w = exp.(out.log_density); w ./= sum(w)
        @test isempty(out.truncated)
        @test out.z * w ≈ zeros(d) atol=1e-12
        # the drop-6 truncation removes about e^-6 of the mass
        @test out.z .* w' * out.z' ≈ Matrix(I, d, d) rtol=0.03
        @test all(v -> v >= -6.0, out.log_density)
    end
end

const LINT_DATA = (; y=[0.2, -0.4, 0.8, 0.3, 1.1, 0.6, -0.2], idx=[1, 2, 1, 4, 3, 3, 5], n=5)

# location and scale outer (d = 2), plus an offset for d = 3, or the scale
# alone (d = 1) with the location fixed at 0.2
function lint_model(; offset=false, single=false)
    data = LINT_DATA
    model = if single
        @brm data begin
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(0.2, scale)
            prediction = field[idx]
            y ~ Normal(prediction, 0.6)
        end
    elseif offset
        @brm data begin
            offset ~ Normal(0, 1)
            location ~ Normal(0.2, 0.8)
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(location, scale)
            prediction = offset + field[idx]
            y ~ Normal(prediction, 0.6)
        end
    else
        @brm data begin
            location ~ Normal(0.2, 0.8)
            scale ~ LogNormal(0, 0.4)
            field[1:n] .~ Normal.(location, scale)
            prediction = field[idx]
            y ~ Normal(prediction, 0.6)
        end
    end
    rk = RKBRMI(model)
    p = brm_laplace_problem(rk; ad_backend=NLME_AD, integrate=(:field,))
    names = offset ? (:offset, :location, :scale) : single ? (:scale,) : (:location, :scale)
    read = BRM._rk_value_reader(rk, names)
    location(x) = single ? 0.2 : x.location
    B = [Float64(j == data.idx[i]) for i in eachindex(data.y), j in 1:data.n]
    values(phi) = (u = zeros(length(p.u)); u[p.population] .= phi; read(u))
    function exact(phi)
        x = values(phi)
        o = offset ? x.offset : 0.0
        covariance = 0.6^2 * I + x.scale^2 * B * B'
        r = logpdf(MvNormal(o .+ location(x) .* B * ones(data.n), Matrix(covariance)), data.y)
        r += logpdf(LogNormal(0, 0.4), x.scale) + log(x.scale)
        single || (r += logpdf(Normal(0.2, 0.8), x.location))
        offset && (r += logpdf(Normal(0, 1), x.offset))
        r
    end
    # the field's conditional mean given the outer values
    function field_mean(phi)
        x = values(phi)
        o = offset ? x.offset : 0.0
        Q = I / x.scale^2 + B' * B / 0.36
        Q \ (fill(location(x) / x.scale^2, data.n) .+ B' * (data.y .- o) / 0.36)
    end
    (; rk, p, exact, field_mean)
end

# Brute-force quadrature of the exact marginal on a fine rectangular grid in
# the target's population coordinates, ±7 curvature standard errors.
function lint_reference(m, mode, se; n=81)
    d = length(mode)
    axes1 = [range(mode[i] - 7se[i], mode[i] + 7se[i]; length=n) for i in 1:d]
    pts = vec(collect(Iterators.product(axes1...)))
    lv = [m.exact(collect(t)) for t in pts]
    w = exp.(lv .- maximum(lv)); Z = sum(w); w ./= Z
    X = reduce(hcat, collect.(pts))
    μ = X * w
    sd = sqrt.((X .- μ) .^ 2 * w)
    logZ = maximum(lv) + log(Z) + sum(log(step(a)) for a in axes1)
    field = sum(w[k] * m.field_mean(X[:, k]) for k in eachindex(w))
    (; mean=μ, sd, logZ, field)
end

@stestset "grid at d = 2 agrees with quadrature of the exact marginal" begin
    m = lint_model()
    p = m.p
    @test length(p.population) == 2
    out = brm_laplace_integration(p; draws=4000, rng=Xoshiro(7))
    @test out.design === :grid
    @test LogDensityProblems.logdensity(p, out.point) ≈ m.exact(out.point) atol=1e-8
    ref = lint_reference(m, out.point, sqrt.(diag(inv(-out.standardization.hessian))))
    @test sum(out.weights) ≈ 1
    @test out.diagnostics.nonfinite == 0 && out.diagnostics.max_gain <= 1e-8
    @test out.mean ≈ ref.mean atol=0.02 * maximum(ref.sd)
    @test out.sd ≈ ref.sd rtol=0.03
    @test out.log_evidence ≈ ref.logZ atol=0.01
    for (i, mg) in enumerate(out.marginals)
        @test mg.name === p.names[i]
        @test length(mg.x) == 401 && all(>=(0), mg.density)
        # a kernel mixture that keeps the design's moments
        @test mg.mean ≈ out.mean[i] atol=1e-3 * out.sd[i]
        @test mg.sd ≈ out.sd[i] rtol=1e-3
        @test mg.mean ≈ ref.mean[i] atol=0.02 * ref.sd[i]
    end
    # draws: population rows are design points; effects are joint draws
    @test size(out.draws) == (length(p.u), 4000)
    D = out.draws
    @test all(any(col ≈ D[p.population, k] for col in eachcol(out.points)) for k in 1:50)
    @test vec(mean(D[p.integrated, :]; dims=2)) ≈ ref.field atol=0.05
    @test all(>(0), vec(std(D[p.integrated, :]; dims=2)))
end

@stestset "CCD at d = 3 agrees with quadrature of the exact marginal" begin
    m = lint_model(; offset=true)
    p = m.p
    @test length(p.population) == 3
    out = brm_laplace_integration(p; draws=2000, rng=Xoshiro(9))
    @test out.design === :ccd
    @test size(out.z, 2) == 15
    @test out.log_evidence === nothing
    ref = lint_reference(m, out.point, sqrt.(diag(inv(-out.standardization.hessian))); n=41)
    @test out.mean ≈ ref.mean atol=0.15 * maximum(ref.sd)
    @test out.sd ≈ ref.sd rtol=0.15
    for (i, mg) in enumerate(out.marginals)
        @test mg.mean ≈ out.mean[i] atol=1e-3 * out.sd[i]
        @test mg.sd ≈ out.sd[i] rtol=1e-3
        @test mg.mean ≈ ref.mean[i] atol=0.15 * ref.sd[i]
        @test mg.sd ≈ ref.sd[i] rtol=0.15
    end
    @test vec(mean(out.draws[p.integrated, :]; dims=2)) ≈ ref.field atol=0.08
    # the grid on the same target, forced, is closer
    grid = brm_laplace_integration(p; point=out.point, design=:grid, draws=0)
    @test grid.design === :grid && size(grid.draws, 2) == 0
    @test grid.mean ≈ ref.mean atol=0.03 * maximum(ref.sd)
    @test grid.sd ≈ ref.sd rtol=0.03
end

@stestset "grid at d = 1: the table interpolates the exact marginal" begin
    m = lint_model(; single=true)
    p = m.p
    @test length(p.population) == 1
    out = brm_laplace_integration(p; draws=500, rng=Xoshiro(5))
    @test out.design === :grid
    ref = lint_reference(m, out.point, sqrt.(diag(inv(-out.standardization.hessian))); n=401)
    @test out.mean ≈ ref.mean atol=0.01 * ref.sd[1]
    @test out.sd ≈ ref.sd rtol=0.01
    @test out.log_evidence ≈ ref.logZ atol=0.01
    mg = only(out.marginals)
    @test mg.mean ≈ ref.mean[1] atol=0.02 * ref.sd[1]
    @test mg.sd ≈ ref.sd[1] rtol=0.03
    # the table is the exact marginal, normalized, between the grid's ends
    x = mg.x[200]
    logmass = ref.logZ
    @test mg.density[200] ≈ exp(m.exact([x]) - logmass) rtol=0.02
    @test vec(mean(out.draws[p.integrated, :]; dims=2)) ≈ ref.field atol=0.05
end

@stestset "refusals name their reason" begin
    m = lint_model()
    p = m.p
    point = brm_laplace_mode(p).draws[p.population, 1]
    @test_throws "max_dimension = 1" brm_laplace_integration(p; point, max_dimension=1)
    @test_throws "design = :foo" brm_laplace_integration(p; point, design=:foo)
    @test_throws "f0 = 1.0" brm_laplace_integration(p; point, f0=1.0)
    @test_throws "not negative definite" brm_laplace_integration(p; point,
        curvature=(; hessian=[1.0 0.0; 0.0 1.0], positive_definite=false))
    @test_throws "point has length 3" brm_laplace_integration(p; point=zeros(3))
end
