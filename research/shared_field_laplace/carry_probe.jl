# Data-only conditional field factor from the streaming lane's joint global
# Gaussian (mu, rho, B). No inference API or posterior projection is supplied.
# Run: julia research/shared_field_laplace/carry_probe.jl
using LinearAlgebra, Random, Test

# For joint covariance diag(exp(2rho)) + B*B', condition the field on the
# remaining globals. Only rank-sized matrices are factored. The two index
# sets describe the ENTIRE carried global factor, in arbitrary packed order.
function field_conditional(mu, rho, B, globals, field)
    length(mu) == length(rho) == size(B, 1) || throw(DimensionMismatch("global factor"))
    sort(vcat(globals, field)) == collect(eachindex(mu)) ||
        throw(ArgumentError("global and field coordinates must partition the carried factor"))
    Bg, Bf = B[globals, :], B[field, :]
    dg, df = exp.(2 .* rho[globals]), exp.(2 .* rho[field])
    rank = size(B, 2)
    R = cholesky(Symmetric(Matrix{Float64}(I, rank, rank) + Bg' * (Bg ./ dg)))
    # R is the conditional precision of the common rank-dimensional normal.
    L = Bf / R.U
    W = cholesky(Symmetric(Matrix{Float64}(I, rank, rank) + L' * (L ./ df)))
    (; mg=mu[globals], mf=mu[field], dg, df, Bg, Bf, R, L, W)
end

function field_conditional_mean(c, g)
    c.mf + c.Bf * (c.R \ (c.Bg' * ((g .- c.mg) ./ c.dg)))
end

function field_conditional_precision(c, v)
    a = v ./ c.df
    a - (c.L ./ c.df) * (c.W \ (c.L' * a))
end

function field_conditional_density_gradient(c, g, z)
    delta = z - field_conditional_mean(c, g)
    score = field_conditional_precision(c, delta)
    ld = sum(log, c.df) + logdet(c.W)
    value = -(length(z)*log(2π) + ld + dot(delta, score))/2
    gg = (c.Bg * (c.R \ (c.Bf' * score))) ./ c.dg
    (; value, gg, gz=-score)
end

normal_logpdf(x, mean, variance) = -(log(2π*variance) + (x-mean)^2/variance)/2

@testset "low-rank conditional field preserves the carried joint Gaussian" begin
    rng = Xoshiro(62)
    # Dense matrices below are bounded eight-coordinate reference objects.
    n = 8
    mu, rho = randn(rng, n), 0.2 .* randn(rng, n)
    field, globals = [7, 2, 4, 1, 6], [8, 3, 5]
    for rank in (0, 1, 3)
        B = 0.4 .* randn(rng, n, rank)
        c = field_conditional(mu, rho, B, globals, field)
        C = Matrix(Diagonal(exp.(2 .* rho)) + B*B')
        Cg = C[globals, globals]
        cross = C[field, globals]
        conditional = Symmetric(C[field, field] - cross * (Cg \ cross'))
        g, z = randn(rng, length(globals)), randn(rng, length(field))
        mean = mu[field] + cross * (Cg \ (g .- mu[globals]))
        @test field_conditional_mean(c, g) ≈ mean
        @test field_conditional_precision(c, z) ≈ conditional \ z
        result = field_conditional_density_gradient(c, g, z)
        delta = z - mean
        reference = -(length(z)*log(2π) + logdet(conditional) +
            dot(delta, conditional \ delta))/2
        @test result.value ≈ reference
        @test result.gz ≈ -(conditional \ delta)
        @test result.gg ≈ Cg \ (cross' * (conditional \ delta))
    end
    @test_throws ArgumentError field_conditional(mu, rho, zeros(n, 1), [1, 2], [2, 3])
end

@testset "conditional carry recovers a two-window Gaussian marginal" begin
    # theta,z independently N(0,1). Window 1: y1 ~ N(theta+z,1).
    # Window 2: y2 ~ N(theta-z,1). The first posterior has covariance
    # (1/3)I + [1,-1]*[1,-1]'/3, exactly a diagonal-plus-rank-one factor.
    y1, y2 = 1.7, -0.4
    mu = fill(y1/3, 2)
    rho = fill(log(1/3)/2, 2)
    B = reshape([1.0, -1.0] ./ sqrt(3), 2, 1)
    c = field_conditional(mu, rho, B, [1], [2])
    @test only(c.df + vec(sum(abs2, c.L; dims=2))) ≈ 0.5
    function target(theta)
        mz = only(field_conditional_mean(c, [theta]))
        normal_logpdf(theta, y1/3, 2/3) + normal_logpdf(y2, theta-mz, 1.5)
    end
    batch(theta) = normal_logpdf(theta, (y1+y2)/3, 1/3)
    for theta in (-1.0, 0.0, 0.7, 2.0)
        @test target(theta)-target(0.0) ≈ batch(theta)-batch(0.0) atol=1e-14
    end
    # Negative control: retaining only the theta marginal and resetting the
    # field loses the first window's theta/field dependence.
    reset(theta) = normal_logpdf(theta, y1/3, 2/3) + normal_logpdf(y2, theta, 2.0)
    @test !isapprox(reset(0.7)-reset(0.0), batch(0.7)-batch(0.0); atol=1e-4)
end

println("Conditional-carry algebra verified; no posterior compression, SVI fit, or streaming API tested.")
