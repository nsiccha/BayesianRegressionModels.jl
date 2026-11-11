using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LinearAlgebra, SparseArrays, Distributions, Enzyme
import DifferentiationInterface as DI
const BRM = BayesianRegressionModels
const EXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)

const GMRF_XY = [0.0 0.0; 1 0; 1 1; 0 1; 0.5 0.5]
const GMRF_TRIANGLES = [1 2 5; 2 3 5; 3 4 5; 4 1 5]

# Tiny independent covariance oracle: the time covariance is exp(-|dt|/ell),
# rather than the emitted transition recurrence, and all factors are dense.
function gmrf_rk_oracle(z, spatial, temporal, L, times)
    mesh = BRM._brm_gmrf_mesh(GMRF_XY, GMRF_TRIANGLES)
    kappa2 = 2 / spatial^2
    K = Matrix(mesh.stiffness) + kappa2 * Diagonal(mesh.mass)
    Q = K * Diagonal(1 ./ mesh.mass) * K / (4pi * kappa2)
    Ct = [exp(-abs(t-s) / temporal) for t in times, s in times]
    covariance = kron(Ct, kron(L * L', inv(Symmetric(Q))))
    logpdf(MvNormal(zeros(length(z)), Symmetric(covariance)), z)
end

@testset "RK sparse GMRF graph native Reverse against dense covariance" begin
    times = [0.0, 0.3, 1.7]
    geometry = BRM.brm_gmrf_geometry(GMRF_XY, GMRF_TRIANGLES, times, 2)
    field = 0.03 .* sin.(1:geometry.ncoordinates)
    spatial, temporal = 0.7, 1.4
    L = [1.0 0.0; 0.45 sqrt(1 - 0.45^2)]
    query = prepare(EXT._gmrf_density_graph)
    differentiated = prepare_ad(query, DI.AutoEnzyme(; mode=Enzyme.Reverse),
        field, spatial, temporal, L, geometry;
        active=(:field, :spatial_length, :temporal_length, :L))
    value, derivatives = ad_value_and_gradient(differentiated,
        field, spatial, temporal, L, geometry)
    @test value ≈ gmrf_rk_oracle(field, spatial, temporal, L, times) atol=2e-9
    for j in eachindex(field)
        plus, minus = copy(field), copy(field)
        h = 2e-6
        plus[j] += h
        minus[j] -= h
        expected = (gmrf_rk_oracle(plus, spatial, temporal, L, times) -
                    gmrf_rk_oracle(minus, spatial, temporal, L, times)) / (2h)
        @test derivatives[1][j] ≈ expected atol=2e-5 rtol=2e-6
    end
    for (index, length_scale) in ((2, spatial), (3, temporal))
        h = 2e-6 * length_scale
        plus = index == 2 ? (spatial + h, temporal) : (spatial, temporal + h)
        minus = index == 2 ? (spatial - h, temporal) : (spatial, temporal - h)
        expected = (gmrf_rk_oracle(field, plus..., L, times) -
                    gmrf_rk_oracle(field, minus..., L, times)) / (2h)
        @test derivatives[index] ≈ expected atol=2e-5 rtol=2e-6
    end
    for (i, j) in ((1, 1), (2, 1), (2, 2))
        plus, minus = copy(L), copy(L)
        h = 2e-6
        plus[i, j] += h
        minus[i, j] -= h
        expected = (gmrf_rk_oracle(field, spatial, temporal, plus, times) -
                    gmrf_rk_oracle(field, spatial, temporal, minus, times)) / (2h)
        @test derivatives[4][i, j] ≈ expected atol=2e-5 rtol=2e-6
    end
    @test derivatives[4][1, 2] == 0.0 # The law reads a lower-triangular factor.
end

@testset "RK visible sparse LDL and normalized GMRF density" begin
    factor_query = prepare(EXT._gmrf_factor_graph)
    query = prepare(EXT._gmrf_density_graph)
    for times in ([0.0], [0.0, 0.003, 1.7]), C in (1, 2)
        geometry = BRM.brm_gmrf_geometry(GMRF_XY, GMRF_TRIANGLES, times, C)
        L = C == 1 ? ones(1, 1) : [1.0 0.0; 0.45 sqrt(1-0.45^2)]
        z = 0.03 .* sin.(1:geometry.ncoordinates)
        for spatial in (0.2, 1.3), temporal in (0.1, 2.1)
            factor = factor_query(1/spatial, spatial/2, geometry)
            n = length(geometry.mass)
            lower = sparse(geometry.lrow, geometry.columns, copy(factor), n, n)
            diagonal = factor[geometry.lptr[1:n]]
            for j in 1:n
                lower[j, j] = 1.0
            end
            mesh = BRM._brm_gmrf_mesh(GMRF_XY, GMRF_TRIANGLES)
            @test lower * Diagonal(diagonal) * lower' ≈
                (spatial/2) * mesh.factor_stiffness + Diagonal(geometry.mass ./ spatial)
            @test query(z, spatial, temporal, L, geometry) ≈
                gmrf_rk_oracle(z, spatial, temporal, L, times) atol=2e-8 rtol=2e-10
        end
    end
end

@testset "Outside the support the RK GMRF density is -Inf, never an error" begin
    # A line search proposes lengths far beyond the mesh (snag gmrf-neighbors-d-952fd2fb):
    # B's mass term rounds away against its stiffness term and the factor loses a pivot.
    times = [0.0, 0.3, 1.7]
    geometry = BRM.brm_gmrf_geometry(GMRF_XY, GMRF_TRIANGLES, times, 2)
    field = 0.03 .* sin.(1:geometry.ncoordinates)
    L = [1.0 0.0; 0.45 sqrt(1 - 0.45^2)]
    query = prepare(EXT._gmrf_density_graph)
    differentiated = prepare_ad(query, DI.AutoEnzyme(; mode=Enzyme.Reverse),
        field, 0.7, 1.4, L, geometry; active=(:field, :spatial_length, :temporal_length, :L))
    for (spatial, temporal) in ((1e10, 1.4), (8.9e64, 1.4), (Inf, 1.4), (0.0, 1.4),
            (0.7, Inf), (0.7, 0.0))
        @test query(field, spatial, temporal, L, geometry) == -Inf
        value, derivatives = ad_value_and_gradient(differentiated, field, spatial, temporal, L, geometry)
        @test value == -Inf
        @test length(derivatives[1]) == length(field)
    end
    # Long lengths the factor still resolves keep their exact density. The Float64
    # oracle above inverts a near-singular Q there (off by 444 at 1e3, PosDefException
    # from 1e4), so these compare with the joint precision in 512-bit arithmetic.
    function big_oracle(z, spatial, temporal)
        setprecision(BigFloat, 512) do
            mesh = BRM._brm_gmrf_mesh(GMRF_XY, GMRF_TRIANGLES)
            kappa2 = 2 / big(spatial)^2
            M = big.(mesh.mass)
            K = big.(Matrix(mesh.stiffness)) + kappa2 * Diagonal(M)
            Q = K * Diagonal(1 ./ M) * K / (4 * big(pi) * kappa2)
            Ct = [exp(-abs(big(a) - big(b)) / big(temporal)) for a in times, b in times]
            P = kron(inv(Ct), kron(inv(big.(L) * big.(L)'), Q))
            x = big.(z)
            Float64((logdet(P) - dot(x, P * x) - length(x) * log(2 * big(pi))) / 2)
        end
    end
    for (spatial, temporal) in ((1e3, 1.4), (1e7, 1.4), (0.7, 1e6), (0.7, 1e12))
        @test query(field, spatial, temporal, L, geometry) ≈ big_oracle(field, spatial, temporal) rtol=1e-10
    end
end
