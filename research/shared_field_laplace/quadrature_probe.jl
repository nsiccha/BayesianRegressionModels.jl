# No AD compilation or model preparation. Run with the repository test env.
using BayesianRegressionModels, LinearAlgebra, Random, Test
const KrylovKit = BayesianRegressionModels.KrylovKit
include("quadrature.jl")

@testset "log quadrature bounds against a bounded dense oracle" begin
    rng = Xoshiro(80)
    n = 16
    Q = Matrix(qr(randn(rng,n,n)).Q)
    for condition in (1.0, 10.0, 1e4)
        values = exp.(range(0,log(condition);length=n))
        H = Symmetric(Q*Diagonal(values)*Q')
        r = randn(rng,n)
        exact = dot(abs2.(Q'*r), log.(values))
        # Every early interval must enclose the independent eigenbasis answer.
        for budget in (1, 2, 4, 8, 16)
            out = shared_logquadratic(v -> H*v, r; spectral_lower=1.0,
                atol=1e-11, rtol=1e-11, maxdim=budget)
            @test out.lower-out.roundoff-1e-10 <= exact <= out.upper+out.roundoff+1e-10
            if out.converged
                @test abs(out.value-exact) <= (out.upper-out.lower)/2 + out.roundoff+1e-10
            else
                @test out.iterations == budget
            end
        end
    end
    @test shared_logquadratic(identity,zeros(4);spectral_lower=1).value == 0
    @test_throws ArgumentError shared_logquadratic(identity,ones(4);spectral_lower=0)
    @test_throws ArgumentError shared_logquadratic(identity,ones(4);spectral_lower=2)
    @test_throws ArgumentError shared_logquadratic(v -> -v,ones(4);spectral_lower=1)
    @test_throws ErrorException shared_logdet(Xoshiro(1),v -> [v[1],100v[2]],2;
        spectral_lower=1, probes=2,maxdim=1)
end

@testset "trace probes and bounded memory" begin
    # Diagonal H gives the exact trace for EVERY Rademacher vector, so this
    # distinguishes quadrature error from random-probe error without luck.
    for d in (32, 1024, 8192)
        diagonal = 1 .+ 4 .* collect(range(0,1;length=d))
        H(v) = diagonal .* v
        run() = shared_logdet(Xoshiro(81),H,d;spectral_lower=1.0,
            probes=4,maxdim=32,atol=1e-9,rtol=1e-9)
        out = run()
        exact = sum(log,diagonal)
        @test out.value ≈ exact rtol=2e-9
        @test out.mc_se < 1e-9
        @test run() == out
        bytes = @allocated run()
        @test bytes < 6000d + 8_000_000
        println((; d,bytes,out.max_iterations,out.operations,
            error=abs(out.value-exact),quadrature_width=out.quadrature_upper-out.quadrature_lower))
    end
    # For a nondiagonal 8-coordinate reference, averaging ALL sign vectors
    # removes trace sampling error exactly; no probabilistic pass/fail test.
    d = 8
    Q = Matrix(qr(randn(Xoshiro(82),d,d)).Q)
    values = exp.(range(0,log(50);length=d))
    H = Symmetric(Q*Diagonal(values)*Q')
    average, lower, upper = 0.0, 0.0, 0.0
    for mask in 0:2^d-1
        r = [iszero(mask & (1 << (j-1))) ? -1.0 : 1.0 for j in 1:d]
        out = shared_logquadratic(v -> H*v,r;spectral_lower=1.0,maxdim=8)
        @test out.converged
        average += out.value/2^d
        lower += out.lower/2^d
        upper += out.upper/2^d
    end
    @test average ≈ sum(log,values) rtol=1e-10
    @test lower-1e-10 <= sum(log,values) <= upper+1e-10
end
