# Research only. Run in an environment with MUMPS 1.7 and Julia 1.10.
# Public MUMPS low-level interface; UMFPACK remains the solve/selected-inverse
# factor. No selected-inverse implementation or dense large matrix here.
using MUMPS, SparseArrays, LinearAlgebra, Random, Test
import MUMPS: MPI

MPI.Init()

function augmented(D, B, W)
    n = size(D, 1); r = size(B, 1)
    C = B / sqrt(2.0); E = spdiagm(0 => ones(r))
    [D C' C'; C E + W/2 -W/2; C -W/2 -E + W/2]
end

function inertia_workspace()
    m = Mumps{Float64}(mumps_symmetric,
        get_icntl(det=true, nthreads=1, tree_parallelism=false),
        copy(default_cntl64); backend=MUMPS.Sequential())
    # Exact inertia, no rank-revealing perturbations, and no retained factors
    # needed for this role. UMFPACK owns the solve factor independently.
    MUMPS.set_icntl!(m, 13, 1; displaylevel=0)
    MUMPS.set_icntl!(m, 31, 1; displaylevel=0)
    m
end

function inertia!(m, A; analyze=false)
    MUMPS.associate_matrix!(m, A)
    MUMPS.set_job!(m, analyze ? 4 : 2)
    MUMPS.invoke_mumps!(m)
    (; negative=Int(m.infog[12]),
        logabsdet=log(abs(m.rinfog[12])) + m.infog[34] * log(2.0),
        sign=sign(m.rinfog[12]), entries=Int(m.infog[35]),
        workspace_mb=Int(m.infog[19]))
end

function arrow(n; scale=1.0)
    D = spdiagm(0 => [0.0; fill(scale, n-1)])
    B = sparse(fill(1,n), 1:n, [1.0; fill(.01,n-1)], 1,n)
    augmented(D,B,sparse([scale;;]))
end

@testset "inertia and independent workspace lifecycle" begin
    m = inertia_workspace(); other = inertia_workspace()
    try
        rng = Xoshiro(728)
        # Reuse symbolic analysis as numerical pivots and inertia change.
        for pass in 1:20
            A = randn(rng,12,12); A = sparse(A + A')
            a = inertia!(m,A; analyze=pass==1)
            @test a.negative == count(<(0), eigvals(Symmetric(Matrix(A))))
            @test a.logabsdet ≈ first(logabsdet(A)) atol=1e-10
            @test a.sign == last(logabsdet(A))
            b = inertia!(other, -A; analyze=pass==1)
            @test b.negative == 12-a.negative
            @test a.logabsdet ≈ b.logabsdet atol=1e-10
        end
    finally
        finalize(m); finalize(other)
    end
    for d in (-1.0,0.0,1.0,2.0), b in (.1,1.0,2.0), w in (-2.0,-1.0,0.0,1.0,2.0,4.0)
        h = d + b*b*w
        iszero(h) && continue
        m = inertia_workspace()
        try
            K = augmented(sparse([d;;]),sparse([b;;]),sparse([w;;]))
            a = inertia!(m,K; analyze=true)
            @test a.negative == 1 + (h < 0)
            @test a.logabsdet ≈ log(abs(h)) atol=1e-10
        finally
            finalize(m)
        end
    end
end

function measure_inertia(n)
    K = arrow(n); m = inertia_workspace()
    try
        fresh = @timed inertia!(m,K; analyze=true)
        F = lu(K)
        inertia!(m,K); lu!(F,K)
        mruns = [@timed inertia!(m,K) for _ in 1:3]
        lruns = [@timed lu!(F,K) for _ in 1:3]
        a = first(mruns).value
        @test a.negative == 1
        @test abs(a.logabsdet) < 1e-9
        @test norm(K*(F\ones(n+2))-ones(n+2),Inf) < 1e-8
        result=(; n, input_entries=nnz(K), factor_entries=a.entries,
            mumps_workspace_mb=a.workspace_mb,
            analysis_and_factor_seconds=fresh.time,
            inertia_seconds=minimum(x.time for x in mruns),
            inertia_bytes=minimum(x.bytes for x in mruns),
            lu_seconds=minimum(x.time for x in lruns),
            lu_bytes=minimum(x.bytes for x in lruns))
        println(result)
        result
    finally
        finalize(m)
    end
end

measure_inertia(32)
results = map(measure_inertia,(512,4096,32768))
@test results[3].inertia_bytes < 10results[2].inertia_bytes
MPI.Finalize()
