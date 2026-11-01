# Research only: public KLU solves with a separately serialized MUMPS
# inertia/log-determinant check. No selected inverse is implemented here.
using KLU, MUMPS, SparseArrays, LinearAlgebra, Random, Test
import MUMPS: MPI

MPI.Init(; threadlevel=:serialized)
const inertia_gate = ReentrantLock()

function checked_inertia(A)
    lock(inertia_gate) do
        m = Mumps{Float64}(mumps_symmetric,
            get_icntl(det=true,nthreads=1,tree_parallelism=false),
            copy(default_cntl64);backend=MUMPS.Sequential())
        try
            MUMPS.set_icntl!(m,13,1;displaylevel=0)
            MUMPS.set_icntl!(m,31,1;displaylevel=0)
            MUMPS.associate_matrix!(m,A)
            MUMPS.set_job!(m,4)
            MUMPS.invoke_mumps!(m)
            (;negative=Int(m.infog[12]),
                logabsdet=log(abs(m.rinfog[12]))+m.infog[34]*log(2.0))
        finally
            finalize(m)
        end
    end
end

function augmented_blocks(D,B,W)
    r = size(B,1); C=B/sqrt(2.0); E=spdiagm(0=>ones(r))
    [D C' C';C E+W/2 -W/2;C -W/2 -E+W/2]
end

function block_checks(D,B,W,rng)
    n=size(D,1);r=size(B,1)
    H=Matrix(D+B'*W*B); K=augmented_blocks(D,B,W)
    expected=eigvals(Symmetric(H))
    minimum(abs,expected) > 1e-9 || error("singular research fixture")
    inertia=checked_inertia(K)
    F=klu(K)
    b=randn(rng,n)
    x=F\[b;zeros(2r)]
    (; inertia=inertia.negative == r+count(<(0),expected),
        determinant=isapprox(inertia.logabsdet,first(logabsdet(H));atol=1e-9),
        solve=isapprox(x[1:n],H\b;atol=1e-9,rtol=1e-9),
        residual=norm(K*x-[b;zeros(2r)],Inf) < 1e-9,
        btf=nnz(F.F) == 0, # symmetric graph: BTF components are diagonal blocks
        factors=isapprox(F.L*F.U,Diagonal(1 ./ F.Rs)*K[F.p,F.q];atol=1e-10))
end

function check_blocks(D,B,W,rng)
    for passed in values(block_checks(D,B,W,rng))
        @test passed
    end
end

@testset "pivot changes and overlapping augmented cells" begin
    rng=Xoshiro(772)
    for d in (-1.0,0.0,1.0,2.0), b in (.1,1.0,2.0), w in (-2.0,-1.0,0.0,1.0,2.0,4.0)
        iszero(d+b*b*w) && continue
        check_blocks(sparse([d;;]),sparse([b;;]),sparse([w;;]),rng)
    end
    B=sparse([1.0 .2 0 0 0 0;.1 1 .3 0 0 0;0 0 1 .2 .1 0;0 0 0 0 1 .5])
    for pass in 1:30
        D=spdiagm(0=>vcat(0.0,1 .+ rand(rng,5)))
        W=spdiagm(0=>[1.0,randn(rng),0.0,randn(rng)])
        check_blocks(D,B,W,rng)
    end
end

@testset "independent concurrent workspaces with serialized inertia" begin
    @test Threads.nthreads() >= 2
    tasks=map(1:4) do chain
        Threads.@spawn begin
            rng=Xoshiro(900+chain)
            map(1:20) do pass
                n=12; B=sprandn(rng,3,n,.25)
                D=spdiagm(0=>1 .+rand(rng,n))
                W=spdiagm(0=>rand(rng,3))
                block_checks(D,B,W,rng)
            end
        end
    end
    # Record assertions in the parent testset, not the worker's task-local
    # fallback testset, so all concurrent results are counted and inspected.
    for task in tasks, result in fetch(task), passed in values(result)
        @test passed
    end
end
MPI.Finalize()
