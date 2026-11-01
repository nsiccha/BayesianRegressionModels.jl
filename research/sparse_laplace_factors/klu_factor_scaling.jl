# Research only: a second mature pivoting sparse factorization, public API.
using KLU, SparseArrays, LinearAlgebra, Test

function measure_klu(n)
    D = spdiagm(0 => [0.0; ones(n-1)])
    B = sparse(fill(1,n),1:n,[1.0;fill(.01,n-1)],1,n)/sqrt(2.0)
    A = [D B' B'; B sparse([1.5;;]) sparse([-.5;;]);
        B sparse([-.5;;]) sparse([-.5;;])]
    F = klu(A)
    klu!(F,A)
    fresh = [@timed klu(A) for _ in 1:3]
    reuse = [@timed klu!(F,A) for _ in 1:3]
    @test norm(A*(F\ones(n+2))-ones(n+2),Inf) < 1e-8
    @test nnz(F.F) == 0 # this connected symmetric graph has one BTF block
    if n == 32
        @test F.L * F.U + F.F ≈ Diagonal(1 ./ F.Rs) * A[F.p,F.q]
    end
    println((;n,entries=nnz(F.L)+nnz(F.U)+nnz(F.F),
        fresh_seconds=minimum(x.time for x in fresh),
        reuse_seconds=minimum(x.time for x in reuse),
        fresh_bytes=minimum(x.bytes for x in fresh),
        reuse_bytes=minimum(x.bytes for x in reuse)))
end

foreach(measure_klu,(32,512,4096,32768))
