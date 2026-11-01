# Research only: assess the whole factorization, not just selected inversion.
using SparseArrays, LinearAlgebra, Test
const UMF = SparseArrays.UMFPACK

function lu_arrow(n; rotated=true)
    D = spdiagm(0 => [0.0; ones(n-1)])
    B = sparse(fill(1,n),1:n,[1.0;fill(.01,n-1)],1,n)/sqrt(2.0)
    rotated || return [D sqrt(2.0)*B' spzeros(n,1);
        sqrt(2.0)*B spzeros(1,1) sparse([1.0;;]);
        spzeros(1,n) sparse([1.0;;]) sparse([1.0;;])]
    [D B' B'; B sparse([1.5;;]) sparse([-.5;;]);
        B sparse([-.5;;]) sparse([-.5;;])]
end

function factor_scaling(n, strategy)
    K = lu_arrow(n; rotated=!(strategy in (:unrotated,:unrotated_symmetric)))
    control = UMF.get_umfpack_control(Float64,Int)
    if strategy in (:symmetric,:leaves_symmetric,:unrotated_symmetric)
        # UMFPACK public header: C index 5 (Julia index 6) is STRATEGY;
        # value 3 selects symmetric AMD ordering with numerical pivoting.
        control[6] = 3
    end
    q = strategy in (:leaves_first,:leaves_symmetric) ? [collect(2:n);1;n+1;n+2] : nothing
    F = lu(K; control, q)
    lu!(F,K)
    fresh = [@timed lu(K; control, q) for _ in 1:3]
    reuse = [@timed lu!(F,K) for _ in 1:3]
    @test norm(K*(F\ones(n+2))-ones(n+2),Inf) < 1e-8
    println((;n,strategy,entries=nnz(F.L)+nnz(F.U),
        fresh_seconds=minimum(x.time for x in fresh),
        reuse_seconds=minimum(x.time for x in reuse),
        fresh_bytes=minimum(x.bytes for x in fresh),
        reuse_bytes=minimum(x.bytes for x in reuse)))
end

factor_scaling(32,:default)
strategies = isempty(ARGS) ? (:default,:symmetric,:leaves_first,:leaves_symmetric,
    :unrotated,:unrotated_symmetric) : Symbol.(ARGS)
for strategy in strategies, n in (512,4096,32768)
    factor_scaling(n,strategy)
end
