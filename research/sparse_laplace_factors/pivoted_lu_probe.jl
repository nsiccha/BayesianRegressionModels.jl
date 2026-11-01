# Research only: feasibility of a package-level pivoted-LU selected inverse.
# Not loaded by BRM. Scalar specialization of Algorithm 1 in Jacquelin, Lin,
# Yang (2017), https://arxiv.org/abs/1708.04539. UMFPACK owns factorization.
using SparseArrays, LinearAlgebra, Random, Test

function probe_lu_selected(F, A)
    L = F.L; U = F.U
    # UMFPACK may omit an exactly cancelled entry. Keep the symbolic closure
    # of the authored pattern under the final numerical permutations.
    n = size(A, 1)
    cols = [Set{Int}((k,)) for k in 1:n]
    rows = [Set{Int}((k,)) for k in 1:n]
    # Include requested original-coordinate diagonals, even when A_ii=0.
    pattern = copy(A); fill!(nonzeros(pattern), 1.0)
    ai, aj, _ = findnz((pattern + I)[F.p, F.q])
    for (i,j) in zip(ai,aj)
        push!(cols[j],i); push!(rows[i],j)
    end
    for k in 1:n
        lower = [i for i in cols[k] if i > k]
        upper = [j for j in rows[k] if j > k]
        for i in lower, j in upper
            push!(cols[j],i); push!(rows[i],j)
        end
    end
    zi = Int[]; zj = Int[]
    for j in 1:n, i in cols[j]
        push!(zi,i); push!(zj,j)
    end
    Zt = sparse(zi,zj,zeros(length(zi)),n,n)
    for k in size(L, 1):-1:1
        ls = [i for i in cols[k] if i > k]
        us = [j for j in rows[k] if j > k]
        ukk = U[k, k]
        for j in us
            Zt[k, j] = -sum(i -> L[i,k] * Zt[i,j], ls; init=0.0)
        end
        Zt[k, k] = inv(ukk) - sum(j -> Zt[k,j] * U[k,j] / ukk, us; init=0.0)
        for i in ls
            Zt[i, k] = -sum(j -> Zt[i,j] * U[k,j] / ukk, us; init=0.0)
        end
    end
    # (Rs .* A)[p,q] = L*U, hence A^-1[q,p] = (LU)^-1 .* Rs[p]'.
    i, j, v = findnz(Zt)
    sparse(F.q[j], F.p[i], v .* F.Rs[F.p[i]], size(Zt)...)
end

function probe_augmented(D, B, W)
    r = size(B, 1); E = spdiagm(0 => ones(r)); C = B / sqrt(2)
    [D C' C'; C E + W / 2 -W / 2; C -W / 2 -E + W / 2]
end

@testset "pivoted LU selected inverse feasibility" begin
    for d in (-1.0, 0.0, 1.0, 2.0), b in (0.1, 1.0, 2.0), w in (-2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
        d + b^2 * w > 0 || continue
        K = probe_augmented(sparse([d;;]), sparse([b;;]), sparse([w;;]))
        F = lu(K); Z = probe_lu_selected(F,K); exact = inv(Matrix(K))
        i, j, _ = findnz(K)
        @test all(isapprox(Z[a,b], exact[a,b]; atol=1e-11, rtol=1e-11) for (a,b) in zip(i,j))
    end
    rng = Xoshiro(38)
    for n in (5, 20, 50), trial in 1:5
        A = sprandn(rng, n, n, 3 / n) + spdiagm(0 => randn(rng, n))
        F = lu(A); Z = probe_lu_selected(F,A); exact = inv(Matrix(A))
        i, j, _ = findnz(A)
        @test all(isapprox(Z[b,a], exact[b,a]; atol=1e-9, rtol=1e-9) for (a,b) in zip(i,j))
    end
end

function probe_scaling(n)
    D = spdiagm(0 => [0.0; ones(n-1)])
    B = sparse(fill(1,n), 1:n, [1.0; fill(0.01,n-1)], 1, n)
    K = probe_augmented(D, B, sparse([1.0;;]))
    F = lu(K)
    probe_lu_selected(F,K)
    runs = [@timed probe_lu_selected(F,K) for _ in 1:3]
    Z = first(runs).value
    @test Z[1,1] ≈ 1 + (n-1)*0.0001
    @test Z[2,2] ≈ 1
    @test sum(K .* transpose(Z)) ≈ n+2
    println((; n, factor_entries=nnz(F.L)+nnz(F.U), inverse_entries=nnz(Z),
        seconds=minimum(x.time for x in runs), bytes=minimum(x.bytes for x in runs)))
end
probe_scaling(32)
for n in (128, 512, 4096, 32768)
    probe_scaling(n)
end
