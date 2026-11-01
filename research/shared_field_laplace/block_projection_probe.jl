# Bounded-draw moment projection into the common-factor carry family.
# Standalone: julia research/shared_field_laplace/block_projection_probe.jl
# This is a numerical control, not the streaming API or a claim that a fixed
# draw/rank budget gives dimension-independent statistical accuracy.
include("block_carry_probe.jl")
using Statistics, Serialization
BLAS.set_num_threads(1)

function project_block_carry(draws,blocks,reference;rank=8,shrinkage=0.01)
    n,M = size(draws)
    2 <= M <= 256 || throw(ArgumentError("projection needs 2:256 joint draws"))
    0 <= rank <= min(M-1,64) || throw(ArgumentError("projection rank exceeds fixed cap"))
    0 < shrinkage <= 1 || throw(ArgumentError("shrinkage must be in (0,1]"))
    all(isfinite,draws) || throw(ArgumentError("projection draws must be finite"))
    # Validate blocks and fixed positive reference covariances before allocating
    # the projection. The numerical reference need not be an authored prior.
    block_carry(zeros(n),blocks,reference,zeros(n,0))
    mu = vec(mean(draws;dims=2))
    X = (draws .- mu) / sqrt(M) # empirical-distribution covariance convention
    V = eigen(Symmetric(X'*X)).vectors[:,end-rank+1:end]
    A = X*V
    B = sqrt(1-shrinkage)*A
    covariance = map(zip(blocks,reference)) do (block,T)
        E = X[block,:]-A[block,:]*V'
        # Residual Gram, never a subtraction of covariance matrices and never
        # clipped eigenvalues. Fixed positive shrinkage makes every block PD.
        (1-shrinkage)*(E*E')+shrinkage*T
    end
    factor = block_carry(mu,blocks,covariance,B)
    metadata = (;draw_count=M,rank,shrinkage,covariance_convention=:empirical,
        reference_blocks=deepcopy(reference),draw_limit=256,rank_limit=64)
    (;factor,metadata)
end

function block_covariance_action(c,v)
    out = c.B*(c.B'*v)
    for (block,F) in zip(c.blocks,c.factors)
        out[block] += F.L*(F.L'*v[block])
    end
    out
end

# Algebraic spectral bound; floating-point arithmetic here is not outward
# rounded and this is NOT an interval-certified bound. If the window log
# likelihood is concave in the field, its precision adds a PSD contribution.
function block_precision_lower_bound(c)
    base_upper = maximum(c.factors) do F
        maximum(vec(sum(abs,F.L*F.L';dims=2)))
    end
    1/(base_upper+sum(abs2,c.B))
end

@testset "fixed-draw block projection" begin
    rng = Xoshiro(811)
    draws = randn(rng,5,64)
    draws[5,:] += draws[1,:]+0.5draws[3,:]
    blocks = [[1,2],[3],[4,5]]
    reference = [Matrix{Float64}(I,length(b),length(b)) for b in blocks]
    out = project_block_carry(draws,blocks,reference;rank=2,shrinkage=.02)
    c = out.factor
    # Bounded five-dimensional oracle only.
    C = hcat([block_covariance_action(c,Float64.(1:5 .== j)) for j in 1:5]...)
    X = draws.-mean(draws;dims=2)
    empirical = X*X'/size(draws,2)
    @test c.mu ≈ vec(mean(draws;dims=2))
    @test isposdef(Symmetric(C))
    for (b,T) in zip(blocks,reference)
        @test C[b,b] ≈ .98empirical[b,b]+.02T atol=2e-14
    end
    @test out.metadata.covariance_convention == :empirical
    order = [5,3,1,4,2]
    permuted = project_block_carry(draws[order,:],[invperm(order)[b] for b in blocks],
        reference;rank=2,shrinkage=.02).factor
    v = randn(rng,5)
    @test block_covariance_action(permuted,v[order]) ≈ (C*v)[order] atol=1e-12
    @test permuted.mu == c.mu[order]
    conditional = block_field_conditional(c,[4,5],[1,2,3])
    cf = conditional.field_state
    Cf = hcat([block_covariance_action(cf,Float64.(1:2 .== j)) for j in 1:2]...)
    lower = block_precision_lower_bound(cf)
    @test 0 < lower <= minimum(eigvals(inv(Symmetric(Cf))))*(1+1e-12)
    @test_throws ArgumentError project_block_carry(draws,blocks,reference;shrinkage=0)
    @test_throws ArgumentError project_block_carry(draws,blocks,reference;rank=64)
    @test_throws ArgumentError project_block_carry(zeros(5,257),blocks,reference)
    # Rank zero remains a documented product across blocks; delta=1 returns T.
    independent = project_block_carry(draws,blocks,reference;rank=0).factor
    @test isempty(independent.B)
    regularized = project_block_carry(draws,blocks,reference;rank=2,shrinkage=1).factor
    @test block_covariance_action(regularized,v) ≈ v
    mktemp() do path,io
        serialize(io,out); close(io)
        restored = deserialize(path)
        @test block_density_gradient(restored.factor,v) == block_density_gradient(c,v)
        @test restored.metadata == out.metadata
    end
end

@testset "field and repeated-unit predictive covariance survives projection" begin
    C = [2/3 0.0 -1/3;0.0 2/3 -1/3;-1/3 -1/3 5/6]
    L = Matrix(cholesky(Symmetric(C)).L)
    mu = fill(.425,3)
    # Exact moment set for this THREE-coordinate oracle, not product draw count.
    draws = mu .+ sqrt(3)*hcat(L,-L)
    blocks,reference = [[1],[2],[3]],[ones(1,1) for _ in 1:3]
    c = project_block_carry(draws,blocks,reference;rank=3,shrinkage=1e-8).factor
    direction = [0.,1.,1.]
    predictive_variance = 1+dot(direction,block_covariance_action(c,direction))
    @test predictive_variance ≈ 11/6 atol=2e-8
    @test norm(block_covariance_action(c,direction)-C*direction) < 2e-8
    cross = block_covariance_action(c,[0.,1.,0.])[3]
    @test cross ≈ -1/3 atol=1e-8
    conditional = block_field_conditional(c,[3],[1,2])
    for outer in ([.4,.2],[-1.,.5])
        result = block_conditional_density_gradient(conditional,outer,[.7])
        @test only(result.mean) ≈ (1.7-sum(outer))/2 atol=2e-8
        @test only(block_covariance_action(conditional.field_state,[1.])) ≈ .5 atol=2e-8
    end
    println((;predictive_variance,field_unit_covariance=cross,
        scope="exact small moment control; sample/rank/shrinkage errors remain separate"))
end

@testset "projection sampling and sequential order controls" begin
    C = [2/3 0.0 -1/3;0.0 2/3 -1/3;-1/3 -1/3 5/6]
    L = Matrix(cholesky(Symmetric(C)).L)
    mu,blocks,reference = fill(.425,3),[[1],[2],[3]],[ones(1,1) for _ in 1:3]
    direction = [0.,1.,1.]
    target_variance = 1+dot(direction,C*direction)
    # Report draw noise separately from rank/shrinkage: no monotonic-improvement
    # assertion for one Monte Carlo realization and no post-hoc tolerance.
    for seed in (821,822), M in (32,128), rank in (1,3)
        draws = mu.+L*randn(Xoshiro(seed),3,M)
        c = project_block_carry(draws,blocks,reference;rank,shrinkage=.02).factor
        prediction = 1+dot(direction,block_covariance_action(c,direction))
        println((;seed,draws=M,rank,shrinkage=.02,
            predictive_mean_error=dot(direction,c.mu-mu),
            predictive_variance_error=prediction-target_variance))
        @test isfinite(prediction) && prediction > 1
    end
    # Exact small Gaussian update followed by projection, in either window
    # order. Full rank removes truncation, while finite shrinkage still makes
    # assumed-density streaming order-sensitive.
    function update_and_project(m,S,a,y,delta)
        v = S*a
        updated_mu = m+v*((y-dot(a,m))/(1+dot(a,v)))
        updated_C = S-v*v'/(1+dot(a,v))
        root = Matrix(cholesky(Symmetric(updated_C)).L)
        draws = updated_mu.+sqrt(3)*hcat(root,-root)
        c = project_block_carry(draws,blocks,reference;rank=3,shrinkage=delta).factor
        covariance = hcat([block_covariance_action(c,Float64.(1:3 .== j)) for j in 1:3]...)
        c.mu,covariance
    end
    windows = [([1.,0.,1.],.8),([0.,1.,1.],-.4)]
    function sequential(order,delta)
        m,S = zeros(3),Matrix{Float64}(I,3,3)
        for i in order
            a,y = windows[i]
            m,S = update_and_project(m,S,a,y,delta)
        end
        m,S
    end
    A = hcat(first.(windows)... )'; y = last.(windows)
    batch_C = inv(Symmetric(Matrix{Float64}(I,3,3)+A'*A))
    batch_mu = batch_C*A'*y
    near = sequential([1,2],1e-8)
    @test near[1] ≈ batch_mu atol=2e-8
    @test near[2] ≈ batch_C atol=2e-8
    forward,reverse = sequential([1,2],.02),sequential([2,1],.02)
    order_mean = norm(forward[1]-reverse[1])
    order_covariance = norm(forward[2]-reverse[2])
    @test order_mean > 1e-4
    @test order_covariance > 1e-4
    println((;shrinkage=.02,order_mean,order_covariance,
        scope="controlled projection order sensitivity, not an invariance claim"))
end

@testset "projection storage with fixed draw and rank caps" begin
    sizes = Int[]
    for n in (128,1024)
        rng = Xoshiro(812)
        draws = randn(rng,n,32)
        blocks = [[i] for i in 1:n]
        reference = [ones(1,1) for _ in 1:n]
        out = project_block_carry(draws,blocks,reference;rank=4)
        push!(sizes,Base.summarysize(out))
        @test all(isfinite,block_precision(out.factor,ones(n)))
        @test block_precision_lower_bound(out.factor) > 0
    end
    @test sizes[2] < 9sizes[1]
    println((;dimensions=(128,1024),state_bytes=sizes,draws=32,rank=4,
        scope="projection state only; native fit and approximation quality unmeasured"))
end
