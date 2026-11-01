# Data-only carry algebra allowing common-factor rows for unit effects.
# This is NOT a fitted projection, serialization API, or streaming entry.
# Only the bounded oracle fixtures below construct a full covariance.
using LinearAlgebra, Random, Test

function block_carry(mu,blocks,covariances,B;block_limit=16,rank_limit=64)
    length(blocks) == length(covariances) || throw(DimensionMismatch("carry blocks"))
    length(mu) == size(B,1) || throw(DimensionMismatch("carry factor rows"))
    sort(reduce(vcat,blocks;init=Int[])) == collect(eachindex(mu)) ||
        throw(ArgumentError("carry blocks must partition the coordinates"))
    all(b -> 1 <= length(b) <= block_limit,blocks) ||
        throw(ArgumentError("carry block exceeds fixed block_limit=$block_limit"))
    size(B,2) <= rank_limit || throw(ArgumentError("carry rank exceeds rank_limit"))
    for (b,C) in zip(blocks,covariances)
        size(C) == (length(b),length(b)) || throw(DimensionMismatch("carry covariance"))
    end
    factors = [cholesky(Symmetric(C)) for C in covariances]
    base = (;mu=copy(mu),blocks=deepcopy(blocks),factors,B=copy(B))
    DB = block_base_solve(base,B)
    r = size(B,2)
    R = cholesky(Symmetric(Matrix{Float64}(I,r,r)+B'*DB))
    logdeterminant = sum(logdet, factors;init=0.0)+logdet(R)
    (;base...,DB,R,logdeterminant,block_limit,rank_limit)
end

function block_base_solve(c,V::AbstractMatrix)
    out = similar(V)
    for (indices,F) in zip(c.blocks,c.factors)
        out[indices,:] .= F \ V[indices,:]
    end
    out
end
block_base_solve(c,v::AbstractVector) = vec(block_base_solve(c,reshape(v,:,1)))
block_precision(c,v) = block_base_solve(c,v)-c.DB*(c.R \ (c.DB'*v))
function block_density_gradient(c,x)
    delta = x-c.mu
    score = block_precision(c,delta)
    (;value=-(length(x)*log(2π)+c.logdeterminant+dot(delta,score))/2,gradient=-score)
end

# Removing coordinates takes a marginal, not a conditional at zero. Partial
# unit blocks are handled inside their fixed-size covariance only.
function block_marginal(c,keep)
    length(unique(keep)) == length(keep) || throw(ArgumentError("duplicate coordinate"))
    all(i -> i in eachindex(c.mu),keep) || throw(BoundsError(c.mu,keep))
    positions = Dict(old=>new for (new,old) in enumerate(keep))
    blocks, covariances = Vector{Int}[],Matrix{Float64}[]
    for (indices,F) in zip(c.blocks,c.factors)
        selected = findall(i -> haskey(positions,i),indices)
        isempty(selected) && continue
        push!(blocks,[positions[i] for i in indices[selected]])
        L = F.L
        push!(covariances,(L*L')[selected,selected])
    end
    block_carry(c.mu[keep],blocks,covariances,c.B[keep,:];c.block_limit,c.rank_limit)
end

# The field must consist of entire base blocks: cross-block dependence is
# carried by B. A partition cutting a base block is explicitly unsupported.
function block_field_conditional(c,field,outer)
    sort(vcat(field,outer)) == collect(eachindex(c.mu)) ||
        throw(ArgumentError("field and outer must partition the carry"))
    selected = Set(field)
    all(b -> all(in(selected),b) || all(i -> !(i in selected),b),c.blocks) ||
        throw(ArgumentError("field partition cuts a carry base block"))
    marginal = block_marginal(c,outer)
    field_base = block_marginal(c,field)
    Bf = c.B[field,:]
    conditional_factor = Bf/marginal.R.U
    field_state = block_carry(c.mu[field],field_base.blocks,
        [F.L*F.L' for F in field_base.factors],conditional_factor;
        c.block_limit,c.rank_limit)
    (;marginal,field_state,Bf)
end

function block_conditional_density_gradient(c,outer,field)
    centered = outer-c.marginal.mu
    mean = c.field_state.mu+c.Bf*(c.marginal.R \ (c.marginal.DB'*centered))
    delta = field-mean
    score = block_precision(c.field_state,delta)
    value = -(length(field)*log(2π)+c.field_state.logdeterminant+dot(delta,score))/2
    outer_gradient = c.marginal.DB*(c.marginal.R \ (c.Bf'*score))
    (;value,outer_gradient,field_gradient=-score,mean)
end

@testset "common factor over field and unit coordinates" begin
    rng = Xoshiro(91)
    n = 5
    blocks = [[3,1],[5],[2,4]]
    covariances = [[0.8 0.2;0.2 1.2],reshape([0.7],1,1),[1.1 -0.3;-0.3 0.9]]
    mu,B = randn(rng,n),0.4randn(rng,n,3)
    c = block_carry(mu,blocks,covariances,B)
    exact = B*B'
    for (indices,C) in zip(blocks,covariances)
        exact[indices,indices] .+= C
    end
    x = randn(rng,n)
    out = block_density_gradient(c,x)
    @test block_precision(c,x) ≈ exact \ x
    @test out.gradient ≈ -(exact \ (x-mu))
    @test out.value ≈ -(n*log(2π)+logdet(Symmetric(exact))+dot(x-mu,exact \ (x-mu)))/2
    @test c.logdeterminant ≈ logdet(Symmetric(exact))
    keep = [5,2,1] # arbitrary order, cuts both small unit blocks
    marginal = block_marginal(c,keep)
    @test block_precision(marginal,x[keep]) ≈ exact[keep,keep] \ x[keep]
    field,outer = [5],[3,1,4,2]
    conditional = block_field_conditional(c,field,outer)
    density = block_conditional_density_gradient(conditional,x[outer],x[field])
    mean = mu[field]+exact[field,outer]*(exact[outer,outer] \ (x[outer]-mu[outer]))
    covariance = exact[field,field]-exact[field,outer]*(exact[outer,outer] \ exact[outer,field])
    @test density.mean ≈ mean
    @test density.field_gradient ≈ -(covariance \ (x[field]-mean))
    @test density.outer_gradient+block_density_gradient(conditional.marginal,x[outer]).gradient ≈ out.gradient[outer]
    @test density.value+block_density_gradient(conditional.marginal,x[outer]).value ≈ out.value
    @test_throws ArgumentError block_field_conditional(c,[1],[2,3,4,5])
    @test_throws ArgumentError block_carry(mu,blocks,covariances,B;block_limit=1)
end

@testset "shared unit rows preserve the exact carry counterexample" begin
    C = [2/3 0.0 -1/3;0.0 2/3 -1/3;-1/3 -1/3 5/6]
    B = Matrix(cholesky(Symmetric(C-0.2I)).L) # bounded three-coordinate oracle
    c = block_carry(fill(.425,3),[[1],[2],[3]],fill(reshape([.2],1,1),3),B)
    conditional = block_field_conditional(c,[3],[1,2])
    @test block_precision(c,ones(3)) ≈ C \ ones(3)
    for g in (-1.0,0.4,1.3), eta in (-.5,0.2)
        result = block_conditional_density_gradient(conditional,[g,eta],[.7])
        @test only(result.mean) ≈ (1.7-g-eta)/2
        @test only(block_precision(conditional.field_state,[1.0])) ≈ 2.0
    end
    # New-window predictive variance retains eta/z covariance exactly.
    direction = [0.0,1.0,1.0]
    action = 0.2direction+c.B*(c.B'*direction)
    @test 1+dot(direction,action) ≈ 11/6
end

@testset "fixed rank and bounded blocks use linear state storage" begin
    sizes = Int[]
    for n in (128,1024)
        rng = Xoshiro(92)
        c = block_carry(zeros(n),[[i,i+1] for i in 1:2:n],
            [Matrix{Float64}(I,2,2) for _ in 1:2:n],0.1randn(rng,n,3))
        v = randn(rng,n)
        @test block_precision(c,v+c.B*(c.B'*v)) ≈ v
        push!(sizes,Base.summarysize(c))
    end
    @test sizes[2] < 9sizes[1]
    println((;dimensions=(128,1024),state_bytes=sizes,
        scope="carry algebra/storage only; no projection or streaming fit"))
end
