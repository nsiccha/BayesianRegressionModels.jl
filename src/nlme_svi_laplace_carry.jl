# Gaussian common-factor algebra. These helpers own their returned buffers;
# supplied means, covariances, block dictionaries and draws are read-only.
# No model replay, checkpoint identity or inference policy is implied here.
_brm_svi_carry_error(message) = throw(ArgumentError("GaussianBlockLaplace carry: " * message))

function _brm_svi_carry_factor(C, n)
    size(C) == (n,n) || throw(DimensionMismatch("carry covariance block"))
    all(isfinite,C) && issymmetric(C) || _brm_svi_carry_error("covariance must be finite and symmetric")
    cholesky(Symmetric(Matrix{Float64}(C)))
end

function _brm_svi_carry_gaussian(mu, blocks, covariances, B; block_limit=16, rank_limit=64)
    1 <= block_limit <= 16 && 0 <= rank_limit <= 64 || _brm_svi_carry_error("invalid fixed block/rank cap")
    length(blocks) == length(covariances) || throw(DimensionMismatch("carry blocks"))
    length(mu) == size(B,1) || throw(DimensionMismatch("carry factor rows"))
    sort(reduce(vcat,blocks;init=Int[])) == collect(eachindex(mu)) ||
        _brm_svi_carry_error("blocks must partition the coordinates")
    all(b -> 1 <= length(b) <= block_limit,blocks) || _brm_svi_carry_error("block exceeds fixed cap")
    size(B,2) <= rank_limit || _brm_svi_carry_error("rank exceeds fixed cap")
    all(isfinite,mu) && all(isfinite,B) || _brm_svi_carry_error("nonfinite Gaussian parameters")
    factors = [_brm_svi_carry_factor(C,length(b)) for (b,C) in zip(blocks,covariances)]
    base = (;mu=Vector{Float64}(mu),blocks=[Vector{Int}(b) for b in blocks],
        factors,B=Matrix{Float64}(B))
    DB = _brm_svi_carry_base_solve(base,base.B)
    r = size(B,2)
    R = cholesky(Symmetric(Matrix{Float64}(I,r,r)+base.B'*DB))
    logdeterminant = sum(logdet,factors;init=0.)+logdet(R)
    (;base...,DB,R,logdeterminant,block_limit,rank_limit)
end

function _brm_svi_carry_base_solve(c,V::AbstractMatrix)
    size(V,1) == length(c.mu) || throw(DimensionMismatch("carry action rows"))
    out = Matrix{Float64}(undef,size(V))
    for (indices,F) in zip(c.blocks,c.factors)
        out[indices,:] .= F \ V[indices,:]
    end
    out
end
_brm_svi_carry_base_solve(c,v::AbstractVector) = vec(_brm_svi_carry_base_solve(c,reshape(v,:,1)))
_brm_svi_carry_precision(c,v) = _brm_svi_carry_base_solve(c,v)-c.DB*(c.R \ (c.DB'*v))
function _brm_svi_carry_covariance(c,v::AbstractVector)
    length(v) == length(c.mu) || throw(DimensionMismatch("carry action rows"))
    out = c.B*(c.B'*v)
    for (indices,F) in zip(c.blocks,c.factors)
        out[indices] += F.L*(F.L'*v[indices])
    end
    out
end
function _brm_svi_carry_density_gradient(c,x)
    delta = x-c.mu
    score = _brm_svi_carry_precision(c,delta)
    (;value=-(length(x)*log(2pi)+c.logdeterminant+dot(delta,score))/2,gradient=-score)
end

# Arbitrary retained order and partial small blocks are allowed for a marginal.
function _brm_svi_carry_marginal(c,keep)
    length(unique(keep)) == length(keep) || _brm_svi_carry_error("duplicate marginal coordinate")
    all(i -> i in eachindex(c.mu),keep) || throw(BoundsError(c.mu,keep))
    positions = Dict(old=>new for (new,old) in enumerate(keep))
    blocks,covariances = Vector{Int}[],Matrix{Float64}[]
    for (indices,F) in zip(c.blocks,c.factors)
        selected = findall(i -> haskey(positions,i),indices)
        isempty(selected) && continue
        push!(blocks,[positions[i] for i in indices[selected]])
        push!(covariances,(F.L*F.L')[selected,selected])
    end
    _brm_svi_carry_gaussian(c.mu[keep],blocks,covariances,c.B[keep,:];c.block_limit,c.rank_limit)
end

# A conditional partition must preserve each entire residual block.
function _brm_svi_carry_conditional(c,field,outer)
    sort(vcat(field,outer)) == collect(eachindex(c.mu)) ||
        _brm_svi_carry_error("conditional coordinates must partition the carry")
    selected = Set(field)
    all(b -> all(in(selected),b) || all(i -> !(i in selected),b),c.blocks) ||
        _brm_svi_carry_error("conditional partition cuts a residual block")
    marginal = _brm_svi_carry_marginal(c,outer)
    base = _brm_svi_carry_marginal(c,field)
    Bf = c.B[field,:]
    conditional_factor = Bf/marginal.R.U
    field_state = _brm_svi_carry_gaussian(base.mu,base.blocks,
        [F.L*F.L' for F in base.factors],conditional_factor;c.block_limit,c.rank_limit)
    (;marginal,field_state,Bf)
end
function _brm_svi_carry_conditional_density_gradient(c,outer,field)
    centered = outer-c.marginal.mu
    mean = c.field_state.mu+c.Bf*(c.marginal.R \ (c.marginal.DB'*centered))
    delta = field-mean
    score = _brm_svi_carry_precision(c.field_state,delta)
    value = -(length(field)*log(2pi)+c.field_state.logdeterminant+dot(delta,score))/2
    outer_gradient = c.marginal.DB*(c.marginal.R \ (c.Bf'*score))
    (;value,outer_gradient,field_gradient=-score,mean)
end

# Algebraic lower bound on precision; not outward-rounded floating-point
# certification. A concave log likelihood adds a positive semidefinite term.
function _brm_svi_carry_precision_lower_bound(c)
    isempty(c.mu) && _brm_svi_carry_error("precision bound needs nonempty coordinates")
    upper = maximum(c.factors) do F
        maximum(vec(sum(abs,F.L*F.L';dims=2)))
    end
    lower = inv(upper+sum(abs2,c.B))
    isfinite(lower) && lower > 0 || _brm_svi_carry_error("precision bound is not positive finite")
    lower
end

function _brm_svi_carry_psd(D,n)
    size(D) == (n,n) || throw(DimensionMismatch("conditional covariance block"))
    all(isfinite,D) && issymmetric(D) && eigmin(Symmetric(D)) >= 0 ||
        _brm_svi_carry_error("conditional covariance must be finite symmetric PSD")
    nothing
end

# `means` may be draws or conditional means. `conditional` is their expected
# conditional covariance in each small block, with zero conditional cross
# blocks. PCA compresses the means only. The numerical reference is NOT an
# authored prior; shrinkage changes the represented moments explicitly.
function _brm_svi_carry_project(means,blocks,reference;
        rank=8,shrinkage=.01,conditional=nothing,draw_limit=256,
        block_limit=16,rank_limit=64)
    n,M = size(means)
    2 <= draw_limit <= 256 && 2 <= M <= draw_limit || _brm_svi_carry_error("draw count exceeds fixed cap")
    0 <= rank <= min(M-1,rank_limit) || _brm_svi_carry_error("projection rank exceeds cap")
    0 < shrinkage <= 1 || _brm_svi_carry_error("sample projection shrinkage must be in (0,1]")
    all(isfinite,means) || _brm_svi_carry_error("nonfinite projection points")
    _brm_svi_carry_gaussian(zeros(n),blocks,reference,zeros(n,0);block_limit,rank_limit)
    conditional === nothing || length(conditional) == length(blocks) || throw(DimensionMismatch("conditional blocks"))
    local_cov = conditional === nothing ? [zeros(length(b),length(b)) for b in blocks] : conditional
    for (b,D) in zip(blocks,local_cov)
        _brm_svi_carry_psd(D,length(b))
    end
    mu = vec(mean(means;dims=2))
    X = (means.-mu)/sqrt(M)
    V = eigen(Symmetric(X'*X)).vectors[:,end-rank+1:end]
    A = X*V
    B = sqrt(1-shrinkage)*A
    covariances = map(zip(blocks,reference,local_cov)) do (b,T,D)
        E = X[b,:]-A[b,:]*V'
        (1-shrinkage)*(E*E'+D)+shrinkage*T
    end
    factor = _brm_svi_carry_gaussian(mu,blocks,covariances,B;block_limit,rank_limit)
    (;factor,metadata=(;draw_count=M,rank,shrinkage,draw_limit,block_limit,rank_limit,
        covariance_convention=:empirical,projection=:means_gram_plus_local_covariances,
        reference_blocks=deepcopy(reference),conditional_blocks=deepcopy(local_cov)))
end

# Fixed auxiliary representation: eta~N(m,C), x_b|eta~N(a_b+B_b eta,D_b).
function _brm_svi_carry_core(m,C;rank_limit=64)
    0 <= rank_limit <= 64 && length(m) <= rank_limit || _brm_svi_carry_error("auxiliary rank exceeds cap")
    all(isfinite,m) || _brm_svi_carry_error("nonfinite auxiliary mean")
    (;m=Vector{Float64}(m),F=_brm_svi_carry_factor(C,length(m)),rank_limit)
end
function _brm_svi_carry_block(a,B,D;block_limit=16,rank_limit=64)
    1 <= block_limit <= 16 && 1 <= length(a) <= block_limit || _brm_svi_carry_error("conditional block exceeds cap")
    0 <= rank_limit <= 64 && size(B,2) <= rank_limit || _brm_svi_carry_error("auxiliary rank exceeds cap")
    size(B,1) == length(a) || throw(DimensionMismatch("conditional loading"))
    all(isfinite,a) && all(isfinite,B) || _brm_svi_carry_error("nonfinite conditional parameters")
    (;a=Vector{Float64}(a),B=Matrix{Float64}(B),F=_brm_svi_carry_factor(D,length(a)),block_limit,rank_limit)
end

# One-time conversion of an initial common-factor Gaussian. Later windows
# pass only their active blocks; conversion itself necessarily visits all rows.
function _brm_svi_carry_auxiliary(c,labels=collect(eachindex(c.blocks)))
    length(labels) == length(c.blocks) && length(unique(labels)) == length(labels) ||
        _brm_svi_carry_error("auxiliary labels must name every block once")
    r=size(c.B,2)
    core=_brm_svi_carry_core(zeros(r),Matrix{Float64}(I,r,r);rank_limit=c.rank_limit)
    blocks=Dict(key=>_brm_svi_carry_block(c.mu[b],c.B[b,:],Matrix(F.L*F.L');
        c.block_limit,c.rank_limit) for (key,b,F) in zip(labels,c.blocks,c.factors))
    (;core,blocks,order=copy(labels))
end

# The caller supplies active blocks only, in an explicit stable order. No
# absent-row dictionary is visited, copied, or serialized by these helpers.
function _brm_svi_carry_active_marginal(core,blocks,order)
    length(unique(order)) == length(order) && Set(order) == Set(keys(blocks)) ||
        _brm_svi_carry_error("active order must name each supplied block once")
    isempty(order) && _brm_svi_carry_error("active marginal needs a block")
    means,loadings,covariances = Vector{Float64}[],Matrix{Float64}[],Matrix{Float64}[]
    indices = Vector{Int}[]; offset=0
    for key in order
        b=blocks[key]
        size(b.B,2) == length(core.m) || throw(DimensionMismatch("auxiliary core rank"))
        push!(means,b.a+b.B*core.m);push!(loadings,b.B*core.F.L)
        push!(covariances,Matrix(b.F.L*b.F.L'))
        push!(indices,collect(offset+1:offset+length(b.a)));offset+=length(b.a)
    end
    _brm_svi_carry_gaussian(vcat(means...),indices,covariances,vcat(loadings...);rank_limit=core.rank_limit)
end
function _brm_svi_carry_auxiliary_conditional(core,blocks,draws)
    !isempty(blocks) && Set(keys(blocks)) == Set(keys(draws)) ||
        _brm_svi_carry_error("supply all active block draws and no absent blocks")
    r=length(core.m);M=size(first(values(draws)),2)
    1 <= M <= 256 || _brm_svi_carry_error("auxiliary draw count exceeds cap")
    precision=core.F\Matrix{Float64}(I,r,r)
    information=zeros(r,M)
    for key in keys(blocks)
        b,X=blocks[key],draws[key]
        size(X) == (length(b.a),M) && size(b.B,2) == r || throw(DimensionMismatch("active draw/core shape"))
        all(isfinite,X) || _brm_svi_carry_error("nonfinite active draws")
        DB=b.F\b.B
        precision .+= b.B'*DB
        information .+= DB'*(X.-b.a.-b.B*core.m)
    end
    F=cholesky(Symmetric(precision))
    (;means=core.m.+F\information,precision=F)
end
function _brm_svi_carry_lift(rng,core,blocks,draws)
    c=_brm_svi_carry_auxiliary_conditional(core,blocks,draws)
    c.means+c.precision.U\randn(rng,size(c.means))
end

# Return a patch; callers retain absent block objects unchanged. `conditional`
# adds E Cov(x_b|conditioned variables), which must condition on the sampled
# auxiliary core itself. Cross-block conditional covariances are not retained.
# delta=0 is allowed only for exact-moment oracle controls by explicit opt-in.
function _brm_svi_carry_project_active(core,blocks,means,H;
        conditional=Dict(),shrinkage=.01,exact_moments=false)
    (0 < shrinkage <= 1 || (exact_moments && shrinkage == 0)) ||
        _brm_svi_carry_error("active sample projection requires positive shrinkage")
    r,M=size(H)
    r == length(core.m) && 2 <= M <= 256 || _brm_svi_carry_error("active core/draw cap mismatch")
    Set(keys(blocks)) == Set(keys(means)) && issubset(Set(keys(conditional)),Set(keys(blocks))) ||
        _brm_svi_carry_error("active projection keys differ")
    all(isfinite,H) || _brm_svi_carry_error("nonfinite auxiliary draws")
    alpha=1-shrinkage
    m=vec(mean(H;dims=2));Z=(H.-m)/sqrt(M)
    reference=Matrix(core.F.L*core.F.L')
    C=alpha*(Z*Z')+shrinkage*reference
    updated=_brm_svi_carry_core(m,C;rank_limit=core.rank_limit)
    touched=Dict{keytype(blocks),NamedTuple}()
    for key in keys(blocks)
        b,X=blocks[key],means[key]
        size(X) == (length(b.a),M) && size(b.B,2) == r || throw(DimensionMismatch("active moment shape"))
        all(isfinite,X) || _brm_svi_carry_error("nonfinite active means")
        Dlocal=get(conditional,key,nothing)
        Dlocal === nothing || _brm_svi_carry_psd(Dlocal,length(b.a))
        mu=vec(mean(X;dims=2));Y=(X.-mu)/sqrt(M)
        cross=alpha*(Y*Z')
        B=Matrix((updated.F\cross')')
        E=Y-B*Z
        D=alpha*(E*E')+shrinkage*(Matrix(b.F.L*b.F.L')+B*reference*B')
        Dlocal === nothing || (D=D+alpha*Dlocal)
        touched[key]=_brm_svi_carry_block(mu-B*m,B,D;b.block_limit,b.rank_limit)
    end
    (;core=updated,touched,metadata=(;shrinkage,draw_count=M,rank=r,exact_moments,
        projection=:fixed_core_conditional_moments,integrated_blocks=collect(keys(conditional))))
end
