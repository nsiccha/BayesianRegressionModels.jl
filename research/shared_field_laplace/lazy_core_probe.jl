# Fixed auxiliary-core carry: algebra controls, not a streaming product API.
# Method study: MultivariateStats.jl 06d85eb6 src/fa.jl predict uses Gaussian
# factor conditioning, but materializes a large diagonal; its fit uses a full
# data covariance. Here LinearAlgebra handles only rank/block-sized solves.
# Broderick et al2013 Eq2 supplies the previous-posterior-as-prior framework;
# this conditional-block moment projection is a separate approximation.
using LinearAlgebra, Random, Statistics, Test, Serialization
BLAS.set_num_threads(1)

function auxiliary_core(m,C)
    length(m)<=64 || error("auxiliary core exceeds fixed rank cap")
    size(C)==(length(m),length(m)) || throw(DimensionMismatch("core covariance"))
    (;m=copy(m),F=cholesky(Symmetric(C)))
end
function auxiliary_block(a,B,D)
    1<=length(a)<=16 || error("conditional block exceeds fixed block cap")
    size(B,1)==length(a) && size(B,2)<=64 || throw(DimensionMismatch("block loading"))
    size(D)==(length(a),length(a)) || throw(DimensionMismatch("block covariance"))
    (;a=copy(a),B=copy(B),F=cholesky(Symmetric(D)))
end

# No absent-row dictionary is read. Whole active blocks are supplied by label.
function auxiliary_conditional(core,blocks,draws)
    r=length(core.m)
    precision=core.F\Matrix{Float64}(I,r,r)
    M=size(first(values(draws)),2)
    information=zeros(r,M)
    Set(keys(blocks))==Set(keys(draws)) || error("active draw/block keys differ")
    for key in keys(blocks)
        b,X=blocks[key],draws[key]
        size(X)==(length(b.a),M) || throw(DimensionMismatch("active draws"))
        size(b.B,2)==r || throw(DimensionMismatch("core rank"))
        DB=b.F\b.B
        precision .+= b.B'*DB
        information .+= DB'*(X.-b.a.-b.B*core.m)
    end
    F=cholesky(Symmetric(precision))
    (;means=core.m.+F\information,precision=F)
end
function auxiliary_lift(rng,core,blocks,draws)
    c=auxiliary_conditional(core,blocks,draws)
    c.means+c.precision.U\randn(rng,size(c.means))
end

# The joint empirical covariance is blended with blockdiag(core_reference,
# residual_reference_b). Residual Gram form avoids covariance subtraction or
# eigenvalue clipping. delta=0 is for exact full-rank controls; sampled fits
# must predeclare a positive regularization policy.
function auxiliary_project(core,blocks,draws,H;delta=.01)
    0<=delta<=1 || throw(ArgumentError("delta must be in [0,1]"))
    r,M=size(H)
    r==length(core.m) && 2<=M<=256 || error("fixed core/draw cap violated")
    Set(keys(blocks))==Set(keys(draws)) || error("active draw/block keys differ")
    all(isfinite,H) || error("nonfinite auxiliary draws")
    m=vec(mean(H;dims=2));Z=(H.-m)/sqrt(M)
    reference=Matrix(core.F.L*core.F.L')
    C=(1-delta)*(Z*Z')+delta*reference
    updated=auxiliary_core(m,C)
    touched=Dict{Symbol,NamedTuple}()
    for key in keys(blocks)
        b,X=blocks[key],draws[key]
        size(X)==(length(b.a),M) || throw(DimensionMismatch("active draws"))
        all(isfinite,X) || error("nonfinite active draws")
        mu=vec(mean(X;dims=2));Y=(X.-mu)/sqrt(M)
        cross=(1-delta)*(Y*Z')
        B=Matrix((updated.F\cross')')
        E=Y-B*Z
        D=(1-delta)*(E*E')+delta*(Matrix(b.F.L*b.F.L')+B*reference*B')
        touched[key]=auxiliary_block(mu-B*m,B,D)
    end
    # Return a patch. Applying it to an owned mutable store is O(active rows);
    # copying/serializing the entire checkpoint remains O(all retained rows).
    (;core=updated,touched,metadata=(;delta,draw_count=M,rank=r,
        projection=:fixed_core_conditional_blocks))
end

# Dense oracles below are capped independently and never used by the above.
function auxiliary_dense(core,store,order)
    n=sum(length(store[k].a) for k in order);r=length(core.m)
    n+r<=16 || error("dense auxiliary oracle cap exceeded")
    a=vcat([store[k].a for k in order]...)
    B=vcat([store[k].B for k in order]...)
    D=zeros(n,n);positions=Dict{Symbol,UnitRange{Int}}();offset=0
    for k in order
        b=store[k];indices=offset+1:offset+length(b.a)
        D[indices,indices]=b.F.L*b.F.L';positions[k]=indices
        offset+=length(b.a)
    end
    C=Matrix(core.F.L*core.F.L')
    (;mean=vcat(core.m,a+B*core.m),covariance=[C C*B';B*C D+B*C*B'],positions,r)
end
function exact_linear_update(mean,covariance,direction,y,variance)
    length(mean)<=16 || error("dense Gaussian update cap exceeded")
    v=covariance*direction;s=variance+dot(direction,v)
    (;mean=mean+v*((y-dot(direction,mean))/s),covariance=covariance-v*v'/s)
end
function exact_moment_points(mean,C)
    d=length(mean);d<=16 || error("exact moment control cap exceeded")
    L=Matrix(cholesky(Symmetric(C)).L)
    mean.+sqrt(d)*hcat(L,-L)
end
function exact_active_projection(core,store,order,posterior,active;delta=0.)
    structure=auxiliary_dense(core,store,order)
    selected=vcat(collect(1:structure.r),
        [collect(structure.r.+structure.positions[k]) for k in active]...)
    X=exact_moment_points(posterior.mean[selected],posterior.covariance[selected,selected])
    offset=structure.r;draws=Dict{Symbol,Matrix{Float64}}()
    for k in active
        n=length(store[k].a);draws[k]=X[offset+1:offset+n,:];offset+=n
    end
    auxiliary_project(core,Dict(k=>store[k] for k in active),draws,X[1:structure.r,:];delta)
end

@testset "fixed auxiliary core carries absent units" begin
    core=auxiliary_core([.2,-.1],[1. .2;.2 .7])
    store=Dict(:a=>auxiliary_block([.1],[.8 .3],reshape([.5],1,1)),
        :b=>auxiliary_block([-.2],[-.4 .7],reshape([.6],1,1)),
        :c=>auxiliary_block([.3],[.5 -.6],reshape([.4],1,1)))
    order=[:a,:b,:c]
    prior=auxiliary_dense(core,store,order)
    active=Dict(:a=>store[:a]);draws=Dict(:a=>reshape([-.4,.7,1.2],1,:))
    conditional=auxiliary_conditional(core,active,draws)
    j=prior.r+first(prior.positions[:a]);C=prior.covariance
    expected_mean=core.m.+(C[1:2,j]/C[j,j])*(draws[:a].-prior.mean[j])
    expected_covariance=C[1:2,1:2]-C[1:2,j]*C[j,1:2]'/C[j,j]
    @test conditional.means≈expected_mean atol=2e-14
    @test conditional.precision\Matrix{Float64}(I,2,2)≈expected_covariance atol=2e-14
    @test auxiliary_lift(Xoshiro(714),core,active,draws)==auxiliary_lift(Xoshiro(714),core,active,draws)

    # End-to-end active marginal -> old-core conditional -> projection, using
    # independent exact moment sets instead of Monte Carlo sampling noise.
    observed=exact_linear_update(prior.mean,prior.covariance,[0.,0.,1.,0.,0.],.7,.4)
    Xnodes=reshape(observed.mean[3].+sqrt(observed.covariance[3,3]).*[-1.,1.],1,:)
    lifted=auxiliary_conditional(core,active,Dict(:a=>Xnodes))
    noise=lifted.precision.U\(sqrt(2)*hcat(Matrix{Float64}(I,2,2),-Matrix{Float64}(I,2,2)))
    H=hcat([lifted.means[:,j].+noise for j in 1:2]...)
    X=Dict(:a=>repeat(Xnodes;inner=(1,4)))
    restored=auxiliary_project(core,active,X,H;delta=0.)
    joined=auxiliary_dense(restored.core,merge(copy(store),restored.touched),order)
    @test joined.mean≈observed.mean atol=3e-13
    @test joined.covariance≈observed.covariance atol=3e-13

    # Single-block observations preserve conditional independence exactly.
    # A leaves, B is observed twice, then A returns; C is absent throughout.
    dense=(;mean=prior.mean,covariance=prior.covariance)
    for (label,y,variance) in ((:a,.7,.4),(:b,-.2,.6),(:b,.5,.8),(:a,-.4,.3))
        before=copy(store)
        direction=zeros(length(dense.mean));direction[prior.r+first(prior.positions[label])]=1
        dense=exact_linear_update(dense.mean,dense.covariance,direction,y,variance)
        patch=exact_active_projection(core,store,order,dense,[label])
        core=patch.core;merge!(store,patch.touched)
        actual=auxiliary_dense(core,store,order)
        @test actual.mean≈dense.mean atol=3e-13
        @test actual.covariance≈dense.covariance atol=3e-13
        @test all(store[k]===before[k] for k in order if k!=label)
    end
    @test abs(auxiliary_dense(core,store,order).mean[end]-prior.mean[end])>1e-3
    mktemp() do path,io
        serialize(io,(;core,store));close(io);saved=deserialize(path)
        @test auxiliary_dense(saved.core,saved.store,order)==auxiliary_dense(core,store,order)
    end

    # Two active blocks couple their residuals. Projection preserves core/
    # each-block moments, but deliberately drops that conditional covariance.
    before=auxiliary_dense(core,store,order)
    direction=[0.,0.,1.,1.,0.]
    dense=exact_linear_update(before.mean,before.covariance,direction,.2,.5)
    patch=exact_active_projection(core,store,order,dense,[:a,:b])
    updated=merge(copy(store),patch.touched)
    actual=auxiliary_dense(patch.core,updated,order)
    residual=dense.covariance[3,4]-dot(dense.covariance[3,1:2],
        dense.covariance[1:2,1:2]\dense.covariance[1:2,4])
    @test actual.mean≈dense.mean atol=3e-13
    @test diag(actual.covariance)≈diag(dense.covariance) atol=3e-13
    @test abs(residual)>1e-3
    @test actual.covariance[3,4]-dense.covariance[3,4]≈-residual atol=3e-13
    @test actual.covariance[:,5]≈dense.covariance[:,5] atol=3e-13
    println((;coupled_active_residual=residual,
        discarded_sum_covariance_contribution=2residual,
        projected_minus_exact_sum_variance=-2residual,
        scope="controlled projection loss, not an accuracy pass"))

    # New blocks independent of the fixed core can learn dependence that this
    # representation cannot retain. Expose the blind spot rather than hide it.
    newcore=auxiliary_core([0.],ones(1,1))
    newstore=Dict(k=>auxiliary_block([0.],zeros(1,1),ones(1,1)) for k in [:x,:y])
    old=auxiliary_dense(newcore,newstore,[:x,:y])
    dense=exact_linear_update(old.mean,old.covariance,[0.,1.,1.],1.,1.)
    patch=exact_active_projection(newcore,newstore,[:x,:y],dense,[:x,:y])
    actual=auxiliary_dense(patch.core,patch.touched,[:x,:y])
    @test dense.covariance[2,3]≈-1/3
    @test actual.covariance[2,3]==0

    # Predetermined positive regularization survives a singular empirical core
    # and preserves the intended regularized marginal block/core moments.
    H=repeat([.2,-.1],1,8);X=Dict(:a=>reshape(collect(1.:8.),1,:))
    patch=auxiliary_project(core,Dict(:a=>store[:a]),X,H;delta=.02)
    marginal=auxiliary_dense(patch.core,patch.touched,[:a])
    expected=.98var(vec(X[:a]);corrected=false)+.02only(store[:a].F.L*store[:a].F.L')
    @test marginal.covariance[end,end]≈expected atol=3e-13
    @test isposdef(Symmetric(marginal.covariance))
end
