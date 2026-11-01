# Bounded algebra controls; no model fitting, replay, or AD preparation.
using Test, BayesianRegressionModels, LinearAlgebra, Random, Statistics
const BRM = BayesianRegressionModels
isdefined(BRM,:_brm_svi_carry_gaussian) ||
    Base.include(BRM,joinpath(@__DIR__,"../src/nlme_svi_laplace_carry.jl"))

# Only test oracles assemble full covariances, independently capped at 16.
function carry_test_covariance(c)
    n=length(c.mu);n<=16 || error("dense test oracle cap")
    C=c.B*c.B'
    for (b,F) in zip(c.blocks,c.factors)
        C[b,b].+=F.L*F.L'
    end
    C
end
function carry_test_augmented(core,store,order)
    r=length(core.m);n=sum(length(store[k].a) for k in order)
    n+r<=16 || error("dense auxiliary test oracle cap")
    a=vcat([store[k].a for k in order]...)
    B=vcat([store[k].B for k in order]...)
    D=zeros(n,n);offset=0
    for key in order
        b=store[key];indices=offset+1:offset+length(b.a)
        D[indices,indices]=b.F.L*b.F.L';offset+=length(b.a)
    end
    C=Matrix(core.F.L*core.F.L')
    (;mean=vcat(core.m,a+B*core.m),covariance=[C C*B';B*C D+B*C*B'])
end
function carry_test_points(mu,C)
    n=length(mu);n<=16 || error("moment test oracle cap")
    mu.+sqrt(n)*hcat(Matrix(cholesky(Symmetric(C)).L),-Matrix(cholesky(Symmetric(C)).L))
end

@testset "common-factor covariance and conditional density" begin
    rng=Xoshiro(91)
    blocks=[[3,1],[5],[2,4]]
    Ds=[[.8 .2;.2 1.2],reshape([.7],1,1),[1.1 -.3;-.3 .9]]
    mu=randn(rng,5);B=.4randn(rng,5,3);x=randn(rng,5)
    original=(copy(mu),deepcopy(blocks),deepcopy(Ds),copy(B))
    c=BRM._brm_svi_carry_gaussian(mu,blocks,Ds,B)
    C=carry_test_covariance(c)
    result=BRM._brm_svi_carry_density_gradient(c,x)
    @test BRM._brm_svi_carry_precision(c,x)≈C\x
    @test BRM._brm_svi_carry_covariance(c,x)≈C*x
    @test result.gradient≈-(C\(x-mu))
    @test result.value≈-(5log(2pi)+logdet(Symmetric(C))+dot(x-mu,C\(x-mu)))/2
    @test (mu,blocks,Ds,B)==original
    auxiliary=BRM._brm_svi_carry_auxiliary(c)
    restored=BRM._brm_svi_carry_active_marginal(auxiliary.core,auxiliary.blocks,auxiliary.order)
    blockorder=vcat(blocks...)
    @test restored.mu≈mu[blockorder]
    @test carry_test_covariance(restored)≈C[blockorder,blockorder]
    keep=[5,2,1]
    marginal=BRM._brm_svi_carry_marginal(c,keep)
    @test carry_test_covariance(marginal)≈C[keep,keep]
    field,outer=[5],[3,1,4,2]
    conditional=BRM._brm_svi_carry_conditional(c,field,outer)
    d=BRM._brm_svi_carry_conditional_density_gradient(conditional,x[outer],x[field])
    expected_mean=mu[field]+C[field,outer]*(C[outer,outer]\(x[outer]-mu[outer]))
    expected_cov=C[field,field]-C[field,outer]*(C[outer,outer]\C[outer,field])
    @test d.mean≈expected_mean
    @test carry_test_covariance(conditional.field_state)≈expected_cov
    marginal_density=BRM._brm_svi_carry_density_gradient(conditional.marginal,x[outer])
    @test d.value+marginal_density.value≈result.value
    @test d.outer_gradient+marginal_density.gradient≈result.gradient[outer]
    lower=BRM._brm_svi_carry_precision_lower_bound(conditional.field_state)
    @test 0<lower<=minimum(eigvals(inv(Symmetric(expected_cov))))*(1+1e-12)
    # A cut residual block invalidates this conditional rank formula.
    @test_throws ArgumentError BRM._brm_svi_carry_conditional(c,[1],[2,3,4,5])
    @test_throws ArgumentError BRM._brm_svi_carry_gaussian(mu,blocks,Ds,B;block_limit=1)
end

@testset "conditional means Gram projection preserves local covariance" begin
    rng=Xoshiro(812);X=randn(rng,5,64)
    blocks=[[1,2],[3],[4,5]]
    reference=[Matrix{Float64}(I,length(b),length(b)) for b in blocks]
    local_cov=[zeros(2,2),reshape([.7],1,1),[.3 .1;.1 .4]]
    projected=BRM._brm_svi_carry_project(X,blocks,reference;
        conditional=local_cov,rank=2,shrinkage=.02)
    C=carry_test_covariance(projected.factor)
    A=(X.-mean(X;dims=2))/sqrt(size(X,2))
    for (b,D,T) in zip(blocks,local_cov,reference)
        @test C[b,b]≈.98*(A[b,:]*A[b,:]'+D)+.02T atol=2e-13
    end
    @test projected.metadata.reference_blocks==reference
    @test projected.metadata.conditional_blocks==local_cov
    @test projected.factor.mu≈vec(mean(X;dims=2))
    @test isposdef(Symmetric(C))
    # The rank basis is computed from the means, not from added local noise.
    plain=BRM._brm_svi_carry_project(X,blocks,reference;rank=2,shrinkage=.02)
    @test projected.factor.B≈plain.factor.B
    zero=BRM._brm_svi_carry_project(X,blocks,reference;rank=0,shrinkage=1.)
    @test isempty(zero.factor.B)
    @test carry_test_covariance(zero.factor)≈Matrix{Float64}(I,5,5)
    # Negative conditional covariance and unregularized sample projection
    # violate the PSD/positive-definiteness requirements.
    @test_throws ArgumentError BRM._brm_svi_carry_project(X,blocks,reference;shrinkage=0.)
    @test_throws ArgumentError BRM._brm_svi_carry_project(X,blocks,reference;
        conditional=[zeros(2,2),-ones(1,1),zeros(2,2)])
end

@testset "fixed auxiliary core, active lift and conditional moment patch" begin
    core=BRM._brm_svi_carry_core([.2,-.1],[1. .2;.2 .7])
    store=Dict(:active=>BRM._brm_svi_carry_block([.1],[.8 .3],reshape([.5],1,1)),
        :absent=>BRM._brm_svi_carry_block([-.2],[-.4 .7],reshape([.6],1,1)),
        :new=>BRM._brm_svi_carry_block([0.],zeros(1,2),ones(1,1)))
    prior=carry_test_augmented(core,store,[:active,:absent,:new])
    active=Dict(:active=>store[:active],:new=>store[:new])
    marginal=BRM._brm_svi_carry_active_marginal(core,active,[:new,:active])
    @test marginal.mu≈prior.mean[[5,3]]
    @test carry_test_covariance(marginal)≈prior.covariance[[5,3],[5,3]]
    draws=Dict(:active=>reshape([-.4,.7,1.2],1,:),:new=>reshape([2.,-1.,.5],1,:))
    lifted=BRM._brm_svi_carry_auxiliary_conditional(core,active,draws)
    C=prior.covariance
    @test lifted.means≈core.m.+(C[1:2,3]/C[3,3])*(draws[:active].-prior.mean[3])
    @test lifted.precision\Matrix{Float64}(I,2,2)≈C[1:2,1:2]-C[1:2,3]*C[3,1:2]'/C[3,3]
    @test BRM._brm_svi_carry_lift(Xoshiro(714),core,active,draws)==
        BRM._brm_svi_carry_lift(Xoshiro(714),core,active,draws)

    # Independent exact Gaussian observation update of the active scalar.
    v=C[:,3];s=C[3,3]+.4
    postmean=prior.mean+v*((.7-prior.mean[3])/s)
    postcov=C-v*v'/s
    selected=[1,2,3,5]
    points=carry_test_points(postmean[selected],postcov[selected,selected])
    fullmeans=Dict(:active=>points[3:3,:],:new=>points[4:4,:])
    # Integrate active scalar conditional on eta; new scalar stays a draw.
    Q=postcov[3:3,1:2]/postcov[1:2,1:2]
    conditional_mean=postmean[3].+Q*(points[1:2,:].-postmean[1:2])
    variance=postcov[3,3]-only(Q*postcov[1:2,3])
    integrated=Dict(:active=>reshape(conditional_mean,1,:),:new=>points[4:4,:])
    for shrinkage in (0.,.01)
        raw=BRM._brm_svi_carry_project_active(core,active,fullmeans,points[1:2,:];
            shrinkage,exact_moments=true)
        rb=BRM._brm_svi_carry_project_active(core,active,integrated,points[1:2,:];
            conditional=Dict(:active=>reshape([variance],1,1)),shrinkage,exact_moments=true)
        @test rb.core.m≈raw.core.m
        @test Matrix(rb.core.F)≈Matrix(raw.core.F)
        for key in keys(active)
            @test rb.touched[key].a≈raw.touched[key].a
            @test rb.touched[key].B≈raw.touched[key].B
            @test Matrix(rb.touched[key].F)≈Matrix(raw.touched[key].F) atol=2e-13
        end
        combined=merge(store,rb.touched)
        @test combined[:absent]===store[:absent]
        if shrinkage==0
            restored=carry_test_augmented(rb.core,combined,[:active,:absent,:new])
            @test restored.mean≈postmean atol=3e-13
            @test restored.covariance≈postcov atol=3e-13
        end
    end
end

@testset "active projection discloses residual cross-block loss" begin
    core=BRM._brm_svi_carry_core([0.],ones(1,1))
    blocks=Dict(:a=>BRM._brm_svi_carry_block([0.],reshape([.6],1,1),ones(1,1)),
        :b=>BRM._brm_svi_carry_block([0.],reshape([-.4],1,1),ones(1,1)))
    prior=carry_test_augmented(core,blocks,[:a,:b])
    direction=[0.,1.,1.];v=prior.covariance*direction
    C=prior.covariance-v*v'/(1+dot(direction,v))
    X=carry_test_points(zeros(3),C)
    patch=BRM._brm_svi_carry_project_active(core,blocks,
        Dict(:a=>X[2:2,:],:b=>X[3:3,:]),X[1:1,:];shrinkage=0.,exact_moments=true)
    actual=carry_test_augmented(patch.core,patch.touched,[:a,:b]).covariance
    residual=C[2,3]-C[2,1]*C[1,3]/C[1,1]
    @test abs(residual)>.1
    @test diag(actual)≈diag(C)
    @test actual[1,:]≈C[1,:]
    @test actual[2,3]-C[2,3]≈-residual
end

@testset "fixed-rank carry storage" begin
    sizes=Int[]
    for n in (128,1024)
        rng=Xoshiro(92)
        c=BRM._brm_svi_carry_gaussian(zeros(n),[[i,i+1] for i in 1:2:n],
            [Matrix{Float64}(I,2,2) for _ in 1:2:n],.1randn(rng,n,3))
        v=randn(rng,n)
        @test BRM._brm_svi_carry_precision(c,v+c.B*(c.B'*v))≈v
        @test size(c.DB)==(n,3)
        @test size(c.R)==(3,3)
        push!(sizes,Base.summarysize(c))
    end
    @test sizes[2]<9sizes[1]
end
