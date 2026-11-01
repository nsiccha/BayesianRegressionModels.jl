# Small independent quadratic controls for the carried-prior hooks. No native
# AD/model preparation: derivative dispatch uses a known Gaussian likelihood.
using Test, BayesianRegressionModels, LinearAlgebra, Random
const BRM = BayesianRegressionModels
isdefined(BRM,:_BRMSVILaplaceTarget) ||
    Base.include(BRM,joinpath(@__DIR__,"../src/nlme_svi_laplace.jl"))
isdefined(BRM,:_brm_svi_carry_gaussian) ||
    Base.include(BRM,joinpath(@__DIR__,"../src/nlme_svi_laplace_carry.jl"))
isdefined(BRM,:_BRMSVILaplaceCarryPrior) ||
    Base.include(BRM,joinpath(@__DIR__,"../src/nlme_svi_laplace_prior.jl"))

struct CarryPriorQuadraticAD
    A::Matrix{Float64}
    y::Vector{Float64}
end
function BRM._rk_svi_laplace_value_gradient(ad::CarryPriorQuadraticAD,u,part)
    # The sampler prior is intentionally different from the carry. Accidentally
    # adding it would fail the independent carried-density expectation below.
    residual=ad.y-ad.A*u
    value=-(length(residual)*log(2pi)+sum(abs2,residual))/2
    gradient=ad.A'*residual
    part === :likelihood && return value,gradient
    part === :sampler && return value-sum(abs2,u)/2-length(u)*log(2pi)/2,gradient-u
    error("unknown quadratic test query")
end
BRM._rk_svi_laplace_hvp(ad::CarryPriorQuadraticAD,u,v)=-ad.A'*(ad.A*v)

function carry_prior_test_fixture()
    n=6
    # Units [3,4] and [5] are independent given coordinates [1,2,6].
    A=[.2 1. 1. .3 0. .5; -.1 1. -.4 1. 0. .2; -.3 1. 0. 0. 1. -.7]
    ad=CarryPriorQuadraticAD(A,[.4,-.2,.1])
    certificate=(;globals=[1],latent=collect(2:n),field=[6],
        unit_blocks=[[3,4],[5]],prior_mean=[.1,.2,-.1,.3,0.],
        prior_precision=[1.,2.,3.,4.,1.])
    m=(;unit=zeros(n))
    t=BRM._BRMSVILaplaceTarget(m,certificate,ad,nothing,BRM.GaussianBlockLaplace(),
        [1],Vector{Int}[],zeros(Int,5),Dict{Symbol,Float64}())
    blocks=[[1],[2],[3,4],[5],[6]]
    Ds=[reshape([.8],1,1),ones(1,1),[.7 .1;.1 .9],reshape([.6],1,1),reshape([.5],1,1)]
    B=[.4 -.2;.3 .1;-.5 .4;.2 .3;.6 -.1;-.2 .7]
    mu=[.1,-.2,.3,-.1,.2,.4]
    carry=BRM._brm_svi_carry_gaussian(mu,blocks,Ds,B)
    C=B*B'
    for (b,D) in zip(blocks,Ds)
        C[b,b].+=D
    end
    (;t,carry,C,A,blocks,Ds,B,mu)
end

@testset "carried density replaces authored prior and retains mixed curvature" begin
    f=carry_prior_test_fixture()
    t=BRM._brm_svi_laplace_with_carry(f.t,f.carry)
    u=[.3,-.2,.8,-.4,.1,.5]
    value,gradient=BRM._brm_svi_laplace_value_gradient(t,u)
    residual=t.ad.y-f.A*u;Q=inv(Symmetric(f.C));delta=u-f.mu
    expected=-(length(residual)*log(2pi)+sum(abs2,residual)+
        length(u)*log(2pi)+logdet(Symmetric(f.C))+dot(delta,Q*delta))/2
    @test value≈expected
    @test gradient≈f.A'*residual-Q*delta
    v=[.2,-.3,.5,.7,-.1];embedded=vcat(0.,v)
    joint=BRM._brm_svi_laplace_joint_hvp(t.prior,t,u,v)
    @test joint≈-(f.A'*f.A+Q)*embedded
    @test abs((Q*embedded)[1])>1e-4 # omitted mixed prior term is detectably wrong
    @test BRM._brm_svi_laplace_prior_precision(t.prior,t,v)≈Q[2:6,2:6]*v
    @test BRM._brm_svi_laplace_precision(t,u,v)≈(f.A'*f.A+Q)[2:6,2:6]*v
    @test 0<BRM._brm_svi_laplace_lower_bound(t.prior,t)<=minimum(eigvals(Q[2:6,2:6]))*(1+1e-12)
    @test f.t.prior===nothing
    @test all(iszero,f.t.counts)
    # Conditioning across only part of a residual block is unsupported by
    # the conditional spectral-bound representation, and must be explicit.
    bad=merge(f.t.certificate,(;latent=[2,3,5,6],globals=[1,4]))
    @test_throws ArgumentError BRM._brm_svi_laplace_carry_prior(bad,f.carry)
end

@testset "colored likelihood unit moments with initial and conditional priors" begin
    f=carry_prior_test_fixture();u=[.3,-.2,.8,-.4,.1,.5]
    initial=BRM._brm_svi_laplace_unit_moments(f.t,u)
    g=f.A'*(f.t.ad.y-f.A*u)
    for (indices,C) in zip(f.t.certificate.unit_blocks,initial.covariances)
        positions=indices.-1
        precision=Diagonal(f.t.certificate.prior_precision[positions])
        Q=Matrix(precision)+f.A[:,indices]'*f.A[:,indices]
        score=g[indices]-precision*(u[indices]-f.t.certificate.prior_mean[positions])
        @test initial.means[indices]≈u[indices]+Q\score
        @test C≈inv(Symmetric(Q))
    end
    @test initial.means[[1,2,6]]==u[[1,2,6]]
    @test initial.calls==(;likelihood_gradients=1,colored_likelihood_hvps=2)

    t=BRM._brm_svi_laplace_with_carry(f.t,f.carry)
    aux=BRM._brm_svi_carry_auxiliary(f.carry)
    priors=[aux.blocks[3],aux.blocks[4]];eta=[.4,-.3]
    result=BRM._brm_svi_laplace_unit_moments(t,u;priors,eta)
    # Independent bounded augmented-prior precision oracle, in (eta,x) order.
    Dinv=zeros(6,6)
    for (b,D) in zip(f.blocks,f.Ds)
        Dinv[b,b]=inv(Symmetric(D))
    end
    P=[Matrix{Float64}(I,2,2)+f.B'*Dinv*f.B -f.B'*Dinv;-Dinv*f.B Dinv]
    prior_score=-P*(vcat(eta,u)-vcat(zeros(2),f.mu))
    for (indices,C) in zip(t.certificate.unit_blocks,result.covariances)
        aug=indices.+2
        Q=P[aug,aug]+f.A[:,indices]'*f.A[:,indices]
        @test result.means[indices]≈u[indices]+Q\(g[indices]+prior_score[aug])
        @test C≈inv(Symmetric(Q))
    end
    @test result.calls==(;likelihood_gradients=1,colored_likelihood_hvps=2)
    # Marginal carry unit precision has cross-unit entries: coloring that
    # prior would mix columns from distinct units and change the target.
    Q=inv(Symmetric(f.C))
    @test norm(Q[[3,4],[5]])>1e-3
    @test_throws ArgumentError BRM._brm_svi_laplace_unit_moments(t,u)
    @test_throws ArgumentError BRM._brm_svi_laplace_unit_moments(t,u;priors)
end
