using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, DifferentiationInterface, Random, LinearAlgebra
const BRM=BayesianRegressionModels
include("nlme_svi_laplace_streaming_fixture.jl")

function laplace_identity_same_carry(a,b)
    a.core.m==b.core.m && a.core.F.L==b.core.F.L && a.q==b.q &&
        a.units==b.units && a.projection==b.projection && keys(a.blocks)==keys(b.blocks) &&
        all(a.blocks[k].a==b.blocks[k].a && a.blocks[k].B==b.blocks[k].B &&
            a.blocks[k].F.L==b.blocks[k].F.L for k in keys(a.blocks))
end

@testset "GaussianBlockLaplace output draw count and absent marginals" begin
    builder=laplace_window_builder()
    method=GaussianBlockLaplace(averaging_window=3,carry_draws=12,carry_rank=3)
    first=brm_svi_fit(builder(laplace_window_data([1,2,3]));laplace=method,
        ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(740),
        family=LowRankGlobal(2),max_iterations=4,ndraws=0)
    previous=brm_svi_checkpoint(first);saved=deepcopy(previous)
    function update(ndraws)
        brm_svi_fit(previous,builder(laplace_window_data([3,4];shift=.7));
            ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(741),
            max_iterations=4,ndraws)
    end
    short=update(3);long=update(25)
    a,b=brm_svi_checkpoint(short),brm_svi_checkpoint(long)
    @test short.draws==long.draws[:,1:3]
    @test short.diagnostics.elbo==long.diagnostics.elbo
    @test laplace_identity_same_carry(a,b)
    @test laplace_identity_same_carry(previous,saved)
    key=(:unit,1);block=a.blocks[key];old=previous.blocks[key]
    @test block.a==old.a && block.B==old.B && block.F.L==old.F.L
    marginal=BRM._brm_svi_carry_active_marginal(a.core,Dict(key=>block),[key])
    expected_mean=block.a+block.B*a.core.m
    expected_covariance=Matrix(block.F.L*block.F.L')+
        block.B*Matrix(a.core.F.L*a.core.F.L')*block.B'
    @test marginal.mu≈expected_mean atol=1e-12
    @test BRM._brm_svi_carry_covariance(marginal,ones(1))≈vec(expected_covariance) atol=1e-12
    old_mean=old.a+old.B*previous.core.m
    old_covariance=Matrix(old.F.L*old.F.L')+old.B*Matrix(previous.core.F.L*previous.core.F.L')*old.B'
    @test norm(expected_mean-old_mean)+norm(expected_covariance-old_covariance)>1e-10
end

@testset "GaussianBlockLaplace semantic within-unit permutation" begin
    builder=BRM.brm(quote
        mu ~ 1 + (1 + time | probe) + hsgp(time;k=2,by=probe,
            cor=multigroup((sx,sy),2.;k=2))
        @plate for p in groups(probe)
            y[p] ~ Normal(mu[p],.7)
        end
    end;mod=@__MODULE__)
    fit=brm_svi_fit(builder(laplace_window_data([1,2,3]));
        laplace=GaussianBlockLaplace(averaging_window=2,carry_draws=8,carry_rank=2),
        ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(742),
        max_iterations=2,ndraws=0)
    state=brm_svi_checkpoint(fit);permuted=deepcopy(state)
    for label in keys(permuted.units)
        key=(:unit,label);block=permuted.blocks[key];unit=permuted.units[label]
        @test length(unit.keys)==2
        p=[2,1]
        permuted.units[label]=merge(unit,(;keys=unit.keys[p]))
        permuted.blocks[key]=BRM._brm_svi_carry_block(block.a[p],block.B[p,:],
            Matrix(block.F.L*block.F.L')[p,p])
    end
    t=BRM._brm_svi_laplace_target(fit.diagnostics.model,state.method)
    layout=BRM._brm_svi_laplace_layout(t)
    original=BRM._brm_svi_laplace_active(t,state,layout).carry
    reordered=BRM._brm_svi_laplace_active(t,permuted,layout).carry
    @test original.mu≈reordered.mu atol=1e-12
    for i in eachindex(original.mu)
        e=Float64.(eachindex(original.mu).==i)
        @test BRM._brm_svi_carry_covariance(original,e)≈
            BRM._brm_svi_carry_covariance(reordered,e) atol=1e-11
    end
end
