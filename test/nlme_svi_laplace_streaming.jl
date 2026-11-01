# Public frozen replay with fitted carry: bounded updates test transport and
# target construction, not posterior accuracy or optimizer convergence.
using Test, BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, DifferentiationInterface, LinearAlgebra, Random, Serialization
const BRM=BayesianRegressionModels
include("nlme_svi_laplace_streaming_fixture.jl")

laplace_same_block(a,b)=a.a==b.a && a.B==b.B && a.F.L==b.F.L

# Independent dense arithmetic is confined to this <=32-coordinate fixture.
# Build the carried prior directly from its auxiliary conditional blocks,
# then integrate the native likelihood's Gaussian latent slice by unit probes.
function laplace_window_exact(t,u,prior)
    length(u)<=32 || error("test-only dense carried Gaussian oracle cap")
    c=t.certificate;point=copy(u);point[c.latent].=0
    value,gradient=BRM._rk_svi_laplace_value_gradient(t.ad,point,:likelihood)
    Q=inv(cov(prior));h=gradient[c.latent]-(Q*(point-mean(prior)))[c.latent]
    H=copy(Q[c.latent,c.latent])
    for (j,i) in enumerate(c.latent)
        point[i]=1
        _,score=BRM._rk_svi_laplace_value_gradient(t.ad,point,:likelihood)
        H[:,j].+=gradient[c.latent]-score[c.latent]
        point[i]=0
    end
    @test H≈H' atol=1e-9
    F=cholesky(Symmetric(H));mu=F\h
    (;mu,H,value=value+logpdf(prior,point)+dot(h,mu)/2+
        length(c.latent)*log(2pi)/2-logdet(F)/2)
end

function laplace_window_target_check(fit,previous)
    m=fit.diagnostics.model
    t=BRM._brm_svi_laplace_target(m,fit.diagnostics.laplace.method)
    layout=BRM._brm_svi_laplace_layout(t)
    active=BRM._brm_svi_laplace_active(t,previous,layout)
    length(m.unit)<=32 || error("test-only dense carried covariance cap")
    a=zeros(length(m.unit));B=zeros(length(m.unit),length(previous.core.m))
    C=zeros(length(m.unit),length(m.unit))
    for (key,indices) in zip(layout.order,layout.blocks)
        if haskey(previous.blocks,key)
            block=previous.blocks[key]
            # This fixture has one coordinate per unit, so no semantic
            # within-unit permutation is required in the independent oracle.
            a[indices]=block.a;B[indices,:]=block.B
            C[indices,indices]=Matrix(block.F.L*block.F.L')
        else
            @test first(key)===:unit
            C[indices,indices]=Matrix{Float64}(I,length(indices),length(indices))
        end
    end
    C+=B*Matrix(previous.core.F.L*previous.core.F.L')*B'
    prior=MvNormal(a+B*previous.core.m,Symmetric(C))
    @test active.carry.mu≈mean(prior) atol=1e-12
    for i in eachindex(a)
        e=Float64.(eachindex(a).==i)
        @test BRM._brm_svi_carry_covariance(active.carry,e)≈C*e atol=1e-11
    end
    carried=BRM._brm_svi_laplace_with_carry(t,active.carry)
    point=copy(m.unit);point[t.globals]=fit.diagnostics.q.μ
    exact=laplace_window_exact(carried,point,prior)
    state=BRM._brm_svi_laplace_mode(carried,point)
    @test state.mode[t.certificate.latent]≈exact.mu atol=2e-8
    @test eigmin(Symmetric(exact.H))>=carried.prior.lower_bound-1e-9
    @test fit.diagnostics.laplace.spectral_lower_bound≈carried.prior.lower_bound
    direction=randn(Xoshiro(734),length(t.certificate.latent))
    @test BRM._brm_svi_laplace_precision(carried,state.mode,direction)≈exact.H*direction atol=3e-8
    root=cholesky(Symmetric(exact.H)).U\Matrix{Float64}(I,length(direction),length(direction))
    trace=sum(BRM._brm_svi_laplace_third(carried,state.mode,w) for w in eachcol(root))
    gradient=state.gradient[t.globals]+trace[t.globals]/2
    epsilon=1e-5
    difference=map(t.globals) do i
        plus,minus=copy(point),copy(point);plus[i]+=epsilon;minus[i]-=epsilon
        (laplace_window_exact(carried,plus,prior).value-
            laplace_window_exact(carried,minus,prior).value)/(2epsilon)
    end
    @test gradient≈difference atol=5e-6 rtol=5e-6
end

@testset "GaussianBlockLaplace frozen windows and retained units" begin
    builder=laplace_window_builder()
    method=GaussianBlockLaplace(averaging_window=3,carry_draws=12,carry_rank=3)
    a=brm_svi_fit(builder(laplace_window_data([1,2,3]));laplace=method,
        ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(730),
        family=LowRankGlobal(2),max_iterations=4,ndraws=12)
    first=brm_svi_checkpoint(a)
    # Check the plain checkpoint round trip here, then resume that state in
    # a fresh Julia process after comparing the native carried target.
    io=IOBuffer();serialize(io,first);seekstart(io);state=deserialize(io)
    @test state.core.m==first.core.m && state.core.F.L==first.core.F.L
    @test keys(state.blocks)==keys(first.blocks)
    @test all(laplace_same_block(state.blocks[k],first.blocks[k]) for k in keys(first.blocks))
    @test state.source_id==first.source_id
    @test Set(keys(state.units))==Set([1,2,3])
    b=brm_svi_fit(state,builder(laplace_window_data([3,4];shift=.2));
        ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(731),
        max_iterations=4,ndraws=12)
    second=brm_svi_checkpoint(b)
    @test second.updates==1
    @test laplace_same_block(second.blocks[(:unit,1)],first.blocks[(:unit,1)])
    @test laplace_same_block(second.blocks[(:unit,2)],first.blocks[(:unit,2)])
    @test Set(keys(second.units))==Set([1,2,3,4])
    @test b.diagnostics.laplace.scope.streaming
    @test b.diagnostics.laplace.fields==a.diagnostics.laplace.fields
    @test second.shared_coordinates==first.shared_coordinates
    @test second.projection.numerical_reference===:previous_auxiliary_core_and_local_residual_covariances
    @test !b.diagnostics.converged
    laplace_window_target_check(b,first)
    mktempdir() do dir
        input,output=joinpath(dir,"input.jls"),joinpath(dir,"output.jls")
        serialize(input,(;state,data=laplace_window_data([3,4];shift=.2)))
        script=joinpath(@__DIR__,"nlme_svi_laplace_streaming_process.jl")
        run(`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $script $input $output`)
        @test deserialize(output)==laplace_window_summary(b)
    end
    c=brm_svi_fit(second,builder(laplace_window_data([4,1];shift=-.2));
        ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),rng=Xoshiro(732),
        max_iterations=2,ndraws=3)
    third=brm_svi_checkpoint(c)
    @test third.updates==2
    @test third.units[4].locations==second.units[4].locations
    @test third.units[1].locations==first.units[1].locations
    @test laplace_same_block(third.blocks[(:unit,2)],second.blocks[(:unit,2)])
    @test laplace_same_block(third.blocks[(:unit,3)],second.blocks[(:unit,3)])
    @test third.source_id==first.source_id
    @test all(isfinite,c.draws)
    laplace_window_target_check(c,second)
    @test_throws ArgumentError brm_svi_fit(second,builder(laplace_window_data([3,4]));
        laplace=GaussianBlockLaplace(carry_rank=2),max_iterations=1,ndraws=0)
    @test_throws ArgumentError brm_svi_fit(second,builder(laplace_window_data([4,1];move_new=true));
        ad_backend=AutoEnzyme(;mode=Enzyme.Reverse),max_iterations=1,ndraws=0)
    @test_throws ArgumentError brm_svi_fit(second,laplace_window_data([4,1]);max_iterations=1)
end
