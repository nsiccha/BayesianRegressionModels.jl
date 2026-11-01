# Conditional moments in a fixed auxiliary-core projection. Research helper.
# For r(h,z)=q(h)*p_new(z|h), lifting eta|h,z gives
# r(h,z,eta)=[q(h)/p_new(h)]*p_new(h,z,eta). Conditional unit integration
# given h,z_without_units,eta therefore preserves this ideal fitted law.
# This does not hold for arbitrary independent VI unit factors. Native
# finite mode/root solves introduce their separately measured numerical error.
include("lazy_core_probe.jl")

function auxiliary_project_moments(core,blocks,means,H,conditional;delta=.01)
    Set(keys(conditional))<=Set(keys(blocks)) || error("conditional block is not active")
    for (key,D) in conditional
        n=length(blocks[key].a)
        size(D)==(n,n) || throw(DimensionMismatch("conditional covariance"))
        all(isfinite,D) && issymmetric(D) && eigmin(Symmetric(D))>=0 ||
            error("conditional covariance must be finite symmetric PSD")
    end
    patch=auxiliary_project(core,blocks,means,H;delta)
    touched=copy(patch.touched)
    for (key,D) in conditional
        b=touched[key]
        touched[key]=auxiliary_block(b.a,b.B,Matrix(b.F.L*b.F.L')+(1-delta)*D)
    end
    (;core=patch.core,touched,metadata=(;patch.metadata...,
        projection=:fixed_core_conditional_moments,integrated_blocks=sort!(collect(keys(conditional)))))
end

# These two routines are bounded dense mathematical oracles, never fit code.
function control_reweight_marginal(mean,C,h,newmean,newcov)
    length(mean)<=16 || error("reweight oracle cap")
    rest=setdiff(eachindex(mean),h)
    A=C[rest,h]/C[h,h]
    D=C[rest,rest]-A*C[h,rest]
    outmean=copy(mean);outmean[h]=newmean
    outmean[rest]=mean[rest]+A*(newmean-mean[h])
    outcov=copy(C);outcov[h,h]=newcov
    outcov[rest,h]=A*newcov;outcov[h,rest]=outcov[rest,h]'
    outcov[rest,rest]=D+A*newcov*A'
    (;mean=outmean,covariance=outcov)
end
function control_conditional_moments(mean,C,U,points)
    length(mean)<=16 || error("conditional oracle cap")
    K=setdiff(eachindex(mean),U)
    A=C[U,K]/C[K,K]
    D=Matrix(Symmetric(C[U,U]-A*C[K,U]))
    means=copy(points);means[U,:]=mean[U].+A*(points[K,:].-mean[K])
    (;means,D)
end

@testset "conditional moments preserve collapsed law and fixed core" begin
    core=auxiliary_core([.2,-.1],[1. .2;.2 .7])
    store=Dict(:h=>auxiliary_block([.1],[.3 -.2],reshape([.5],1,1)),
        :field=>auxiliary_block([.2],[.8 .3],reshape([.4],1,1)),
        :u=>auxiliary_block([-.2],[-.4 .7],reshape([.6],1,1)),
        :new=>auxiliary_block([0.],zeros(1,2),ones(1,1)),
        :absent=>auxiliary_block([.3],[.5 -.6],reshape([.4],1,1)))
    order=[:h,:field,:u,:new,:absent]
    prior=auxiliary_dense(core,store,order)
    # Independent unit observations conditional on the shared physical field.
    updated=exact_linear_update(prior.mean,prior.covariance,[0.,0.,.2,1.,1.,0.,0.],.7,.4)
    updated=exact_linear_update(updated.mean,updated.covariance,[0.,0.,-.1,1.,0.,1.,0.],-.3,.6)
    # Deliberately imperfect q(h), but exact joint conditional for all others.
    h=[3];q=control_reweight_marginal(updated.mean,updated.covariance,h,
        updated.mean[h].+[.4],1.3updated.covariance[h,h])
    points=exact_moment_points(q.mean,q.covariance)
    U=[5,6];conditional=control_conditional_moments(updated.mean,updated.covariance,U,points)
    Y=conditional.means;D=conditional.D
    @test abs(D[1,2])<1e-13
    mu=vec(mean(Y;dims=2));A=(Y.-mu)/sqrt(size(Y,2))
    C=A*A';C[U,U].+=D
    @test mu≈q.mean atol=3e-13
    @test C≈q.covariance atol=3e-13
    @test norm(q.mean-updated.mean)>.4

    # Exact old-core conditional lift of the reweighted active posterior.
    active=[:h,:field,:u,:new]
    draws=Dict(key=>points[prior.r.+prior.positions[key],:] for key in active)
    lifted=auxiliary_conditional(core,Dict(k=>store[k] for k in active),draws)
    # Absent coordinates are marginalized, not observed in the core lift.
    exact_core=control_conditional_moments(q.mean[1:6],q.covariance[1:6,1:6],[1,2],points[1:6,:])
    @test lifted.means≈exact_core.means[1:2,:] atol=3e-13
    @test lifted.precision\Matrix{Float64}(I,2,2)≈exact_core.D atol=3e-13
    # The new block has zero loading and adds no information to the old core.
    without=filter(!=(:new),active)
    oldonly=auxiliary_conditional(core,Dict(k=>store[k] for k in without),
        Dict(k=>draws[k] for k in without))
    @test lifted.means≈oldonly.means atol=3e-13
    @test Matrix(lifted.precision)≈Matrix(oldonly.precision) atol=3e-13

    integrated=Dict(:u=>reshape([D[1,1]],1,1),:new=>reshape([D[2,2]],1,1))
    mean_draws=Dict(key=>Y[prior.r.+prior.positions[key],:] for key in active)
    active_store=Dict(k=>store[k] for k in active)
    for delta in (0.,.01)
        patch=auxiliary_project_moments(core,active_store,mean_draws,Y[1:2,:],integrated;delta)
        # Exact joint moment nodes provide an independent projection input.
        sampled=auxiliary_project(core,active_store,draws,points[1:2,:];delta)
        @test patch.core.m≈sampled.core.m atol=2e-13
        @test Matrix(patch.core.F)≈Matrix(sampled.core.F) atol=2e-13
        for key in active
            a,b=patch.touched[key],sampled.touched[key]
            @test a.a≈b.a atol=3e-13
            @test a.B≈b.B atol=3e-13
            @test Matrix(a.F)≈Matrix(b.F) atol=3e-13
        end
        combined=merge(copy(store),patch.touched)
        @test combined[:absent]===store[:absent]
        actual=auxiliary_dense(patch.core,combined,order)
        if delta==0
            @test actual.mean≈q.mean atol=3e-13
            @test diag(actual.covariance)≈diag(q.covariance) atol=3e-13
            @test actual.covariance[1:2,:]≈q.covariance[1:2,:] atol=3e-13
            @test actual.covariance[:,end]≈q.covariance[:,end] atol=3e-13
            @test abs(actual.covariance[5,6]-q.covariance[5,6])>1e-3
            println((;lost_unit_cross_covariance=actual.covariance[5,6]-q.covariance[5,6],
                scope="projection drops conditional dependence; local integration itself is invariant"))
        end
    end

    # Arbitrary VI factorization does not inherit the h-only ratio proof.
    perturbed_mean=copy(q.mean);perturbed_mean[5]+=.5
    badpoints=exact_moment_points(perturbed_mean,q.covariance)
    refreshed=control_conditional_moments(updated.mean,updated.covariance,U,badpoints)
    @test norm(vec(mean(refreshed.means;dims=2))-perturbed_mean)>.49
    @test_throws ErrorException auxiliary_project_moments(core,active_store,mean_draws,Y[1:2,:],
        Dict(:absent=>ones(1,1)))
    @test_throws ErrorException auxiliary_project_moments(core,active_store,mean_draws,Y[1:2,:],
        Dict(:u=>-ones(1,1)))
end
