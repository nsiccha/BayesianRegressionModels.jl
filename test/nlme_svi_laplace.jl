# Structural refusals and a bounded independent Gaussian marginal oracle.
include("nlme_svi_laplace_plan.jl")
using Enzyme, DifferentiationInterface, LinearAlgebra, Random, LogDensityProblems

function laplace_test_conditional(full,u,latent)
    length(u)<=32 || error("test-only dense Gaussian oracle cap")
    point=copy(u);point[latent].=0
    f,g=LogDensityProblems.logdensity_and_gradient(full,point)
    h=g[latent];H=zeros(length(latent),length(latent))
    for (k,j) in enumerate(latent)
        point[j]=1
        _,score=LogDensityProblems.logdensity_and_gradient(full,point)
        H[:,k]=h-score[latent];point[j]=0
    end
    @test H≈H' atol=1e-10
    F=cholesky(Symmetric(H));mu=F\h
    (;H,F,mu,value=f+dot(h,mu)/2+length(latent)*log(2pi)/2-logdet(F)/2)
end

@testset "GaussianBlockLaplace bounded numerical kernels" begin
    @test_throws ArgumentError GaussianBlockLaplace(trace_probes=0)
    @test_throws ArgumentError GaussianBlockLaplace(logdet_probes=1)
    @test_throws ArgumentError GaussianBlockLaplace(mode_tolerance=NaN)
    rng=Xoshiro(721);A=randn(rng,17,17);H=.4I+A*A'
    E=eigen(Symmetric(H));logH=E.vectors*Diagonal(log.(E.values))*E.vectors'
    for r in (randn(rng,17),ones(17),zeros(17))
        q=BRM._brm_svi_laplace_logquadratic(v->H*v,r;spectral_lower=.4,maxdim=32)
        @test q.converged
        @test q.lower-1e-10<=dot(r,logH*r)<=q.upper+1e-10
    end
    capped=BRM._brm_svi_laplace_logquadratic(v->H*v,ones(17);
        spectral_lower=.4,maxdim=1,rtol=0.,atol=0.)
    @test !capped.converged
    @test_throws ErrorException BRM._brm_svi_laplace_logdet(Xoshiro(1),v->H*v,17;
        probes=2,spectral_lower=.4,maxdim=1,rtol=0.,atol=0.)
    # A curvature the certificate rules out is refused, never given a log.
    @test_throws ArgumentError BRM._brm_svi_laplace_logquadratic(
        v->Diagonal([-1.,2.,3.])*v,ones(3);spectral_lower=.1)
    @test_throws ArgumentError BRM._brm_svi_laplace_logquadratic(
        v->Diagonal([.5,2.,3.])*v,ones(3);spectral_lower=1.)
    # An unconverged step reads the same two rules as the textbook construction.
    r=ones(17);a=.2;mass=sum(abs2,r)
    it=BRM.KrylovKit.LanczosIterator(v->H*v,r,BRM.KrylovKit.ModifiedGramSchmidt2())
    fact=BRM.KrylovKit.initialize(it)
    for k in 1:5
        k>1 && BRM.KrylovKit.expand!(it,fact)
        T=BRM.KrylovKit.rayleighquotient(fact);beta=BRM.KrylovKit.normres(fact)
        shifted=SymTridiagonal(T.dv.-a,copy(T.ev));rhs=zeros(k);rhs[end]=1.
        radau=SymTridiagonal(vcat(T.dv,a+beta^2*(shifted\rhs)[end]),vcat(T.ev,beta))
        E,R=eigen(T),eigen(radau)
        @test minimum(R.values)≈a atol=1e-10
        q=BRM._brm_svi_laplace_logquadratic(v->H*v,r;spectral_lower=2a,maxdim=k,rtol=0.,atol=0.)
        @test q.upper≈mass*dot(abs2.(E.vectors[1,:]),log.(E.values)) atol=1e-10
        @test q.lower≈mass*dot(abs2.(R.vectors[1,:]),log.(R.values)) atol=1e-10
    end
    # Two spectra on which independent rounding of the two rules crossed them. The
    # shape of nearly exchangeable groups: an unobserved block at the bound, then
    # near-duplicates. And a rank-deficient likelihood with a far larger observed
    # direction beside that block.
    clustered(top)=rng->begin
        values=vcat(fill(1.,6),1 .+ [.0017,.0025,.003].*exp.(.3 .*randn(rng,3)),
            3.92 .+ .007 .*randn(rng,2),top*exp(.1*randn(rng)))
        Q=Matrix(qr(randn(rng,12,12)).Q)
        Symmetric(Q*Diagonal(values)*Q')
    end
    wide(d,rank,bound,spread)=rng->begin
        J=randn(rng,rank,d).*exp.(spread.*randn(rng,rank))
        Symmetric(bound*I+J'J)
    end
    spectra=((1.,1000,clustered(22.3)),(1.,1000,clustered(500.)),(1.,300,wide(13,9,1.,3.)),
        (1e-4,300,wide(13,9,1e-4,2.)),(1e-3,300,wide(25,9,1e-3,3.)),(1e-5,300,wide(64,20,1e-5,3.)))
    @testset "the bounds stay ordered on spectrum $case" for case in eachindex(spectra)
        bound,trials,spectrum=spectra[case]
        rng=Xoshiro(case);failures=0;excess=-Inf;used=0
        for _ in 1:trials
            A=spectrum(rng);d=size(A,1);v=ifelse.(rand(rng,Bool,d),1.,-1.)
            E=eigen(A)
            # Past this, double precision cannot hold A: even the reference fails.
            last(E.values)/bound<1e14 || continue
            used+=1
            exact=dot(abs2.(E.vectors'v),log.(E.values))
            q=try
                BRM._brm_svi_laplace_logquadratic(w->A*w,v;spectral_lower=bound,
                    maxdim=min(d,32),rtol=1e-8,atol=1e-8)
            catch e
                e isa ErrorException || rethrow()
                failures+=1;continue
            end
            # The bounds are exact-arithmetic ones: T's entries and the dense
            # reference both still carry about eps*cond per unit of mass.
            noise=2e-8+64eps()*d*last(E.values)/bound
            excess=max(excess,abs(q.value-exact)-noise-(q.upper-q.lower))
        end
        @test used>=trials/2
        @test failures==0
        @test excess<=0
    end
end

@testset "GaussianBlockLaplace native conditional target" begin
    fixture=gaussian_certificate_fixture(population=:(effect(mu, Intercept) ~ Normal(.4,1.5)))
    m=brm_nlme_model(fixture.backend;ad_backend=AutoEnzyme(;mode=Enzyme.Reverse))
    t=BRM._brm_svi_laplace_target(m,GaussianBlockLaplace())
    cert=t.certificate
    @test minimum(cert.prior_precision)≈inv(1.5^2)
    start=first(BRM._brm_nlme_prior_median_point(m))
    full=rk_logdensity_problem(m.backend;ad_backend=AutoEnzyme(;mode=Enzyme.Reverse))
    for shift in (0.,.15)
        point=copy(start);point[t.globals].+=shift
        exact=laplace_test_conditional(full,point,cert.latent)
        state=BRM._brm_svi_laplace_mode(t,point)
        @test state.mode[cert.latent]≈exact.mu atol=2e-8
        @test eigmin(Symmetric(exact.H))>=minimum(cert.prior_precision)-1e-10
        v=randn(Xoshiro(722),length(cert.latent))
        @test BRM._brm_svi_laplace_precision(t,state.mode,v)≈exact.H*v atol=2e-9
        # Complete orthogonal trace and logdet controls are bounded to this
        # <=32-coordinate test; the fit uses fixed-count random probes.
        root=exact.F.U\Matrix{Float64}(I,length(v),length(v))
        trace=sum(BRM._brm_svi_laplace_third(t,state.mode,w) for w in eachcol(root))
        @test norm(trace[cert.latent])<1e-8
        gradient=state.gradient[t.globals]+trace[t.globals]/2
        eps=1e-5
        difference=map(t.globals) do i
            plus,minus=copy(point),copy(point);plus[i]+=eps;minus[i]-=eps
            (laplace_test_conditional(full,plus,cert.latent).value-
                laplace_test_conditional(full,minus,cert.latent).value)/(2eps)
        end
        @test gradient≈difference atol=3e-6 rtol=3e-6
        determinant=sum(1:length(v)) do j
            e=Float64.((1:length(v)).==j)
            BRM._brm_svi_laplace_logquadratic(w->BRM._brm_svi_laplace_precision(t,state.mode,w),e;
                spectral_lower=minimum(cert.prior_precision),maxdim=32).value
        end
        @test state.value+length(v)*log(2pi)/2-determinant/2≈exact.value atol=2e-7
    end
    # Reproducibility of actual public fitting, including fixed-budget status
    # and restored Gaussian latents. Fresh-process coverage is separate.
    options=(;laplace=GaussianBlockLaplace(averaging_window=3,carry_draws=12,carry_rank=3),family=LowRankGlobal(2),
        max_iterations=4,ndraws=3)
    a=brm_svi_fit(m;rng=Xoshiro(723),options...)
    b=brm_svi_fit(m;rng=Xoshiro(723),options...)
    @test a.draws==b.draws
    @test a.diagnostics.elbo==b.diagnostics.elbo
    @test a.diagnostics.q==b.diagnostics.q
    @test a.diagnostics.iterations==4
    @test a.diagnostics.averaging_window==2:4
    @test !a.diagnostics.converged && !a.diagnostics.convergence_assessed
    @test a.diagnostics.stop_reason===:fixed_update_budget
    @test a.diagnostics.laplace.scope.window===:whole
    @test a.diagnostics.laplace.authored_prior_precision==cert.prior_precision
    @test a.diagnostics.laplace.spectral_lower_bound==minimum(cert.prior_precision)
    @test size(a.draws)==(length(m.unit),3) && all(isfinite,a.draws)
    @test a.coordinates==m.view.coordinates
    @test_throws ArgumentError brm_svi_fit(m;laplace=GaussianBlockLaplace(),max_iterations=0)
    @test_throws ArgumentError brm_svi_fit(m;laplace=GaussianBlockLaplace(),batch_size=1)
    checkpoint=brm_svi_checkpoint(a)
    @test checkpoint isa BRMSVILaplaceState
    @test checkpoint.projection.draw_count==12
    @test checkpoint.updates==0
    @test checkpoint.source_id==a.diagnostics.source_id
    @test checkpoint.globals==a.coordinates[a.diagnostics.globals]
end
