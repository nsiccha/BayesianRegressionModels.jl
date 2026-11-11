# Structural, NO-REFIT diagnostic after the failed joint-latent endpoint.
# Keep the saved marginal q(h) over three hyperparameters. Replace q(beta|h)
# by the exact joint conditional of ALL affine Gaussian-prior coordinates:
# population coefficient, unit draws, and HSGP weights. Select by structure,
# not by observed posterior error. Dense algebra is a hard-bounded oracle.
# The KL split measures the saved conditional's mismatch; its components
# include optimization error, not only unavoidable family restrictions.
# julia --project=test research/shared_field_laplace/gaussian_block_comparison.jl 'fixed hyperparameter Gaussian block'
include("stationarity_audit.jl")

const GAUSSIAN_BLOCK_PAIRS = 8192
const GAUSSIAN_BLOCK_SEED = 1967

function affine_gaussian_block(rk,data,p,view)
    population = findall(==(:population),view.role)
    length(population)==1 || error("bounded control requires one affine Gaussian population coordinate")
    beta = only(population)
    record = only(r for r in BRM._rk_coordinate_records(rk.plan)
        if r.kind===:population && BRM._brm_nlme_owned(view.coordinates[beta],r.declaration))
    plan = BRM._brm_nlme_regression(rk.plan)
    prior = only(r for r in plan.population_priors
        if r.predictor===record.predictor && r.addressee===record.coefficient)
    prior.family===:Normal || error("authored population prior is not Gaussian: $(prior.family)")
    location,scale = prior.args
    isfinite(location) && isfinite(scale) && scale>0 || error("invalid Gaussian prior")
    column = BRM._brm_nlme_population_column(plan,record.predictor,record.coefficient,length(data.y))
    column===nothing && error("population coordinate lacks an affine design certificate")
    latent = sort(vcat(p.latent,population))
    globals = setdiff(eachindex(view.coordinates),latent)
    length(latent)==23 && length(globals)==3 && length(view.coordinates)==26 &&
        length(data.y)<=128 || error("Gaussian-block oracle size/shape cap exceeded")
    p3 = ExactGaussianReference(p.full,copy(p.start),globals,latent)
    (;p3,beta,position=only(findall(==(beta),latent)),column,
        prior=(;family=prior.family,location,scale,predictor=record.predictor,
            coefficient=record.coefficient))
end

function gaussian_global_partition(candidate,p,block)
    k = only(findall(==(block.beta),p.globals))
    hi = findall(!=(k),eachindex(p.globals))
    p.globals[hi]==block.p3.globals || error("hyperparameter coordinate order mismatch")
    covariance = candidate.C*candidate.C'
    H = cholesky(Symmetric(covariance[hi,hi]))
    regression = H\covariance[hi,k]
    variance = covariance[k,k]-dot(covariance[hi,k],regression)
    variance>0 || error("nonpositive q(beta|h) variance")
    (;k,hi,mu=candidate.mu[hi],C=Matrix(H.L),regression,variance,
        beta_mean=candidate.mu[k],covariance)
end

function gaussian_block_point(block,q,h)
    c = gaussian_reference_conditional(block.p3,h)
    covariance = inv(c.F)
    j = block.position
    m,v = c.mu[j],covariance[j,j]
    a = q.beta_mean+dot(q.regression,h-q.mu)
    mean_gap = (a-m)^2/(2v)
    ratio = q.variance/v
    delta = ratio-1
    # Avoid cancellation in delta-log1p(delta) near equality.
    variance_gap = abs(delta)<1e-4 ?
        (delta^2/2-delta^3/3+delta^4/4-delta^5/5+delta^6/6)/2 :
        (delta-log1p(delta))/2
    gap = mean_gap+variance_gap
    loading = covariance[:,j]/v
    old_mean = copy(c.u)
    old_mean[block.p3.latent] = c.mu+loading*(a-m)
    old_variance = diag(covariance)+(q.variance-v)*loading.^2
    minimum(old_variance)>0 || error("nonpositive reconstructed old conditional variance")
    old_second = old_mean.^2
    old_second[block.p3.latent] .+= old_variance
    new_mean = copy(c.u);new_second = new_mean.^2
    new_second[block.p3.latent] .+= diag(covariance)
    entropy_h = 1.5*(1+log(2pi))+sum(log,diag(q.C))
    (;c,covariance,m,v,a,gap,mean_gap,variance_gap,
        old_mean,old_second,new_mean,new_second,
        old_elbo=c.value+entropy_h-gap,new_elbo=c.value+entropy_h)
end

function gaussian_block_control(p,block,q,h,mean_query,data,rng)
    s = gaussian_block_point(block,q,h)
    z = randn(rng,length(block.p3.latent))
    u = copy(s.c.u);u[block.p3.latent] = z
    value,score = LogDensityProblems.logdensity_and_gradient(p.full,u)
    quadratic_error = abs(value-(s.c.f0+dot(s.c.h,z)-dot(z,s.c.H*z)/2))
    score_error = maximum(abs.(score[block.p3.latent]-(s.c.h-s.c.H*z)))
    # Independent observation-space conditional beta posterior. The original
    # 22 unit/field coordinates have independent standard Normal priors.
    u[block.p3.latent] .= 0
    offset = Base.invokelatest(mean_query,u)
    A = hcat(map(p.latent) do i
        u[i]=1
        col = Base.invokelatest(mean_query,u)-offset
        u[i]=0
        col
    end...)
    R = cholesky(Symmetric(.7^2*Matrix{Float64}(I,length(data.y),length(data.y))+A*A'))
    variance = inv(inv(block.prior.scale^2)+dot(block.column,R\block.column))
    mean = variance*(block.prior.location/block.prior.scale^2+
        dot(block.column,R\(data.y-offset)))
    variance_error,mean_error = abs(s.v-variance),abs(s.m-mean)
    # Two beta nodes exactly integrate the quadratic four-global log marginal
    # and its affine conditional latent means / quadratic second moments.
    old_value=0.;old_mean=zeros(length(u));old_second=zeros(length(u))
    for sign in (-1,1)
        g=zeros(length(p.globals));g[q.hi]=h;g[q.k]=s.a+sign*sqrt(q.variance)
        c=gaussian_reference_conditional(p,g)
        second=c.u.^2;second[p.latent].+=diag(inv(c.F))
        old_value+=c.value/2;old_mean.+=c.u/2;old_second.+=second/2
    end
    old_value+=(1.5*(1+log(2pi))+sum(log,diag(q.C)))+
        .5*(1+log(2pi*q.variance))
    identity_error=abs(s.new_elbo-old_value-s.gap)
    moment_error=max(maximum(abs.(old_mean-s.old_mean)),maximum(abs.(old_second-s.old_second)))
    (;quadratic_error,score_error,variance_error,mean_error,identity_error,moment_error)
end

@stestset "fixed hyperparameter Gaussian block" begin
    base=joinpath(@__DIR__, "results")
    candidate=deserialize(joinpath(base,"joint-latent-fit-v1/candidate.jls"))
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    endpoint=deserialize(joinpath(base,"joint-latent-endpoint-v1/audit.jls"))
    reference.reference_ok && candidate.identity.reference_signature==reference.signature &&
        endpoint.identity.candidate_signature==candidate.identity.signature &&
        endpoint.identity.reference_signature==reference.signature || error("reference/candidate identity failed")
    sources=[joinpath(@__DIR__,name) for name in
        ("joint_latent_fit.jl","profile_optimization.jl","stationarity_audit.jl",
         "exact_reference.jl","operator_probe.jl","trace_probe.jl","nonlinear_probe.jl",
         "whole_window_probe.jl","quadrature.jl","joint_svi_probe.jl")]
    append!(sources,[joinpath(@__DIR__,"../../src/nlme_svi.jl"),joinpath(@__DIR__,"../../test/Manifest.toml")])
    bytes2hex(sha256(reduce(vcat,read.(sources))))==candidate.identity.signature ||
        error("candidate source/environment changed")
    directory=joinpath(base,"gaussian-block-comparison-v1")
    ispath(directory) && error("preserve previous Gaussian block comparison")
    mkpath(directory)
    audit_sources=vcat([@__FILE__],sources,
        [joinpath(@__DIR__,"../../src/nlme_view.jl"),joinpath(@__DIR__,"../../src/rk_backend.jl")])
    identity=(;signature=bytes2hex(sha256(reduce(vcat,read.(audit_sources)))),
        repository_revision=readchomp(`git rev-parse HEAD`),
        candidate_signature=candidate.identity.signature,reference_signature=reference.signature,
        seed=GAUSSIAN_BLOCK_SEED,npairs=GAUSSIAN_BLOCK_PAIRS,julia=string(VERSION))
    serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Fixed-hyperparameter Gaussian block comparison") do progress
        snapshot_progress(progress,joinpath(directory,"comparison.progress")) do
            fixture=with_progress(progress;description="Verify authored affine Gaussian partition") do _
                (;rk,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2,prepare_model=false)
                merge((;rk,data),exact_gaussian_reference(rk,data))
            end
            (;rk,data,p,view,field,units)=fixture
            view.coordinates==candidate.coordinates==reference.coordinates || error("coordinate identity failed")
            block=affine_gaussian_block(rk,data,p,view)
            q=gaussian_global_partition(candidate,p,block)
            observations=prepare_query(rk.model,RKEXT._rk_translated(rk),:observations)
            mean_query(u)=reduce(vcat,Base.invokelatest(observations,u).y.location)
            controls=map((zeros(3),[.5,-.4,.6],[-1.,.5,-.3])) do shift
                gaussian_block_control(p,block,q,q.mu+shift,mean_query,data,Xoshiro(1968))
            end
            for result in controls, metric in values(result)
                @test metric<2e-8
            end
            all(r->all(<(2e-8),values(r)),controls) || error("structural Gaussian control failed")
            serialize(joinpath(directory,"controls.jls"),(;identity,controls,prior=block.prior,
                beta=block.beta,latent=block.p3.latent,globals=block.p3.globals,q))
            n=length(p.start)
            pairs=zeros(8,GAUSSIAN_BLOCK_PAIRS) # old/new ELBO, KL, mean/variance KL, m, m², v
            moments=[zeros(2n,GAUSSIAN_BLOCK_PAIRS) for _ in 1:2]
            variance_extrema=[Inf,-Inf];min_gap=Inf
            rng=Xoshiro(GAUSSIAN_BLOCK_SEED)
            with_progress(progress,GAUSSIAN_BLOCK_PAIRS;description="Fixed no-refit conditional expectations") do node
                for j in 1:GAUSSIAN_BLOCK_PAIRS
                    xi=randn(rng,3)
                    for sign in (-1,1)
                        h=q.mu+sign*q.C*xi
                        s=gaussian_block_point(block,q,h)
                        pairs[:,j].+=[s.old_elbo,s.new_elbo,s.gap,s.mean_gap,s.variance_gap,s.m,s.m^2,s.v]/2
                        moments[1][:,j].+=vcat(s.old_mean,s.old_second)/2
                        moments[2][:,j].+=vcat(s.new_mean,s.new_second)/2
                        variance_extrema[1]=min(variance_extrema[1],s.v)
                        variance_extrema[2]=max(variance_extrema[2],s.v)
                        min_gap=min(min_gap,s.gap)
                    end
                    j%128==0 && update_progress!(node,j)
                end
            end
            @test all(isfinite,pairs) && all(x->all(isfinite,x),moments)
            @test min_gap>=-1e-12
            summary=paired_summary(pairs)
            @test summary.average[3]≈sum(summary.average[4:5]) atol=1e-12
            @test summary.average[2]-summary.average[1]≈summary.average[3] atol=1e-10
            batches=[paired_summary(pairs[:,(b-1)*512+1:b*512]) for b in 1:16]
            batch_mcse=vec(std(hcat(getproperty.(batches,:average)...);dims=2))/4
            centered=(pairs.-summary.average).^2
            variance_share=vec(maximum(centered;dims=2))./vec(sum(centered;dims=2))
            reports=map(enumerate(moments)) do (i,points)
                s=paired_summary(points)
                mu=s.average[1:n];variance=s.average[n+1:2n]-mu.^2
                known=i==1 ? p.globals : block.p3.globals
                ki=i==1 ? collect(1:4) : q.hi
                mu[known]=candidate.mu[ki];variance[known]=diag(q.covariance)[ki]
                sd=sqrt.(variance)
                mean_error=abs.(mu-reference.ref_mean)./reference.ref_sd
                sd_ratio=sd./reference.ref_sd
                influence=points[n+1:2n,:].-s.average[n+1:2n].-
                    2mu.*(points[1:n,:].-s.average[1:n])
                sd_mcse=vec(std(influence;dims=2))/sqrt(GAUSSIAN_BLOCK_PAIRS)./(2sd)
                sd_mcse[known].=0
                accuracy_ok=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
                (;mu,sd,mean_error,sd_ratio,sd_mcse,accuracy_ok)
            end
            @test reports[1].mu[block.p3.globals]==reports[2].mu[block.p3.globals]
            @test reports[1].sd[block.p3.globals]==reports[2].sd[block.p3.globals]
            critical=quantile(TDist(GAUSSIAN_BLOCK_PAIRS-1),.975)
            gain_interval=summary.average[3].+(-1,1).*(critical*summary.mcse[3])
            result=(;identity,controls,prior=block.prior,q,summary,batches,batch_mcse,variance_share,
                gain_interval,reports,variance_extrema,min_gap,pairs,moments,field,units,
                beta=block.beta,globals=block.p3.globals,
                exact_conditional_variance_mean=summary.average[8],
                exact_conditional_mean_variance=summary.average[7]-summary.average[6]^2,
                scope="fixed saved q(h); exact conditional block replacement, no fit or stationarity/scaling claim; MCSE excludes reference uncertainty")
            serialize(joinpath(directory,"comparison.jls"),result)
            println((;prior=block.prior,controls))
            println((;elbos=summary.average[1:2],gain=summary.average[3],gain_interval,
                mean_gap=summary.average[4],variance_gap=summary.average[5],
                mcse=summary.mcse[3:5],batch_mcse=batch_mcse[3:5],
                maximum_variance_share=maximum(variance_share[3:5]),
                q_conditional_variance=q.variance,exact_conditional_variance_extrema=variance_extrema))
            for (i,r) in enumerate(reports)
                println((;stage=i==1 ? "saved Gaussian conditional" : "exact Gaussian block conditional",
                    r.accuracy_ok,max_mean_error=maximum(r.mean_error),
                    original_global_sd_ratio=r.sd_ratio[p.globals],
                    unit_sd_extrema=extrema(r.sd_ratio[units]),field_sd_extrema=extrema(r.sd_ratio[field]),
                    maximum_sd_mcse=maximum(r.sd_mcse)))
            end
        end
    end
end
