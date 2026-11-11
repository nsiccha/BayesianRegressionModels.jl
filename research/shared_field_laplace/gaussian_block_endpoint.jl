# One independent, predeclared audit of the saved native three-global fit.
# Fixed32768 antithetic pairs/seed1969; exact23-latent expectations, no refit,
# reference resampling, or changed posterior tolerances. Dense oracle only.
# julia --project=test research/shared_field_laplace/gaussian_block_endpoint.jl 'native Gaussian block endpoint'
include("gaussian_block_fit.jl")

const GAUSSIAN_ENDPOINT_PAIRS=32768
const GAUSSIAN_ENDPOINT_SEED=1969

function gaussian_endpoint_point(p,mu,C,xi)
    h=mu+C*xi
    c=gaussian_reference_conditional(p,h)
    d=length(p.latent);D=length(p.globals)
    d==23 && D==3 || error("bounded Gaussian-block endpoint shape required")
    root=c.F.U\Matrix{Float64}(I,d,d)
    score=zeros(D);u=copy(c.u)
    for j in 1:d,sign in (-1,1)
        u[p.latent]=c.mu+sign*sqrt(d)*root[:,j]
        _,w=LogDensityProblems.logdensity_and_gradient(p.full,u)
        score.+=w[p.globals]/(2d)
    end
    white=C'*score
    covariance=(white*xi'+xi*white')/2+I
    residual=vcat(white,[covariance[i,j] for j in 1:D for i in 1:j],
        1 .+ score.*(h-mu))
    mean=copy(c.u);second=mean.^2
    second[p.latent].+=vec(sum(abs2,root;dims=2))
    elbo=c.value+D/2*(1+log(2pi))+sum(log,diag(C))
    (;score,residual,mean,second,elbo)
end

@stestset "native Gaussian block endpoint" begin
    base=joinpath(@__DIR__, "results")
    saved=deserialize(joinpath(base,"native-gaussian-block-fit-v1/fit.jls"))
    native_comparison=deserialize(joinpath(base,"native-gaussian-block-fit-v1/comparison.jls"))
    reference=deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    baseline=deserialize(joinpath(base,"gaussian-block-comparison-v1/comparison.jls"))
    reference.reference_ok && saved.identity==native_comparison.identity &&
        saved.identity.reference_signature==baseline.identity.reference_signature==reference.signature ||
        error("native endpoint reference/fit identity failed")
    sources=[joinpath(@__DIR__,name) for name in
        ("gaussian_block_fit.jl","gaussian_block_comparison.jl","stationarity_audit.jl",
         "exact_reference.jl","operator_probe.jl","whole_window_probe.jl",
         "joint_svi_probe.jl","nonlinear_probe.jl","trace_probe.jl","quadrature.jl")]
    append!(sources,[joinpath(@__DIR__,"../../src",name) for name in
        ("nlme_view.jl","nlme_svi.jl","rk_backend.jl")])
    push!(sources,joinpath(@__DIR__,"../../test/Manifest.toml"))
    bytes2hex(sha256(reduce(vcat,read.(sources))))==saved.identity.signature ||
        error("native fit source/environment changed")
    directory=joinpath(base,"native-gaussian-block-endpoint-v1")
    ispath(directory) && error("preserve native Gaussian endpoint artifacts")
    mkpath(directory)
    identity=(;signature=bytes2hex(sha256(reduce(vcat,read.(vcat([@__FILE__],sources))))),
        fit_signature=saved.identity.signature,reference_signature=reference.signature,
        baseline_signature=baseline.identity.signature,
        seed=GAUSSIAN_ENDPOINT_SEED,npairs=GAUSSIAN_ENDPOINT_PAIRS)
    serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Independent native Gaussian block endpoint") do progress
        snapshot_progress(progress,joinpath(directory,"endpoint.progress")) do
            fixture=with_progress(progress;description="Prepare bounded exact conditional score oracle") do _
                (;rk,data)=shared_field_fixture(;groups=6,rows=12,temporal=4,spatial=2,prepare_model=false)
                merge((;rk,data),exact_gaussian_reference(rk,data))
            end
            (;rk,data,p,view,field,units)=fixture
            view.coordinates==saved.coordinates==reference.coordinates || error("endpoint coordinate identity failed")
            block=affine_gaussian_block(rk,data,p,view)
            block.p3.globals==saved.globals && block.p3.latent==saved.latent ||
                error("endpoint Gaussian block identity failed")
            baseline.globals==block.p3.globals && baseline.beta==block.beta ||
                error("fixed-hyperparameter baseline partition differs")
            q=saved.fit.q;mu=copy(q.μ)
            C=Matrix(cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(q)))).L)
            xi=[.3,-.5,.7]
            check=gaussian_endpoint_point(block.p3,mu,C,xi)
            _,score=LogDensityProblems.logdensity_and_gradient(block.p3,mu+C*xi)
            @test check.score≈score atol=2e-11
            maximum(abs.(check.score-score))<=2e-11 || error("independent endpoint score validation failed")
            labels=vcat(["whitened_mean_$i" for i in 1:3],
                ["covariance_$(i)_$j" for j in 1:3 for i in 1:j],
                ["row_scale_$i" for i in 1:3])
            pairs=zeros(length(labels),GAUSSIAN_ENDPOINT_PAIRS)
            means=zeros(length(p.start),GAUSSIAN_ENDPOINT_PAIRS);seconds=similar(means)
            elbos=zeros(1,GAUSSIAN_ENDPOINT_PAIRS)
            rng=Xoshiro(GAUSSIAN_ENDPOINT_SEED)
            with_progress(progress,GAUSSIAN_ENDPOINT_PAIRS;description="Fixed independent hyperparameter pairs") do node
                for j in 1:GAUSSIAN_ENDPOINT_PAIRS
                    xi=randn(rng,3)
                    a,b=gaussian_endpoint_point(block.p3,mu,C,xi),gaussian_endpoint_point(block.p3,mu,C,-xi)
                    pairs[:,j]=(a.residual+b.residual)/2
                    means[:,j]=(a.mean+b.mean)/2;seconds[:,j]=(a.second+b.second)/2
                    elbos[1,j]=(a.elbo+b.elbo)/2
                    j%128==0 && update_progress!(node,j)
                end
            end
            summary,elbo=paired_summary(pairs),paired_summary(elbos)
            critical=quantile(TDist(GAUSSIAN_ENDPOINT_PAIRS-1),1-.05/(2length(labels)))
            lower,upper=summary.average-critical*summary.mcse,summary.average+critical*summary.mcse
            material=(lower .> AUDIT_PRACTICAL_TOLERANCE).|(upper .< -AUDIT_PRACTICAL_TOLERANCE)
            local_stationarity_ok=all((lower .>= -AUDIT_PRACTICAL_TOLERANCE).&
                (upper .<= AUDIT_PRACTICAL_TOLERANCE))
            verdict=any(material) ? "material nonstationarity" :
                local_stationarity_ok ? "small local residuals" : "inconclusive"
            batches=[paired_summary(pairs[:,(b-1)*2048+1:b*2048]) for b in 1:16]
            batch_mcse=vec(std(hcat(getproperty.(batches,:average)...);dims=2))/4
            squared=(pairs.-summary.average).^2
            variance_share=map(vec(maximum(squared;dims=2)),vec(sum(squared;dims=2))) do a,b
                iszero(b) ? 0. : a/b
            end
            mean_summary,second_summary=paired_summary(means),paired_summary(seconds)
            posterior_mean=copy(mean_summary.average)
            posterior_variance=second_summary.average-posterior_mean.^2
            posterior_mean[block.p3.globals]=mu
            posterior_variance[block.p3.globals]=vec(sum(abs2,C;dims=2))
            @test all(>(0),posterior_variance)
            posterior_sd=sqrt.(posterior_variance)
            influence=seconds.-second_summary.average.-2posterior_mean.*(means.-mean_summary.average)
            sd_mcse=vec(std(influence;dims=2))/sqrt(GAUSSIAN_ENDPOINT_PAIRS)./(2posterior_sd)
            sd_mcse[block.p3.globals].=0
            mean_mcse=copy(mean_summary.mcse);mean_mcse[block.p3.globals].=0
            mean_error=abs.(posterior_mean-reference.ref_mean)./reference.ref_sd
            sd_ratio=posterior_sd./reference.ref_sd
            moment_accuracy_ok=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
            # Difference from the earlier fixed-q(h) reconstruction, NOT
            # improvement from this native fit's prior-median initialization.
            gain=only(elbo.average)-baseline.summary.average[2]
            gain_mcse=hypot(only(elbo.mcse),baseline.summary.mcse[2])
            gain_critical=quantile(TDist(min(GAUSSIAN_ENDPOINT_PAIRS,baseline.identity.npairs)-1),.975)
            gain_interval=gain.+(-1,1).*(gain_critical*gain_mcse)
            @test all(isfinite,pairs) && all(isfinite,sd_mcse)
            result=(;identity,verdict,local_stationarity_ok,moment_accuracy_ok,
                native_draw_accuracy_ok=native_comparison.accuracy_ok,
                native_converged=saved.fit.converged,labels,summary,lower,upper,critical,material,
                elbo,gain,gain_mcse,gain_interval,batches,batch_mcse,variance_share,
                posterior_mean,posterior_sd,mean_mcse,sd_mcse,mean_error,sd_ratio,
                pairs,means,seconds,elbos,field,units,beta=block.beta,globals=block.p3.globals,
                scope="independent exact Gaussian-conditional audit of saved native fit; intervals approximate; reference uncertainty excluded; no fit, streaming or scaling")
            serialize(joinpath(directory,"audit.jls"),result)
            println((;verdict,local_stationarity_ok,material_labels=labels[material],
                elbo=elbo.average,mcse=elbo.mcse,gain,gain_interval,
                maximum_variance_share=maximum(variance_share)))
            println((;moment_accuracy_ok,native_draw_accuracy_ok=native_comparison.accuracy_ok,
                max_mean_error=maximum(mean_error),original_global_sd_ratio=sd_ratio[p.globals],
                unit_sd_extrema=extrema(sd_ratio[units]),field_sd_extrema=extrema(sd_ratio[field]),
                maximum_sd_mcse=maximum(sd_mcse)))
        end
    end
end
