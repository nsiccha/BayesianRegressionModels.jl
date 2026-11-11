# Independent, predeclared endpoint audit for joint_latent_fit.jl.
# No refit or budget extension: 32768 fresh antithetic pairs, seed1966,
# followed by the original 1000-draw/seed701 posterior comparison.
# julia --project=test research/shared_field_laplace/joint_latent_endpoint.jl 'joint latent endpoint audit'
include("joint_latent_fit.jl")

const JOINT_ENDPOINT_PAIRS = 32768
const JOINT_ENDPOINT_SEED = 1966

function joint_endpoint_point(p,mu,C,xi)
    g = mu+C*xi
    c = gaussian_reference_conditional(p,g)
    d = length(p.latent)
    root = c.F.U\Matrix{Float64}(I,d,d)
    score = zeros(length(g));u = copy(c.u)
    # Exact Gaussian cubature of the native partial score; identical Fisher
    # identity to the accepted reference, with its conditional moments retained.
    for j in 1:d,sign in (-1,1)
        u[p.latent] = c.mu+sign*sqrt(d)*root[:,j]
        _,w = LogDensityProblems.logdensity_and_gradient(p.full,u)
        score .+= w[p.globals]/(2d)
    end
    white = C'*score
    covariance = (white*xi'+xi*white')/2+I
    residual = vcat(white,[covariance[i,j] for j in 1:4 for i in 1:j],
        1 .+ score.*(g-mu))
    mean = copy(c.u);second = mean.^2
    second[p.latent] .+= vec(sum(abs2,root;dims=2))
    elbo = c.value+2*(1+log(2pi))+sum(log,diag(C))
    (;residual,mean,second,elbo,score)
end

@stestset "joint latent endpoint audit" begin
    base = joinpath(@__DIR__, "results")
    candidate = deserialize(joinpath(base,"joint-latent-fit-v1/candidate.jls"))
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    baseline = deserialize(joinpath(base,"joint-latent-comparison-v1/comparison.jls"))
    @test candidate.identity.reference_signature == baseline.reference_signature == reference.signature
    @test candidate.identity.warm_start_signature == baseline.candidate_signature
    @test reference.reference_ok
    reference.reference_ok &&
        candidate.identity.reference_signature == baseline.reference_signature == reference.signature &&
        candidate.identity.warm_start_signature == baseline.candidate_signature ||
        error("endpoint prerequisite reference/warm-start identity failed")
    sources = [joinpath(@__DIR__,name) for name in
        ("joint_latent_fit.jl","profile_optimization.jl","stationarity_audit.jl",
         "exact_reference.jl","operator_probe.jl","trace_probe.jl",
         "nonlinear_probe.jl","whole_window_probe.jl","quadrature.jl","joint_svi_probe.jl")]
    append!(sources,[joinpath(@__DIR__,"../../src/nlme_svi.jl"),
        joinpath(@__DIR__,"../../test/Manifest.toml")])
    bytes2hex(sha256(reduce(vcat,read.(sources)))) == candidate.identity.signature ||
        error("joint fit source/environment identity changed")
    directory = joinpath(base,"joint-latent-endpoint-v1")
    ispath(directory) && error("preserve existing joint latent endpoint audit")
    mkpath(directory)
    identity = (;signature=bytes2hex(sha256(reduce(vcat,read.(vcat([@__FILE__],sources))))),
        candidate_signature=candidate.identity.signature,reference_signature=reference.signature,
        npairs=JOINT_ENDPOINT_PAIRS,seed=JOINT_ENDPOINT_SEED)
    serialize(joinpath(directory,"identity.jls"),identity)
    with_progress(:state;description="Independent joint latent endpoint audit") do progress
        snapshot_progress(progress,joinpath(directory,"endpoint.progress")) do
            fixture = with_progress(progress;description="Prepare exact Gaussian-latent oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test view.coordinates == candidate.coordinates == reference.coordinates
            view.coordinates == candidate.coordinates == reference.coordinates ||
                error("endpoint coordinate identity failed")
            mu,C = candidate.mu,candidate.C
            labels = vcat(["whitened_mean_$i" for i in 1:4],
                ["covariance_$(i)_$j" for j in 1:4 for i in 1:j],
                ["row_scale_$i" for i in 1:4])
            check_xi = [.3,-.5,.7,-.2]
            check = joint_endpoint_point(p,mu,C,check_xi)
            _,score = LogDensityProblems.logdensity_and_gradient(p,mu+C*check_xi)
            @test check.score ≈ score atol=2e-11
            maximum(abs.(check.score-score)) <= 2e-11 || error("endpoint score validation failed")
            pairs = zeros(length(labels),JOINT_ENDPOINT_PAIRS)
            means = zeros(length(p.start),JOINT_ENDPOINT_PAIRS)
            seconds = similar(means)
            elbos = zeros(1,JOINT_ENDPOINT_PAIRS)
            rng = Xoshiro(JOINT_ENDPOINT_SEED)
            with_progress(progress,JOINT_ENDPOINT_PAIRS;description="Fixed independent global pairs") do node
                for j in 1:JOINT_ENDPOINT_PAIRS
                    xi = randn(rng,4)
                    a,b = joint_endpoint_point(p,mu,C,xi),joint_endpoint_point(p,mu,C,-xi)
                    pairs[:,j] = (a.residual+b.residual)/2
                    means[:,j] = (a.mean+b.mean)/2
                    seconds[:,j] = (a.second+b.second)/2
                    elbos[1,j] = (a.elbo+b.elbo)/2
                    j%128 == 0 && update_progress!(node,j)
                end
            end
            summary,elbo = paired_summary(pairs),paired_summary(elbos)
            critical = quantile(TDist(JOINT_ENDPOINT_PAIRS-1),1-.05/(2length(labels)))
            lower,upper = summary.average-critical*summary.mcse,summary.average+critical*summary.mcse
            material = (lower .> AUDIT_PRACTICAL_TOLERANCE) .| (upper .< -AUDIT_PRACTICAL_TOLERANCE)
            local_stationarity_ok = all((lower .>= -AUDIT_PRACTICAL_TOLERANCE) .&
                (upper .<= AUDIT_PRACTICAL_TOLERANCE))
            verdict = any(material) ? "material nonstationarity" :
                local_stationarity_ok ? "small local residuals" : "inconclusive"
            batches = [paired_summary(pairs[:,(b-1)*2048+1:b*2048]) for b in 1:16]
            batch_mcse = vec(std(hcat(getproperty.(batches,:average)...);dims=2))/4
            squared = (pairs.-summary.average).^2
            variance_share = map(vec(maximum(squared;dims=2)),vec(sum(squared;dims=2))) do a,b
                iszero(b) ? 0. : a/b
            end
            mean_summary,second_summary = paired_summary(means),paired_summary(seconds)
            posterior_mean = copy(mean_summary.average)
            posterior_variance = second_summary.average-posterior_mean.^2
            # Known Gaussian global moments have no audit integration error.
            posterior_mean[p.globals] = mu
            posterior_variance[p.globals] = vec(sum(abs2,C;dims=2))
            @test all(>(0),posterior_variance)
            posterior_sd = sqrt.(posterior_variance)
            influence = seconds.-second_summary.average .-
                2posterior_mean.*(means.-mean_summary.average)
            sd_mcse = vec(std(influence;dims=2))/sqrt(JOINT_ENDPOINT_PAIRS)./(2posterior_sd)
            sd_mcse[p.globals] .= 0
            mean_mcse = copy(mean_summary.mcse);mean_mcse[p.globals] .= 0
            mean_error = abs.(posterior_mean-reference.ref_mean)./reference.ref_sd
            sd_ratio = posterior_sd./reference.ref_sd
            # Point estimates only: neither these moments nor their MCSEs
            # incorporate uncertainty in the retained reference chains.
            moment_accuracy_ok = maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
            serialize(joinpath(directory,"completed-audit.jls"),(;identity,verdict,
                local_stationarity_ok,moment_accuracy_ok,labels,summary,lower,upper,
                critical,material,elbo,batches,batch_mcse,variance_share,
                posterior_mean,posterior_sd,mean_mcse,sd_mcse,mean_error,sd_ratio,
                pairs,means,seconds,elbos,field,units,globals=p.globals))
            draws = with_progress(progress,1000;description="Original fixed draw-count comparison") do node
                # Same original seed/count/tolerances, but changed roots and
                # latent grouping mean these are not paired with older draws.
                U = repeat(p.start,1,1000);rng = Xoshiro(701)
                for j in 1:1000
                    g = mu+C*randn(rng,4)
                    c = gaussian_reference_conditional(p,g)
                    U[p.globals,j] = g
                    U[p.latent,j] = c.mu+c.F.U\randn(rng,length(p.latent))
                    j%50 == 0 && update_progress!(node,j)
                end
                U
            end
            draw_mean_error = abs.(vec(mean(draws;dims=2))-reference.ref_mean)./reference.ref_sd
            draw_sd_ratio = vec(std(draws;dims=2))./reference.ref_sd
            draw_accuracy_ok = maximum(draw_mean_error)<=.3 && all(x->.8<=x<=1.2,draw_sd_ratio)
            gain = only(elbo.average)-baseline.summary.average[2]
            gain_mcse = hypot(only(elbo.mcse),baseline.summary.mcse[2])
            gain_critical = quantile(TDist(min(JOINT_ENDPOINT_PAIRS,baseline.npairs)-1),.975)
            gain_interval = (gain-gain_critical*gain_mcse,gain+gain_critical*gain_mcse)
            @test all(isfinite,pairs) && all(isfinite,draws) && all(isfinite,sd_mcse)
            result = (;identity,verdict,local_stationarity_ok,moment_accuracy_ok,draw_accuracy_ok,
                labels,summary,lower,upper,critical,material,elbo,gain,gain_mcse,gain_interval,
                batches,batch_mcse,variance_share,posterior_mean,posterior_sd,mean_mcse,sd_mcse,
                mean_error,sd_ratio,draw_mean_error,draw_sd_ratio,pairs,means,seconds,elbos,draws,
                field,units,globals=p.globals,optimizer_reason=candidate.optimizer.reason,
                scope="independent fixed-budget joint-latent oracle endpoint; no native fit/scaling or global-optimum claim")
            serialize(joinpath(directory,"audit.jls"),result)
            println((;verdict,local_stationarity_ok,elbo=elbo.average,mcse=elbo.mcse,gain,gain_interval,
                material_labels=labels[material],maximum_variance_share=maximum(variance_share)))
            println((;moment_accuracy_ok,max_mean_error=maximum(mean_error),
                global_sd_ratio=sd_ratio[p.globals],unit_sd_ratio=sd_ratio[units],
                field_sd_extrema=extrema(sd_ratio[field]),maximum_sd_mcse=maximum(sd_mcse)))
            println((;draw_accuracy_ok,max_mean_error=maximum(draw_mean_error),
                global_sd_ratio=draw_sd_ratio[p.globals],unit_sd_ratio=draw_sd_ratio[units],
                field_sd_extrema=extrema(draw_sd_ratio[field])))
        end
    end
end
