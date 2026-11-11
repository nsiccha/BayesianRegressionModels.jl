# Independent endpoint check after the SINGLE bounded optimization experiment.
# Same predeclared 32768-pair precision budget, new seed 1957. No extensions
# based on stationarity or the unchanged posterior tolerances. Both diagnostics
# and failures are saved. This validates an oracle family, not a production fit.
# julia --project=test research/shared_field_laplace/profile_endpoint.jl 'profile endpoint audit'
include("profile_optimization.jl")

const ENDPOINT_PAIRS = 32768
const ENDPOINT_SEED = 1957

function endpoint_point(p,field,units,q,C,M,variance,epsilon,zeta)
    g = BRM._brm_svi_draw_global(q,epsilon,zeta)
    a = gaussian_unit_expectation(p,field,units,g,M,variance)
    xi = C\(g-q.μ)
    ws = C'*a.w
    covariance = (ws*xi'+xi*ws')/2+I
    residual = vcat(ws,[covariance[i,j] for j in 1:4 for i in 1:j],
        1 .+ a.w.*(g-q.μ),sqrt.(variance).*a.unit_score,a.unit_logsd)
    packed = BRM._brm_svi_global_gradient(q,a.w,epsilon,zeta)
    full_mean = copy(p.start)
    full_mean[p.globals] = g
    full_mean[p.latent] = a.latent_mean
    raw_second = full_mean.^2
    raw_second[p.latent] .+= vec(sum(abs2,a.root;dims=2))
    (;residual,packed,elbo=a.value+audit_entropy(C,variance),full_mean,raw_second)
end

function oracle_conditional_draws(rng,p,field,units,q,n)
    n==1000 || error("fixed original draw-count diagnostic")
    result = repeat(p.start,1,n)
    M,variance = audit_unit_state(q)
    for j in 1:n
        g = BRM._brm_svi_draw_global(q,randn(rng,4),randn(rng,4))
        r = profile_quadratic(p,field,units,g)
        eta = M+sqrt.(variance).*randn(rng,length(M))
        z = r.z0-r.coupling*eta+r.F.U\randn(rng,length(field))
        result[p.globals,j] = g
        result[units,j] = eta
        result[field,j] = z
    end
    result
end

@stestset "profile endpoint audit" begin
    base = joinpath(@__DIR__, "results")
    candidate = deserialize(joinpath(base,"profile-optimization-v1/candidate.jls"))
    baseline = deserialize(joinpath(base,"stationarity-precision-v1/audit.jls"))
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    @test candidate.fit_signature==baseline.fit_signature==AUDIT_FIT_SIGNATURE
    @test candidate.reference_signature==reference.signature && reference.reference_ok
    profile_sources = [joinpath(@__DIR__,"profile_optimization.jl"),
        joinpath(@__DIR__,"stationarity_audit.jl"),joinpath(@__DIR__,"exact_reference.jl"),
        joinpath(@__DIR__,"operator_probe.jl"),joinpath(@__DIR__,"../../src/nlme_svi.jl"),
        joinpath(@__DIR__,"../../test/Manifest.toml")]
    @test bytes2hex(sha256(reduce(vcat,read.(profile_sources))))==candidate.signature
    directory = get(ENV,"KB_LAPLACE_ENDPOINT_DIR",joinpath(base,"profile-endpoint-v1"))
    ispath(directory) && error("preserve existing endpoint audit: $directory")
    mkpath(directory)
    with_progress(:state;description="Independent bounded-optimizer endpoint audit") do progress
        snapshot_progress(progress,joinpath(directory,"endpoint.progress")) do
            fixture = with_progress(progress;description="Prepare native exact-latent oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test view.coordinates==candidate.coordinates==reference.coordinates
            q = candidate.q
            M,variance = audit_unit_state(q)
            C = Matrix(cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(q)))).L)
            labels = baseline.labels
            pairs = zeros(length(labels),ENDPOINT_PAIRS)
            packed_pairs = zeros(length(BRM._brm_svi_global_vector(q)),ENDPOINT_PAIRS)
            elbos = zeros(1,ENDPOINT_PAIRS)
            means = zeros(length(p.start),ENDPOINT_PAIRS)
            seconds = similar(means)
            rng = Xoshiro(ENDPOINT_SEED)
            with_progress(progress,ENDPOINT_PAIRS;description="Fixed independent endpoint pairs") do node
                for j in 1:ENDPOINT_PAIRS
                    epsilon,zeta = randn(rng,4),randn(rng,4)
                    a = endpoint_point(p,field,units,q,C,M,variance,epsilon,zeta)
                    b = endpoint_point(p,field,units,q,C,M,variance,-epsilon,-zeta)
                    pairs[:,j] = (a.residual+b.residual)/2
                    packed_pairs[:,j] = (a.packed+b.packed)/2
                    elbos[1,j] = (a.elbo+b.elbo)/2
                    means[:,j] = (a.full_mean+b.full_mean)/2
                    seconds[:,j] = (a.raw_second+b.raw_second)/2
                    j%128==0 && update_progress!(node,j)
                end
            end
            summary,packed,elbo = paired_summary(pairs),paired_summary(packed_pairs),paired_summary(elbos)
            critical = quantile(TDist(ENDPOINT_PAIRS-1),1-.05/(2length(labels)))
            lower,upper = summary.average-critical*summary.mcse,summary.average+critical*summary.mcse
            material = (lower .> AUDIT_PRACTICAL_TOLERANCE) .|
                (upper .< -AUDIT_PRACTICAL_TOLERANCE)
            local_stationarity_ok = all((lower .>= -AUDIT_PRACTICAL_TOLERANCE) .&
                (upper .<= AUDIT_PRACTICAL_TOLERANCE))
            verdict = any(material) ? "material nonstationarity" :
                local_stationarity_ok ? "small local residuals" : "inconclusive"
            batches = [paired_summary(pairs[:,(b-1)*2048+1:b*2048]) for b in 1:16]
            batch_mcse = vec(std(hcat(getproperty.(batches,:average)...);dims=2))/4
            squared = (pairs .- summary.average).^2
            variance_total = vec(sum(squared;dims=2))
            constant_components = iszero.(variance_total)
            variance_share = map(vec(maximum(squared;dims=2)),variance_total) do largest,total
                iszero(total) ? 0.0 : largest/total
            end
            mean_summary,second_summary = paired_summary(means),paired_summary(seconds)
            posterior_mean = mean_summary.average
            posterior_variance = second_summary.average-posterior_mean.^2
            @test all(>(0),posterior_variance)
            posterior_sd = sqrt.(posterior_variance)
            variance_influence = seconds .- second_summary.average .-
                2posterior_mean.*(means .- posterior_mean)
            sd_mcse = vec(std(variance_influence;dims=2))/sqrt(ENDPOINT_PAIRS)./(2posterior_sd)
            mean_error = abs.(posterior_mean-reference.ref_mean)./reference.ref_sd
            sd_ratio = posterior_sd./reference.ref_sd
            # Point-estimate gates: the endpoint MCSE excludes uncertainty in
            # the retained reference chains. It is not a confidence certificate.
            moment_accuracy_ok = maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio)
            draws = with_progress(progress;description="Original 1000-draw comparison") do _
                oracle_conditional_draws(Xoshiro(701),p,field,units,q,1000)
            end
            draw_mean_error = abs.(vec(mean(draws;dims=2))-reference.ref_mean)./reference.ref_sd
            draw_sd_ratio = vec(std(draws;dims=2))./reference.ref_sd
            # Same seed/count/tolerances, but not paired with the earlier fit:
            # the covariance decomposition and conditional-root implementation differ.
            draw_accuracy_ok = maximum(draw_mean_error)<=.3 && all(x->.8<=x<=1.2,draw_sd_ratio)
            gain = only(elbo.average-baseline.elbo.average)
            gain_mcse = hypot(only(elbo.mcse),only(baseline.elbo.mcse))
            gain_critical = quantile(TDist(ENDPOINT_PAIRS-1),.975)
            gain_interval = (gain-gain_critical*gain_mcse,gain+gain_critical*gain_mcse)
            @test all(isfinite,pairs) && all(isfinite,draws) && all(isfinite,sd_mcse)
            signature = bytes2hex(sha256(reduce(vcat,read.(vcat([@__FILE__],profile_sources)))))
            result = (;verdict,local_stationarity_ok,moment_accuracy_ok,draw_accuracy_ok,
                labels,summary,packed,elbo,lower,upper,material,critical,pairs,packed_pairs,
                elbos,batches,batch_mcse,variance_share,constant_components,means,seconds,mean_summary,second_summary,
                posterior_mean,posterior_sd,sd_mcse,mean_error,sd_ratio,draws,
                draw_mean_error,draw_sd_ratio,gain,gain_mcse,gain_interval,signature,
                candidate_signature=candidate.signature,baseline_signature=baseline.signature,
                reference_signature=reference.signature,seed=ENDPOINT_SEED,npairs=ENDPOINT_PAIRS,
                coordinates=view.coordinates,field,units,globals=p.globals,
                optimizer_reason=candidate.optimizer.reason,
                scope="same-family oracle endpoint; no production accuracy or global-optimum claim")
            serialize(joinpath(directory,"audit.jls"),result)
            println((;stage="independent_endpoint",verdict,local_stationarity_ok,
                elbo=elbo.average,elbo_mcse=elbo.mcse,gain,gain_interval,
                maximum_variance_share=maximum(variance_share),
                material_labels=labels[material]))
            println((;stage="posterior_moments",moment_accuracy_ok,
                maximum_mean_error=maximum(mean_error),global_sd_ratio=sd_ratio[p.globals],
                unit_sd_ratio=sd_ratio[units],field_sd_extrema=extrema(sd_ratio[field]),
                maximum_sd_mcse=maximum(sd_mcse)))
            println((;stage="original_draw_gate",draw_accuracy_ok,
                maximum_mean_error=maximum(draw_mean_error),global_sd_ratio=draw_sd_ratio[p.globals],
                unit_sd_ratio=draw_sd_ratio[units],field_sd_extrema=extrema(draw_sd_ratio[field])))
            for i in eachindex(labels)
                println((;label=labels[i],mean=summary.average[i],mcse=summary.mcse[i],
                    lower=lower[i],upper=upper[i],batch_mcse=batch_mcse[i]))
            end
        end
    end
end
