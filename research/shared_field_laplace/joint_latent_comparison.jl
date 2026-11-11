# Fixed-q(g) diagnostic: replace independent q(unit) by the exact joint
# conditional of units+field. No refit, no reference resampling, no gate change.
# Global posterior errors necessarily remain unchanged. Dense operations are
# confined to the existing <=32-latent, <=40-coordinate Gaussian oracle.
# julia --project=test research/shared_field_laplace/joint_latent_comparison.jl 'fixed global joint latent comparison'
include("profile_optimization.jl")

const JOINT_LATENT_PAIRS = 8192
const JOINT_LATENT_SEED = 1964

function fixed_conditional_moments(p,r,g,M,V)
    mu = zeros(length(p.latent))
    mu[r.ei] = M
    mu[r.fi] = r.z0-r.coupling*M
    covariance = zeros(length(mu),length(mu))
    covariance[r.ei,r.ei] = V
    covariance[r.ei,r.fi] = -V*r.coupling'
    covariance[r.fi,r.ei] = -r.coupling*V
    covariance[r.fi,r.fi] = inv(r.F)+r.coupling*V*r.coupling'
    fullmean = copy(p.start);fullmean[p.globals] = g;fullmean[p.latent] = mu
    (;fullmean,covariance)
end

@stestset "fixed global joint latent comparison" begin
    base = joinpath(@__DIR__, "results")
    candidate = deserialize(joinpath(base,"profile-optimization-v1/candidate.jls"))
    reference = deserialize(joinpath(base,"exact-reference-v1/reference-summary.jls"))
    @test candidate.reference_signature == reference.signature && reference.reference_ok
    profile_sources = [joinpath(@__DIR__,"profile_optimization.jl"),
        joinpath(@__DIR__,"stationarity_audit.jl"),joinpath(@__DIR__,"exact_reference.jl"),
        joinpath(@__DIR__,"operator_probe.jl"),joinpath(@__DIR__,"../../src/nlme_svi.jl"),
        joinpath(@__DIR__,"../../test/Manifest.toml")]
    @test bytes2hex(sha256(reduce(vcat,read.(profile_sources)))) == candidate.signature
    directory = joinpath(base,"joint-latent-comparison-v1")
    ispath(directory) && error("preserve existing joint-latent comparison")
    mkpath(directory)
    signature = bytes2hex(sha256(reduce(vcat,read.(vcat([@__FILE__],profile_sources)))))
    serialize(joinpath(directory,"identity.jls"),(;signature,
        candidate_signature=candidate.signature,reference_signature=reference.signature,
        seed=JOINT_LATENT_SEED,npairs=JOINT_LATENT_PAIRS,julia=string(VERSION)))
    with_progress(:state;description="Fixed-global joint latent conditional comparison") do progress
        snapshot_progress(progress,joinpath(directory,"comparison.progress")) do
            fixture = with_progress(progress;description="Prepare native Gaussian-latent oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test view.coordinates == candidate.coordinates == reference.coordinates
            q = candidate.q
            M,variance = audit_unit_state(q)
            V = Diagonal(variance)
            Cg = Matrix(BRM._brm_svi_global_covariance(q))
            Lg = Matrix(cholesky(Symmetric(Cg)).L)
            hg = 2*(1+log(2pi))+sum(log,diag(Lg))
            he = length(M)/2*(1+log(2pi))+sum(log,variance)/2
            n = length(p.start)
            # Full means/raw seconds plus selected cross moments retain
            # conditional covariance without 26x26x8192 arrays.
            cross = [(p.globals[3],units[1]),(units[1],units[2]),
                (units[1],field[1]),(p.globals[3],field[1])]
            pairs = zeros(3,JOINT_LATENT_PAIRS) # old ELBO, joint ELBO, exact KL gain
            moments = [zeros(2n+length(cross),JOINT_LATENT_PAIRS) for _ in 1:2]
            latent_position = Dict(k=>j for (j,k) in enumerate(p.latent))
            max_identity_error,min_gap = 0.,Inf
            rng = Xoshiro(JOINT_LATENT_SEED)
            with_progress(progress,JOINT_LATENT_PAIRS;description="Fixed paired conditional expectations") do node
                for j in 1:JOINT_LATENT_PAIRS
                    xi = randn(rng,4)
                    for sign in (-1,1)
                        g = q.μ+sign*Lg*xi
                        r = profile_quadratic(p,field,units,g)
                        unit_mean = r.S\r.b
                        delta = M-unit_mean
                        gap = (dot(diag(r.S),variance)+dot(delta,r.S*delta)-length(M)-
                            logdet(Symmetric(r.S))-sum(log,variance))/2
                        old = r.base+dot(r.b,M)-dot(M,r.S*M)/2-dot(diag(r.S),variance)/2+hg+he
                        joint = r.c.value+hg
                        max_identity_error = max(max_identity_error,abs(joint-old-gap))
                        min_gap = min(min_gap,gap)
                        pairs[:,j] .+= [old,joint,gap]/2
                        before = fixed_conditional_moments(p,r,g,M,V)
                        after_mean = copy(r.c.u)
                        after = (;fullmean=after_mean,covariance=inv(r.c.F))
                        for (i,state) in enumerate((before,after))
                            mu = state.fullmean
                            second = mu.^2
                            second[p.latent] .+= diag(state.covariance)
                            products = map(cross) do (a,b)
                                value = mu[a]*mu[b]
                                if haskey(latent_position,a) && haskey(latent_position,b)
                                    value += state.covariance[latent_position[a],latent_position[b]]
                                end
                                value
                            end
                            moments[i][:,j] .+= vcat(mu,second,products)/2
                        end
                    end
                    j%128 == 0 && update_progress!(node,j)
                end
            end
            @test max_identity_error < 2e-8
            @test min_gap >= -1e-10
            @test all(isfinite,pairs) && all(x->all(isfinite,x),moments)
            summary = paired_summary(pairs)
            # The exact global moments are identical before/after, without
            # global Monte Carlo error. These failed global ratios cannot be
            # repaired by changing the latent conditional alone.
            # moment_mcse contains uncertainty of raw first/second/cross
            # moments, not of SDs or centered covariances. Global moment point
            # estimates below are exact; their raw MC diagnostics are retained.
            # All comparison gates exclude reference-chain uncertainty.
            comparisons = map(moments) do values
                stats = paired_summary(values)
                mu = stats.average[1:n]
                raw = stats.average[n+1:2n]
                mu[p.globals] = q.μ
                raw[p.globals] = q.μ.^2+diag(Cg)
                sd = sqrt.(raw-mu.^2)
                mean_error = abs.(mu-reference.ref_mean)./reference.ref_sd
                sd_ratio = sd./reference.ref_sd
                crosscov = [stats.average[2n+i]-mu[a]*mu[b]
                    for (i,(a,b)) in enumerate(cross)]
                (;mu,sd,mean_error,sd_ratio,crosscov,
                    moment_accuracy_ok=maximum(mean_error)<=.3 && all(x->.8<=x<=1.2,sd_ratio),
                    moment_mcse=stats.mcse)
            end
            @test comparisons[1].mu[p.globals] == comparisons[2].mu[p.globals]
            @test comparisons[1].sd[p.globals] == comparisons[2].sd[p.globals]
            batches = [paired_summary(pairs[:,(b-1)*512+1:b*512]) for b in 1:16]
            batch_mcse = vec(std(hcat(getproperty.(batches,:average)...);dims=2))/4
            squared = (pairs.-summary.average).^2
            variance_share = vec(maximum(squared;dims=2))./vec(sum(squared;dims=2))
            critical = quantile(TDist(JOINT_LATENT_PAIRS-1),.975)
            gap_interval = (summary.average[3]-critical*summary.mcse[3],
                summary.average[3]+critical*summary.mcse[3])
            result = (;summary,gap_interval,batches,batch_mcse,variance_share,
                comparisons,cross,pairs,moments,max_identity_error,min_gap,signature,
                reference_signature=reference.signature,candidate_signature=candidate.signature,
                seed=JOINT_LATENT_SEED,npairs=JOINT_LATENT_PAIRS,
                scope="fixed q(global), exact Gaussian-latent conditional diagnostic; no fit or new acceptance gate")
            serialize(joinpath(directory,"comparison.jls"),result)
            println((;elbos=summary.average[1:2],gain=summary.average[3],
                gain_mcse=summary.mcse[3],gap_interval,batch_mcse,variance_share,
                max_identity_error,min_gap))
            for (name,c) in zip((:before,:joint_latent),comparisons)
                println((;name,c.moment_accuracy_ok,max_mean_error=maximum(c.mean_error),
                    global_sd_ratio=c.sd_ratio[p.globals],unit_sd_ratio=c.sd_ratio[units],
                    field_sd_extrema=extrema(c.sd_ratio[field]),c.crosscov))
            end
        end
    end
end
