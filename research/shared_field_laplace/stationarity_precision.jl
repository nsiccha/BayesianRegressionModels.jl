# One independently seeded, fixed-budget precision audit after the initial
# 64-pair audit and GH7/11 proved inconclusive for globals. The pilot's largest
# MCSE (~.32 at 64 pairs) forecasts simultaneous half-width ~.05 at 32768 pairs.
# That forecast is not guaranteed; there is no optional extension of this run.
# julia --project=test research/shared_field_laplace/stationarity_precision.jl 'independent precision audit'
include("stationarity_audit.jl")

const PRECISION_PAIRS = 32768
const PRECISION_SEED = 1955
const PRECISION_BATCHES = 16

@stestset "independent precision audit" begin
    base = joinpath(@__DIR__, "results")
    saved = deserialize(joinpath(base,"joint-fit-v1/fit.jls"))
    pilot = deserialize(joinpath(base,"stationarity-v1/audit.jls"))
    @test saved.signature == pilot.fit_signature == AUDIT_FIT_SIGNATURE
    audit_sources = [joinpath(@__DIR__,"stationarity_audit.jl"),
        joinpath(@__DIR__,"exact_reference.jl"),joinpath(@__DIR__,"operator_probe.jl"),
        joinpath(@__DIR__,"../../test/Manifest.toml")]
    @test bytes2hex(sha256(reduce(vcat,read.(audit_sources)))) == pilot.signature
    directory = get(ENV,"KB_LAPLACE_PRECISION_DIR",joinpath(base,"stationarity-precision-v1"))
    ispath(directory) && error("preserve prior precision audit: $directory")
    mkpath(directory)
    with_progress(:state;description="Independent fixed-precision stationarity audit") do progress
        snapshot_progress(progress,joinpath(directory,"precision.progress")) do
            fixture = with_progress(progress;description="Prepare native exact-latent oracle") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                exact_gaussian_reference(rk,data)
            end
            (;p,view,field,units) = fixture
            @test view.coordinates == pilot.coordinates
            q = saved.fit.q
            M,variance = audit_unit_state(q)
            C = Matrix(cholesky(Symmetric(Matrix(BRM._brm_svi_global_covariance(q)))).L)
            D = length(q.μ)
            pairs = zeros(length(pilot.labels),PRECISION_PAIRS)
            packed_pairs = zeros(length(BRM._brm_svi_global_vector(q)),PRECISION_PAIRS)
            elbos = zeros(1,PRECISION_PAIRS)
            rng = Xoshiro(PRECISION_SEED)
            with_progress(progress,PRECISION_PAIRS;description="Predeclared independent pairs") do node
                for j in 1:PRECISION_PAIRS
                    epsilon,zeta = randn(rng,D),randn(rng,size(q.Β,2))
                    plus = audit_point(p,field,units,q,C,M,variance,epsilon,zeta)
                    minus = audit_point(p,field,units,q,C,M,variance,-epsilon,-zeta)
                    pairs[:,j] = (plus.residual+minus.residual)/2
                    packed_pairs[:,j] = (plus.packed+minus.packed)/2
                    elbos[1,j] = (plus.elbo+minus.elbo)/2
                    j%128==0 && update_progress!(node,j)
                end
            end
            summary,packed,elbo = paired_summary(pairs),paired_summary(packed_pairs),paired_summary(elbos)
            labels = pilot.labels
            critical = quantile(TDist(PRECISION_PAIRS-1),1-.05/(2length(labels)))
            lower,upper = summary.average-critical*summary.mcse,summary.average+critical*summary.mcse
            material = (lower .> AUDIT_PRACTICAL_TOLERANCE) .|
                (upper .< -AUDIT_PRACTICAL_TOLERANCE)
            locally_small = all((lower .>= -AUDIT_PRACTICAL_TOLERANCE) .&
                (upper .<= AUDIT_PRACTICAL_TOLERANCE))
            verdict = any(material) ? "material nonstationarity" :
                locally_small ? "small local residuals" : "inconclusive"
            # Predetermined disjoint batches and single-pair variance shares
            # expose rare-contribution domination. Neither is a tail guarantee.
            batch_size = div(PRECISION_PAIRS,PRECISION_BATCHES)
            batches = [paired_summary(pairs[:,(b-1)*batch_size+1:b*batch_size])
                for b in 1:PRECISION_BATCHES]
            batch_means = hcat(getproperty.(batches,:average)...)
            squared = (pairs .- summary.average).^2
            max_variance_share = vec(maximum(squared;dims=2)./sum(squared;dims=2))
            batch_mcse = vec(std(batch_means;dims=2))/sqrt(PRECISION_BATCHES)
            @test all(isfinite,pairs) && all(isfinite,packed_pairs) && all(isfinite,elbos)
            sources = vcat([@__FILE__],audit_sources,[joinpath(@__DIR__,"../../src/nlme_svi.jl")])
            signature = bytes2hex(sha256(reduce(vcat,read.(sources))))
            result = (;verdict,labels,summary,packed,elbo,lower,upper,material,critical,
                pairs,packed_pairs,elbos,batches,max_variance_share,batch_mcse,signature,
                fit_signature=saved.signature,pilot_signature=pilot.signature,
                seed=PRECISION_SEED,npairs=PRECISION_PAIRS,nbatches=PRECISION_BATCHES,
                practical_tolerance=AUDIT_PRACTICAL_TOLERANCE,
                scope="fresh independent fixed-budget audit; same saved q; approximate simultaneous intervals")
            serialize(joinpath(directory,"audit.jls"),result)
            println((;stage="precision_audit",verdict,npairs=PRECISION_PAIRS,
                elbo=elbo.average,elbo_mcse=elbo.mcse,critical,
                maximum_variance_share=maximum(max_variance_share),material_labels=labels[material]))
            for i in eachindex(labels)
                println((;label=labels[i],mean=summary.average[i],mcse=summary.mcse[i],
                    lower=lower[i],upper=upper[i],variance_share=max_variance_share[i],
                    batch_mcse=batch_mcse[i]))
            end
        end
    end
end
