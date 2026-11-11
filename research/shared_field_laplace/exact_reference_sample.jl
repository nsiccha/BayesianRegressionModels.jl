# Run only after exact_reference.jl passes. This samples the SAME fixture's
# exact four-dimensional marginal, then restores all Gaussian latents jointly.
# julia --project=test research/shared_field_laplace/exact_reference_sample.jl 'Gaussian marginal sampling'
include("exact_reference.jl")
using WarmupHMC, MCMCDiagnosticTools, Serialization, Statistics, SHA

function reference_diagnostics(arrays)
    n = minimum(size(a,2) for a in arrays)
    samples = permutedims(cat((a[:,end-n+1:end]' for a in arrays)...;dims=3),(1,3,2))
    (;rhat=maximum(MCMCDiagnosticTools.rhat(samples)),
        bulk=minimum(MCMCDiagnosticTools.ess(samples;kind=:bulk)),
        tail=minimum(MCMCDiagnosticTools.ess(samples;kind=:tail)))
end
reference_mixing_ok(d) = isfinite(d.rhat) && d.rhat < 1.01 &&
    min(d.bulk,d.tail) >= 400

@stestset "Gaussian marginal sampling" begin
    directory = get(ENV,"KB_LAPLACE_REFERENCE_DIR",
        joinpath(@__DIR__, "results", "exact-reference-v1"))
    mkpath(directory)
    signature = bytes2hex(sha256(reduce(vcat,read.([
        @__FILE__,joinpath(@__DIR__,"exact_reference.jl"),
        joinpath(@__DIR__,"operator_probe.jl"),
        joinpath(@__DIR__,"../../test/Manifest.toml")]))))
    identity = joinpath(directory,"identity.txt")
    isfile(identity) ? (readchomp(identity) == signature || error(
        "reference code/environment changed; use a fresh directory")) :
        write(identity,signature*"\n")
    with_progress(:state;description="Exact Gaussian-marginal reference") do progress
        snapshot_progress(progress,joinpath(directory,"reference.progress")) do
            fixture = with_progress(progress;description="Prepare native marginal") do _
                (;rk,data) = shared_field_fixture(;groups=6,rows=12,temporal=4,
                    spatial=2,prepare_model=false)
                merge((;rk,data),exact_gaussian_reference(rk,data))
            end
            (;p,rk,view) = fixture
            saved = joinpath(directory,"global-chains.jls")
            global_record = if isfile(saved)
                deserialize(saved)
            else
                # Construct private prepared native targets; do not deepcopy RK
                # graphs or let concurrent chains share a mutable AD workspace.
                problems = [ExactGaussianReference(rk_logdensity_problem(rk;
                    ad_backend=NLME_AD),copy(p.start),copy(p.globals),copy(p.latent))
                    for _ in 1:4]
                init = [(;position=p.start[p.globals]+0.05randn(Xoshiro(600+i),4),
                    squared_scale=ones(4)) for i in 1:4]
                checkpoint_dir = joinpath(directory,"completion")
                resume = isfile(joinpath(checkpoint_dir,"completion_manifest.jls"))
                # All four must complete and be admitted: no runtime-selected
                # subset is accepted as this accuracy reference.
                seconds = @elapsed outcome = try
                    completion_warmup_mcmc([Xoshiro(500+i) for i in 1:4],problems;
                        min_completed=4,init,n_draws=2000,target_acceptance_rate=0.99,
                        checkpoint_dir,resume,progress,description="Marginal reference")
                catch e
                    e isa WarmupHMC.CompletionQuorumError || rethrow()
                    e.outcome
                end
                chains = [(;position=Matrix{Float64}(r.posterior_position),
                    divergences=r.n_divergent_samples,chain=r.chain_index)
                    for r in outcome.results]
                record = (;chains,completion=outcome.completion,seconds,signature)
                serialize(saved,record)
                record
            end
            @test global_record.signature == signature
            @test global_record.completion.quorum_met
            @test global_record.completion.n_omitted == 0
            global_record.completion.quorum_met &&
                global_record.completion.n_omitted == 0 || error("incomplete reference quorum")
            global_diagnostics = reference_diagnostics([c.position for c in global_record.chains])
            divergences = sum(c.divergences for c in global_record.chains)
            println((;stage="global reference",global_diagnostics...,divergences,
                draws=sum(size(c.position,2) for c in global_record.chains)))
            serialize(joinpath(directory,"global-diagnostics.jls"),
                (;global_diagnostics,divergences,signature))
            global_ok = reference_mixing_ok(global_diagnostics) && divergences == 0
            @test global_ok
            global_ok || error("exact marginal reference failed its diagnostic gate")
            records = with_progress(progress;description="Restore joint Gaussian latents") do restoration
                map(global_record.chains) do chain
                    saved_chain = joinpath(directory,"reference-$(chain.chain).jls")
                    isfile(saved_chain) && return deserialize(saved_chain)
                    n = size(chain.position,2); q = length(p.latent)
                    position = zeros(length(p.start),n)
                    conditional_mean,conditional_second = zeros(q,n),zeros(q,n)
                    rng = Xoshiro(900+chain.chain)
                    with_progress(restoration,n;description="Chain $(chain.chain)") do child
                        for j in 1:n
                            g = chain.position[:,j]
                            c = gaussian_reference_conditional(p,g)
                            C = c.F.U \ Matrix{Float64}(I,q,q)
                            position[p.globals,j] = g
                            position[p.latent,j] = c.mu+C*randn(rng,q)
                            conditional_mean[:,j] = c.mu
                            conditional_second[:,j] = c.mu.^2+vec(sum(abs2,C;dims=2))
                            update_progress!(child,j)
                        end
                    end
                    record = (;position,conditional_mean,conditional_second,
                        chain=chain.chain,divergences=chain.divergences,signature)
                    serialize(saved_chain,record)
                    record
                end
            end
            @test all(c -> c.signature == signature,records)
            joint_diagnostics = reference_diagnostics([r.position for r in records])
            # Gaussian restoration noise can conceal unmixed global scales.
            # The conditional first AND raw second moments must also mix.
            moment_diagnostics = reference_diagnostics([
                vcat(r.conditional_mean,r.conditional_second) for r in records])
            reference_ok = reference_mixing_ok(joint_diagnostics) &&
                reference_mixing_ok(moment_diagnostics)
            combined = reduce(hcat,(r.position for r in records))
            ref_mean = vec(mean(combined;dims=2))
            ref_second = vec(mean(abs2,combined;dims=2))
            ref_mean[p.latent] = vec(mean(reduce(hcat,(r.conditional_mean for r in records));dims=2))
            ref_second[p.latent] = vec(mean(reduce(hcat,(r.conditional_second for r in records));dims=2))
            ref_sd = sqrt.(ref_second-ref_mean.^2)
            summary = (;reference_ok,global_diagnostics,joint_diagnostics,moment_diagnostics,
                divergences,ref_mean,ref_sd,coordinates=view.coordinates,
                field=fixture.field,units=fixture.units,globals=p.globals,signature,
                scope="exact small Gaussian marginal; four of four chains required")
            serialize(joinpath(directory,"reference-summary.jls"),summary)
            println((;stage="restored reference",joint_diagnostics,moment_diagnostics,reference_ok))
            @test reference_ok
        end
    end
end
