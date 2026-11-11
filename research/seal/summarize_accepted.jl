# Post-process the admitted fit only. Never samples or edits checkpoints.
using JSON
const repo = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(repo, "research", "seal", "summarize.jl"))

function summarize_accepted_seal(checkpoints, results)
    draws_file = joinpath(checkpoints, "model-frame-draws.jls")
    ispath(results) && error("refusing to overwrite an existing result directory")
    expected_source = first(split(first(readlines(joinpath(checkpoints, "source-hashes.txt")))))
    @assert bytes2hex(sha256(read(joinpath(@__DIR__, "grey_seal_brm.jl")))) == expected_source
    summary_file = joinpath(checkpoints, "completion_attempts", "attempt_1", "run_summary.json")
    admission = JSON.parsefile(summary_file)
    @assert admission["quorum_met"] && admission["completed_chain_indices"] == [1,2,3,4]
    @assert admission["n_omitted"] == 0
    @assert admission["min_completed"] == 3 && admission["max_admitted_divergent_fraction"] == 0.1
    @assert admission["n_draws"] == 1500
    cache_hash = bytes2hex(sha256(read(draws_file)))
    saved = deserialize(draws_file) # sample_quorum cache: model-frame draws
    @assert saved.n_chains == 4 && length(saved.chains) == 4 && isempty(saved.omitted)
    @assert saved.divergent == admission["n_divergent_samples"]
    @assert [size(c, 2) for c in saved.chains] == admission["chain_n_retained_samples"]
    sb = SBBRMI(grey_seal_brm_model())
    @assert bytes2hex(sha256(stan_code(sb))) == strip(read(joinpath(checkpoints,"program-sha256.txt"),String))
    @assert bytes2hex(sha256(repr(grey_seal_brm_fixture()))) == strip(read(joinpath(checkpoints,"data-sha256.txt"),String))
    problem = StanBlocks.stan_instantiate(sb)
    target = adaptive_centering_problem(sb, problem, AutoEnzyme())
    descriptor = brm_descriptor(sb; name=:seal)
    names = constrained_names(problem)
    counts = grey_seal_brm_fixture().obs_aerial_count
    constrained = Matrix{Float64}[]
    diagnostics, predictions, hashes = Tuple[], Tuple[], Tuple[]
    for (chain, draws) in enumerate(saved.chains)
        file = joinpath(checkpoints, "chain_$chain", "cp_latest.jls")
        before = bytes2hex(sha256(read(file)))
        cp = deserialize(file)
        @assert size(cp.posterior_position, 2) == size(draws, 2)
        divergent = cp.n_divergent_samples
        @assert divergent == round(Int, admission["chain_divergent_fractions"][chain] * size(draws,2))
        restored = WarmupHMC.back_transform(cp, target, cp.posterior_position)
        @assert isapprox(restored, draws; atol=1e-12, rtol=1e-12)
        C = constrain_chain(problem, draws, 20261111 + chain)
        @assert all(isfinite, C)
        push!(constrained, C)
        simulated = brm_output_draws(descriptor, C, names;
            logical=:obs_aerial_count, role=:posterior_predictive)
        for year in eachindex(counts)
            qs = quantile(simulated[:, year], [0.025, 0.25, 0.5, 0.75, 0.975])
            push!(predictions, (chain, year, counts[year], round.(Int, qs)...))
        end
        push!(diagnostics, (chain, size(draws,2), size(draws,2), divergent,
            round(100divergent/size(draws,2); digits=3), "admitted"))
        @assert before == bytes2hex(sha256(read(file)))
        push!(hashes, (chain, before))
    end
    quantities = [
        (; name="Female adult survival", logical=:phi_a_sc, unit="fraction/year", transform=x -> 0.8 + 0.2x),
        (; name="Initial population", logical=:population_init_size, unit="seals", transform=identity),
        (; name="Carrying capacity", logical=:carrying_capacity, unit="seals", transform=identity),
        (; name="Aerial fraction", logical=:aerial_mu, unit="fraction", transform=identity),
    ]
    rows = map(quantities) do spec
        chains = quantity_draws(descriptor, names, constrained, merge(spec, (; kind=:scalar)))
        d = chain_diagnostics(chains)
        (spec.name, spec.unit, round(d.rhat; digits=3), round(Int, d.ess_bulk), round(Int, d.ess_tail))
    end
    all_diagnostics = [chain_diagnostics([collect(c[j,:]) for c in saved.chains])
        for j in axes(first(saved.chains), 1)]
    max_rhat = maximum(d -> d.rhat, all_diagnostics)
    min_bulk_ess = minimum(d -> d.ess_bulk, all_diagnostics)
    min_tail_ess = minimum(d -> d.ess_tail, all_diagnostics)
    @assert all(isfinite, (max_rhat, min_bulk_ess, min_tail_ess))
    mkpath(results)
    write_tsv(joinpath(results,"chains.tsv"),
        ["chain","saved_states","accepted_states","divergences","divergence_pct","status"],diagnostics)
    write_tsv(joinpath(results,"aerial_simulations.tsv"),
        ["chain","year","observed","q025","q25","q50","q75","q975"],predictions)
    write_tsv(joinpath(results,"diagnostics.tsv"),["quantity","unit","rhat","ess_bulk","ess_tail"],rows)
    density_response_table(results)
    write_tsv(joinpath(results,"receipt.tsv"),["key","value"],[
        ("status","admitted: four chains; no omitted chain"),
        ("sampled_commit",strip(read(joinpath(checkpoints,"source-commit.txt"),String))),
        ("data","synthetic; 3 ages x 2 sexes x 3 years"),
        ("seed","20261111"),("saved_states",join([size(c,2) for c in saved.chains],", ") * "; all four chains admitted"),
        ("target_acceptance_rate","0.99"),("divergence_admission_limit","0.10"),
        ("warmup_divergence_gate","0.10"),
        ("divergences",string(saved.divergent," (",round(100saved.divergent/admission["n_samples"];digits=5),"%)")),
        ("max_rhat_all_coordinates",max_rhat),("min_bulk_ess_all_coordinates",min_bulk_ess),
        ("min_tail_ess_all_coordinates",min_tail_ess),
        ("program_sha256",bytes2hex(sha256(stan_code(sb)))),
        ("data_sha256",bytes2hex(sha256(repr(grey_seal_brm_fixture())))),
        ("model_frame_draw_cache_sha256",cache_hash),
        ("completion_summary_sha256",bytes2hex(sha256(read(summary_file)))),
        ("StanBlocks",string(pkgversion(StanBlocks))),("WarmupHMC",string(pkgversion(WarmupHMC))),
        ("BridgeStan",string(pkgversion(BridgeStan))),("summary_julia",string(VERSION)),
        ("summary_method","saved model-frame draws; generated quantities seeds 20261112-20261115; no sampling; checkpoints unchanged"),
        [("checkpoint_chain_$i",h) for (i,h) in hashes]...,
    ])
    @assert cache_hash == bytes2hex(sha256(read(draws_file)))
    println("SEAL_ACCEPTED_SUMMARY results=",results," max_rhat=",max_rhat,
        " min_bulk_ess=",min_bulk_ess," min_tail_ess=",min_tail_ess)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: summarize_accepted.jl CHECKPOINTS NEW_RESULTS_DIRECTORY")
    summarize_accepted_seal(ARGS...)
end
