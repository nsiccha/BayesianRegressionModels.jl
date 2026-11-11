# Summarize a saved, rejected diagnostic run; this script never samples.
using BayesianRegressionModels, StanBlocks, BridgeStan, WarmupHMC, Enzyme
using Serialization, Statistics, Random, MCMCDiagnosticTools, SHA, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
include("grey_seal_brm.jl")
include("../mcmc_harness.jl")

# Call the model's actual scalar helper at each point of the display grid.
StanBlocks.@deffun density_curve(N::vector[n], b0::real, theta0::real, theta1::real)::vector[n] = begin
    birth::vector[n]
    for j in 1:n
        birth[j] = update_birth_rate(b0, theta0, theta1, N[j])
    end
    birth
end

function density_response_table(results)
    input = (; N=collect(0.0:1500.0:150000.0), b_max=0.9, loss_scaled=0.4,
        b_reference=0.7, b_capacity=0.2, probe=0.0)
    model = @slic input begin
        log_K ~ normal(log(60000.0), 0.3)
        theta0 = compute_density_dependence_intercept(b_max, loss_scaled)
        theta1 = compute_density_dependence_slope(exp(log_K), b_capacity, b_reference, theta0)
        births = density_curve(N, b_reference, theta0, theta1)
        # Retain log_K as a parameter and births as transformed parameters.
        # This is a numerical probe: no density is used and no fit is run.
        probe ~ normal(sum(births), 1.0)
    end
    problem = StanBlocks.stan_instantiate(model)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=false)
    coords = [only(findall(==("births.$j"), names)) for j in eachindex(input.N)]
    rows = Tuple[]
    for K in (30000.0, 60000.0, 120000.0)
        values = BridgeStan.param_constrain(problem.model, [log(K)]; include_tp=true)
        birth = values[coords]
        @assert birth[1] ≈ input.b_reference
        @assert birth[only(findall(==(K), input.N))] ≈ input.b_capacity
        @assert all(diff(birth) .<= 0)
        append!(rows, [(Int(K), Int(N), round(b; sigdigits=8)) for (N,b) in zip(input.N, birth)])
    end
    write_tsv(joinpath(results, "density_response.tsv"), ["capacity", "population", "birth_rate"], rows)
end

function summarize_seal(checkpoints; results=joinpath(@__DIR__, "results"))
    mkpath(results)
    sb = SBBRMI(grey_seal_brm_model())
    problem = StanBlocks.stan_instantiate(sb)
    # Same target and centering structure as test/seal_density_fit.jl.
    target = adaptive_centering_problem(sb, problem, AutoEnzyme())
    descriptor = brm_descriptor(sb; name=:seal)
    names = constrained_names(problem)
    counts = grey_seal_brm_fixture().obs_aerial_count
    diagnostics, predictions, hashes = Tuple[], Tuple[], Tuple[]
    constrained = Matrix{Float64}[]
    for chain in 1:4
        file = joinpath(checkpoints, "chain_$chain", "cp_latest.jls")
        before = bytes2hex(sha256(read(file)))
        cp = deserialize(file)
        draws = WarmupHMC.back_transform(cp, target, cp.posterior_position)
        saved = size(draws, 2)
        saved == 300 || error("expected the strict 300-state run; chain $chain has $saved")
        divergent = cp.n_divergent_samples
        divergent / saved > 0.1 || error("this script describes the rejected run, not a newly accepted fit")
        for q in eachcol(draws)
            lp, gradient = BridgeStan.log_density_gradient(problem.model, collect(q))
            @assert isfinite(lp) && all(isfinite, gradient)
        end
        C = constrain_chain(problem, draws, 20261111 + chain)
        push!(constrained, C)
        simulated = brm_output_draws(descriptor, C, names;
            logical=:obs_aerial_count, role=:posterior_predictive)
        for year in eachindex(counts)
            qs = quantile(simulated[:, year], [0.025, 0.25, 0.5, 0.75, 0.975])
            push!(predictions, (chain, year, counts[year], round.(Int, qs)...))
        end
        push!(diagnostics, (chain, saved, 0, divergent,
            round(100divergent/saved; digits=1), "rejected"))
        @assert before == bytes2hex(sha256(read(file)))
        push!(hashes, (chain, before))
        println("SEAL_SUMMARY chain=$chain saved=$saved divergences=$divergent accepted=0")
        flush(stdout)
    end
    write_tsv(joinpath(results, "chains.tsv"),
        ["chain", "saved_states", "accepted_states", "divergences", "divergence_pct", "status"], diagnostics)
    write_tsv(joinpath(results, "aerial_simulations.tsv"),
        ["chain", "year", "observed", "q025", "q25", "q50", "q75", "q975"], predictions)
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
    write_tsv(joinpath(results, "diagnostics.tsv"), ["quantity", "unit", "rhat", "ess_bulk", "ess_tail"], rows)
    density_response_table(results)
    write_tsv(joinpath(results, "receipt.tsv"), ["key", "value"], [
        ("status", "rejected: exploratory simulations only; no accepted posterior fit"),
        ("sampled_commit", "21c3e34ab807c812dbfd546252415ff43313a77a"),
        ("data", "synthetic; 3 ages x 2 sexes x 3 years"),
        ("seed", "20261111"), ("saved_states", "4 chains x 300; zero admitted"),
        ("target_acceptance_rate", "0.99"), ("divergence_admission_limit", "0.10"),
        ("warmup_divergence_gate", "0.10; two restarts per chain"),
        ("program_sha256", bytes2hex(sha256(stan_code(sb)))),
        ("data_sha256", bytes2hex(sha256(repr(grey_seal_brm_fixture())))),
        ("StanBlocks", string(pkgversion(StanBlocks))),
        ("WarmupHMC", string(pkgversion(WarmupHMC))),
        ("BridgeStan", string(pkgversion(BridgeStan))),
        ("summary_julia", string(VERSION)),
        ("summary_method", "back_transform with exact target; no sampling; checkpoints unchanged"),
        [("checkpoint_chain_$i" , h) for (i,h) in hashes]...,
    ])
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=test research/seal/summarize.jl CHECKPOINT_DIRECTORY")
    summarize_seal(only(ARGS))
end
