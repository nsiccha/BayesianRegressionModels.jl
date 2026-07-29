#!/usr/bin/env julia

# Focused executable refresh for the two historical Beta-binomial rows. The
# combined candidate control separately exercises both public parameterizations;
# this harness gives each translated mean/precision formula its own receipt.

using BayesianRegressionModels
using LogDensityProblems
using StanBlocks

include(joinpath(@__DIR__, "probe.jl"))

const TARGETS = Set([
    "brms:cbpp_beta_binomial",
    "mcelreath:ucbadmit_beta_binomial",
])

length(ARGS) >= 1 || error(
    "usage: refresh_beta_binomial.jl <expected-brm-sha> [refresh-output.tsv] [capability-results.tsv]",
)

expected_brm_sha = ARGS[1]
refresh_output = length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "beta_binomial_refresh.tsv")
capability_path = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(@__DIR__, "capability_results.tsv")
translation_path = joinpath(@__DIR__, "translations.tsv")
control_path = joinpath(@__DIR__, "beta_binomial_surface_control.tsv")

brm_root = dirname(dirname(pathof(BayesianRegressionModels)))
brm_sha = readchomp(`git -C $brm_root rev-parse HEAD`)
brm_sha == expected_brm_sha ||
    error("active BRM SHA $brm_sha != expected $expected_brm_sha")
stanblocks_root = dirname(dirname(pathof(StanBlocks)))
stanblocks_sha = readchomp(`git -C $stanblocks_root rev-parse HEAD`)

candidate_control = only(read_tsv(control_path))
candidate_control["candidate_sha"] == brm_sha ||
    error("candidate control SHA $(candidate_control["candidate_sha"]) != active $brm_sha")
candidate_control["stanblocks_sha"] == stanblocks_sha ||
    error("candidate control StanBlocks SHA $(candidate_control["stanblocks_sha"]) != active $stanblocks_sha")

translations = read_tsv(translation_path)
targets = filter(translations) do row
    row["variant"] == "inferred-family" &&
        string(row["source"], ':', row["key"]) in TARGETS
end
length(targets) == length(TARGETS) ||
    error("expected $(length(TARGETS)) target rows, found $(length(targets))")
Set(string(row["source"], ':', row["key"]) for row in targets) == TARGETS ||
    error("target row-key mismatch")
all(row -> row["family_selected"] == "beta_binomial" &&
           row["translation_status"] == "ready" &&
           occursin("BetaBinomial2", row["current_brm_body"]), targets) ||
    error("Beta-binomial translation contract drift")

capability_columns = split(first(readlines(capability_path)), '\t'; keepempty=true)
capabilities = read_tsv(capability_path)
capability_by = Dict(
    (row["source"], row["key"], row["variant"]) => row for row in capabilities
)

finite_values(values_iter) = all(
    value -> value isa Number ? isfinite(value) : all(isfinite, value),
    values_iter,
)

refresh_rows = Dict{String,String}[]
failed_rows = String[]

for row in targets
    row_key = string(row["source"], ':', row["key"])
    prior = capability_by[(row["source"], row["key"], row["variant"])]
    probe_id = sha16(row["current_brm_body"] * "\0" * row["data_columns"] * "\0" *
                     row["group_columns"] * "\0" * row["family_selected"])

    state = empty_state(probe_id)
    built = build_static(row, state)
    operations = ""
    prediction_outputs = ""
    pointwise_outputs = ""
    prediction_rows = ""
    pointwise_rows = ""
    prediction_finite = "not-run"
    pointwise_finite = "not-run"
    operation_error = ""

    if !isnothing(built) && state["stanc"] == "pass"
        try
            operation_names = Symbol[operation.name for operation in built.descriptor.operations]
            operations = join(string.(operation_names), ',')
            all(operation -> operation in operation_names,
                (:fit, :predict, :pointwise_loglik)) ||
                error("missing required operation; offered=$operation_names")

            problem = brm_execute(built.descriptor, :fit)
            state["bridgestan_instantiate"] = "pass"
            dimension = LogDensityProblems.dimension(problem)
            state["dimension"] = string(dimension)
            q = zeros(dimension)
            log_density, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
            state["log_density"] = string(log_density)
            state["gradient_finite"] = string(
                isfinite(log_density) && all(isfinite, gradient),
            )
            state["gradient_finite"] == "true" ||
                error("non-finite BridgeStan density/gradient at zero")

            prediction = brm_execute(
                built.descriptor, :predict; problem, draws=q, seed=20260729,
            )
            pointwise = brm_execute(
                built.descriptor, :pointwise_loglik; problem, draws=q, seed=20260729,
            )
            outcome = outcome_name(row["current_brm_body"])
            expected_prediction = Set([Symbol(outcome * "_gen")])
            expected_pointwise = Set([Symbol(outcome * "_likelihood")])
            Set(keys(prediction)) == expected_prediction ||
                error("unexpected prediction outputs $(keys(prediction))")
            Set(keys(pointwise)) == expected_pointwise ||
                error("unexpected pointwise outputs $(keys(pointwise))")

            prediction_outputs = join(sort!(string.(collect(keys(prediction)))), ',')
            pointwise_outputs = join(sort!(string.(collect(keys(pointwise)))), ',')
            prediction_rows = join(sort!(string.(length.(collect(values(prediction))))), ',')
            pointwise_rows = join(sort!(string.(length.(collect(values(pointwise))))), ',')
            prediction_finite = string(finite_values(values(prediction)))
            pointwise_finite = string(finite_values(values(pointwise)))
            prediction_finite == "true" || error("non-finite prediction output")
            pointwise_finite == "true" || error("non-finite pointwise output")
        catch err
            state["bridgestan_instantiate"] == "not-run" &&
                (state["bridgestan_instantiate"] = "fail")
            state["runtime_error"] = compact_error(err)
            operation_error = state["runtime_error"]
        end
    end

    passed = state["descriptor"] == "pass" && state["stanc"] == "pass" &&
        state["bridgestan_instantiate"] == "pass" &&
        state["gradient_finite"] == "true" && prediction_finite == "true" &&
        pointwise_finite == "true"
    passed || push!(failed_rows, row_key)

    for (key, value) in state
        prior[key] = value
    end
    for key in (
        "row_index", "deployed", "source", "key", "variant",
        "family_selected", "family_selected_provenance", "semantic_route",
        "translation_status", "surface_support_class", "surface_secondary_gap",
        "current_brm_body", "data_shape_assumptions",
    )
        prior[key] = row[key]
    end
    prior["historical_formula"] = row["formula_claim"]
    prior["evidence_kind"] = "direct"
    prior["evidence_from"] = ""

    push!(refresh_rows, Dict(
        "row_key" => row_key,
        "row_index" => row["row_index"],
        "probe_id" => probe_id,
        "candidate_sha" => brm_sha,
        "stanblocks_sha" => stanblocks_sha,
        "descriptor" => state["descriptor"],
        "stanc" => state["stanc"],
        "stan_code_sha256" => state["stan_code_sha256"],
        "stan_data_sha256" => state["stan_data_sha256"],
        "bridgestan_instantiate" => state["bridgestan_instantiate"],
        "dimension" => state["dimension"],
        "log_density_zero" => state["log_density"],
        "gradient_finite" => state["gradient_finite"],
        "descriptor_operations" => operations,
        "prediction_outputs" => prediction_outputs,
        "prediction_rows" => prediction_rows,
        "prediction_finite" => prediction_finite,
        "pointwise_outputs" => pointwise_outputs,
        "pointwise_rows" => pointwise_rows,
        "pointwise_finite" => pointwise_finite,
        "result" => passed ? "pass" : "fail",
        "error" => isempty(operation_error) ?
            (isempty(state["static_error"]) ? state["runtime_error"] : state["static_error"]) :
            operation_error,
    ))
end

refresh_columns = [
    "row_key", "row_index", "probe_id", "candidate_sha", "stanblocks_sha",
    "descriptor", "stanc", "stan_code_sha256", "stan_data_sha256",
    "bridgestan_instantiate", "dimension", "log_density_zero",
    "gradient_finite", "descriptor_operations", "prediction_outputs",
    "prediction_rows", "prediction_finite", "pointwise_outputs",
    "pointwise_rows", "pointwise_finite", "result", "error",
]
write_tsv(refresh_output, refresh_columns, refresh_rows)
write_tsv(capability_path, capability_columns, capabilities)

println("candidate_sha=$brm_sha")
println("stanblocks_sha=$stanblocks_sha")
println("target_rows=$(length(refresh_rows))")
passed_count = count(row -> row["result"] == "pass", refresh_rows)
println("passed=$passed_count")
println("refresh_output=$(abspath(refresh_output))")
println("capability_output=$(abspath(capability_path))")
isempty(failed_rows) || error(
    "focused Beta-binomial refresh failed for $(join(failed_rows, ','))",
)
