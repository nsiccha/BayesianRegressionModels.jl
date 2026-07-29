#!/usr/bin/env julia

const ROOT = @__DIR__
const HISTORICAL = joinpath(ROOT, "historical_catalog.tsv")
const TRANSLATIONS = joinpath(ROOT, "translations.tsv")
const CAPABILITY = joinpath(ROOT, "capability_results.tsv")
const RECEIPTS = joinpath(ROOT, "receipts", "row_receipt_audit.tsv")
const MANUAL_RECEIPTS = joinpath(ROOT, "receipts", "manual_review.tsv")
const DATASET_RECEIPTS = joinpath(ROOT, "receipts", "row_dataset_receipts.tsv")
const CANDIDATE_FAMILY_CONTROLS = joinpath(ROOT, "candidate_family_controls.tsv")
const ALL_OUTPUT = joinpath(ROOT, "all_source_rows_matrix.tsv")
const DEPLOYED_OUTPUT = joinpath(ROOT, "model_matrix.tsv")
const SOURCE_SUMMARY_OUTPUT = joinpath(ROOT, "final_summary_by_source.tsv")

tsv_unescape(value) = replace(value,
    "\\t" => "\t", "\\n" => "\n", "\\r" => "\r", "\\\\" => "\\")
tsv_escape(value) = replace(string(value),
    '\\' => "\\\\", '\t' => "\\t", '\n' => "\\n", '\r' => "\\r")

function read_tsv(path; escaped=true)
    lines = readlines(path)
    columns = split(first(lines), '\t'; keepempty=true)
    [Dict(columns .=> (escaped ? tsv_unescape.(values) : String.(values)))
     for line in Iterators.drop(lines, 1)
     for values in (split(line, '\t'; keepempty=true),)]
end

function write_tsv(path, columns, rows)
    open(path, "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            println(io, join((tsv_escape(get(row, column, "")) for column in columns), '\t'))
        end
    end
end

function capability_tier(translation, capability)
    status = translation["translation_status"]
    status == "ready" || return status
    capability["warmuphmc"] == "pass" && return "warmuphmc-sampled"
    capability["gradient_finite"] == "true" && return "bridgestan-finite-density-gradient"
    capability["bridgestan_instantiate"] == "pass" && return "bridgestan-nonfinite-or-runtime-error"
    capability["stanc"] == "pass" && return "stanc-accepted"
    stage = capability["static_error_stage"]
    isempty(stage) ? "not-run" : "failed-$stage"
end

function projection_contract(route, status)
    if route == "ordinary_brm" && status == "ready"
        return "authoritative brm_descriptor/brm_execute; experimental SbPMX semantic_app mount adapter"
    elseif route == "brm_kernel"
        return "authoritative kernel-capable BRMDescriptor substrate; row-specific experimental semantic_app adapter required"
    elseif route == "stanblocks_plate"
        return "StanBlocks plate substrate; no authoritative BRM descriptor for this row; experimental semantic_app adapter required"
    end
    "no executable descriptor until translation is resolved"
end

historical = read_tsv(HISTORICAL)
translations = read_tsv(TRANSLATIONS)
capabilities = read_tsv(CAPABILITY)
receipts = read_tsv(RECEIPTS; escaped=false)
manual_receipts = read_tsv(MANUAL_RECEIPTS; escaped=false)
dataset_receipts = read_tsv(DATASET_RECEIPTS; escaped=false)
candidate_family_control = only(read_tsv(CANDIDATE_FAMILY_CONTROLS; escaped=false))
candidate_control_families = Set(["zero_inflated_poisson", "negativebinomial", "student_t"])

translation_by = Dict((row["row_index"], row["variant"]) => row for row in translations)
capability_by = Dict((row["row_index"], row["variant"]) => row for row in capabilities)
receipt_by = Dict(replace(row["row_key"], '/' => ':') => row for row in receipts)
manual_by = Dict(replace(row["row_key"], '/' => ':') => row for row in manual_receipts)
datasets_by = Dict{String,Vector{Dict{String,String}}}()
for row in dataset_receipts
    push!(get!(datasets_by, replace(row["row_key"], '/' => ':'), Dict{String,String}[]), row)
end

rows = Dict{String,String}[]
for historical_row in historical
    index = historical_row["row_index"]
    row_key = historical_row["source"] * ":" * historical_row["key"]
    exact = translation_by[(index, "exact-metadata")]
    inferred = translation_by[(index, "inferred-family")]
    exact_cap = capability_by[(index, "exact-metadata")]
    inferred_cap = capability_by[(index, "inferred-family")]
    receipt = receipt_by[row_key]
    manual = get(manual_by, row_key, nothing)
    dataset_rows = get(datasets_by, row_key, Dict{String,String}[])

    inferred_family = inferred["family_selected"]
    candidate_family_applies = inferred_family in candidate_control_families
    renderer_family = receipt["deployed_renderer_family_claim"]
    family_discrepancy = if receipt["deployed_family_provenance"] == "explicit_metadata"
        inferred_family == renderer_family ? "none" : "explicit-metadata-vs-inference-review"
    elseif inferred["family_selected_provenance"] == "authoritative-audit-genuinely-indeterminate"
        "renderer-default-unsubstantiated; authoritative audit genuinely indeterminate"
    elseif isempty(inferred_family)
        "renderer-default-unsubstantiated; semantic family unresolved"
    elseif inferred_family == renderer_family
        "renderer-default-unsubstantiated; inference happens to agree"
    else
        "renderer-default-contradicts semantic inference"
    end

    push!(rows, Dict(
        "row_index" => index,
        "deployed" => historical_row["deployed"],
        "historical_sha" => historical_row["historical_sha"],
        "source" => historical_row["source"],
        "key" => historical_row["key"],
        "row_key" => row_key,
        "source_path" => historical_row["source_path"],
        "source_line" => historical_row["source_line"],
        "name_claim" => historical_row["name_claim"],
        "dataset_claim" => historical_row["dataset_claim"],
        "formula_claim" => historical_row["formula_claim"],
        "formula_return" => historical_row["formula_return"],
        "formula_claim_matches_return" => historical_row["formula_match"],
        "family_metadata_claim" => historical_row["family_claim"],
        "family_description_claim" => historical_row["family_text_claim"],
        "renderer_family_claim" => renderer_family,
        "renderer_family_provenance" => receipt["deployed_family_provenance"],
        "inferred_family" => inferred_family,
        "inferred_family_provenance" => inferred["family_selected_provenance"],
        "family_audit_evidence_class" => inferred["family_audit_evidence_class"],
        "family_audit_evidence_strength" => inferred["family_audit_evidence_strength"],
        "family_audit_authoritative_surface" => inferred["family_audit_authoritative_surface"],
        "family_audit_authoritative_revision" => inferred["family_audit_authoritative_revision"],
        "family_audit_authoritative_path_anchor" => inferred["family_audit_authoritative_path_anchor"],
        "family_audit_retrieved_at_utc" => inferred["family_audit_retrieved_at_utc"],
        "family_audit_http_status" => inferred["family_audit_http_status"],
        "family_audit_explicit_evidence" => inferred["family_audit_explicit_evidence"],
        "family_audit_semantic_evidence" => inferred["family_audit_semantic_evidence"],
        "family_audit_dataset_evidence" => inferred["family_audit_dataset_evidence"],
        "family_audit_negative_evidence" => inferred["family_audit_negative_evidence"],
        "family_audit_decision_rationale" => inferred["family_audit_decision_rationale"],
        "family_discrepancy" => family_discrepancy,
        "row_source_claim" => historical_row["row_source_claim"],
        "receipt_retrieved_at" => receipt["retrieved_at"],
        "receipt_http_status" => receipt["http_status"],
        "receipt_final_url" => receipt["final_url"],
        "citation_authority_class" => receipt["citation_authority_class"],
        "data_only_citation" => receipt["data_only_citation"],
        "formula_support" => receipt["formula_support"],
        "family_support" => receipt["family_support"],
        "dataset_support" => receipt["dataset_support"],
        "source_fidelity_first_pass" => receipt["semantic_support_verdict"],
        "source_fidelity_manual_reviewed" => string(!isnothing(manual)),
        "source_fidelity_verdict" => isnothing(manual) ? receipt["semantic_support_verdict"] : manual["manual_verdict"],
        "source_fidelity_reason" => isnothing(manual) ? receipt["semantic_support_reason"] : manual["manual_support_reason"],
        "source_correction_with_evidence" => isnothing(manual) ? receipt["correction_with_evidence"] : manual["manual_correction"],
        "receipt_evidence_anchor" => isnothing(manual) ? receipt["evidence_anchor"] : manual["manual_evidence_anchor"],
        "receipt_body_sha256" => receipt["body_sha256"],
        "dataset_receipt_urls" => join(get.(dataset_rows, "dataset_source_url", ""), " | "),
        "dataset_receipt_statuses" => join(get.(dataset_rows, "dataset_receipt_status", ""), " | "),
        "semantic_route" => inferred["semantic_route"],
        "semantic_route_note" => inferred["route_note"],
        "exact_translation_status" => exact["translation_status"],
        "exact_surface_support_class" => exact["surface_support_class"],
        "exact_surface_secondary_gap" => exact["surface_secondary_gap"],
        "exact_translation_note" => exact["translation_note"],
        "exact_current_brm_body" => exact["current_brm_body"],
        "exact_capability_tier" => capability_tier(exact, exact_cap),
        "exact_probe_id" => exact_cap["probe_id"],
        "exact_evidence_kind" => exact_cap["evidence_kind"],
        "inferred_translation_status" => inferred["translation_status"],
        "inferred_surface_support_class" => inferred["surface_support_class"],
        "inferred_surface_secondary_gap" => inferred["surface_secondary_gap"],
        "family_adapter_landed_applies" => string(candidate_family_applies),
        "family_adapter_validation_sha" => candidate_family_applies ? candidate_family_control["candidate_sha"] : "",
        "family_adapter_clean_reviewed_sha" => candidate_family_applies ? "a707af21d138b0019810f8dce9d655109dc97ff6" : "",
        "family_adapter_canonical_sha" => candidate_family_applies ? "11031f2d3bbd0c9cad42bed53a4a8dd193ab9d2e" : "",
        "family_adapter_control_tier" => candidate_family_applies ? "landed-surface-control-stanc-bridgestan-finite-predict-pointwise" : "",
        "family_adapter_control_scope" => candidate_family_applies ? "combined synthetic family-surface control only; never inherited as row validation" : "",
        "inferred_translation_note" => inferred["translation_note"],
        "inferred_current_brm_body" => inferred["current_brm_body"],
        "inferred_capability_tier" => capability_tier(inferred, inferred_cap),
        "probe_id" => inferred_cap["probe_id"],
        "probe_evidence_kind" => inferred_cap["evidence_kind"],
        "probe_evidence_from_row" => inferred_cap["evidence_from"],
        "data_shape_assumptions" => inferred_cap["data_shape_assumptions"],
        "brmi_parse" => inferred_cap["brmi_parse"],
        "brmi_eval" => inferred_cap["brmi_eval"],
        "sbbrmi_lower" => inferred_cap["sbbrmi_lower"],
        "descriptor" => inferred_cap["descriptor"],
        "stan_transpile" => inferred_cap["stan_transpile"],
        "stanc" => inferred_cap["stanc"],
        "stan_code_sha256" => inferred_cap["stan_code_sha256"],
        "stan_data_sha256" => inferred_cap["stan_data_sha256"],
        "bridgestan_instantiate" => inferred_cap["bridgestan_instantiate"],
        "dimension" => inferred_cap["dimension"],
        "log_density" => inferred_cap["log_density"],
        "gradient_finite" => inferred_cap["gradient_finite"],
        "warmuphmc" => inferred_cap["warmuphmc"],
        "warmuphmc_draws" => inferred_cap["warmuphmc_draws"],
        "failure_stage" => inferred_cap["static_error_stage"],
        "failure_evidence" => isempty(inferred_cap["static_error"]) ?
            inferred_cap["runtime_error"] : inferred_cap["static_error"],
        "descriptor_mount_contract" => projection_contract(
            inferred["semantic_route"], inferred["translation_status"]),
        "sbpmx_semantic_pattern" => "@options + option_domain/option_records + semantic_card under semantic_app",
        "sbpmx_pattern_provenance" => "SbPMX origin/main@a8fd3c02e1e2305683a56c91384701bfeba6b528:design/{ASKS.md,SbPMX.jl,mock_runtime.jl},compat/check-semantic-model-pipeline.jl",
    ))
end

columns = collect(keys(first(rows)))
write_tsv(ALL_OUTPUT, columns, rows)
deployed = filter(row -> row["deployed"] == "true", rows)
write_tsv(DEPLOYED_OUTPUT, columns, deployed)

source_summary = Dict{String,String}[]
for source in sort(unique(get.(deployed, "source", "")))
    source_rows = filter(row -> row["source"] == source, deployed)
    push!(source_summary, Dict(
        "source" => source,
        "rows" => string(length(source_rows)),
        "confirmed" => string(count(row -> row["source_fidelity_verdict"] == "confirmed", source_rows)),
        "adapted_but_defensible" => string(count(row -> row["source_fidelity_verdict"] == "adapted-but-defensible", source_rows)),
        "mismatch" => string(count(row -> row["source_fidelity_verdict"] == "mismatch", source_rows)),
        "unverifiable" => string(count(row -> row["source_fidelity_verdict"] == "unverifiable", source_rows)),
        "dead_source" => string(count(row -> row["source_fidelity_verdict"] == "dead-source", source_rows)),
        "ordinary_brm" => string(count(row -> row["semantic_route"] == "ordinary_brm", source_rows)),
        "brm_kernel" => string(count(row -> row["semantic_route"] == "brm_kernel", source_rows)),
        "stanblocks_plate" => string(count(row -> row["semantic_route"] == "stanblocks_plate", source_rows)),
        "ready_translation" => string(count(row -> row["inferred_translation_status"] == "ready", source_rows)),
        "stanc_accepted" => string(count(row -> row["stanc"] == "pass", source_rows)),
        "finite_density_gradient" => string(count(row -> row["gradient_finite"] == "true", source_rows)),
        "surface_verbatim" => string(count(row -> row["inferred_surface_support_class"] == "already-expressible-verbatim", source_rows)),
        "surface_semantic_rewrite" => string(count(row -> row["inferred_surface_support_class"] == "already-expressible-via-semantic-rewrite", source_rows)),
        "surface_missing_brm" => string(count(row -> row["inferred_surface_support_class"] == "genuinely-missing-brm-surface", source_rows)),
        "surface_missing_stanblocks" => string(count(row -> row["inferred_surface_support_class"] == "genuinely-missing-stanblocks-substrate", source_rows)),
        "surface_historical_unresolved" => string(count(row -> row["inferred_surface_support_class"] == "historical-semantics-unresolved", source_rows)),
        "audit_explicit_family_recovery" => string(count(row -> row["family_audit_evidence_class"] == "explicit-family-recovery", source_rows)),
        "audit_defensible_semantic_inference" => string(count(row -> row["family_audit_evidence_class"] == "defensible-semantic-inference", source_rows)),
        "audit_genuinely_indeterminate" => string(count(row -> row["family_audit_evidence_class"] == "genuinely-indeterminate", source_rows)),
    ))
end
source_columns = collect(keys(first(source_summary)))
write_tsv(SOURCE_SUMMARY_OUTPUT, source_columns, source_summary)

function counts(values)
    result = Dict{String,Int}()
    for value in values
        result[value] = get(result, value, 0) + 1
    end
    join(("$key=$(result[key])" for key in sort(collect(keys(result)))), ",")
end

println("all_source_rows=$(length(rows))")
println("deployed_rows=$(length(deployed))")
println("source_fidelity=" * counts(get.(deployed, "source_fidelity_verdict", "")))
println("semantic_routes=" * counts(get.(deployed, "semantic_route", "")))
println("inferred_translation=" * counts(get.(deployed, "inferred_translation_status", "")))
println("inferred_capability=" * counts(get.(deployed, "inferred_capability_tier", "")))
println("surface_support=" * counts(get.(deployed, "inferred_surface_support_class", "")))
println("family_audit=" * counts(get.(deployed, "family_audit_evidence_class", "")))
println("output=$(abspath(DEPLOYED_OUTPUT))")
