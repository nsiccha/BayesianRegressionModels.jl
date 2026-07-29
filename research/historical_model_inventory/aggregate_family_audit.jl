#!/usr/bin/env julia

const ROOT = @__DIR__
const DEFAULT_CATALOG = joinpath(ROOT, "historical_catalog.tsv")
const DEFAULT_OUTPUT = joinpath(ROOT, "family_audit.tsv")

tsv_unescape(value) = replace(value,
    "\\t" => "\t", "\\n" => "\n", "\\r" => "\r", "\\\\" => "\\")
tsv_escape(value) = replace(string(value),
    '\\' => "\\\\", '\t' => "\\t", '\n' => "\\n", '\r' => "\\r")

function read_tsv(path; escaped=false)
    lines = readlines(path)
    isempty(lines) && error("empty TSV: $path")
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

# The audit lanes retain upstream family spelling in their evidence columns.
# This field is the stable inventory vocabulary consumed by translate.jl.
const FAMILY_ALIASES = Dict(
    "beta_binomial2" => "beta_binomial",
    "gaussian (tarsus); gaussian (back)" => "multivariate_gaussian",
    "skew_normal (tarsus); gaussian (back)" => "multivariate_skew_normal_gaussian",
    "normal (trial-level report); linked population regression" =>
        "action_gaussian_composite",
    "categorical (trial-level deck choice); linked population regression" =>
        "action_categorical_composite",
)

function aggregate_family_audit(lanes;
                                catalog=DEFAULT_CATALOG,
                                output=DEFAULT_OUTPUT)
    isempty(lanes) && error("provide one or more lane audit TSVs")
    audit_rows = reduce(vcat, read_tsv.(lanes))
    columns = collect(keys(first(audit_rows)))
    required = [
        "row_key", "recovered_family", "evidence_class", "evidence_strength",
        "authoritative_surface", "authoritative_revision",
        "authoritative_path_anchor", "retrieved_at_utc", "http_status",
        "explicit_evidence", "semantic_evidence", "dataset_evidence",
        "negative_evidence", "decision_rationale",
    ]
    Set(columns) == Set(required) ||
        error("lane schema mismatch: $(join(columns, ','))")

    by_key = Dict{String,Dict{String,String}}()
    for row in audit_rows
        key = row["row_key"]
        haskey(by_key, key) && error("duplicate row_key across lanes: $key")
        row["recovered_family"] = get(
            FAMILY_ALIASES, row["recovered_family"], row["recovered_family"])
        by_key[key] = row
    end

    catalog_rows = read_tsv(catalog; escaped=true)
    catalog_keys = [row["source"] * ":" * row["key"] for row in catalog_rows]
    audited_keys = Set(keys(by_key))
    unknown = setdiff(audited_keys, Set(catalog_keys))
    isempty(unknown) || error("audit contains unknown rows: $(join(sort(collect(unknown)), ','))")
    ordered_keys = filter(in(audited_keys), catalog_keys)
    length(ordered_keys) == length(audit_rows) ||
        error("audit-key ordering lost rows: ordered=$(length(ordered_keys)) audit=$(length(audit_rows))")

    ordered = [by_key[key] for key in ordered_keys]
    write_tsv(output, required, ordered)
    println("family_audit_rows=$(length(ordered))")
    for class in sort(unique(row["evidence_class"] for row in ordered))
        class_count = count(row -> row["evidence_class"] == class, ordered)
        println("$class=$class_count")
    end
    println("genuinely_indeterminate=" * join(
        (row["row_key"] for row in ordered
         if row["evidence_class"] == "genuinely-indeterminate"), ','))
    println("output=$(abspath(output))")
    ordered
end

if abspath(PROGRAM_FILE) == @__FILE__
    aggregate_family_audit(ARGS)
end
