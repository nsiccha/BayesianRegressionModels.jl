#!/usr/bin/env julia

using Dates

const DEFAULT_SHA = "05c3f465e7987e8d7caa7e214fedddd90415a922"
const DEFAULT_OUTPUT = joinpath(@__DIR__, "historical_catalog.tsv")

const SOURCE_PATHS = Dict(
    :action_models => "scripts/examples/action_models.jl",
    :bambi => "scripts/examples/bambi.jl",
    :bmm => "scripts/examples/bmm.jl",
    :brms => "scripts/examples/brms.jl",
    :burkner_papers => "scripts/examples/burkner_papers.jl",
    :epidist => "scripts/examples/epidist.jl",
    :epinowcast => "scripts/examples/epinowcast.jl",
    :flocker => "scripts/examples/flocker.jl",
    :glm_jl => "scripts/examples/glm_jl.jl",
    :glmmtmb => "scripts/examples/glmmtmb.jl",
    :inla => "scripts/examples/inla.jl",
    :kruschke => "scripts/examples/kruschke.jl",
    :lme4 => "scripts/examples/lme4.jl",
    :mcelreath => "scripts/examples/mcelreath.jl",
    :mcmcglmm => "scripts/examples/mcmcglmm.jl",
    :mixed_models_jl => "scripts/examples/mixed_models_jl.jl",
    :mvgam => "scripts/examples/mvgam.jl",
    :rstanarm => "scripts/examples/rstanarm.jl",
    :vasishth => "scripts/examples/vasishth.jl",
)

git_show(sha, path) = read(Cmd(["git", "show", "$(sha):$(path)"]), String)

function yaml_scalar(raw::AbstractString)
    value = strip(raw)
    isempty(value) && return ""
    if startswith(value, '"') && endswith(value, '"')
        parsed = try
            Meta.parse(value)
        catch
            value
        end
        return parsed isa String ? parsed : value
    end
    if startswith(value, '\'') && endswith(value, '\'')
        return replace(value[nextind(value, firstindex(value)):prevind(value, lastindex(value))],
                       "''" => "'")
    end
    value == "true" && return true
    value == "false" && return false
    value
end

function split_claim_doc(doc::AbstractString)
    parts = split(doc, r"\n---+\n"; limit=2)
    header = first(parts)
    description = length(parts) == 2 ? last(parts) : ""
    metadata = Dict{String,Any}()
    for line in eachline(IOBuffer(header))
        m = match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        isnothing(m) && continue
        metadata[m.captures[1]] = yaml_scalar(m.captures[2])
    end
    metadata, description
end

function source_line(text::AbstractString, offset::Integer)
    offset <= firstindex(text) && return 1
    count(==('\n'), SubString(text, firstindex(text), prevind(text, offset))) + 1
end

function preceding_doc(text::AbstractString, offset::Integer)
    closing = findprev("\"\"\"", text, prevind(text, offset))
    isnothing(closing) && error("no closing docstring before source offset $offset")
    opening = findprev("\"\"\"", text, prevind(text, first(closing)))
    isnothing(opening) && error("no opening docstring before source offset $offset")
    doc_start = nextind(text, last(opening))
    doc_end = prevind(text, first(closing))
    strip(String(SubString(text, doc_start, doc_end))), first(opening)
end

function extract_dataset_receipts(text::AbstractString, path::AbstractString)
    receipts = Dict{String,NamedTuple}()
    pattern = r"(?:function\s+)?load\(::Val\{:(\w+)\}\)"
    for m in eachmatch(pattern, text)
        doc, doc_offset = preceding_doc(text, m.offset)
        metadata, _ = split_claim_doc(doc)
        key = m.captures[1]
        receipts[key] = (
            source=get(metadata, "source", ""),
            name=get(metadata, "name", ""),
            path=path,
            line=source_line(text, doc_offset),
        )
    end
    receipts
end

function extract_example(text::AbstractString, path::AbstractString, key::Symbol)
    key_string = string(key)
    header_pattern = Regex("function\\s+examples\\(::Val\\{:" * key_string * "\\}\\)")
    header = match(header_pattern, text)
    isnothing(header) && error("no documented examples(::Val{:$key}) method in $path")
    doc, doc_offset = preceding_doc(text, header.offset)
    metadata, description = split_claim_doc(doc)

    search_after = nextind(text, header.offset, length(header.match))
    next_method = match(r"\nfunction examples\(", text, search_after)
    block_end = isnothing(next_method) ? lastindex(text) : prevind(text, next_method.offset)
    block = SubString(text, header.offset, block_end)
    returned = match(r"(?s)return\s*\(\s*(\"(?:\\.|[^\"])*\")", block)
    isnothing(returned) && error("no literal formula return in $path for $key")
    returned_formula = Meta.parse(returned.captures[1])
    returned_formula isa String || error("formula return is not a string in $path for $key")

    family_text = let fm = match(r"(?i)\bfamily:\s*([A-Za-z0-9_-]+)", description)
        isnothing(fm) ? "" : lowercase(fm.captures[1])
    end
    (
        metadata=metadata,
        description=description,
        returned_formula=returned_formula,
        family_text=family_text,
        line=source_line(text, doc_offset),
    )
end

function tsv_escape(value)
    replace(string(value), '\\' => "\\\\", '\t' => "\\t", '\n' => "\\n", '\r' => "\\r")
end

function write_tsv(path, columns, rows)
    open(path, "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            println(io, join((tsv_escape(getproperty(row, column)) for column in columns), '\t'))
        end
    end
end

function extract_catalog(; sha=DEFAULT_SHA, output=DEFAULT_OUTPUT)
    all_path = "scripts/examples/all.jl"
    all_text = git_show(sha, all_path)
    catalog_pattern = r"\(source=:(\w+),\s*key=:(\w+)\)"
    catalog = [(source=Symbol(m.captures[1]), key=Symbol(m.captures[2]))
               for m in eachmatch(catalog_pattern, all_text)]

    source_text = Dict(source => git_show(sha, path) for (source, path) in SOURCE_PATHS)
    dataset_receipts = Dict(
        source => extract_dataset_receipts(source_text[source], SOURCE_PATHS[source])
        for source in keys(SOURCE_PATHS)
    )

    rows = NamedTuple[]
    for (row_index, entry) in enumerate(catalog)
        source, key = entry.source, entry.key
        path = get(SOURCE_PATHS, source) do
            error("catalog source $source has no source path")
        end
        example = extract_example(source_text[source], path, key)
        metadata = example.metadata
        formula_claim = string(get(metadata, "formula", ""))
        dataset_claim = string(get(metadata, "dataset", ""))
        dataset_receipt = get(dataset_receipts[source], dataset_claim, nothing)
        family_claim = lowercase(string(get(metadata, "family", "")))
        family_provenance = !isempty(family_claim) ? "explicit" :
            !isempty(example.family_text) ? "inferred-from-description" : "unresolved"
        hidden = get(metadata, "hidden", false) === true
        push!(rows, (
            row_index=row_index,
            deployed=!hidden,
            historical_sha=sha,
            source=source,
            key=key,
            source_path=path,
            source_line=example.line,
            name_claim=get(metadata, "name", ""),
            example_claim=get(metadata, "example", ""),
            dataset_claim=dataset_claim,
            dataset_source_claim=isnothing(dataset_receipt) ? "" : dataset_receipt.source,
            formula_claim=formula_claim,
            formula_return=example.returned_formula,
            formula_match=formula_claim == example.returned_formula,
            family_claim=family_claim,
            family_text_claim=example.family_text,
            family_claim_provenance=family_provenance,
            row_source_claim=get(metadata, "source", ""),
            verified_claim=get(metadata, "verified", false) === true,
            hidden_claim=hidden,
        ))
    end

    columns = propertynames(first(rows))
    write_tsv(output, columns, rows)

    deployed = count(row -> row.deployed, rows)
    explicit_family = count(row -> !isempty(row.family_claim), rows)
    formula_mismatches = filter(row -> !row.formula_match, rows)
    println("historical_sha=$sha")
    println("catalog_rows=$(length(rows))")
    println("deployed_rows=$deployed")
    println("hidden_rows=$(length(rows) - deployed)")
    println("explicit_family_rows=$explicit_family")
    println("formula_claim_return_mismatches=$(length(formula_mismatches))")
    for row in formula_mismatches
        println("FORMULA-MISMATCH\t$(row.source):$(row.key)\t$(row.formula_claim)\t$(row.formula_return)")
    end
    println("output=$(abspath(output))")
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    sha = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_SHA
    output = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT
    extract_catalog(; sha, output)
end
