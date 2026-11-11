module BRMDocsComparisons

using Markdown
using JSON
using BayesianRegressionModels
using Distributions
using StanBlocks
import AlgebraOfVega
import ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels
const EXAMPLE_MODULES = Dict{Symbol,Module}()
const PANE_IDS = "brm|ir|stanblocks|stan"
const PANE_LABELS = "BRM authoring|IR|StanBlocks model|Stan source"
const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

html_attribute(value) = replace(string(value), '&' => "&amp;", '"' => "&quot;",
                                '<' => "&lt;", '>' => "&gt;", '\'' => "&#39;")

"""
    figure(spec; caption, alt=caption)

Embed an AlgebraOfVega figure in a Documenter `@eval` block. The page carries
the generated Vega-Lite JSON; the theme calls AlgebraOfVega's browser runtime
after hydration. It follows light/dark switches. No rendered asset is committed.
"""
function figure(spec; caption::AbstractString, alt::AbstractString=caption)
    json = AlgebraOfVega.to_json(spec)
    html = """
    <figure class="brm-figure" v-pre>
      <div class="brm-figure-plot" data-brm-vega="$(html_attribute(json))" aria-label="$(html_attribute(alt))">
        <p role="status">Loading figure…</p>
      </div>
      <figcaption>$(html_attribute(caption))</figcaption>
    </figure>
    """
    return Markdown.parse("```@raw html\n" * html * "\n```\n")
end

"""Return one stable evaluation module for all executable blocks on a page."""
example_module(name::Symbol) = get!(EXAMPLE_MODULES, name) do
    m = Module(name)
    # A bare module has no `include`; a repository script may include a sibling.
    Core.eval(m, :(include(path) = Base.include($m, path)))
    m
end

function evaluate_source(mod::Module, displayed::AbstractString)
    parsed = Meta.parseall(displayed; filename="brm-docs-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    value = nothing
    for expression in expressions
        expression isa LineNumberNode && continue
        value = Core.eval(mod, expression)
    end
    return value
end

function definition_name(expression)
    expression isa Expr || return nothing
    if expression.head === :function
        signature = expression.args[1]
        signature isa Expr && signature.head === :call || return nothing
        name = first(signature.args)
        return name isa Symbol ? name : nothing
    end
    for argument in expression.args
        name = definition_name(argument)
        isnothing(name) || return name
    end
    return nothing
end

function repository_source(relative_path::AbstractString)
    path = normpath(joinpath(REPOSITORY_ROOT, relative_path))
    relative = relpath(path, REPOSITORY_ROOT)
    (!isempty(splitpath(relative)) && first(splitpath(relative)) == "..") &&
        error("repository source path escapes the repository root: $relative_path")
    return read(path, String)
end

"""Return one function declaration exactly as written in a repository source file."""
function source_function(relative_path::AbstractString, name::Symbol)
    source = repository_source(relative_path)
    marker = "function $name"
    matches = findall(marker, source)
    length(matches) == 1 || error(
        "expected one `$marker` declaration in $relative_path, found $(length(matches))")
    start = first(only(matches))
    expression, next_position = Meta.parse(source, start; greedy=true, raise=true)
    definition_name(expression) === name || error(
        "source at `$marker` in $relative_path did not parse as that function")
    stop = prevind(source, next_position)
    return strip(SubString(source, start, stop), '\n')
end

"""Return an exact, marker-bounded source region from a repository file."""
function source_region(relative_path::AbstractString;
                       starting_at::AbstractString,
                       ending_before::AbstractString)
    source = repository_source(relative_path)
    starts = findall(starting_at, source)
    length(starts) == 1 || error(
        "expected one `$starting_at` marker in $relative_path, found $(length(starts))")
    start = first(only(starts))

    stops = findall(ending_before, source)
    stops = filter(r -> first(r) > start, stops)
    length(stops) == 1 || error(
        "expected one `$ending_before` marker after `$starting_at` in " *
        "$relative_path, found $(length(stops))")
    stop = prevind(source, first(only(stops)))
    return strip(SubString(source, start, stop), '\n')
end

"""
Render rows of a tab-separated results file as a Markdown table, so a page
states only numbers its scripts wrote. `columns` pairs each column with its
heading; `select` keeps the rows it returns `true` for.
"""
function results_table(relative_path::AbstractString;
                       columns::Vector{<:Pair{Symbol,<:AbstractString}},
                       select=row -> true)
    lines = readlines(normpath(joinpath(REPOSITORY_ROOT, relative_path)))
    header = Symbol.(split(lines[1], '\t'))
    rows = [NamedTuple{Tuple(header)}(Tuple(String.(split(line, '\t'))))
            for line in lines[2:end]]
    kept = Base.filter(select, rows)
    isempty(kept) && error("$relative_path: no rows selected")
    for (column, _) in columns
        column in header || error("$relative_path has no column $column")
    end
    cells = [[String(label) for (_, label) in columns]]
    for row in kept
        push!(cells, [getproperty(row, column) for (column, _) in columns])
    end
    return Markdown.MD([Markdown.Table(cells, fill(:l, length(columns)))])
end

"""
The rows of several fits' tables of the same shape in one table, each file's
rows prefixed with its name under the column header `label`.
"""
function results_tables(labelled::Vector{<:Pair{<:AbstractString,<:AbstractString}};
                        columns::Vector{<:Pair{Symbol,<:AbstractString}},
                        label::AbstractString="fit")
    cells = [vcat([String(label)], [String(l) for (_, l) in columns])]
    for (name, relative_path) in labelled
        lines = readlines(normpath(joinpath(REPOSITORY_ROOT, relative_path)))
        length(lines) > 1 || error("$relative_path: no result rows")
        header = Symbol.(split(lines[1], '\t'))
        for (column, _) in columns
            column in header || error("$relative_path has no column $column")
        end
        for line in lines[2:end]
            row = NamedTuple{Tuple(header)}(Tuple(String.(split(line, '\t'))))
            push!(cells, vcat([String(name)], [getproperty(row, column) for (column, _) in columns]))
        end
    end
    length(cells) > 1 || error("no rows in $(last.(labelled))")
    return Markdown.MD([Markdown.Table(cells, fill(:l, length(cells[1])))])
end

"The receipts of several fits side by side: one row per key, one column per fit."
function receipts_table(labelled::Vector{<:Pair{<:AbstractString,<:AbstractString}})
    function read_receipt(relative_path)
        lines = readlines(normpath(joinpath(REPOSITORY_ROOT, relative_path)))
        length(lines) > 1 || error("$relative_path: empty receipt")
        [String.(split(line, '\t'; limit=2)) for line in lines[2:end]]
    end
    receipts = [name => read_receipt(path) for (name, path) in labelled]
    keys_in_order = String[]
    for (_, pairs) in receipts, (k, _) in pairs
        k in keys_in_order || push!(keys_in_order, k)
    end
    value(pairs, k) = (i = findlast(p -> p[1] == k, pairs); isnothing(i) ? "" : pairs[i][2])
    cells = [vcat([""], [String(name) for (name, _) in labelled])]
    for k in keys_in_order
        push!(cells, vcat([k], [value(pairs, k) for (_, pairs) in receipts]))
    end
    return Markdown.MD([Markdown.Table(cells, fill(:l, length(cells[1])))])
end

"""Render a drift-proof repository source region as a documentation code block."""
source_code_region(relative_path::AbstractString; language="julia", kwargs...) =
    Markdown.MD([Markdown.Code(language, source_region(relative_path; kwargs...))])

"""Evaluate a source file's shared prelude, stopping before a named function."""
function evaluate_source_prelude(mod::Module, relative_path::AbstractString;
                                 before::Symbol,
                                 starting_at::Union{Nothing,AbstractString}=nothing)
    source = repository_source(relative_path)
    if !isnothing(starting_at)
        matches = findall(starting_at, source)
        length(matches) == 1 || error(
            "expected one `$starting_at` marker in $relative_path, " *
            "found $(length(matches))")
        source = SubString(source, first(only(matches)), lastindex(source))
    end
    Core.eval(mod, :(using BayesianRegressionModels, Distributions, StanBlocks))
    # An absolute file name makes the script's `@__DIR__` its real directory.
    parsed = Meta.parseall(source;
                           filename=normpath(joinpath(REPOSITORY_ROOT, relative_path)))
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        definition_name(expression) === before && return nothing
        Core.eval(mod, expression)
    end
    error("$relative_path does not define `$before`; shared prelude was not bounded")
end

function exception_text(backend::AbstractString, err)
    reason = sprint(showerror, err)
    "$backend unsupported for this BRM example\n\n$reason"
end

function stan_emissions(brmi::BRM.BRMI, mod::Module;
                        required::Bool=false, total_groups=())
    try
        sb = Base.invokelatest(BRM.SBBRMI, brmi; mod=mod, total_groups)
        sb_source = strip(sprint(show, sb), '\n')
        stan_source = strip(Base.invokelatest(BRM.stan_code, sb), '\n')
        return sb_source, stan_source
    catch err
        required && rethrow()
        reason = exception_text("StanBlocks", err)
        return reason, "Stan emission unavailable because StanBlocks lowering failed.\n\n" *
                       sprint(showerror, err)
    end
end

const JAX_JULIA_USAGE = """
using ReactiveKernels, ReactiveKernelsPPL
rk = RKBRMI(model)
write("model.py", jax_source(rk.model, rk_bound_plan(rk)))
inputs = rk_bound_data(rk)   # the data prepare_data reads, by name"""

const JAX_PYTHON_USAGE = """
import jax
import model as m

prepared = m.prepare_data(**{k: inputs[k] for k in m.DATA})
lp = m.log_density(u, **prepared)               # u: unconstrained, length m.dimension(prepared)
lp = jax.jit(m.log_density)(u, **prepared)      # one compiled function for every dataset of these shapes"""

# The module's `UNTRANSLATED` entries, name => reason, or nothing when empty.
# The module writes the dict as a JSON object.
function jax_untranslated(text::AbstractString)
    found = match(r"(?m)^UNTRANSLATED\s*=\s*(\{.*\})\s*$", text)
    isnothing(found) && return nothing
    entries = JSON.parse(found.captures[1])
    isempty(entries) && return nothing
    return sort!([String(k) => String(v) for (k, v) in entries]; by=first)
end

# Wrap a sentence into comment lines of at most `width` characters.
function comment_lines(text::AbstractString; width=84)
    lines = String[]
    current = ""
    for word in split(text)
        candidate = isempty(current) ? word : current * " " * word
        if length(candidate) > width && !isempty(current)
            push!(lines, current)
            current = word
        else
            current = candidate
        end
    end
    isempty(current) || push!(lines, current)
    return lines
end

function jax_untranslated_note(entries)
    lines = ["", "# prepare_data takes these data-only values as inputs, from rk_bound_data(rk),",
             "# because they have no Python translation (m.UNTRANSLATED):"]
    for (name, reason) in entries
        wrapped = comment_lines(reason; width=80)
        push!(lines, "#   $name: $(wrapped[1])")
        append!(lines, "#       " .* wrapped[2:end])
    end
    return join(lines, '\n')
end

"""
The JAX pane of a comparison: how to write the module, how Python calls it,
and the module `jax_source` emits for this model's RKBRMI. A model that does
not translate shows the `JAXEmissionError` (or other construction error).
"""
function jax_pane(brmi::BRM.BRMI)
    module_text = try
        rk = Base.invokelatest(BRM.RKBRMI, brmi)
        plan = Base.invokelatest(BRM.rk_bound_plan, rk)
        string(Base.invokelatest(ReactiveKernelsPPL.jax_source, rk.model, plan))
    catch err
        return [Markdown.Code("julia", JAX_JULIA_USAGE),
                Markdown.Code("text", "This model has no JAX module yet.\n\n" *
                                      sprint(showerror, err))]
    end
    untranslated = jax_untranslated(module_text)
    usage = isnothing(untranslated) ? JAX_PYTHON_USAGE :
        JAX_PYTHON_USAGE * "\n" * jax_untranslated_note(untranslated)
    return [Markdown.Code("julia", JAX_JULIA_USAGE),
            Markdown.Code("python", usage),
            Markdown.Code("python", strip(module_text, '\n'))]
end

"""
Evaluate one displayed BRM example and emit the build-generated comparison:
BRM authoring, IR, StanBlocks model and Stan source, plus the JAX module of
its RKBRMI with `jax=true`. Every pane but the first is derived during this
build.
"""
function comparison(mod::Module, code::AbstractString, brmi_name::Symbol;
                    title=replace(string(brmi_name), '_' => ' '),
                    require_stan::Bool=false, total_groups=(), jax::Bool=false)
    displayed = strip(code, '\n')
    Core.eval(mod, :(using BayesianRegressionModels, Distributions))
    evaluate_source(mod, displayed)
    candidate = Core.eval(mod, brmi_name)
    brmi = candidate isa Function ? Base.invokelatest(candidate) : candidate
    brmi isa BRM.BRMI || error(
        "docs comparison `$brmi_name` did not evaluate to or construct a BRMI")
    ir_source = strip(sprint(show, brmi), '\n')
    sb_source, stan_source = stan_emissions(brmi, mod; required=require_stan, total_groups)
    jax_blocks = jax ? jax_pane(brmi) : Markdown.Code[]
    sentinel = string(title)
    isempty(jax_blocks) ||
        (sentinel *= "\npanes: brm|ir|stanblocks|stan|jax:$(length(jax_blocks))")
    return Markdown.MD([
        Markdown.Code("brm-comparison", sentinel),
        Markdown.Code("julia", displayed),
        Markdown.Code("julia", ir_source),
        Markdown.Code("julia", sb_source),
        Markdown.Code("stan", stan_source),
        jax_blocks...,
    ])
end

"""
Render a StanBlocks-backend (`@slic`) model that is NOT expressed through the `@brm`
formula surface: its exact model source (read from the repository, drift-proof) plus
the generated Stan it emits. There is no `@brm` authoring pane because
such a model does not go through the formula DSL — this is the honest two-pane view
for a model that lives at BRM's backend layer. Errors if the emission fails.
"""
function slic_model_panes(mod::Module, code::AbstractString, model_name::Symbol;
                          title=replace(string(model_name), '_' => ' '))
    displayed = strip(code, '\n')
    Core.eval(mod, :(using BayesianRegressionModels, Distributions, StanBlocks))
    evaluate_source(mod, displayed)
    candidate = Core.eval(mod, model_name)
    model = candidate isa Function ? Base.invokelatest(candidate) : candidate
    stan_source = strip(Base.invokelatest(StanBlocks.stan_code, model), '\n')
    return Markdown.MD([
        Markdown.Paragraph([Markdown.Bold(string(title))]),
        Markdown.Code("julia", displayed),
        Markdown.Code("stan", stan_source),
    ])
end

"""
Render a StanBlocks `compile_slic_bundle` model (a multi-part `@slic`/`@deffun`
workspace that is not a single `@brm` or `@slic` function): `include` its source file,
show the parent-body source held in `body_name`, then the generated Stan from
`builder_name()` (a zero-arg builder returning the bundle result with a `.code`
field). Two panes — there is no `@brm` authoring pane. Errors closed.
"""
function bundle_model_panes(mod::Module, include_relpath::AbstractString,
                            body_name::Symbol, builder_name::Symbol; title)
    Core.eval(mod, :(using BayesianRegressionModels, Distributions, StanBlocks))
    path = normpath(joinpath(REPOSITORY_ROOT, include_relpath))
    Base.invokelatest(Base.include, mod, path)
    body = strip(Core.eval(mod, body_name), '\n')
    result = Base.invokelatest(Core.eval(mod, builder_name))
    stan_source = strip(result.code, '\n')
    return Markdown.MD([
        Markdown.Paragraph([Markdown.Bold(string(title))]),
        Markdown.Code("julia", body),
        Markdown.Code("stan", stan_source),
    ])
end

"""Assert that a generated page contains complete required Stan comparisons."""
function validate_required_stan_outputs(path::AbstractString,
                                        brmi_names::Tuple)
    source = read(path, String)
    expected = length(brmi_names)
    comparisons = length(collect(eachmatch(
        r"(?m)^```brm-comparison\s*$", source)))
    comparisons == expected || error(
        "$(basename(path)) generated $comparisons required comparisons; " *
        "expected $expected")

    stan_fences = length(collect(eachmatch(r"(?m)^```stan\s*$", source)))
    stan_fences == expected || error(
        "$(basename(path)) generated $stan_fences Stan panes; expected $expected")

    for name in brmi_names
        occursin("function $name", source) || error(
            "$(basename(path)) omitted the exact `$name` declaration")
    end

    for fallback in ("StanBlocks unsupported for this BRM example",
                     "Stan emission unavailable because StanBlocks lowering failed")
        occursin(fallback, source) && error(
            "$(basename(path)) contains forbidden required-backend fallback: " *
            fallback)
    end
    return nothing
end

function validate_generated_templates(paths)
    for path in paths
        source = read(path, String)
        generated = length(collect(eachmatch(
            r"Main\.BRMDocsComparisons\.comparison\(", source)))
        generated > 0 || error(
            "$(basename(path)) contains no build-generated comparison")
        occursin("data-backend-comparison", source) && error(
            "$(basename(path)) hand-authors the comparison wrapper; " *
            "BRMDocsComparisons.comparison owns the fixed four panes")
        occursin(r"(?m)^```stan\s*$", source) && error(
            "$(basename(path)) contains hand-authored Stan; generated examples " *
            "must call BRMDocsComparisons.comparison")
    end
    return nothing
end

function validate_no_bypasses(paths)
    executable_fences = Set((
        "julia", "jldoctest", "@eval", "@example", "@repl", "@setup"))
    for path in paths
        source = read(path, String)
        occursin("BRMDocsComparisons.source_excerpt", source) && error(
            "$path bypasses the build-generated four-pane comparison")
        for block in eachmatch(r"(?ms)^```([^\n]*)\n(.*?)^```\s*$", source)
            info, code = block.captures
            fence_parts = split(strip(info))
            language = isempty(fence_parts) ? "" : first(fence_parts)
            language in executable_fences || continue
            occursin("@brm", code) || continue
            occursin("Main.BRMDocsComparisons.comparison(", code) && continue
            error("$path contains a standalone executable `@brm` example " *
                  "in a `$language` fence; use the generated four-pane " *
                  "comparison")
        end
    end
    return nothing
end

end
