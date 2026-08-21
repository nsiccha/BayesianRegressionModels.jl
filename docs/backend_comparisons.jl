module BRMDocsComparisons

using Markdown
using BayesianRegressionModels
using Distributions
using StanBlocks
using Turing

const BRM = BayesianRegressionModels
const EXAMPLE_MODULES = Dict{Symbol,Module}()
const PANE_IDS = "brm|stanblocks|stan|turing"
const PANE_LABELS = "BRM authoring|StanBlocks model|Stan source|Turing model"

"""Return one stable evaluation module for all executable blocks on a page."""
example_module(name::Symbol) = get!(EXAMPLE_MODULES, name) do
    Module(name)
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

function exception_text(backend::AbstractString, err)
    reason = sprint(showerror, err)
    "$backend unsupported for this BRM example\n\n$reason"
end

function stan_emissions(brmi::BRM.BRMI)
    try
        sb = Base.invokelatest(BRM.SBBRMI, brmi)
        sb_source = strip(sprint(show, sb), '\n')
        stan_source = strip(Base.invokelatest(BRM.stan_code, sb), '\n')
        return sb_source, stan_source
    catch err
        reason = exception_text("StanBlocks", err)
        return reason, "Stan emission unavailable because StanBlocks lowering failed.\n\n" *
                       sprint(showerror, err)
    end
end

function model_macro_name(expression::Expr)
    expression.head === :macrocall || return nothing
    function_expression = findfirst(arg -> arg isa Expr && arg.head === :function,
                                    expression.args)
    isnothing(function_expression) && return nothing
    signature = expression.args[function_expression].args[1]
    signature isa Expr && signature.head === :call || return nothing
    name = first(signature.args)
    return name isa Symbol ? name : nothing
end

function collect_turing_models!(models::Dict{Symbol,String}, expression)
    expression isa Expr || return models
    name = model_macro_name(expression)
    if !isnothing(name)
        function_expression = only(arg for arg in expression.args
                                   if arg isa Expr && arg.head === :function)
        cleaned = deepcopy(function_expression)
        Base.remove_linenums!(cleaned)
        models[name] = "Turing.@model " *
            sprint(io -> Base.show_unquoted(io, cleaned))
    end
    foreach(arg -> collect_turing_models!(models, arg), expression.args)
    return models
end

function turing_model_sources()
    extension = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    isnothing(extension) && error(
        "Turing extension did not load in the docs environment")
    parsed = Meta.parseall(read(pathof(extension), String);
                           filename=pathof(extension))
    collect_turing_models!(Dict{Symbol,String}(), parsed)
end

const TURING_MODEL_SOURCES = turing_model_sources()

function turing_emission(brmi::BRM.BRMI)
    try
        backend = Base.invokelatest(BRM.TuringBRMI, brmi)
        model_name = nameof(backend.model.f)
        source = get(TURING_MODEL_SOURCES, model_name, nothing)
        isnothing(source) && error(
            "selected Turing executor `$model_name` has no build-visible " *
            "`Turing.@model` definition")
        return strip(source, '\n')
    catch err
        return exception_text("Turing", err)
    end
end

"""
Render a reviewed source excerpt without evaluating it during the docs build.

This is for complete examples whose surrounding helper definitions and runtime
acceptance harness live in a linked repository source file. Ordinary standalone
`@brm` examples should use `comparison` so every backend pane is generated.
"""
function source_excerpt(code::AbstractString)
    return Markdown.MD([Markdown.Code("julia", strip(code, '\n'))])
end

"""
Evaluate one displayed BRM example and emit the fixed four-pane comparison.
The StanBlocks, Stan, and Turing panes are always derived during this build.
"""
function comparison(mod::Module, code::AbstractString, brmi_name::Symbol;
                    title=replace(string(brmi_name), '_' => ' '))
    displayed = strip(code, '\n')
    Core.eval(mod, :(using BayesianRegressionModels, Distributions))
    evaluate_source(mod, displayed)
    brmi = Core.eval(mod, brmi_name)
    brmi isa BRM.BRMI || error(
        "docs comparison `$brmi_name` did not evaluate to a BRMI")
    sb_source, stan_source = stan_emissions(brmi)
    turing_source = turing_emission(brmi)
    return Markdown.MD([
        Markdown.Code("brm-comparison", string(title)),
        Markdown.Code("julia", displayed),
        Markdown.Code("julia", sb_source),
        Markdown.Code("stan", stan_source),
        Markdown.Code("julia", turing_source),
    ])
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
        for block in eachmatch(r"(?ms)^```([^\n]*)\n(.*?)^```\s*$", source)
            info, code = block.captures
            fence_parts = split(strip(info))
            language = isempty(fence_parts) ? "" : first(fence_parts)
            language in executable_fences || continue
            occursin("@brm", code) || continue
            occursin("Main.BRMDocsComparisons.comparison(", code) && continue
            occursin("Main.BRMDocsComparisons.source_excerpt(", code) && continue
            error("$path contains a standalone executable `@brm` example " *
                  "in a `$language` fence; use the generated four-pane " *
                  "comparison")
        end
        if endswith(path, "turing-backend.md")
            occursin("Main.BRMDocsComparisons.comparison", source) && error(
                "backend-specific pages must be overview-only; move examples " *
                "to a backend-neutral page")
        end
    end
    return nothing
end

end
