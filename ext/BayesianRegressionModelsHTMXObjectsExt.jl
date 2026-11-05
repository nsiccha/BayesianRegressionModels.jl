module BayesianRegressionModelsHTMXObjectsExt

import BayesianRegressionModels as BRM
import Markdown
using HTMXObjects

_role(role) = replace(string(role), '_' => ' ')
_role(::Val{:response_data}) = "Data · response"
_role(::Val{:data}) = "Data"
_role(::Val{:parameter}) = "Parameter"
_role(::Val{:predictor}) = "Derived · formula predictor"
_role(::Val{:assignment}) = "Derived · assignment"
_role(::Val{:cell}) = "Derived · cell result"
_role(::Val{:component}) = "Derived · component result"
_role(::Val{:prior_modifier}) = "Prior specification"
_role(::Val{R}) where R = _role(R)

_rank(::Nothing) = "Unknown"
_rank(n::Integer) = string(n)
_shape(::Nothing, ::Nothing) = "Not inferred"
_shape(::Nothing, rank::Integer) = _shape_name(rank) * " · dimensions not bound"
_shape(dims::Tuple, rank) = isempty(dims) ? "Scalar" :
    _shape_name(length(dims)) * " · " * join(dims, " × ")
_shape_name(n) = n == 0 ? "Scalar" : n == 1 ? "Vector" : n == 2 ? "Matrix" : "$n-dimensional array"

function _rows(records)
    notes = any(r -> !isempty(r.detail), records)
    map(records) do r
        row = (; quantity=string(r.name), role=_role(Val(r.role)), rank=_rank(r.rank),
            shape=_shape(r.dims, r.rank))
        notes ? merge(row, (; note=r.detail)) : row
    end
end

function _inventory(title, records; empty="No quantities in this category.")
    SemanticSection(title, isempty(records) ? SemanticProse(empty) : SemanticTable(_rows(records)))
end

# Query only documentation attached to the actual callable's binding. Do not
# call the function, infer a meaning from its spelling, or dump fallback help.
function _documentation(callables)
    entries = Pair{String,Any}[]
    for callable in callables
        callable isa Union{Function,Type} || continue
        callable isa BRM.BRMComponent && continue  # retained authored records own these docs
        mod, name = parentmodule(callable), nameof(callable)
        # Arithmetic syntax names a broad Base generic, whose docs include
        # unrelated string/regex overloads. Custom operators still have docs.
        mod in (Base, Core) && Base.isoperator(name) && continue
        isdefined(mod, name) && getfield(mod, name) === callable || continue
        binding = Base.Docs.Binding(mod, name)
        metadata = Base.Docs.meta(mod; autoinit=false)
        isnothing(metadata) && continue
        haskey(metadata, binding) || continue
        doc = Base.Docs.doc(binding)
        isnothing(doc) || push!(entries, string(mod, ".", name) => doc)
    end
    sort!(entries; by=first)
    isempty(entries) && return nothing
    SemanticSection("Callable documentation",
        [SemanticDisclosure(name, SemanticProse(doc)) for (name, doc) in entries]...)
end

_path_label(path) = isempty(path) ? "Root" : join(
    [Base.isidentifier(string(part)) ? string(part) : repr(part) for part in path], " › ")

function _authored_documentation(records)
    isempty(records) && return nothing
    SemanticSection("Authored documentation", [
        SemanticSection(_path_label(doc.path), SemanticProse(Markdown.parse(doc.markdown)),
            SemanticDisclosure("Declaration and provenance",
                SemanticCode(:julia, _source_text(doc.statement)),
                SemanticFields(; path=repr(doc.path), owner=string(doc.owner),
                    file=isnothing(doc.file) ? "Not recorded" : string(doc.file),
                    line=doc.line > 0 ? string(doc.line) : "Not recorded")))
        for doc in records]...)
end

function _documentation_panel(records, callables)
    authored = _authored_documentation(records)
    callable = _documentation(callables)
    isnothing(authored) && isnothing(callable) && return nothing
    HTMXObjects.h.div(authored, callable)
end

_source_text(expression) = sprint(BRM._brm_print_source, expression)

function _call_instance(call)
    SemanticSection(_path_label(call.path), SemanticFields(; path=repr(call.path)),
        isnothing(call.call) ? SemanticProse("Root definition") :
            SemanticCode(:julia, _source_text(call.call)))
end

function _component_definitions(records)
    definitions = NamedTuple[]
    indices = IdDict{Any,Int}()
    for record in records
        index = get!(indices, record.definition) do
            push!(definitions, (; definition=record.definition, source=record.source,
                scope=record.scope, instances=NamedTuple[]))
            length(definitions)
        end
        push!(definitions[index].instances, record)
    end
    definitions
end

function _source_panel(source, components; root_definition=nothing)
    isempty(components) && return SemanticCode(:julia, source)
    definitions = _component_definitions(components)
    names = [string(record.scope, ".", nameof(record.definition)) for record in definitions]
    counts = Dict{String,Int}()
    for name in names
        counts[name] = get(counts, name, 0) + 1
    end
    labels = [counts[name] == 1 ? name : string(name, " · definition ", i)
        for (i, name) in enumerate(names)]
    panels = Pair{String,Any}[]
    if isnothing(root_definition)
        instances = [SemanticSection(labels[i], _call_instance.(record.instances)...)
            for (i, record) in enumerate(definitions)]
        push!(panels, "Model" => HTMXObjects.h.div(SemanticCode(:julia, source),
            SemanticSection("Component instances", instances...)))
    end
    for (i, record) in enumerate(definitions)
        push!(panels, labels[i] => SemanticSection("Definition · " * labels[i],
            SemanticFields(; defining_module=string(record.scope),
                inputs=join(string.(BRM.brm_columns(record.definition)), ", ")),
            SemanticCode(:julia, _source_text(record.source)),
            SemanticSection("Call instances", _call_instance.(record.instances)...)))
    end
    HTMXObjects.h.div(
        SemanticProse("Each component definition is shown once, with its call-instance paths. " *
            "Paths identify authored call sites; plate iterations share that site."),
        tabset(panels...; url_param=nothing, label="Model and component source"))
end

function _declaration(facts; title, note, extra=(), data_title="Data")
    panels = Pair{String,Any}[
        "Source" => _source_panel(facts.source, facts.components),
        "Roles and ranks" => HTMXObjects.h.div(
            _inventory(data_title, facts.inputs; empty="No bound data inputs."),
            _inventory("Model quantities", facts.quantities), extra...),
    ]
    documentation = _documentation_panel(facts.documentation, facts.callables)
    isnothing(documentation) || push!(panels, "Documentation" => documentation)
    SemanticSection(title,
        SemanticProse(note),
        tabset(panels...; url_param=nothing, label=string(title, " details")))
end

function BRM.brm_model_view(model::BRM.BRMI; title="Model")
    _declaration(BRM._brm_view_facts(model); title,
        note="Roles come from the bound declaration. Rank is the number of value axes: " *
             "0 scalar, 1 vector, 2 matrix. Unknown means inspection has no value-shape " *
             "metadata; it does not mean scalar. Supplied backend metadata can resolve " *
             "parameter dimensions. Helper and component results remain unknown unless " *
             "the supplied descriptor records their shape; inspection never evaluates them.")
end

function BRM.brm_model_view(builder::BRM.BRMBuilder; title="Model")
    inputs = [BRM._brm_view_record(name, :input;
        detail="Potential data input; observation/parameter role depends on binding.")
        for name in BRM.brm_columns(builder)]
    panels = Pair{String,Any}[
        "Source" => SemanticCode(:julia, string(builder)),
        "Potential data inputs" => _inventory("Potential data inputs", inputs;
            empty="No potential data inputs."),
    ]
    documentation = _authored_documentation(BRM.brm_documentation(builder))
    isnothing(documentation) || push!(panels, "Documentation" => documentation)
    SemanticSection(title,
        SemanticProse("Unbound declaration. Bind data to determine observed versus latent " *
            "quantities; supply a prepared backend for inferred parameter dimensions."),
        tabset(panels...; url_param=nothing, label=string(title, " details")))
end

function BRM.brm_model_view(component::BRM.BRMComponent; title="Model component")
    inputs = [(; input=string(name),
        passing=name in component.keywords ? "Keyword" : "Positional",
        requirement=haskey(component.defaults, name) ? "Optional" : "Required",
        default=haskey(component.defaults, name) ? _source_text(component.defaults[name]) : "None")
        for name in BRM.brm_columns(component)]
    panels = Pair{String,Any}[
        "Source" => _source_panel(string(component), BRM.brm_components(component);
            root_definition=component),
        "Component inputs" => SemanticSection("Component inputs",
            SemanticProse("Default expressions are shown without evaluation. " *
                "An omitted optional input uses its authored default at the call site."),
            isempty(inputs) ? SemanticProse("No explicit component inputs.") : SemanticTable(inputs)),
    ]
    documentation = _authored_documentation(BRM.brm_documentation(component))
    isnothing(documentation) || push!(panels, "Documentation" => documentation)
    SemanticSection(title,
        SemanticProse("Reusable component definition. Each call owns its local declarations " *
            "and reads the explicit inputs below. Inspection does not bind data or prepare a backend."),
        tabset(panels...; url_param=nothing, label=string(title, " details")))
end

function BRM.brm_model_view(d::BRM.BRMDescriptor; title="Model")
    facts = BRM._brm_view_facts(d.plan.parent)
    logical = Dict{Symbol,Vector{BRM.BRMOutput}}()
    for output in d.outputs
        isnothing(output.logical) && continue
        output.role in (:posterior_predictive, :pointwise_loglik, :membership) && continue
        push!(get!(logical, output.logical, BRM.BRMOutput[]), output)
    end
    quantities = map(facts.quantities) do row
        candidates = get(logical, row.name, BRM.BRMOutput[])
        length(candidates) == 1 && !isnothing(only(candidates).dims) || return row
        dims = only(candidates).dims
        merge(row, (; rank=length(dims), dims))
    end
    facts = merge(facts, (; quantities))
    # Value dimensions belong to the carrier itself. In particular, a predictive
    # draw is not a rank inference for its observation or for a linear predictor.
    outputs = [(
        quantity=isnothing(o.logical) ? "Internal carrier" : string(o.logical),
        role=_role(o.role), kind=_role(o.kind),
        rank=isnothing(o.dims) ? "Unknown" : string(length(o.dims)),
        shape=_shape(o.dims, isnothing(o.dims) ? nothing : length(o.dims)),
        carrier=string(o.name),
    ) for o in d.outputs]
    details = [SemanticDisclosure(string(o.name),
        SemanticFields(; logical=isnothing(o.logical) ? "Internal carrier" : string(o.logical),
            role=_role(o.role), kind=_role(o.kind), declared_type=string(o.type),
            declared_dimensions=string(o.size), constraints=string(o.constraints),
            labels=isnothing(o.labels) ? "None" : join(o.labels, ", "),
            segment_ends=isnothing(o.segments) ? "None" : join(o.segments, ", ")))
        for o in d.outputs]
    _declaration(facts; title,
        note="Prepared Stan descriptor. Carrier ranks and dimensions below describe constrained " *
             "values, not the number of free parameters. Logical quantities and their " *
             "generated carriers are listed separately.",
        extra=(
            SemanticSection("Prepared values", SemanticTable(outputs)),
            SemanticDisclosure("Carrier details", details...)))
end

function BRM.brm_model_view(rk::BRM.RKBRMI; title="Model")
    parameters = BRM._brm_view_rk_parameters(rk)
    facts = BRM._brm_view_facts(BRM.brm_source(rk))
    if isnothing(parameters)
        return _declaration(facts; title, data_title="Declaration data",
            note="This RK replay shares an earlier build. The source and data below belong " *
                 "to the original declaration. Current parameter dimensions are unresolved: " *
                 "the retained layout describes a different binding. Viewing does not bind " *
                 "or evaluate the replay to infer a new layout.")
    end
    declared = Set(row.name for row in facts.quantities if row.role === :parameter)
    bypath = Dict(row.path => row for row in parameters)
    quantities = map(facts.quantities) do row
        metadata = get(bypath, (row.name,), nothing)
        row.role === :parameter && !isnothing(metadata) || return row
        merge(row, (; rank=metadata.rank, dims=metadata.shape))
    end
    facts = merge(facts, (; quantities))
    # Paths are shown as tuples to retain authored namespace boundaries:
    # (:child, :b) is different from (Symbol("child.b"),).
    rows = [(
        parameter_path=repr(p.path),
        role=length(p.path) == 1 && only(p.path) in declared ?
            "Declared parameter" : "Backend parameter",
        rank=_rank(p.rank), shape=_shape(p.shape, p.rank),
        free_coordinates=p.ncoordinates,
    ) for p in parameters]
    elements = [SemanticDisclosure("Element shapes · " * repr(p.path),
        SemanticTable([(; element=i, rank=length(shape),
            shape=_shape(shape, length(shape)))
            for (i, shape) in enumerate(p.element_shapes)]))
        for p in parameters if !isnothing(p.element_shapes)]
    _declaration(facts; title, data_title="Declaration data",
        note="Prepared RK model. Parameter ranks describe constrained values; free coordinates " *
             "are counted separately. Backend parameter paths include generated coefficient " *
             "carriers and cell latents; they are separate from the declaration's quantities. " *
             "The view reads this supplied model without fitting or evaluating it.",
        extra=(SemanticSection("Prepared RK parameters", SemanticTable(rows), elements...),))
end

end
