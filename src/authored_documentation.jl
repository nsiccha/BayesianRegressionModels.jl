"""
    BRMDocumentation

An authored Markdown docstring attached to a logical declaration. `path` names
the declaration, with enclosing plate/component names before it; `owner` is
its top-level operation. `statement` is the authored syntax, and `file`/`line`
record Julia's source location when supplied. Documentation is not executable
model code. Access records through [`brm_documentation`](@ref), which returns
copies so changing an expression cannot mutate a model's metadata.
"""
struct BRMDocumentation
    path::Tuple
    owner::Symbol
    markdown::String
    statement::Expr
    file::Union{Nothing,Symbol}
    line::Int
end

function _brm_is_doc(x)
    Meta.isexpr(x, :macrocall) && length(x.args) == 4 || return false
    head = x.args[1]
    head === Symbol("@doc") || (head isa GlobalRef && head.mod === Core && head.name === Symbol("@doc")) ||
        (Meta.isexpr(head, :., 2) && head.args[1] in (:Core, :Base) &&
         head.args[2] == QuoteNode(Symbol("@doc")))
end
_brm_undoc(x) = _brm_is_doc(x) ? _brm_undoc(x.args[4]) : x

# Julia inserts Core.@doc in top-level blocks, but retains a literal followed
# by its declaration as two statements inside loops. Canonicalize that surface
# before either collecting metadata or compiling the scientific declaration.
_brm_normalize_docs(x) = x
function _brm_normalize_docs(x::Expr)
    x.head in (:quote, :$) && return deepcopy(x)
    x.head === :block || return Expr(x.head, map(_brm_normalize_docs, x.args)...)
    statements = Any[]
    location = LineNumberNode(0)
    i = 1
    while i <= length(x.args)
        statement = x.args[i]
        statement isa LineNumberNode && (location = statement)
        if statement isa String
            next = findnext(a -> !(a isa LineNumberNode), x.args, i + 1)
            if next !== nothing
                target = x.args[next]
                if _brm_statement_lhs(target) !== nothing ||
                        _is_brm_plate(target) || Meta.isexpr(target, :function)
                    push!(statements, Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
                        location, statement, _brm_normalize_docs(target)))
                    i = next + 1
                    continue
                end
            end
        end
        push!(statements, _brm_normalize_docs(statement))
        i += 1
    end
    Expr(:block, statements...)
end
_brm_strip_docs(x) = _brm_strip_normalized_docs(_brm_normalize_docs(x))
_brm_strip_normalized_docs(x) = x
_brm_strip_normalized_docs(x::Expr) = _brm_is_doc(x) ? _brm_strip_normalized_docs(x.args[4]) :
    x.head in (:quote, :$) ? deepcopy(x) : Expr(x.head, map(_brm_strip_normalized_docs, x.args)...)

function _brm_documentation_lhs(lhs)
    lhs isa Symbol && return (lhs,)
    lhs isa Expr || return ()
    _is_optional_prior_lhs(lhs) && return _brm_documentation_lhs(_optional_prior_address(lhs))
    if _is_hyper_lhs(lhs)
        return (_brm_hyper_statement_key(_brm_hyper_address(lhs)),)
    elseif _is_effect_lhs(lhs)
        return (_brm_prior_statement_key(_prior_address(lhs.args[1], map(_prior_slot, lhs.args[2:end]))),)
    elseif Meta.isexpr(lhs, :tuple)
        return Tuple(key for item in lhs.args for key in _brm_documentation_lhs(item))
    elseif Meta.isexpr(lhs, :vect)
        return (_joint_response_operation_key(Tuple(lhs.args)),)
    elseif lhs.head in (:ref, :(::))
        return _brm_documentation_lhs(lhs.args[1])
    elseif Meta.isexpr(lhs, :call)
        return _brm_documentation_lhs(last(lhs.args))
    end
    ()
end

function _brm_documentation_targets(statement)
    x = _brm_strip_docs(statement)
    if _is_brm_plate(x)
        i, _, body = _brm_plate_loop(x)
        stmts, outputs = _brm_plate_outputs(body, i)
        return (isempty(outputs) ? _brm_plate_effect_name(stmts, i) : last(outputs),)
    elseif Meta.isexpr(x, :function)
        signature = first(x.args)
        return (first(signature.args),)
    end
    lhs = _brm_statement_lhs(x)
    _brm_documentation_lhs(lhs)
end

function _brm_collect_documentation!(docs, x, path=(), owner=nothing,
        location=LineNumberNode(0))
    x isa Expr || return docs
    if _brm_is_doc(x)
        markdown, statement = x.args[3:4]
        markdown isa String || throw(ArgumentError(
            "BRM docstrings must be literal Markdown strings; documentation is retained without evaluating Julia expressions"))
        loc = x.args[2] isa LineNumberNode ? x.args[2] : location
        targets = _brm_documentation_targets(statement)
        isempty(targets) && throw(ArgumentError("BRM docstring has no declaration target: $statement"))
        for target in targets
            push!(docs, BRMDocumentation((path..., target), something(owner, target),
                markdown, deepcopy(statement), loc.file, loc.line))
        end
        return _brm_collect_documentation!(docs, statement, path, owner, loc)
    elseif _is_brm_plate(x)
        key = only(_brm_documentation_targets(x))
        _, _, body = _brm_plate_loop(x)
        return _brm_collect_documentation!(docs, body, (path..., key), something(owner, key), location)
    elseif x.head === :block
        for statement in x.args
            if statement isa LineNumberNode
                location = statement
            else
                _brm_collect_documentation!(docs, statement, path, owner, location)
            end
        end
    elseif x.head in (:for, :while, :let) || _is_brm_scan(x)
        # A cell's nested lexical body keeps the enclosing operation identity.
        _brm_collect_documentation!(docs, last(x.args), path, owner, location)
    end
    docs
end
_brm_source_documentation(source) =
    Tuple(_brm_collect_documentation!(BRMDocumentation[], _brm_normalize_docs(source)))
_brm_copy_documentation(d::BRMDocumentation) = BRMDocumentation(d.path, d.owner,
    d.markdown, deepcopy(d.statement), d.file, d.line)

# Emission sites supply the exact records for their known logical target.
# Wrapping never infers identity from generated names or evaluates @doc.
function _brm_documented_statement(statement::Expr, docs)
    result = deepcopy(statement)
    for doc in reverse(docs)
        result = Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
            LineNumberNode(doc.line, something(doc.file, :none)), doc.markdown, result)
    end
    result
end

# BRM's own documentation of a definition it emits into an RKPPL program (a
# component submodel, a block, a statistical helper). RKPPL retains it as the
# definition's record (`rkppl_documentation`, owner `:definition`), and RK's
# JAX export renders it as the docstring of that definition's Python function.
# The text is written where the definition's body is built, so it describes
# exactly that body; `@_rk_definition_doc` records that line of BRM's source.
struct _RKDefinitionDoc
    markdown::String
    location::LineNumberNode
end
macro _rk_definition_doc(markdown)
    file = Symbol("BayesianRegressionModels/src/", basename(String(__source__.file)))
    :(_RKDefinitionDoc($(esc(markdown)), $(QuoteNode(LineNumberNode(__source__.line, file)))))
end
_rk_documented_definition(definition, ::Nothing) = definition
_rk_documented_definition(definition, doc::_RKDefinitionDoc) = Expr(:macrocall,
    GlobalRef(Core, Symbol("@doc")), doc.location, doc.markdown, definition)

# An outer `"doc" @brm function ... end` belongs to Julia's documentation
# macro. Read its original strings rather than re-rendering Markdown. This is
# Julia's existing store, and autoinit=false makes this a strictly read-only
# operation even in an otherwise undocumented defining module.
function _brm_binding_docstring(scope::Module, name::Symbol)
    metadata = Base.Docs.meta(scope; autoinit=false)
    metadata === nothing && return nothing
    binding = Base.Docs.Binding(scope, name)
    docs = get(metadata, binding, nothing)
    docs === nothing ? nothing : get(docs.docs, Union{}, nothing)
end

function _brm_binding_documentation(scope::Module, name::Symbol, statement::Expr;
        previous=nothing)
    doc = _brm_binding_docstring(scope, name)
    (doc === nothing || doc === previous) && return ()
    all(x -> x isa AbstractString, doc.text) && !isempty(doc.text) ||
        throw(ArgumentError("BRM binding documentation must contain Markdown strings"))
    file = get(doc.data, :path, "")
    (BRMDocumentation((name,), name, join(doc.text), deepcopy(statement),
        isempty(file) ? nothing : Symbol(file), get(doc.data, :linenumber, 0)),)
end

_brm_documentation(model) = _brm_documentation(brm_source(model))
_brm_documentation(source::Expr) = _brm_source_documentation(source)
_brm_documentation(artifact::NamedTuple) = haskey(artifact, :meta) ?
    get(artifact.meta, :documentation, ()) : throw(ArgumentError("not a BRM source artifact"))

function _brm_select_documentation(model, operations)
    BRMI(operations, Tuple(d for d in model.documentation if haskey(operations, d.owner)))
end

function _brm_merge_documentation(a, b)
    replaced = Set(k for (k, v) in pairs(b.operations) if !_brm_is_read_binding(v))
    (Tuple(d for d in a.documentation if !(d.owner in replaced))..., b.documentation...)
end

"""
    brm_documentation(model; path=nothing)

Read authored declaration docstrings from a builder, bound model or backend,
without preparing a backend or running a scientific helper. Records retain raw
Markdown, logical paths and source provenance. A tuple-LHS docstring belongs to
each expanded declaration. `path` selects an exact logical path (a `Symbol`
selects a top-level declaration). Later declarations in `merge(::BRMI, ...)`
replace earlier documentation at the same operation, including an undocumented
replacement. Builder merges retain the authored statements and their docs.
"""
function brm_documentation(model; path=nothing)
    docs = _brm_documentation(model)
    selected = path isa Symbol ? (path,) : path
    Tuple(_brm_copy_documentation(d) for d in docs if selected === nothing || d.path == selected)
end

function _brm_print_doc(io, markdown)
    # `repr` preserves quotes, dollar signs and newlines as a Julia literal.
    println(io, repr(markdown))
end

# Julia's generic Expr printer spells Core.@doc as a long macro call. Within
# a statement block, ordinary literal strings recover the authored spelling
# and let Julia's printer handle function/control-flow indentation itself.
_brm_display_source(x, documentation) = x
function _brm_display_source(x::Expr, documentation)
    x.head in (:quote, :$) && return deepcopy(x)
    args = Any[]
    for arg in x.args
        if arg isa LineNumberNode
            # A macro still needs its location slot, but a display has no
            # reason to print the source-file annotation there.
            x.head === :macrocall && push!(args, nothing)
            continue
        end
        if x.head in (:block, :toplevel)
            while _brm_is_doc(arg)
                documentation && push!(args, arg.args[3])
                arg = arg.args[4]
            end
        end
        push!(args, _brm_display_source(arg, documentation))
    end
    Expr(x.head, args...)
end

function _brm_documentation_index(documentation)
    index = Dict{Tuple,Vector{Tuple{Int,BRMDocumentation}}}()
    for (i, doc) in enumerate(documentation)
        push!(get!(Vector{Tuple{Int,BRMDocumentation}}, index, doc.path), (i, doc))
    end
    index
end

# Multiple tuple targets can interleave records; selection preserves their
# original order without rescanning unrelated documentation.
function _brm_documentation_records(index, targets)
    selected = Tuple{Int,BRMDocumentation}[]
    for target in unique(targets)
        append!(selected, get(index, target, ()))
    end
    sort!(selected; by=first)
    last.(selected)
end

function _brm_documentation_owners(documentation)
    owners = Dict{Symbol,Vector{BRMDocumentation}}()
    for doc in documentation
        push!(get!(Vector{BRMDocumentation}, owners, doc.owner), doc)
    end
    owners
end

# Restore wrappers onto the retained source, which also contains the grouping
# bindings needed to replay a cell. Never substitute a pre-normalization body.
_brm_restore_documentation(x, docs, path=()) =
    _brm_restore_indexed_documentation(x, _brm_documentation_index(docs), path)
_brm_restore_indexed_documentation(x, docs, path) = x
function _brm_restore_indexed_documentation(x::Expr, docs, path)
    targets = _brm_documentation_targets(x)
    result = deepcopy(x)
    if _is_brm_plate(x)
        loop = last(result.args)
        loop.args[end] = _brm_restore_indexed_documentation(last(loop.args), docs, (path..., only(targets)))
    elseif x.head in (:block, :toplevel)
        result.args = Any[_brm_restore_indexed_documentation(a, docs, path) for a in x.args]
    elseif x.head in (:for, :while, :let) || _is_brm_scan(x)
        result.args[end] = _brm_restore_indexed_documentation(last(x.args), docs, path)
    end
    attached = Set{String}()
    for doc in _brm_documentation_records(docs, ((path..., target) for target in targets))
        Base.remove_linenums!(_brm_strip_docs(doc.statement)) ==
            Base.remove_linenums!(deepcopy(x)) || continue
        doc.markdown in attached && continue
        push!(attached, doc.markdown)
        result = Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
            LineNumberNode(doc.line, something(doc.file, :none)), doc.markdown, result)
    end
    result
end

function _brm_print_documented_cell(io, source)
    docs = get(io, :brm_cell_documentation, ())
    isempty(docs) && return print(io, source)
    statements = _brm_restore_documentation(_brm_source_block(source), docs).args
    started = false
    for statement in statements
        statement isa LineNumberNode && continue
        started && println(io)
        started = true
        _brm_print_source(io, statement)
    end
end

"""
    brm_documentation_markdown(model)

Render authored declaration documentation as Markdown for HTML and PDF
pipelines. The same records back source printing and semantic model views;
the Markdown remains authored prose, not a claim of generated scientific
description coverage.
"""
function brm_documentation_markdown(model)
    io = IOBuffer()
    for doc in brm_documentation(model)
        path = isempty(doc.path) ? (doc.owner,) : doc.path
        println(io, "### `", join(string.(path), "."), "`\n\n", doc.markdown, "\n")
    end
    String(take!(io))
end
