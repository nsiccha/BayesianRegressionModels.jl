# `brm(source; mod)`: a builder from a runtime `Expr` that evaluates no code.
#
# `_brm` turns a formula block into the code of a `__df__ -> BRMI` builder
# body. `@brm` compiles that body where the macro is written. Evaluating it at
# run time (`Core.eval(mod, _brm(expr))`) instead defines a new method per
# model: a new world, a compile per model, and an `eval` into a module that
# is closed while another package precompiles. `brm` walks the same body:
# every call it makes is to a function that already exists, it defines no
# method and no global, and its names resolve exactly as the compiled body
# resolves them (`let`-locals first, then `mod`). One generator serves both,
# so the two builders cannot diverge.
#
# The body is formula plumbing (bindings, `ExprColumn` construction) plus the
# Julia a `name = constant(expr)` reads. The walk admits the Julia surface such
# code is written in: calls with keywords and splats, broadcasting, indexing
# with `begin`/`end`, tuples and named tuples, array literals, conditionals,
# anonymous functions, generators and comprehensions, string interpolation,
# `do` blocks and macros. Anything that defines or mutates (`function`,
# `struct`, `global`, `const`, `import`, loops, indexed assignment) is refused
# with the expression that does it: define it in `mod` and call it instead.

"""
    brm(source::Expr; mod::Module) -> BRMBuilder
    brm(source::AbstractString; mod::Module) -> BRMBuilder
    brm(df, source; mod::Module) -> BRMI

The model builder of a `@brm` block given as a value, without evaluating code.
`brm(source; mod)` returns the [`BRMBuilder`](@ref) that `@brm begin … end`
written in module `mod` returns; `brm(df, source; mod)` builds it on `df`.
Names the block reads that are not formula names (distribution and term
constructors, user functions, the inputs of `name = constant(expr)` that are
not data fields) resolve as globals of `mod`. A local value enters the block
by interpolation, `\$value`, as in any `Expr`.

```julia
covariates = (:weight, :age)
rhs = Expr(:call, :+, 1, covariates..., :((1 | p | subject)))
model = brm(quote
    log_CL ~ \$rhs
    sigma ~ Exponential(1)
    conc ~ Normal(log_CL, sigma)
end; mod=@__MODULE__)
fit = SBBRMI(model(df); mod=@__MODULE__)
```

No method is defined and no code is compiled for the model, so a builder can
be made at run time, in a precompile workload, or from code compiled before
the model existed, and `merge` of builders is `brm` of their joined sources.
A named function definition, `f(df) = begin … end`, is not a builder value:
write that with `@brm`.
"""
function brm(source::Expr; mod::Module)
    (Meta.isexpr(source, :(=)) && !(source.args[1] isa Symbol)) && error(
        "brm: `$(sprint(Base.show_unquoted, source.args[1])) = …` defines a " *
        "function; pass the model block itself (`quote … end`), or write the " *
        "definition with `@brm`")
    Meta.isexpr(source, :(=)) && (source = Expr(:block, source))
    code = _brm_builder_code(source)
    columns = _brm_interpret(_brm_expand(code.columns, mod), _BRMScope(mod))
    BRMBuilder(_BRMInterpretedBuilder(_brm_expand(code.body, mod), mod),
               code.source, columns, mod)
end
brm(source::AbstractString; mod::Module) = brm(_brm_source_block(source); mod)
brm(df, source::Union{Expr,AbstractString}; mod::Module) = brm(source; mod)(df)

# The builder function of an interpreted `BRMBuilder`: the `_brm` body with
# every macro expanded, walked once per data container.
struct _BRMInterpretedBuilder
    body::Expr
    mod::Module
end
function (b::_BRMInterpretedBuilder)(df)
    scope = _BRMScope(b.mod)
    scope.vars[:__df__] = df
    _brm_interpret(b.body, scope)
end

# ---- macros ------------------------------------------------------------------
#
# Expanded once, when the builder is made. The `_brm` plumbing macros expand
# through their own functions, so a module that only `import`s BRM resolves
# them too; `@__MODULE__` is `mod`. A user macro (inside a constant's
# expression) expands as the compiler would, in `mod`. Quoted code (a `@plate`
# cell body) is data, so its `@scan` blocks stay.
_brm_expand(x, mod::Module) = x
_brm_expand(x::Expr, mod::Module) =
    Meta.isexpr(x, :quote) ? x :
    Meta.isexpr(x, :macrocall) ? _brm_expand_macro(x.args[1], x, mod) :
    Expr(x.head, (_brm_expand(a, mod) for a in x.args)...)

_brm_macro_arguments(x::Expr) = [a for a in x.args[2:end] if !(a isa LineNumberNode)]
function _brm_expand_macro(name, x::Expr, mod::Module)
    expander = _brm_plumbing_macro(name)
    if !isnothing(expander)
        _brm_expand(expander(only(_brm_macro_arguments(x))), mod)
    elseif name === Symbol("@__MODULE__")
        mod
    elseif name === Symbol("@isdefined")
        Expr(:isdefined, only(_brm_macro_arguments(x)))
    else
        _brm_expand(macroexpand(mod, x; recursive=true), mod)
    end
end
_brm_plumbing_macro(name) =
    name === Symbol("@n") ? _n :
    name === Symbol("@x") ? _x :
    name === Symbol("@getproperty") ? _getproperty : nothing

# ---- scopes --------------------------------------------------------------------

# A name a `let` declares holds this until it is assigned.
struct _BRMUnassigned end

# One lexical scope: the builder's, a `let` block's or one call of an
# anonymous function. Names not bound in any scope are globals of `mod`.
struct _BRMScope
    vars::Dict{Symbol,Any}
    parent::Union{_BRMScope,Nothing}
    mod::Module
end
_BRMScope(mod::Module) = _BRMScope(Dict{Symbol,Any}(), nothing, mod)
_brm_child_scope(scope::_BRMScope) = _BRMScope(Dict{Symbol,Any}(), scope, scope.mod)

function _brm_binding_scope(scope::_BRMScope, name::Symbol)
    while !isnothing(scope)
        haskey(scope.vars, name) && return scope
        scope = scope.parent
    end
    nothing
end

function _brm_lookup(scope::_BRMScope, name::Symbol)
    owner = _brm_binding_scope(scope, name)
    if isnothing(owner)
        isdefined(scope.mod, name) || throw(UndefVarError(name))
        return getglobal(scope.mod, name)
    end
    value = owner.vars[name]
    value isa _BRMUnassigned && throw(UndefVarError(name))
    value
end

function _brm_isdefined(scope::_BRMScope, name::Symbol)
    owner = _brm_binding_scope(scope, name)
    isnothing(owner) ? isdefined(scope.mod, name) : !(owner.vars[name] isa _BRMUnassigned)
end

# Assignment writes the innermost scope that declares the name, else makes it
# local to the current scope (a `let` block and a function body are hard
# scopes, so nothing an interpreted body assigns is ever a global).
function _brm_assign!(scope::_BRMScope, name::Symbol, value)
    owner = _brm_binding_scope(scope, name)
    (isnothing(owner) ? scope : owner).vars[name] = value
end

# ---- the walk ------------------------------------------------------------------

_brm_interpret(x, scope::_BRMScope) = x               # a value spliced into the code
_brm_interpret(x::Symbol, scope::_BRMScope) = _brm_lookup(scope, x)
_brm_interpret(x::QuoteNode, scope::_BRMScope) = x.value
_brm_interpret(x::GlobalRef, scope::_BRMScope) = getglobal(x.mod, x.name)
_brm_interpret(::LineNumberNode, scope::_BRMScope) = nothing
_brm_interpret(x::Expr, scope::_BRMScope) = _brm_interpret(Val(x.head), x, scope)

_brm_interpret(::Val, x::Expr, scope::_BRMScope) = error(
    "brm: `$(sprint(Base.show_unquoted, Base.remove_linenums!(deepcopy(x))))` is " *
    "not admitted in a model built without evaluating code: it defines or " *
    "mutates something, or is a form the builder does not walk. Define it as a " *
    "function in $(scope.mod) and call that.")

# Compiler annotations carry no value.
_brm_interpret(::Union{Val{:meta},Val{:inbounds},Val{:boundscheck},Val{:loopinfo}},
               x::Expr, scope::_BRMScope) = nothing

function _brm_interpret(::Val{:block}, x::Expr, scope::_BRMScope)
    value = nothing
    for arg in x.args
        arg isa LineNumberNode || (value = _brm_interpret(arg, scope))
    end
    value
end

_brm_interpret(::Val{:escape}, x::Expr, scope::_BRMScope) = _brm_interpret(only(x.args), scope)
_brm_interpret(::Val{Symbol("hygienic-scope")}, x::Expr, scope::_BRMScope) =
    _brm_interpret(x.args[1], _BRMScope(Dict{Symbol,Any}(), scope, x.args[2]::Module))

function _brm_interpret(::Val{:let}, x::Expr, scope::_BRMScope)
    inner = _brm_child_scope(scope)
    bindings = x.args[1]
    for binding in (Meta.isexpr(bindings, :block) ? bindings.args : Any[bindings])
        if binding isa Symbol
            inner.vars[binding] = _BRMUnassigned()
        elseif Meta.isexpr(binding, :(=), 2) && binding.args[1] isa Symbol
            # `let x = v`: `v` reads the enclosing scope and earlier bindings.
            inner.vars[binding.args[1]] = _brm_interpret(binding.args[2], inner)
        elseif !(binding isa LineNumberNode)
            _brm_interpret(Val(:let_binding), binding, scope)
        end
    end
    _brm_interpret(x.args[2], inner)
end

function _brm_interpret(::Val{:local}, x::Expr, scope::_BRMScope)
    value = nothing
    for declaration in x.args
        name = Meta.isexpr(declaration, :(=)) ? declaration.args[1] : declaration
        name isa Symbol || _brm_interpret(Val(:local_declaration), declaration, scope)
        scope.vars[name] = _BRMUnassigned()
        Meta.isexpr(declaration, :(=)) && (value = _brm_interpret(declaration, scope))
    end
    value
end

function _brm_interpret(::Val{:(=)}, x::Expr, scope::_BRMScope)
    lhs, rhs = x.args
    value = _brm_interpret(rhs, scope)
    _brm_bind!(scope, lhs, value, x)
    value
end

_brm_bind!(scope::_BRMScope, lhs::Symbol, value, _statement) = _brm_assign!(scope, lhs, value)
function _brm_bind!(scope::_BRMScope, lhs::Expr, value, statement)
    if Meta.isexpr(lhs, :tuple) && length(lhs.args) == 1 && Meta.isexpr(lhs.args[1], :parameters)
        # `(; a, b) = x` binds properties by name.
        for name in lhs.args[1].args
            name isa Symbol || _brm_interpret(Val(:destructuring), statement, scope)
            _brm_assign!(scope, name, getproperty(value, name))
        end
    elseif Meta.isexpr(lhs, :tuple)
        # `a, b = x` iterates, exactly as lowered code does.
        state = ()
        for (i, target) in enumerate(lhs.args)
            item, next = Base.indexed_iterate(value, i, state...)
            _brm_bind!(scope, target, item, statement)
            state = (next,)
        end
    else
        _brm_interpret(Val(:assignment), statement, scope)
    end
end

# Positional and keyword arguments of a call (or of a named tuple), with
# splats expanded and `; a` meaning `a = a`.
function _brm_arguments(raw, scope::_BRMScope)
    args = Any[]
    kwargs = Pair{Symbol,Any}[]
    for arg in raw
        if Meta.isexpr(arg, :parameters)
            foreach(p -> _brm_keyword!(kwargs, p, scope), arg.args)
        elseif Meta.isexpr(arg, :kw)
            push!(kwargs, arg.args[1]::Symbol => _brm_interpret(arg.args[2], scope))
        elseif Meta.isexpr(arg, :...)
            append!(args, _brm_interpret(only(arg.args), scope))
        else
            push!(args, _brm_interpret(arg, scope))
        end
    end
    args, kwargs
end
_brm_keyword!(kwargs, p::Symbol, scope) = push!(kwargs, p => _brm_lookup(scope, p))
function _brm_keyword!(kwargs, p::Expr, scope)
    if Meta.isexpr(p, (:kw, :(=)))
        push!(kwargs, p.args[1]::Symbol => _brm_interpret(p.args[2], scope))
    elseif Meta.isexpr(p, :...)
        for (k, v) in pairs(_brm_interpret(only(p.args), scope))
            push!(kwargs, Symbol(k) => v)
        end
    elseif Meta.isexpr(p, :., 2) && p.args[2] isa QuoteNode
        push!(kwargs, p.args[2].value => _brm_interpret(p, scope))   # `; a.b` is `b = a.b`
    else
        _brm_interpret(Val(:keyword), p, scope)
    end
    kwargs
end
_brm_values(raw, scope::_BRMScope) = first(_brm_arguments(raw, scope))

# A dotted operator call `a .+ b` is the broadcast of `+`.
function _brm_callee(f::Symbol, scope::_BRMScope)
    name = string(f)
    if length(name) > 1 && startswith(name, '.') && Base.isoperator(Symbol(name[2:end])) &&
       isnothing(_brm_binding_scope(scope, f)) && !isdefined(scope.mod, f)
        return Base.BroadcastFunction(_brm_lookup(scope, Symbol(name[2:end])))
    end
    _brm_lookup(scope, f)
end
_brm_callee(f, scope::_BRMScope) = _brm_interpret(f, scope)

function _brm_interpret(::Val{:call}, x::Expr, scope::_BRMScope)
    f = _brm_callee(x.args[1], scope)
    args, kwargs = _brm_arguments(@view(x.args[2:end]), scope)
    f(args...; kwargs...)
end

# `f(args...) do params … end` passes the anonymous function first.
function _brm_interpret(::Val{:do}, x::Expr, scope::_BRMScope)
    call, lambda = x.args
    Meta.isexpr(call, :call) || _brm_interpret(Val(:do_block), x, scope)
    args = copy(call.args)
    insert!(args, length(args) >= 2 && Meta.isexpr(args[2], :parameters) ? 3 : 2, lambda)
    _brm_interpret(Expr(:call, args...), scope)
end

function _brm_interpret(::Val{:.}, x::Expr, scope::_BRMScope)
    target, field = x.args
    field isa QuoteNode && return getproperty(_brm_interpret(target, scope), field.value)
    Meta.isexpr(field, :tuple) || _brm_interpret(Val(:dot), x, scope)
    f = _brm_interpret(target, scope)
    args, kwargs = _brm_arguments(field.args, scope)
    isempty(kwargs) ? broadcast(f, args...) : broadcast((xs...) -> f(xs...; kwargs...), args...)
end

function _brm_interpret(::Val{:tuple}, x::Expr, scope::_BRMScope)
    any(a -> Meta.isexpr(a, (:parameters, :(=))), x.args) || return Tuple(_brm_values(x.args, scope))
    # A named tuple: `(; a, b = 1)` or `(a = 1, b = 2)`.
    fields = Pair{Symbol,Any}[]
    for arg in x.args
        Meta.isexpr(arg, :parameters) ? foreach(p -> _brm_keyword!(fields, p, scope), arg.args) :
            _brm_keyword!(fields, arg, scope)
    end
    (; fields...)
end

# Array literals lower to the calls lowered code makes: `[a, b]`, `[a b]`,
# `[a; b]`, `[a b; c d]`, `[a;;]`, and their typed forms `T[a b]`, `T[a; b]`.
_brm_interpret(::Val{:vect}, x::Expr, scope::_BRMScope) = Base.vect(_brm_values(x.args, scope)...)
_brm_interpret(::Val{:hcat}, x::Expr, scope::_BRMScope) = hcat(_brm_values(x.args, scope)...)
_brm_interpret(::Val{:typed_hcat}, x::Expr, scope::_BRMScope) =
    Base.typed_hcat(_brm_values(x.args, scope)...)
_brm_interpret(::Val{:vcat}, x::Expr, scope::_BRMScope) =
    _brm_vcat(x.args, scope, vcat, hvcat)
function _brm_interpret(::Val{:typed_vcat}, x::Expr, scope::_BRMScope)
    T = _brm_interpret(x.args[1], scope)
    _brm_vcat(x.args[2:end], scope, (v...) -> Base.typed_vcat(T, v...),
              (shape, v...) -> Base.typed_hvcat(T, shape, v...))
end
function _brm_vcat(args, scope::_BRMScope, flat, rows)
    any(a -> Meta.isexpr(a, :row), args) || return flat(_brm_values(args, scope)...)
    cells = [Meta.isexpr(a, :row) ? _brm_values(a.args, scope) : Any[_brm_interpret(a, scope)]
             for a in args]
    rows(Tuple(length.(cells)), (v for row in cells for v in row)...)
end
# `[a;;]` is `hvncat(2, a)`. A literal with nested rows (`[a b;;; c d]`) is
# refused.
function _brm_interpret(::Val{:ncat}, x::Expr, scope::_BRMScope)
    any(a -> Meta.isexpr(a, (:row, :nrow)), x.args) && _brm_interpret(Val(:nested_ncat), x, scope)
    Base.hvncat(x.args[1]::Int, _brm_values(x.args[2:end], scope)...)
end

# `a[i, end]`: `begin`/`end` in an index are the bounds of `a` along it.
_brm_mentions(x, name::Symbol) = x === name
_brm_mentions(x::Expr, name::Symbol) =
    !Meta.isexpr(x, :ref) && any(a -> _brm_mentions(a, name), x.args)
function _brm_interpret(::Val{:ref}, x::Expr, scope::_BRMScope)
    collection = _brm_interpret(x.args[1], scope)
    raw = @view x.args[2:end]
    indices = Any[]
    for (d, index) in enumerate(raw)
        inner = scope
        if _brm_mentions(index, :end) || _brm_mentions(index, :begin)
            inner = _brm_child_scope(scope)
            inner.vars[:end] = length(raw) == 1 ? lastindex(collection) : lastindex(collection, d)
            inner.vars[:begin] = length(raw) == 1 ? firstindex(collection) : firstindex(collection, d)
        end
        Meta.isexpr(index, :...) ? append!(indices, _brm_interpret(only(index.args), inner)) :
            push!(indices, _brm_interpret(index, inner))
    end
    getindex(collection, indices...)
end

_brm_interpret(::Val{:if}, x::Expr, scope::_BRMScope) =
    _brm_interpret(x.args[1], scope)::Bool ? _brm_interpret(x.args[2], scope) :
    length(x.args) == 3 ? _brm_interpret(x.args[3], scope) : nothing
_brm_interpret(::Val{:elseif}, x::Expr, scope::_BRMScope) = _brm_interpret(Val(:if), x, scope)
_brm_interpret(::Val{:&&}, x::Expr, scope::_BRMScope) =
    _brm_interpret(x.args[1], scope)::Bool && _brm_interpret(x.args[2], scope)
_brm_interpret(::Val{:||}, x::Expr, scope::_BRMScope) =
    _brm_interpret(x.args[1], scope)::Bool || _brm_interpret(x.args[2], scope)

# `a < b <= c` compares pairwise and stops at the first false.
function _brm_interpret(::Val{:comparison}, x::Expr, scope::_BRMScope)
    left = _brm_interpret(x.args[1], scope)
    for i in 2:2:length(x.args)
        right = _brm_interpret(x.args[i + 1], scope)
        _brm_callee(x.args[i], scope)(left, right) || return false
        left = right
    end
    true
end

_brm_interpret(::Val{:isdefined}, x::Expr, scope::_BRMScope) =
    _brm_isdefined(scope, only(x.args)::Symbol)
_brm_interpret(::Val{:string}, x::Expr, scope::_BRMScope) = string(_brm_values(x.args, scope)...)
_brm_interpret(::Val{:curly}, x::Expr, scope::_BRMScope) =
    Core.apply_type(_brm_values(x.args, scope)...)
_brm_interpret(::Val{:(::)}, x::Expr, scope::_BRMScope) = length(x.args) == 2 ?
    typeassert(_brm_interpret(x.args[1], scope), _brm_interpret(x.args[2], scope)) :
    _brm_interpret(Val(:type_annotation), x, scope)

# `quote … end` is a fresh copy of the code with `$x` interpolated.
_brm_interpret(::Val{:quote}, x::Expr, scope::_BRMScope) = _brm_quote(only(x.args), scope, 0)
_brm_quote(x, scope::_BRMScope, depth::Int) = x
function _brm_quote(x::Expr, scope::_BRMScope, depth::Int)
    Meta.isexpr(x, :$) && depth == 0 && return _brm_interpret(only(x.args), scope)
    inner = depth + Meta.isexpr(x, :quote) - Meta.isexpr(x, :$)
    Expr(x.head, (_brm_quote(a, scope, inner) for a in x.args)...)
end

# ---- anonymous functions ------------------------------------------------------------

# `(a, b) -> body`, closing over the scope it is written in. It is an ordinary
# value of a type BRM defines, so making one defines no method.
struct _BRMClosure <: Function
    parameters::Vector{Symbol}
    body::Any
    scope::_BRMScope
end
function _brm_interpret(::Val{:->}, x::Expr, scope::_BRMScope)
    signature, body = x.args
    parameters = signature isa Symbol ? Any[signature] :
        Meta.isexpr(signature, :tuple) ? signature.args : Any[nothing]
    all(p -> p isa Symbol, parameters) || _brm_interpret(Val(:function_signature), x, scope)
    _BRMClosure(Symbol[parameters...], body, scope)
end
function (f::_BRMClosure)(args...)
    length(args) == length(f.parameters) || throw(MethodError(f, args))
    scope = _brm_child_scope(f.scope)
    for (parameter, arg) in zip(f.parameters, args)
        scope.vars[parameter] = arg
    end
    _brm_interpret(f.body, scope)
end

# ---- generators and comprehensions --------------------------------------------------

# `(body for x in xs, y in ys if cond)`: one fresh scope per element binds the
# iteration variables, as a compiled generator's closure does.
function _brm_interpret(::Val{:generator}, x::Expr, scope::_BRMScope)
    body = x.args[1]
    specs = x.args[2:end]
    condition = nothing
    if length(specs) == 1 && Meta.isexpr(specs[1], :filter)
        condition = specs[1].args[1]
        specs = specs[1].args[2:end]
    end
    all(s -> Meta.isexpr(s, :(=), 2), specs) || _brm_interpret(Val(:generator_spec), x, scope)
    targets = [s.args[1] for s in specs]
    iterators = [_brm_interpret(s.args[2], scope) for s in specs]
    function bind(item)
        inner = _brm_child_scope(scope)
        values = length(targets) == 1 ? (item,) : item
        for (target, value) in zip(targets, values)
            _brm_bind_local!(inner, target, value, x)
        end
        inner
    end
    items = length(iterators) == 1 ? only(iterators) : Iterators.product(iterators...)
    isnothing(condition) ||
        (items = Iterators.filter(item -> _brm_interpret(condition, bind(item))::Bool, items))
    Base.Generator(item -> _brm_interpret(body, bind(item)), items)
end
# A generator's variables are new locals of its own scope, never the
# enclosing scope's names.
function _brm_bind_local!(scope::_BRMScope, target, value, statement)
    for name in _brm_assigned_names!(Symbol[], target)
        scope.vars[name] = _BRMUnassigned()
    end
    _brm_bind!(scope, target, value, statement)
end
_brm_assigned_names!(names, x::Symbol) = push!(names, x)
function _brm_assigned_names!(names, x::Expr)
    Meta.isexpr(x, (:tuple, :parameters)) &&
        foreach(a -> _brm_assigned_names!(names, a), x.args)
    names
end
_brm_assigned_names!(names, _) = names

_brm_interpret(::Val{:flatten}, x::Expr, scope::_BRMScope) =
    Iterators.flatten(_brm_interpret(only(x.args), scope))
_brm_interpret(::Val{:comprehension}, x::Expr, scope::_BRMScope) =
    collect(_brm_interpret(only(x.args), scope))
_brm_interpret(::Val{:typed_comprehension}, x::Expr, scope::_BRMScope) =
    collect(_brm_interpret(x.args[1], scope), _brm_interpret(x.args[2], scope))
