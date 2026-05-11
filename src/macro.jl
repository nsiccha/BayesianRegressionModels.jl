using OrderedCollections

"""
    @brm formula_block
    @brm df formula_block

Parse a brms-style formula block into a [`BRMI`](@ref) (BRM Intermediate).

The one-argument form returns a function `model(df) -> BRMI` (the model
name is gensym-ed). The two-argument form bakes `df` in and returns the
`BRMI` directly. Inside `formula_block`, each line is one of:

- `lhs ~ rhs` — a sampling statement (likelihood or linear predictor).
- `lhs = rhs` — a literal binding (named intermediate).

Multiple `~` lines produce a multi-response / distributional model.

```julia
brmi = @brm df begin
    y ~ Normal(loc, err)
    loc ~ 1 + age + (1 | subj)
    err ~ Exponential(1)
end
```

The resulting `BRMI` feeds either backend: `VBRMI(brmi)` (vectorised
Julia) or `SBBRMI(brmi)` (StanBlocks → Stan).
"""
macro brm(x)
    esc(_brm(x))
end
macro brm(df, x)
    esc(Expr(:call, _brm(x), df))
end

"""
    @n lhs = rhs

Lower a parsed-formula LHS into a [`NamedColumn`](@ref) binding. Used
internally by [`@brm`](@ref); not normally called by hand.
"""
macro n(x)
    esc(_n(x))
end

"""
    @x expr

Lower a parsed-formula RHS into the column-type tree
([`ExprColumn`](@ref) / [`NamedColumn`](@ref)). Used internally by
[`@brm`](@ref); not normally called by hand.
"""
macro x(x)
    esc(_x(x))
end

"""
    @getproperty obj.field

Expands to `hasproperty(obj, :field) ? obj.field : :field` — used by the
`@brm` `df`-baked path to fall back from "column from the dataframe" to
"bare symbol" without erroring on missing columns.
"""
macro getproperty(x)
    esc(_getproperty(x))
end
_getproperty(x::Expr) = begin 
    @assert x.head == :(.)
    @assert length(x.args) == 2
    lhs, qrhs = x.args
    :(hasproperty($lhs, $qrhs) ? $x : $(qrhs.value))
end
begin
isxcall(x, f) = Meta.isexpr(x, :call) && x.args[1] == f
fixcall(x) = x
fixcall(x::Expr) = if Meta.isexpr(x, :call)
    f = x.args[1]
    pargs = []
    args = []
    for arg in fixcall.(x.args[2:end])
        if Meta.isexpr(arg, :parameters)
            append!(pargs, arg.args)
        else
            push!(args, arg)
        end
    end
    if length(pargs) > 0
        Expr(x.head, f, Expr(:parameters, pargs...), args...)
    else
        Expr(x.head, f, args...)
    end
else
    Expr(x.head, fixcall.(x.args)...)
end
"""Marker function for literal-binding (`lhs = rhs`) operations in a `@brm` block. Dispatch tag only — never called."""
function assign end

"""Marker function for the brms `||` zero-correlation random-effects operator. Dispatch tag only — never called."""
function doublepipe end

"""
    gr(group; by=strata)

brms-style stratified grouping marker. `(1 | gr(subj, by=diagnosis))`
gives each level of `by` its own random-effect covariance structure.
Dispatch tag only — backend interpretation lives in `vmeta_sampling_rhs`
(vimpl) / the sbimpl ranef walker.
"""
function gr end

"""
    gp(x; k=K, c=C)

Hilbert-space-approx 1D Gaussian-process predictor (Riutort-Mayol et al.
2022). Allocates a K-basis squared-exponential GP over `x`. Dispatch
tag — see `vmeta_sampling_rhs(::ExprColumn{typeof(gp)}, …)` (vimpl) and
`_sb_hsgp` (sbimpl).
"""
function gp end

"""
    offset(x)

Fixed-slope (no beta) contribution to a linear predictor. brms-style
`offset(...)` — adds `x` directly to the predictor without allocating
any parameters. Dispatch tag only.
"""
function offset end

"""Marker for `zscale(x)` — z-standardise `x` (subtract mean, divide by SD). Dispatch tag only."""
function zscale end

"""Marker for `center(x)` — mean-centre `x`. Dispatch tag only."""
function center end

"""Marker for `standardize(x)` — alias-style mean/SD standardisation. Dispatch tag only."""
function standardize end

"""
    protect(x)

brms `I(...)` analogue: literal-escape wrapper around an expression so
the formula parser doesn't try to re-interpret it. In formula-eval
contexts it's a no-op on `Real` — generic broadcast/materialise paths
handle `protect(expr)` without a dedicated dispatch.
"""
function protect end
protect(x::Real) = x

"""
    factor(x; ref=k)

Treat `x` as a categorical predictor with optional reference level
`ref`. Dispatch tag — backend buckets it into the categorical-predictor
path in `introspection.jl` and `_sb_cat` (sbimpl).
"""
function factor end

"""
    mi(y)

Missing-data marker. Wrap a response LHS (`mi(y) ~ Normal(...)`) to opt
into the brms-style observed/imputed split: observed rows feed the
likelihood, missing rows become parameters drawn from the same family.
See `_sb_emit_mi!` (sbimpl) for backend dispatch.
"""
function mi end
"""
    _brm(formula::Union{Expr,AbstractString}; df=nothing) -> Expr

The macro-free entry point behind [`@brm`](@ref). Parses a formula block
and returns either a function definition (`df=nothing`, default) or a
`let`-wrapped BRMI construction (`df` supplied). Useful when you want to
construct the formula expression programmatically rather than via the
macro.
"""
_brm(x::AbstractString; kwargs...) = _brm(Meta.parse("""
begin
    $x
end
"""); kwargs...)
_brm(x::Expr; df=nothing) = begin
    lhs, x = x.head == :(=) ? x.args : (:($(gensym("model"))(__df__)), x)
    alllocals = OrderedDict{Symbol,Symbol}()
    info = (;alllocals)
    x = parse!(x; info)
    nonlocals = [key for (key, value) in pairs(alllocals) if value == :nonlocal]
    maybelocals = [key for (key, value) in pairs(alllocals) if value == :maybelocal]
    init = quote
        (;$(nonlocals...)) = data(__df__)
        (;$(maybelocals...)) = maybedata(__df__)
    end
    finalize = quote
        $BRMI(;$(keys(alllocals)...))
    end
    if isnothing(df)
        Expr(:(=), lhs, Expr(:block, init, x, finalize))
    else
        Expr(:let, 
            Expr(:block, :(__df__ = $df), :(__ddf__ = $data(__df__)), [:($nonlocal = @getproperty __ddf__.$nonlocal) for nonlocal in nonlocals]...), 
            Expr(:block, :((;$(maybelocals...)) = maybedata(__df__)), x, finalize)
        )
    end
end
brm(df, formula::AbstractString) = eval(_brm(formula; df))

"""
    parse!(x; info) -> Expr

Walk a formula expression once, classifying every bare `Symbol` into
`info.alllocals` (as `:nonlocal` for data columns, `:local` for literal
bindings, `:maybelocal` for response-side names that may or may not be
observed). Returns the rewritten expression with `~`/`=` statements
lowered into [`@n`](@ref)/[`@x`](@ref) calls. Used internally by
[`_brm`](@ref).
"""
parse!(x; info) = x
parse!(x::Expr; info) = if x.head == :block
    Expr(:block, parse!.(x.args; info)...)
elseif x.head == :(=)
    lhs, rhs = x.args
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:local)
    :(@n $lhs = @x $assign($(xname(lhs)), $rhs))
elseif isxcall(x, :~)
    _, lhs, rhs = x.args
    # Shield brms-style `(e | ID | g)` ranef IDs from parselocals! so the bare
    # ID symbol doesn't get registered as a data-column name. Left-associative
    # `|` lowers to `Expr(:call, :|, Expr(:call, :|, e, id), g)`; rewrite to a
    # three-arg `|` whose middle is a QuoteNode so parselocals! treats it as a
    # literal, not a bare Symbol. _x then sees the three-arg form and emits an
    # ExprColumn{|} with the id as a Symbol value.
    rhs = rewrite_ranef_ids(rhs)
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:maybelocal)
    :(@n $lhs = @x $(Expr(:call, :~, lhs, rhs)))
else
    dump(x)
    error("Don't know how to handle parse!($x)!")
end
rewrite_ranef_ids(x) = x
rewrite_ranef_ids(x::Expr) = if isxcall(x, :|) && length(x.args) == 3 &&
                                isxcall(x.args[2], :|) && length(x.args[2].args) == 3
    _quote_ranef_id(x.args[2].args[3], x)
else
    Expr(x.head, rewrite_ranef_ids.(x.args)...)
end
# Dispatches the brms ranef-ID rewrite on the type of the candidate ID.
# Symbol → wrap as QuoteNode and rebuild the three-arg `|`. Anything else
# (already a QuoteNode, an Expr, …) → fall through to the generic recursion.
_quote_ranef_id(id_sym::Symbol, x::Expr) = begin
    inner = x.args[2]
    lhs = rewrite_ranef_ids(inner.args[2])
    g = rewrite_ranef_ids(x.args[3])
    Expr(:call, :|, lhs, QuoteNode(id_sym), g)
end
_quote_ranef_id(_, x::Expr) = Expr(x.head, rewrite_ranef_ids.(x.args)...)
parselocals!(x; kwargs...) = x
parselocals!(x::Symbol; info, val) = get!(info.alllocals, x, val)
parselocals!(x::Expr; info, val) = if Meta.isexpr(x, (:call, :kw))
    parselocals!.(x.args[2:end]; info, val)
else
    parselocals!.(x.args; info, val)
end
_n(x::Expr) = begin 
    @assert x.head == :(=)
    lhs, rhs = x.args
    alhs = xassignable(lhs)
    nlhs = xname(alhs)
    :($alhs = $NamedColumn($nlhs, $rhs))
end
xassignable(x::Symbol) = x
xassignable(x::Expr) = if Meta.isexpr(x, (:tuple, :vect))
    Expr(x.head, xassignable.(x.args)...)
elseif x.head == :call 
    if length(x.args) == 2
        xassignable(x.args[2])
    else
        @warn "Don't know how to handle xassignable($x)!"
        Symbol(x)
    end
else
    dump(x)
    error("Don't know how to handle xassignable($x)!")
end
xname(x::Symbol) = Meta.quot(x)
xname(x::Expr) = if Meta.isexpr(x, (:tuple, :vect))
    Expr(x.head, xname.(x.args)...)
else
    @warn "Don't know how to handle xassignable($x)!"
    Symbol(x)
    # dump(x)
    # error("Don't know how to handle xname($x)!")
end
_x(x) = x
_x(x::Symbol) = x
_x(x::Expr) = if x.head == :call
    Expr(:call, ExprColumn, _x.(x.args)...) |> fixcall
elseif x.head ==  :||
    Expr(:call, ExprColumn, doublepipe, _x.(x.args)...)
else
    Expr(x.head, _x.(x.args)...)
end
"""
    Data(df)

Wrapper that exposes every column of the underlying DataFrame as a
[`NamedColumn`](@ref) over a [`DataColumn`](@ref). Used by [`@brm`](@ref)
to lift `df` columns into the formula's column-type universe.
"""
struct Data{P}
    parent::P
end
Base.parent(d::Data) = getfield(d, :parent)
Base.hasproperty(d::Data, x::Symbol) = hasproperty(parent(d), x)
Base.getproperty(d::Data, x::Symbol) = NamedColumn(x, DataColumn(getproperty(parent(d), x)))
data(x) = Data(x)

"""
    MaybeData(df)

Like [`Data`](@ref), but for columns that may not be present in the
dataframe. Missing columns surface as [`NamedColumn`](@ref) wrapping a
[`MissingColumn`](@ref) — i.e. a parameter to be sampled rather than
observed.
"""
struct MaybeData{P}
    parent::P
end
Base.parent(d::MaybeData) = getfield(d, :parent)
Base.hasproperty(d::MaybeData, x::Symbol) = hasproperty(parent(d), x)
Base.getproperty(d::MaybeData, x::Symbol) = NamedColumn(x, hasproperty(d, x) ? DataColumn(getproperty(parent(d), x)) : MissingColumn())

"""
    maybedata(df) -> MaybeData

Convenience constructor for [`MaybeData`](@ref). Used by [`@brm`](@ref)
to surface response-side names that may be data (observed) or
parameters (sampled).
"""
maybedata(x) = MaybeData(x)

"""
    AbstractColumn

Supertype of every column-shape token the `@brm` parser produces.
Subtypes: [`MissingColumn`](@ref), [`DataColumn`](@ref),
[`NamedColumn`](@ref), [`ExprColumn`](@ref), [`LikelihoodColumn`](@ref),
[`MaterializedColumn`](@ref).

Backend dispatchers (`vmeta_sampling_rhs`, `_sb_emit!`, the
introspection walkers) branch on this hierarchy rather than on
`Symbol` tags.
"""
abstract type AbstractColumn end

"""
    MissingColumn()

LHS marker for response columns that aren't in the dataframe — the
backend treats them as parameters rather than observations.
"""
struct MissingColumn <: AbstractColumn end

"""
    DataColumn(vec)

Wraps an observed vector from the dataframe. Surfaces as the leaf of a
[`NamedColumn`](@ref) for any data-backed predictor or response.
"""
struct DataColumn{P} <: AbstractColumn
    parent::P
end
Base.parent(d::DataColumn) = getfield(d, :parent)

"""
    NamedColumn(name, parent)

Assigns a formula-local symbol `name` to its `parent` column. Every
`lhs ~ rhs` and `lhs = rhs` line produces a `NamedColumn` keyed by
`lhs`. Access via [`name`](@ref) / [`parent`](@ref).
"""
struct NamedColumn{N,P} <: AbstractColumn
    name::N
    parent::P
end

"""
    name(x::NamedColumn) -> Symbol

Return the formula-local name assigned to `x` (i.e. the LHS of the
`~` / `=` line that produced it).
"""
name(x::NamedColumn) = getfield(x, :name)
Base.parent(x::NamedColumn) = getfield(x, :parent)

"""
    ExprColumn(f, args...; kwargs...)

Represents an `f(args...; kwargs...)` formula RHS — the leaves the
`@brm` parser builds for every `:call` Expr (e.g. `Normal(loc, err)`,
`1 + a + (1|g)`, `mo(c)`). Access via [`getf`](@ref) / [`getargs`](@ref)
/ [`getkwargs`](@ref) / [`getop`](@ref).
"""
struct ExprColumn{F,A<:Tuple,K<:NamedTuple} <: AbstractColumn
    f::F
    args::A
    kwargs::K
    ExprColumn(f, args...; kwargs...) = new{typeof(f),typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...))
    ExprColumn(f::Type, args...; kwargs...) = new{Type{f},typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...))
end

"""
    getf(x::ExprColumn) -> F

The callable head of an [`ExprColumn`](@ref) — e.g. `Normal`, `+`, `~`,
`assign`, `mo`. Sole dispatch tag for backend emitters.
"""
getf(x::ExprColumn) = getfield(x, :f)

"""
    getargs(x::ExprColumn) -> Tuple
    getargs(x::ExprColumn, n) -> Tuple

The positional args of an [`ExprColumn`](@ref). The two-arg form asserts
`length == n`. Plus helpers for `+`-flattening:
`getargs(+, x::ExprColumn{typeof(+)})` returns the summands;
`getargs(+, x)` wraps non-`+` shapes as a 1-tuple.
"""
getargs(x::ExprColumn) = getfield(x, :args)
getargs(x::ExprColumn, n) = (rv = getargs(x); @assert length(rv) == n; rv)
getargs(::typeof(+), x::ExprColumn{typeof(+)}) = getargs(x)
getargs(::typeof(+), x::ExprColumn) = (x,)
getargs(::typeof(+), x) = (x,)

"""
    getkwargs(x::ExprColumn) -> NamedTuple

The keyword args of an [`ExprColumn`](@ref). E.g. `gp(x; k=20, c=1.5)`
exposes `(; k=20, c=1.5)`.
"""
getkwargs(x::ExprColumn) = getfield(x, :kwargs)

"""
    getop(x) -> Symbol_or_Function

Symbolic operator name for an [`ExprColumn`](@ref): `:||` for the
[`doublepipe`](@ref) marker, `:(=)` for [`assign`](@ref), otherwise
falls through to [`getf`](@ref). Used by the HTML renderer to print
formula RHS with the original operator surface.
"""
getop(x) = getf(x)
getop(::ExprColumn{typeof(doublepipe)}) = :||
getop(::ExprColumn{typeof(assign)}) = :(=)

"""
    LikelihoodColumn(parent, rhs)

Pairs an observed-response `parent` ([`NamedColumn`](@ref) over a
[`DataColumn`](@ref)) with its distributional `rhs`. vimpl's
`llikelihood!` walks these.
"""
struct LikelihoodColumn{P,R} <: AbstractColumn
    parent::P
    rhs::R
end
Base.parent(d::LikelihoodColumn) = getfield(d, :parent)
rhs(d::LikelihoodColumn) = getfield(d, :rhs)
maybedists(lhs::AbstractColumn, x::AbstractColumn) = LikelihoodColumn(lhs, x)

"""
    BRMI(; ops...)
    BRMI(operations::NamedTuple)

BRM Intermediate — the parsed formula. `operations` is a NamedTuple
mapping each LHS name to its [`NamedColumn`](@ref). The two backends
[`VBRMI`](@ref) and [`SBBRMI`](@ref) walk this representation.

A `BRMI` is what [`@brm`](@ref) ultimately returns (either directly,
when `df` is baked in, or by calling the gensym'd model function on a
dataframe). Introspect via [`outcomes`](@ref) / [`linear_predictors`](@ref)
/ [`predictors`](@ref) / [`grouping_factors`](@ref) /
[`dependencies`](@ref) / [`data_columns`](@ref).
"""
struct BRMI{O<:NamedTuple}
    operations::O
end
BRMI(;kwargs...) = BRMI((;kwargs...))
Base.show(io::IO, (;operations)::BRMI) = begin
    print(io, "BRMI:\n")
    for (key, value::NamedColumn) in pairs(operations)
        print(io, "  ")
        _show_top(io, key, parent(value))
        print(io, "\n")
    end
end
# Top-level entries in a BRMI listing are either ExprColumns whose own show
# already names the LHS (`loc ~ ...`, `(err = ...)`) or other column types
# (DataColumn, MaterializedColumn, ...) whose show prints just the value with
# no name. For the former we strip the outermost parens; for the latter we
# prefix with the operation key so the name doesn't get lost.
_show_top(io::IO, key, op) = print(io, key, ": ", op)
_show_top(io::IO, key, op::ExprColumn{<:Union{typeof(~),typeof(assign)}}) =
    join(io, getargs(op), " $(getop(op)) ")
Base.show(io::IO, d::DataColumn) = begin
    print(io, "data (eltype=", eltype(parent(d)), ")")
end
Base.show(io::IO, x::ExprColumn{<:Union{typeof.((~,*,+,|,doublepipe,assign))...}}) = begin
    print(io, "(", )
    join(io, getargs(x), " $(getop(x)) ")
    print(io, ")")
end
nonemptyjoin(io::IO, iterator, args...; first) = if length(iterator) > 0
    print(io, first)
    join(io, iterator, args...)
end
Base.show(io::IO, x::ExprColumn) = begin
    print(io, getf(x), "(", )
    join(io, getargs(x), ", ")
    nonemptyjoin(io, ["$key=$value" for (key, value) in pairs(getkwargs(x))], ", "; first="; ")
    print(io, ")")
end
Base.show(io::IO, x::NamedColumn) = print(io, name(x))

end