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
Inside an observed family's argument expression, nested `@brm(expr)` is the
explicit opt-in for a coefficient-bearing predictor formula. Unmarked family
arguments remain ordinary expressions.

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
    esc(_brm(x; df=df))
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
    :(hasproperty($lhs, $qrhs) ? $x : $NamedColumn($(Meta.quot(qrhs.value)), $MissingColumn()))
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
    gp(x...; cov=:exp_quad, iso=true, jitter=1e-9)

Exact latent Gaussian-process predictor for the StanBlocks backend. Accepts
one-or-more real-valued axes, with an isotropic squared-exponential kernel by
default; set `iso=false` for one length scale per axis. `jitter` stabilizes the
covariance Cholesky factor. Dispatch tag — lowering lives in `_sb_gp` /
`_sb_gp_aniso` (sbimpl).
"""
function gp end

"""
    hsgp(x...; k=20, c=1.5, cov=:exp_quad, iso=true, by=nothing)

Hilbert-space approximate Gaussian-process predictor. The StanBlocks backend
supports variadic axes, per-axis `k`/`c` tuples, isotropic or anisotropic length
scales, and optional group-specific basis weights via `by=`. In formulas this
method is used only as a dispatch tag.
"""
hsgp(args...; kwargs...) = error("hsgp is a formula marker and cannot be called directly")

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
    (x.head == :block || x.head == :(=) || isxcall(x, :~)) || error(
        "@brm: a standalone predictor fragment such as `@brm(1 + x)` is " *
        "not implemented. Use nested `@brm(1 + x)` inside an outer `@brm` " *
        "likelihood, or write a top-level `lhs ~ rhs` formula.")
    lhs, x = x.head == :(=) ? x.args : (:($(gensym("model"))(__df__)), x)
    alllocals = OrderedDict{Symbol,Symbol}()
    info = (;alllocals)
    x = parse!(x; info)
    nonlocals = [key for (key, value) in pairs(alllocals) if value == :nonlocal]
    maybelocals = [key for (key, value) in pairs(alllocals) if value == :maybelocal]
    finalize = :($_expand_nested_predictor_formulas((;$(keys(alllocals)...))))
    # ONE shared builder body, parameterised on the `__df__` symbol. Nonlocals
    # bind via the @getproperty (hasproperty→MissingColumn) fallback so a name
    # that isn't a df column (e.g. a multi-equation predictor/param like
    # `loc`/`err`) becomes a MissingColumn instead of erroring; `data` and
    # `maybedata` are interpolated as values so the generated code resolves in
    # ANY consumer scope (`data` is unexported). Both forms below reuse THIS
    # body, so the two builder forms cannot diverge.
    body = Expr(:block,
        :(__ddf__ = $data(__df__)),
        [:($nonlocal = @getproperty __ddf__.$nonlocal) for nonlocal in nonlocals]...,
        :((;$(maybelocals...)) = $maybedata(__df__)),
        x,
        finalize,
    )
    if isnothing(df)
        # no-df: `gensym_model(__df__) = body` — a reusable `df -> BRMI` builder.
        Expr(:(=), lhs, body)
    else
        # baked: the SAME body with `__df__` bound to the literal df.
        Expr(:let, Expr(:block, :(__df__ = $df)), body)
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

_is_nested_brm(x) = Meta.isexpr(x, :macrocall) &&
                    !isempty(x.args) && x.args[1] === Symbol("@brm")
_contains_nested_brm_syntax(x) = false
_contains_nested_brm_syntax(x::Expr) =
    _is_nested_brm(x) || any(_contains_nested_brm_syntax, x.args)
function _nested_brm_payload(x::Expr)
    _is_nested_brm(x) || error("@brm: expected a nested `@brm(expr)` marker")
    length(x.args) == 3 || error(
        "@brm: nested `@brm` accepts exactly one parenthesized predictor " *
        "expression, got $(max(length(x.args) - 2, 0)) payloads")
    payload = x.args[3]
    if payload isa Expr &&
       (payload.head == :block || payload.head == :(=) || isxcall(payload, :~))
        error("@brm: nested `@brm(...)` marks one predictor expression; " *
              "blocks, assignments, and `~` statements belong in the outer model")
    end
    _contains_nested_brm_syntax(payload) && error(
        "@brm: a nested predictor formula cannot contain another nested `@brm`")
    payload
end

parselocals!(x; kwargs...) = x
parselocals!(x::Symbol; info, val) = get!(info.alllocals, x, val)
parselocals!(x::Expr; info, val) = if Meta.isexpr(x, :->)
    # Shield a `do`-block lambda (see `_x`): its params are bound inside the
    # block and its body is verbatim SLIC — registering those symbols as data
    # columns would be wrong. Genuine outer params are registered by their own
    # `~`/`=` lines.
    x
elseif _is_nested_brm(x)
    parselocals!(_nested_brm_payload(x); info, val)
elseif Meta.isexpr(x, (:call, :kw))
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
elseif _is_nested_brm(x)
    Expr(:call, NestedPredictorFormula, _x(_nested_brm_payload(x)))
elseif x.head ==  :||
    Expr(:call, ExprColumn, doublepipe, _x.(x.args)...)
elseif x.head == :do
    # At macro-expansion time `term(args...) do params ... end` is
    # `Expr(:do, term(args...), lambda)` (the lambda-first-arg lowering happens
    # LATER). Fold the lambda in as the term's FIRST positional arg, then process
    # the resulting call normally — so the term ExprColumn carries the do-block.
    call, lambda = x.args
    if Meta.isexpr(call, :call)
        newargs = copy(call.args)
        pos = (length(newargs) >= 2 && Meta.isexpr(newargs[2], :parameters)) ? 3 : 2
        insert!(newargs, pos, lambda)
        _x(Expr(:call, newargs...))
    else
        Expr(x.head, _x.(x.args)...)
    end
elseif x.head == :->
    # The folded-in do-block lambda: capture it VERBATIM as an Expr rather than
    # recursing — its body is SLIC to splice into the emitted plate, NOT more @brm
    # formula. No existing formula uses a bare lambda (regression-safe).
    Meta.quot(x)
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
    NestedPredictorFormula(parent)

Internal parser node for an explicit nested `@brm(expr)` predictor formula.
The outer builder consumes every such node before returning a [`BRMI`](@ref),
replacing it with ordinary named `~` operations. It is intentionally not
exported and never reaches a backend.
"""
struct NestedPredictorFormula{P} <: AbstractColumn
    parent::P
end
Base.parent(x::NestedPredictorFormula) = getfield(x, :parent)

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
    _check_term_kwargs(f, kwargs) -> nothing

Per-term keyword validation, run for EVERY [`ExprColumn`](@ref) as it is
constructed — i.e. while `@brm` builds the model, which is the first moment a
consumer can be told anything at all. The default method accepts everything;
a term that has retired part of its keyword surface adds a method next to the
term itself (see `kernel(...)` in `sbimpl.jl`).

Why here and not only in the backend: `@brm` is a pure parser, so a keyword it
does not recognise is captured into [`getkwargs`](@ref) unexamined and only the
backend emitter ever objects — which means a model written with retired syntax
constructs cleanly and complains solely once it is LOWERED (`SBBRMI(...)`).
A consumer whose compatibility gate stops at BRMI construction then sees
retired syntax pass silently.
"""
_check_term_kwargs(f, kwargs) = nothing

"""
    ExprColumn(f, args...; kwargs...)

Represents an `f(args...; kwargs...)` formula RHS — the leaves the
`@brm` parser builds for every `:call` Expr (e.g. `Normal(loc, err)`,
`1 + a + (1|g)`, `mo(c)`). Access via [`getf`](@ref) / [`getargs`](@ref)
/ [`getkwargs`](@ref) / [`getop`](@ref).

Construction runs [`_check_term_kwargs`](@ref) so a term can reject a retired
keyword at the `@brm` call site rather than in the backend.
"""
struct ExprColumn{F,A<:Tuple,K<:NamedTuple} <: AbstractColumn
    f::F
    args::A
    kwargs::K
    ExprColumn(f, args...; kwargs...) =
        (_check_term_kwargs(f, (;kwargs...));
         new{typeof(f),typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...)))
    ExprColumn(f::Type, args...; kwargs...) =
        (_check_term_kwargs(f, (;kwargs...));
         new{Type{f},typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...)))
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

The keyword args of an [`ExprColumn`](@ref). E.g. `hsgp(x; k=20, c=1.5)`
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

# Nested predictor formulas are normalised at builder execution time, after
# data columns have values (categorical K is therefore known) but before a BRMI
# becomes public. Downstream introspection, replay, descriptors, and both
# backends see only the existing explicit `eta ~ formula` representation.
_brm_fit_levels(raw::AbstractVector) = sort(unique(raw))

_nested_contains(::Any) = false
_nested_contains(::NestedPredictorFormula) = true
_nested_contains(x::NamedColumn) = _nested_contains(parent(x))
_nested_contains(x::ExprColumn) =
    any(_nested_contains, getargs(x)) || any(_nested_contains, values(getkwargs(x)))

_nested_path_token(kind::Symbol, value) = Symbol(kind, value)
_nested_predictor_name(target::Symbol, path::Tuple) =
    Symbol(target, :_nested_, join(string.(path), "_"))

function _nested_predictor!(pending, occupied, target, path, formula)
    n = _nested_predictor_name(target, path)
    n in occupied && error(
        "@brm: generated nested-predictor name `$n` collides with an existing " *
        "formula/data name. Rename that binding or write the predictor as an " *
        "explicit top-level `eta ~ ...` statement.")
    lhs = NamedColumn(n, MissingColumn())
    op = NamedColumn(n, ExprColumn(~, lhs, formula))
    push!(pending, n => op)
    push!(occupied, n)
    op
end

function _nested_rewrite!(x::NestedPredictorFormula, target, path,
                          pending, occupied)
    _nested_predictor!(pending, occupied, target, path, parent(x))
end
function _nested_rewrite!(x::ExprColumn, target, path, pending, occupied)
    args = getargs(x)
    new_args = ntuple(length(args)) do i
        token = _nested_path_token(:arg, i)
        _nested_rewrite!(args[i], target, (path..., token), pending, occupied)
    end
    kwargs = getkwargs(x)
    kw_names = keys(kwargs)
    new_kw_values = ntuple(length(kw_names)) do i
        key = kw_names[i]
        token = _nested_path_token(:kw_, key)
        _nested_rewrite!(getfield(kwargs, key), target, (path..., token),
                         pending, occupied)
    end
    new_kwargs = NamedTuple{kw_names}(new_kw_values)
    ExprColumn(getf(x), new_args...; new_kwargs...)
end
_nested_rewrite!(x, _target, _path, _pending, _occupied) = x

_nested_observed_values(lhs::NamedColumn) =
    parent(lhs) isa DataColumn ? parent(parent(lhs)) : nothing
function _nested_observed_values(lhs::ExprColumn)
    args = getargs(lhs)
    length(args) == 1 ? _nested_observed_values(only(args)) : nothing
end
_nested_observed_values(_) = nothing

function _nested_rewrite_categorical!(target, lhs, rhs, pending, occupied)
    args = getargs(rhs)
    kwargs = getkwargs(rhs)
    has_marker = any(_nested_contains, args) || any(_nested_contains, values(kwargs))
    has_marker || return rhs
    (length(args) == 1 && only(args) isa NestedPredictorFormula && isempty(kwargs)) ||
        error("@brm: concise `CategoricalLogit` requires exactly one direct " *
              "marker: `CategoricalLogit(@brm(formula))`. Do not mix marked " *
              "and explicit class predictors.")

    raw = _nested_observed_values(lhs)
    raw isa AbstractVector || error(
        "@brm: `CategoricalLogit(@brm(...))` needs an observed outcome vector " *
        "to determine its non-reference classes; use explicit predictors when " *
        "the outcome schema is unavailable.")
    levels = _brm_fit_levels(raw)
    length(levels) >= 2 || error(
        "@brm: `CategoricalLogit(@brm(...))` needs at least two observed " *
        "outcome levels (got $(length(levels))).")

    formula = parent(only(args))
    refs = ntuple(length(levels) - 1) do j
        class_token = _nested_path_token(:class, j + 1)
        _nested_predictor!(pending, occupied, target,
                           (_nested_path_token(:arg, 1), class_token), formula)
    end
    ExprColumn(getf(rhs), refs...)
end

function _nested_rewrite_likelihood!(target, lhs, rhs::ExprColumn,
                                     pending, occupied)
    if isdefined(@__MODULE__, :CategoricalLogit) && getf(rhs) === CategoricalLogit
        return _nested_rewrite_categorical!(target, lhs, rhs, pending, occupied)
    end
    _nested_rewrite!(rhs, target, (), pending, occupied)
end
_nested_rewrite_likelihood!(target, _lhs, rhs, pending, occupied) =
    rhs isa NestedPredictorFormula ? error(
        "@brm: nested `@brm(...)` must be an argument of the observed " *
        "family, for example `y ~ Normal(@brm(1 + x), 1)`.") : rhs

function _expand_nested_top!(key, value, pending, occupied)
    value isa NamedColumn || return value
    op = parent(value)
    op isa ExprColumn || return value
    _nested_contains(op) || return value
    getf(op) === (~) || error(
        "@brm: nested `@brm(...)` is only valid inside an observed likelihood " *
        "argument, not inside `$key = ...`.")
    lhs, rhs = getargs(op, 2)
    isnothing(_nested_observed_values(lhs)) && error(
        "@brm: nested `@brm(...)` is only valid inside an observed likelihood " *
        "argument, not the linear predictor `$key ~ ...`.")
    new_rhs = _nested_rewrite_likelihood!(key, lhs, rhs, pending, occupied)
    NamedColumn(name(value), ExprColumn(~, lhs, new_rhs))
end

function _expand_nested_predictor_formulas(operations::NamedTuple)
    any(_nested_contains, values(operations)) || return BRMI(operations)
    occupied = Set{Symbol}(keys(operations))
    expanded = Pair{Symbol,Any}[]
    for (key, value) in pairs(operations)
        pending = Pair{Symbol,Any}[]
        new_value = _expand_nested_top!(key, value, pending, occupied)
        append!(expanded, pending)
        push!(expanded, key => new_value)
    end
    names = Tuple(first.(expanded))
    values_ = Tuple(last.(expanded))
    BRMI(NamedTuple{names}(values_))
end

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
