# ==============================================================================
# descriptor.jl — ONE authoritative executable semantic model descriptor.
#
# A `@brm` block is already the authoritative declaration of a model. Until now
# a consumer that wanted to *mount* that model — render a form, offer
# fit/predict, address the draws, label a coefficient — had to assemble the
# answer from six different places:
#
#   * a hand-kept formula string / display name,
#   * `outcomes` / `linear_predictors` / `predictors` / `grouping_factors`
#     (introspection.jl) for the model shape,
#   * `generative_plan` (sbimpl.jl) for what BRM actually emitted,
#   * `popcoefnames` for coefficient labels,
#   * `data_columns` + `sb.preproc` for which dataframe columns are needed,
#   * and StanBlocks for the executable half.
#
# `brm_descriptor` collapses all six into one value derived from the single
# declaration. Nothing here is a second source of truth: every field is read
# off the `GenerativePlan` (which is itself read off the emitted SLIC body) or
# off StanBlocks' own `stan_descriptor`. The two compose BY NAME, and every
# place the correspondence can fail is an explicit, loud failure rather than a
# guess — see "Failing closed" below.
#
# Division of labour with StanBlocks (stanblocks-use §30):
#   StanBlocks' `ModelDescriptor` reports STAN-level structure — the data
#   block, the parameter/TP/GQ outputs, and which operations the traced program
#   supports. It deliberately does not know "this is a random effect", "this
#   column was z-scaled", "this coefficient is the slope on `x`".
#   Those are BRM concepts, and they are what this layer adds.
# ==============================================================================

# ---- inputs -----------------------------------------------------------------

"""
    BRMInput

One entry of the emitted Stan data block, with its BRM provenance attached.

The first eight fields mirror StanBlocks' `ModelInput` (stanblocks-use §30) —
`name`, `type` (the Stan center type), `size` (the declared size *expressions*,
not values), `constraints`, and the four flags:

- `observed` — the model conditions on this input (it is the LHS of a `~`).
- `held_out` — this input is cv-marked, so its likelihood contribution is
  dropped and it re-draws in generated quantities.
- `derived` — this input is another input's declared *size* (`y_n` for
  `vector[y_n] y`). Re-binding `y` re-derives it; never ask a user for it.
- `inlined` — folded into the generated source, never reaches the data JSON.

The last two are BRM's addition — the **schema link** back to the dataframe:

- `column` — the dataframe column this Stan input was built from, or `nothing`
  when it has no single raw column (a design matrix, a level count, a size).
- `transform` — the BRM preprocessing kind applied on the way (`:zscale`,
  `:center`, `:standardize`, `:factor`, `:mo`, `:spline`, `:gp`, `:hsgp`,
  `:multi_membership`, `:protect`),
  or `nothing` when the column is passed through untransformed.

`column`/`transform` are read from the plan's `preproc` record and the BRMI's
own data columns — they are never inferred from the input's *name*.
"""
struct BRMInput
    name::Symbol
    type::Symbol
    size::Tuple
    constraints::NamedTuple
    observed::Bool
    held_out::Bool
    derived::Bool
    inlined::Bool
    column::Union{Nothing,Symbol}
    transform::Union{Nothing,Symbol}
end

# ---- outputs ----------------------------------------------------------------

"""
    BRMOutput

One thing the compiled model produces, with its BRM meaning attached.

Fields `name` … `source` mirror StanBlocks' `ModelOutput` (stanblocks-use §30):
`kind` is `:parameter` / `:transformed_parameter` / `:generated_quantity`, and
`generative` is `:posterior` / `:draw` / `:pointwise_loglik` / `:derived` with
`source` naming the observation a `:draw` or `:pointwise_loglik` belongs to.

BRM adds:

- `role` — what this output *means in the formula*:

  | role | what it is |
  | --- | --- |
  | `:population_effect` | a `popefs` block — the population-level design and its coefficients |
  | `:random_effect` | a `ranef_*` block and its internals |
  | `:group_block` | a `kernel(...)` / `plate` result and everything declared per cell |
  | `:parameter` | an ordinary declared prior (`sigma ~ Exponential(1)`) |
  | `:linear_predictor` | a formula linear predictor (`mu ~ 1 + x + (1\\|g)`) — an assignment, so no `~` declaration binds it |
  | `:posterior_predictive` | a predictive draw of an observation |
  | `:pointwise_loglik` | an observation's per-element log-likelihood |
  | `:stan_derived` | a Stan-level output no BRM declaration owns |

- `declaration` — the `GenerativeDeclaration` this output came from, or
  `nothing` for `:linear_predictor` / `:stan_derived`. Its twelve fields
  (family, dimension, constraints, the verbatim expression, …) are documented
  on [`GenerativeDeclaration`](@ref); reach for them instead of re-parsing.
  An output whose `name` differs from `declaration.target` is one of that
  block's *internals* (`pop_mu_beta_pop` under `pop_mu`).
- `logical` — the BRM-level quantity whose value this emitted Stan output
  physically carries, or `nothing` for an internal. This is deliberately
  narrower than `declaration`: every `kernel(...)` cell local belongs to the
  plate declaration, but only its collected result carries the logical target.
  For a ragged result, `logical == :loc` can therefore accompany an emitted
  `name` such as `:loc__pl_mem_1`; consumers keep the logical identity while
  addressing BridgeStan through the emitted name.

  For a predictive or pointwise-loglik twin it is StanBlocks' `source` — the
  OBSERVED QUANTITY — which is not always a declaration target. A top-level
  `y ~ Normal(mu, sigma)` gives `y_gen` `logical == :y`, where `:y` is also the
  declaration; a plate-nested `yy ~ normal(...)` over column `dv` gives
  `dv_gen` `logical == :dv`, where the declaration is the cell-local `yy` and
  `:dv` is a data column. Both twins of one observation share it, so
  [`brm_output`](@ref) needs `role=` to pick between them.
- `labels` — per-element labels when BRM can name them (population
  coefficients, via [`popcoefnames`](@ref)), otherwise `nothing`. A UI that
  would otherwise print `beta_pop.1`, `beta_pop.2` can print `x`, `g`.
- `segments` — group boundaries for a RAGGED quantity, carried through from
  StanBlocks' `ModelOutput.segments`, otherwise `nothing`. Group `g` occupies
  `segments[g-1]+1 : segments[g]` (with `segments[0] ≡ 0`) *within this
  output's own coordinates* — so it composes directly with the vector
  [`brm_output_coordinates`](@ref) returns, and a consumer never re-derives
  per-subject boundaries from a length column.

`role` is derived from the declaration, never from the output's name — a user
variable may legitimately be called `beta_pop` or end in `_gen`.
"""
struct BRMOutput
    name::Symbol
    kind::Symbol
    type::Symbol
    size::Tuple
    constraints::NamedTuple
    generative::Symbol
    source::Union{Nothing,Symbol}
    role::Symbol
    declaration::Union{Nothing,GenerativeDeclaration}
    logical::Union{Nothing,Symbol}
    labels::Union{Nothing,Vector{Symbol}}
    segments::Union{Nothing,Vector{Int}}
end

# Preserve the earlier positional constructors for downstream code that creates
# display-only BRMOutput values. `segments` and `logical` are appended rather
# than inserted precisely so those calls keep compiling; descriptor-derived
# values always supply both explicitly below.
BRMOutput(name, kind, type, size, constraints, generative, source, role,
          declaration, logical, labels) =
    BRMOutput(name, kind, type, size, constraints, generative, source, role,
              declaration, logical, labels, nothing)
BRMOutput(name, kind, type, size, constraints, generative, source, role,
          declaration, labels) =
    BRMOutput(name, kind, type, size, constraints, generative, source, role,
              declaration, nothing, labels, nothing)

# ---- operations -------------------------------------------------------------

"""
    BRMOperation

One executable operation the declaration supports. Operations are **derived**,
never listed: each appears exactly when the model actually supports it, so a
consumer can render one button per operation without maintaining its own list
and without rendering a button that will fail.

- `name` / `title` — the identifier and a human label.
- `inputs` — the names this operation needs. **Read `origin` to know which
  namespace they live in**: `:stan` operations take Stan data keys (the
  neither-derived-nor-inlined subset), `:brm` operations take *dataframe
  columns*.
- `outputs` — the descriptor output names it produces (empty when it produces
  a new descriptor or the Stan source rather than draws).
- `origin` — `:stan` (delegated to StanBlocks' `stan_execute`), `:brm`
  (BRM-level: replay the declaration on new data), or `:override` (supplied by
  the consumer through `brm_descriptor(...; operations=…)`).
- `run` — the callable `brm_execute` invokes, `(descriptor; kwargs...) -> result`.

See [`brm_descriptor`](@ref) for the derivation table and the extension points.
"""
struct BRMOperation
    name::Symbol
    title::String
    inputs::Tuple{Vararg{Symbol}}
    outputs::Tuple{Vararg{Symbol}}
    origin::Symbol
    run::Any
end

# ---- highlighted Stan definitions ------------------------------------------

"""
    BRMHighlight

One caller-selected Stan definition to feature when presenting a
[`BRMDescriptor`](@ref).

- `name` is the stable definition name resolved by StanBlocks.
- `caption` is optional presentation text supplied by the caller.
- `definition` is StanBlocks' authoritative definition descriptor, including
  its emitted source and direct dependency links.
- `closure` is that definition plus its exact transitive included dependencies
  in StanBlocks' authoritative emission order.

BRM never copies or parses the generated Stan source to construct either.

Pass symbols and/or `name => caption` pairs through the ordered `highlights=`
keyword of [`brm_descriptor`](@ref). The full executable definition inventory
remains available on `d.stan.definitions`; `d.highlights` is only the selected,
ordered presentation layer.
"""
struct BRMHighlight
    name::Symbol
    caption::Union{Nothing,String}
    definition::StanBlocks.ModelDefinition
    closure::Tuple{Vararg{StanBlocks.ModelDefinition}}
end

# ---- the descriptor ---------------------------------------------------------

"""
    BRMDescriptor

The one authoritative, executable, reflectable value for a `@brm` declaration.
Construct with [`brm_descriptor`](@ref).

| field | what it is |
| --- | --- |
| `id` | stable content identity — StanBlocks' descriptor id, the hash of the generated Stan source, which is also the key `instantiate` caches the compiled artifact under. Stable across processes; independent of `name` |
| `name` | informational label (`name=` at construction, else the plan's own) |
| `formula` | the canonical rendering of the parsed declaration (`show` of the `BRMI`). Not the literal characters you typed — it is *derived* from the declaration, so unlike a hand-kept `formula_src` string it cannot drift from the model that runs |
| `plan` | the [`GenerativePlan`](@ref) — every emitted `~` site, in order |
| `stan` | StanBlocks' `ModelDescriptor` for the same model, if you need the Stan-level view verbatim |
| `highlights` | ordered caller-selected [`BRMHighlight`](@ref)s, resolved by stable name against `stan.definitions`; the full included-definition inventory remains on `stan` |
| `inputs` | `Tuple` of [`BRMInput`](@ref) — the data block plus its dataframe provenance |
| `outputs` | `Tuple` of [`BRMOutput`](@ref) — everything the model produces, with BRM roles |
| `operations` | `Tuple` of [`BRMOperation`](@ref) — derived, not listed |
| `columns` | the dataframe columns the declaration reads. **This is the schema** a form should collect for a replay |
| `unpredictable` | observation targets whose predictive draw the emitted Stan program does **not** produce. Empty in the ordinary case; see below |

`brm_columns(d)`, `brm_operation(d, name)`, `brm_execute(d, name; …)` are the
accessors.
"""
struct BRMDescriptor{P,S}
    id::String
    name::Symbol
    formula::String
    plan::P
    stan::S
    highlights::Tuple{Vararg{BRMHighlight}}
    inputs::Tuple{Vararg{BRMInput}}
    outputs::Tuple{Vararg{BRMOutput}}
    operations::Tuple{Vararg{BRMOperation}}
    columns::Tuple{Vararg{Symbol}}
    unpredictable::Tuple{Vararg{Symbol}}
end

# ---- role derivation --------------------------------------------------------

# The BRM meaning of a `:prior` declaration, from the family BRM emitted. These
# are BRM's own submodel names (sbimpl.jl) — a closed set we own, not a guess
# about user code.
_brm_declaration_role(d::GenerativeDeclaration) = begin
    d.role === :observation && return :observation
    # A non-empty `context` means the declaration lives INSIDE a plate cell —
    # a per-group parameter of a `kernel(...)` block, whatever its family.
    # Formula-declared BSV is emitted by the ordinary top-level `ranef_*`
    # declarations and therefore keeps its `:random_effect` role. Only values
    # declared by the plate cell itself belong to this `:group_block` context.
    isempty(d.context) || return :group_block
    f = d.family
    f isa Symbol || return :parameter
    f in (:popefs, :_popefs_normal) && return :population_effect
    startswith(String(f), "ranef") && return :random_effect
    f === :plate && return :group_block
    :parameter
end

# ---- schema: which dataframe column does this Stan input come from? ---------

# A preproc record names its source either directly (a column NAME Symbol, for
# factor/mo/spline) or as an axis-name tuple (gp/hsgp), or as a column-node tree
# (zscale/center/standardize and
# the protect/implicit-fn fallback). Walk the tree to its single data leaf; a
# tree touching several columns has no single source column, and we say so
# rather than picking one.
_brm_raw_column(x::Symbol) = x
_brm_raw_column(x::NamedColumn) = _brm_raw_column_inner(x, parent(x))
_brm_raw_column_inner(x, ::DataColumn) = name(x)
_brm_raw_column_inner(_x, _) = nothing
_brm_raw_column(x::ExprColumn) = begin
    found = Symbol[]
    for a in getargs(x)
        c = _brm_raw_column(a)
        isnothing(c) || push!(found, c)
    end
    unique!(found)
    length(found) == 1 ? only(found) : nothing
end
_brm_raw_column(x::Tuple) = begin
    found = unique(Symbol[c for a in x for c in (_brm_raw_column(a),) if !isnothing(c)])
    length(found) == 1 ? only(found) : nothing
end
_brm_raw_column(_) = nothing

function _brm_input_schema(plan, key::Symbol, df_columns)
    e = get(plan.preproc, key, nothing)
    isnothing(e) || return (_brm_raw_column(e.raw_ref), e.kind)
    key in df_columns && return (key, nothing)
    (nothing, nothing)
end

# ---- construction -----------------------------------------------------------

_brm_plan_of(plan::GenerativePlan) = plan
_brm_plan_of(sb::SBBRMI) = generative_plan(sb)

_brm_highlight_spec(x::Union{Symbol,AbstractString}) =
    (Symbol(x), nothing, nothing)
_brm_highlight_spec(x::Pair{<:Union{Symbol,AbstractString}}) =
    (Symbol(first(x)), isnothing(last(x)) ? nothing : String(last(x)), nothing)
_brm_highlight_spec(x::BRMHighlight) =
    (x.name, x.caption, x.definition.signature)
_brm_highlight_spec(x) = error(
    "brm_descriptor: each `highlights` entry must be a definition name, a " *
    "`name => caption` pair, or a BRMHighlight; got $(repr(x)).")

function _brm_highlights(stan, specs)
    entries = specs isa Union{Symbol,AbstractString,Pair,BRMHighlight} ?
              (specs,) : specs
    selected = BRMHighlight[]
    seen = Set{Symbol}()
    for spec in entries
        name, caption, signature = _brm_highlight_spec(spec)
        name in seen && error(
            "brm_descriptor: definition `$name` is selected more than once in " *
            "`highlights`; each highlight must be unique.")
        push!(seen, name)
        definition = StanBlocks.stan_definition(stan, name; signature)
        closure = StanBlocks.stan_definition_closure(stan, definition)
        push!(selected, BRMHighlight(name, caption, definition, closure))
    end
    Tuple(selected)
end

# Label a population-effect output's elements, when BRM can. `popcoefnames`
# already owns this (and is the documented way not to re-parse `beta_pop.N`);
# it can legitimately return `nothing`, and it errors on shapes it cannot
# resolve (`hsgp(x, by=g)` and friends) — neither is a reason to fail descriptor
# construction, so an unlabelled output simply carries `labels = nothing`.
_brm_labels(brmi, lp::Symbol) =
    try
        v = popcoefnames(brmi, lp)
        isnothing(v) ? nothing : collect(Symbol, v)
    catch
        nothing
    end

# The Stan identifier a declaration resolves to.
#
# At top level that is the target itself. Inside a plate the emitted name is
# the context joined with the target — `kernel_z` declared in the `pred` cell
# becomes the Stan parameter `pred_kernel_z`. That is the same join BRM already
# uses to build a declaration's `draw`, so it is a rule we own, not a guess.
_brm_stan_name(d::GenerativeDeclaration) =
    isempty(d.context) ? d.target : Symbol(join((d.context..., d.target), "_"))

# Which Stan output does a declaration own?
#
# StanBlocks inlines a called submodel under its binding name: `pop_mu ~
# popefs(; X=X_mu)` yields the transformed parameter `pop_mu` (the submodel's
# return) AND the submodel's own internals prefixed with it
# (`pop_mu_beta_pop`, `pop_mu_n_covariates`). So a declaration owns its exact
# resolved name plus everything under `<resolved>_`.
#
# Resolution order, most authoritative first:
#   1. StanBlocks' own `source` link (a `_gen` / `_likelihood` twin). Never
#      parse the suffix ourselves — `source` is the authoritative answer.
#   2. exact name match against a declaration target,
#   3. the LONGEST `<target>_` prefix (nested submodels then resolve to the
#      innermost declaration that owns them, not an outer one).
function _brm_owner(o, by_name, targets)
    isnothing(o.source) || return get(by_name, o.source, nothing)
    haskey(by_name, o.name) && return by_name[o.name]
    best = nothing
    s = String(o.name)
    for t in targets
        p = String(t) * "_"
        startswith(s, p) || continue
        (isnothing(best) || length(String(t)) > length(String(best))) && (best = t)
    end
    isnothing(best) ? nothing : by_name[best]
end

# Find the emitted Stan outputs referenced by a traced logical binding. This
# walks StanBlocks' traced expression graph, not rendered identifiers: a ragged
# plate result is a compile-time `RaggedVector(mem, ends)` view, and only `mem`
# is a posterior-producing descriptor output. Cell locals do not occur in that
# binding even though `_brm_owner` correctly assigns them to the same plate
# declaration.
_brm_output_leaves!(found, x::Symbol, candidates) =
    x in candidates && push!(found, x)
_brm_output_leaves!(found, x::StanBlocks.StanExpr, candidates) =
    _brm_output_leaves!(found, StanBlocks.expr(x), candidates)
_brm_output_leaves!(found, x::StanBlocks.CanonicalExpr, candidates) =
    foreach(a -> _brm_output_leaves!(found, a, candidates), x.args)
_brm_output_leaves!(found, x::Expr, candidates) =
    foreach(a -> _brm_output_leaves!(found, a, candidates), x.args)
_brm_output_leaves!(found, x::Tuple, candidates) =
    foreach(a -> _brm_output_leaves!(found, a, candidates), x)
_brm_output_leaves!(_found, _x, _candidates) = nothing

function _brm_logical_outputs(stan, by_name, targets)
    logical = Dict{Symbol,Symbol}()
    output_names = Set{Symbol}(o.name for o in stan.outputs)
    model_names = Set{Symbol}(keys(stan.model))

    # One assignment point, so every branch below gets the same conflict check.
    # Assigning directly would let a second claim silently overwrite the first,
    # which is precisely the ambiguity this error exists to refuse.
    claim! = (emitted, target) -> begin
        if haskey(logical, emitted) && logical[emitted] !== target
            error(
                "brm_descriptor: emitted output `$emitted` carries two logical " *
                "targets (`$(logical[emitted])` and `$target`). The traced " *
                "model is ambiguous; give the declarations distinct results.")
        end
        logical[emitted] = target
    end

    # StanBlocks' `source` link IS the logical identity of a predictive or
    # pointwise-loglik twin, and it is authoritative — the same link
    # `_brm_owner` trusts first. Resolve on it DIRECTLY rather than through a
    # declaration, because the observed quantity is not always a declaration:
    #
    #   y ~ Normal(mu, sigma)            ->  `y_gen.source === :y`, and `:y` IS
    #                                        a declaration target.
    #
    #   pred ~ kernel(dose, dv, ...) do dd, yy, ls
    #       yy ~ normal(mu, sigma)       ->  `dv_gen.source === :dv`, but the
    #   end                                  declaration is the CELL-LOCAL `yy`
    #                                        (context `(:pred,)`), and `:dv` is
    #                                        a data COLUMN, not a target.
    #
    # Keying off declarations covered only the first shape, so every
    # plate-nested observation's predictive carrier had no logical target at
    # all — which is exactly why a consumer had to filter
    # `descriptor.stan.outputs` and hardcode the emitter-owned `_gen` suffix.
    #
    # Both twins of one observation share a logical target; `role`
    # (`:posterior_predictive` vs `:pointwise_loglik`) separates them, so the
    # query takes `role` to disambiguate.
    for o in stan.outputs
        isnothing(o.source) || claim!(o.name, o.source)
    end

    for (resolved, decl) in by_name
        decl.role === :observation && continue
        owned = Set{Symbol}(o.name for o in stan.outputs
                            if _brm_owner(o, by_name, targets) === decl)
        isempty(owned) && continue

        found = Set{Symbol}()
        if resolved in output_names
            push!(found, resolved)
        elseif resolved in model_names
            _brm_output_leaves!(found, stan.model[resolved], owned)
        end

        foreach(emitted -> claim!(emitted, decl.target), found)
    end
    logical
end

# Attach coefficient labels to the population block's coefficient vector.
#
# `popefs` (sbimpl.jl) declares exactly ONE parameter, `beta_pop`, so within a
# population block the coefficient vector is the unique `:parameter`-kind
# output. Labelling that one is unambiguous; if a future `popefs` ever grew a
# second parameter the uniqueness check fails and nothing is labelled, rather
# than the wrong output being labelled — a missing label is a plain UI, a
# wrong label is a lie about which covariate a posterior column belongs to.
function _brm_label_population!(outputs, brmi, pop_lp)
    for (block, lp) in pop_lp
        idx = findall(o -> o.role === :population_effect && o.kind === :parameter &&
                           !isnothing(o.declaration) && o.declaration.target === block,
                      outputs)
        length(idx) == 1 || continue
        labels = _brm_labels(brmi, lp)
        isnothing(labels) && continue
        o = outputs[idx[1]]
        outputs[idx[1]] = BRMOutput(o.name, o.kind, o.type, o.size, o.constraints,
                                    o.generative, o.source, o.role, o.declaration,
                                    o.logical, labels, o.segments)
    end
    outputs
end

# Ragged group boundaries, read from StanBlocks' `ModelOutput.segments` when the
# pinned StanBlocks emits them. Guarded by `hasproperty` rather than a version
# check: BRM is developed against a moving StanBlocks, and a hard field access
# would turn "your pin predates segments" into a MethodError at descriptor
# construction — i.e. an unrelated model would stop reflecting at all — instead
# of the honest "this consumer has no segment metadata yet".
_brm_output_segments(o) =
    hasproperty(o, :segments) ? getproperty(o, :segments) : nothing

"""
    brm_descriptor(sb::SBBRMI; name=nothing, operations=Dict(), titles=Dict(), highlights=())
    brm_descriptor(plan::GenerativePlan; …)
    brm_descriptor(builder::Function, df; mod=@__MODULE__, cv_groups=Set(), …)

Derive the one authoritative executable semantic descriptor for a `@brm`
declaration. See [`BRMDescriptor`](@ref) for the fields.

The `builder` form is the one to prefer — it keeps the `@brm` builder, so the
descriptor can also offer `:replay` (rebuild the declaration for genuinely new
groups):

```julia
builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    y ~ Normal(mu, sigma)
end
d = brm_descriptor(builder, df; mod=@__MODULE__, name=:my_model)
```

# The derived operations

Each is offered exactly when the declaration supports it. Nothing is listed by
hand, so an operation that appears can be executed.

| operation | origin | offered when |
| --- | --- | --- |
| `:transpile` | `:stan` | always |
| `:instantiate` | `:stan` | always |
| `:fit` | `:stan` | the traced model has ≥1 parameter and ≥1 likelihood term |
| `:predict` | `:stan` | the Stan program emits ≥1 posterior-predictive draw **and** ≥1 BRM observation resolves to it |
| `:pointwise_loglik` | `:stan` | the Stan program emits ≥1 pointwise log-likelihood |
| `:replay` | `:brm` | the descriptor was built from a `@brm` builder (rebuild on a new dataframe, e.g. new subjects) |
| `:reprocess` | `:brm` | the declaration has no random-effect block, or every random-effect block uses typed `mm(...)` preprocessing |

`:replay` / `:reprocess` take the new dataframe positionally and return a NEW
`BRMDescriptor`; their `inputs` are the dataframe columns in `d.columns`, not
Stan data keys.

# Extension points

These three keywords let a consumer extend presentation and operations rather
than fork the descriptor:

- `operations` — a `name => …` mapping applied AFTER derivation:
  * a callable `(d; kwargs...) -> result` adds a new operation (or replaces an
    existing one's `run`), recorded with `origin = :override`;
  * a full `BRMOperation` replaces the entry outright;
  * `nothing` **suppresses** a derived operation (hide a button the surface
    should not show).
- `titles` — a `name => String` mapping relabelling any operation.
- `highlights` — an ordered collection of included Stan definition names, or
  `name => caption` pairs. Names resolve through StanBlocks' authoritative
  definition inventory and fail closed when absent; order and captions are
  presentation metadata only and do not change `d.id` or the executable model.

Overriding a name the model does not offer is allowed (that is how you *add*
one); suppressing a name that was never derived is a loud error, because it
means the caller is holding a stale operation list — exactly the parallel
registry this type exists to remove.

# Failing closed

Five cases raise rather than degrade, because each is unrecoverable in a
consumer that keys on names:

1. **A declaration target that is also a Stan data input.** Every consumer
   keys on the name (form field, result target, BridgeStan lookup), so this
   cannot be auto-resolved. Rename the binding.
2. **Two declarations resolving to the same Stan output.** Ambiguous
   provenance; the descriptor refuses to pick one.
3. **`brm_operation(d, name)` for an operation the model does not offer** —
   errors and names the operations it *does* offer, so the discovery never
   moves into the consumer.
4. **Suppressing an operation that was not derived** (see above).
5. **Selecting an absent or duplicate Stan definition highlight.** BRM accepts
   only names that StanBlocks reports in the executable definition inventory.

One case deliberately does **not** raise, because it is a legitimate model:
an observation whose predictive draw the Stan program does not emit is
recorded in `d.unpredictable` and simply excluded from `:predict`'s outputs.
If that leaves no draws at all, `:predict` is not offered. The consumer never
sees a predict button that would fail — which is the point.
"""
function brm_descriptor(plan_or_sb::Union{GenerativePlan,SBBRMI};
                        name::Union{Nothing,Symbol}=nothing,
                        operations=Dict{Symbol,Any}(),
                        titles=Dict{Symbol,String}(),
                        highlights=())
    plan = _brm_plan_of(plan_or_sb)
    stan = isnothing(name) ? StanBlocks.stan_descriptor(plan.model) :
                             StanBlocks.stan_descriptor(plan.model; name)
    _brm_descriptor(plan, stan, operations, titles, highlights)
end

function brm_descriptor(builder::Function, df;
                        mod::Module=@__MODULE__, cv_groups=Set{Symbol}(),
                        name::Union{Nothing,Symbol}=nothing,
                        operations=Dict{Symbol,Any}(),
                        titles=Dict{Symbol,String}(),
                        highlights=())
    brm_descriptor(generative_plan(builder, df; mod, cv_groups);
                   name, operations, titles, highlights)
end

function _brm_descriptor(plan, stan, operations, titles, highlight_specs)
    brmi = plan.parent
    highlights = _brm_highlights(stan, highlight_specs)

    # --- the dataframe columns this declaration reads -----------------------
    # `data_columns` covers every column the formula references as a bare
    # predictor or grouping factor, but a RESPONSE is a `~` op, not a
    # DataColumn op, so it never appears there. The plan already knows every
    # response: an observation declaration's `data_source` IS its dataframe
    # column (including a plate-local alias, `kernel_y => dv`).
    df_columns = Set{Symbol}(data_columns(brmi))
    for d in plan.declarations
        d.role === :observation && !isnothing(d.data_source) &&
            push!(df_columns, d.data_source)
    end

    # --- inputs: the Stan data block + its dataframe provenance -------------
    inputs = BRMInput[]
    for i in stan.inputs
        col, tf = _brm_input_schema(plan, i.name, df_columns)
        push!(inputs, BRMInput(i.name, i.type, i.size, i.constraints,
                               i.observed, i.held_out, i.derived, i.inlined,
                               col, tf))
        isnothing(col) || push!(df_columns, col)   # e.g. a zscale()'d raw column
    end
    input_names = Set{Symbol}(i.name for i in stan.inputs)

    # --- declarations, indexed by the Stan name they resolve to -------------
    # Composition is BY NAME (stanblocks-use §30), through `_brm_stan_name`.
    # An observation is data, so what it *owns* among the outputs is its
    # predictive draw and its pointwise log-likelihood, not itself — and
    # StanBlocks' `source` is the authoritative link for both.
    #
    # Which observations does the traced model actually produce a draw for?
    # Read it off StanBlocks' `source` link, NOT off the declaration's `draw`
    # field: `draw` is a stable executor-facing NAME the plan assigns, not a
    # promise the program emits it (brm-use, the `generative_plan` section).
    # For a plate-nested observation the two legitimately differ.
    draw_sources = Set{Symbol}(o.source for o in stan.outputs
                               if o.generative === :draw && !isnothing(o.source))
    by_name = Dict{Symbol,GenerativeDeclaration}()
    unpredictable = Symbol[]
    for d in plan.declarations
        sname = _brm_stan_name(d)
        haskey(by_name, sname) && error(
            "brm_descriptor: ambiguous provenance — two declarations both resolve to " *
            "`$sname`. Rename one of them.")
        if d.role === :observation
            # The observation's Stan data key is its `data_source`; `target` is
            # the emitted SLIC binding, which for a plate-nested observation is
            # a plate-local alias (`kernel_y` for the column `dv`).
            (d.target in input_names ||
             (!isnothing(d.data_source) && d.data_source in input_names)) || error(
                "brm_descriptor: observation `$(d.target)` resolves to no data input of " *
                "the emitted model — the plan and the traced model disagree; " *
                "re-derive the plan.")
            (d.target in draw_sources ||
             (!isnothing(d.data_source) && d.data_source in draw_sources)) ||
                push!(unpredictable, d.target)
        else
            d.target in input_names && error(
                "brm_descriptor: `$(d.target)` is both a declared binding and a data input " *
                "of the emitted model. Every consumer keys on the name, so this cannot be " *
                "resolved automatically — rename one of them.")
        end
        by_name[sname] = d
    end
    targets = collect(keys(by_name))
    logical_outputs = _brm_logical_outputs(stan, by_name, targets)

    # Linear-predictor names come from the FORMULA, not from the emitted body:
    # `mu = pop_mu + r_mu_g` is an `=`, so no declaration binds it, yet it is
    # the most meaningful output in the model. `pop_<lp>` is the population
    # block sbimpl emits for `lp` (sbimpl.jl `_sb_linear_predictor!`), which is
    # how a population declaration gets back to the LP whose coefficients it
    # holds — derived forwards from the formula, never parsed off the name.
    # `pop_<emitted>` is keyed by the name sbimpl EMITS the design under, which
    # is the LINKED spelling for an LHS link transformation (`log(Vc) ~ 1 + x`
    # emits `pop_log_Vc_beta_pop`), while the value stays the PUBLIC LP name
    # `popcoefnames` takes (`:Vc`). Deriving it as `Symbol(:pop_, l.name)`
    # matched no emitted block for a linked LHS, so the coefficient vector
    # silently lost its `labels` — the one thing a consumer mounts a descriptor
    # for — while the same model written with an inert `log_Vc` name kept them.
    lp_list = linear_predictors(brmi)
    lps = Set{Symbol}(l.name for l in lp_list)
    pop_lp = Dict{Symbol,Symbol}(
        Symbol(:pop_, _sb_lp_emitted_name(l.name, l.link_lhs_fn)) => l.name
        for l in lp_list)

    # --- outputs ------------------------------------------------------------
    outputs = BRMOutput[]
    for o in stan.outputs
        decl = _brm_owner(o, by_name, targets)
        role = if !isnothing(decl) && decl.role === :observation
            o.generative === :pointwise_loglik ? :pointwise_loglik : :posterior_predictive
        elseif !isnothing(decl)
            _brm_declaration_role(decl)
        elseif o.name in lps
            :linear_predictor
        elseif o.generative === :draw
            :posterior_predictive
        elseif o.generative === :pointwise_loglik
            :pointwise_loglik
        else
            :stan_derived
        end
        push!(outputs, BRMOutput(o.name, o.kind, o.type, o.size, o.constraints,
                                 o.generative, o.source, role, decl,
                                 get(logical_outputs, o.name, nothing), nothing,
                                 _brm_output_segments(o)))
    end
    outputs = _brm_label_population!(outputs, brmi, pop_lp)

    # --- schema -------------------------------------------------------------
    columns = Tuple(sort!(collect(df_columns)))

    # --- operations ---------------------------------------------------------
    ops = _brm_derive_operations(plan, stan, outputs, columns)
    ops = _brm_apply_overrides(ops, operations, titles)

    BRMDescriptor(stan.id, stan.name, sprint(show, brmi), plan, stan, highlights,
                  Tuple(inputs), Tuple(outputs), Tuple(ops), columns,
                  Tuple(unpredictable))
end

# Every operation StanBlocks derived, delegated verbatim, plus the two BRM-level
# replay operations. Nothing is listed: `stan.operations` is itself derived from
# what the traced model supports, and the two BRM ones are gated on the
# declaration.
function _brm_reprocess_supported(plan, outputs)
    any(o -> o.role === :random_effect, outputs) || return true

    ranef_declarations = [d for d in plan.declarations
                          if _brm_declaration_role(d) === :random_effect]
    isempty(ranef_declarations) && return false

    all(ranef_declarations) do d
        haskey(d.keywords, :group_idx) || return false
        idx_key = d.keywords.group_idx
        idx_key isa Symbol || return false
        entry = get(plan.preproc, idx_key, nothing)
        entry isa PreprocEntry && entry.kind === :multi_membership
    end
end

function _brm_derive_operations(plan, stan, outputs, columns)
    ops = BRMOperation[]
    predictive = Symbol[o.name for o in outputs if o.role === :posterior_predictive]
    for so in stan.operations
        # `:predict` is only real if a BRM observation actually resolves to a
        # draw the program emits. `unpredictable` observations are already
        # excluded from `predictive`.
        so.name === :predict && isempty(predictive) && continue
        push!(ops, BRMOperation(so.name, so.title, Tuple(so.inputs),
                                Tuple(so.outputs), :stan,
                                (d; kwargs...) -> StanBlocks.stan_execute(
                                    d.stan, so.name; kwargs...)))
    end

    if !isnothing(plan.builder)
        push!(ops, BRMOperation(
            :replay, "Rebuild the declaration on a new dataframe", columns, (), :brm,
            (d, new_df; highlights=d.highlights, kwargs...) -> brm_descriptor(
                generative_plan(d.plan, new_df); highlights, kwargs...)))
    end
    if _brm_reprocess_supported(plan, outputs)
        push!(ops, BRMOperation(
            :reprocess, "Re-run preprocessing on a new dataframe", columns, (), :brm,
            (d, new_df; freeze_constants::Bool=true, highlights=d.highlights,
             kwargs...) -> brm_descriptor(
                reprocess(d.plan, new_df; freeze_constants); highlights, kwargs...)))
    end
    ops
end

function _brm_apply_overrides(ops, operations, titles)
    derived = Set{Symbol}(o.name for o in ops)
    by_name = Dict{Symbol,BRMOperation}(o.name => o for o in ops)
    order = Symbol[o.name for o in ops]
    # Sorted so a `Dict` of overrides yields the same operation order every
    # time — a consumer rendering one button per operation should not see them
    # reshuffle between processes.
    for key in sort!(Symbol[Symbol(k) for k in keys(operations)])
        v = operations[key]
        if isnothing(v)
            key in derived || error(
                "brm_descriptor: cannot suppress operation `$key` — this model does not " *
                "offer it. Offered: $(sort(collect(derived))). A stale operation list is " *
                "exactly what this descriptor removes.")
            delete!(by_name, key)
            filter!(!=(key), order)
        elseif v isa BRMOperation
            haskey(by_name, key) || push!(order, key)
            by_name[key] = v
        else
            base = get(by_name, key, nothing)
            haskey(by_name, key) || push!(order, key)
            by_name[key] = BRMOperation(
                key,
                isnothing(base) ? String(key) : base.title,
                isnothing(base) ? () : base.inputs,
                isnothing(base) ? () : base.outputs,
                :override, v)
        end
    end
    for (k, t) in pairs(titles)
        key = Symbol(k)
        haskey(by_name, key) || error(
            "brm_descriptor: cannot retitle operation `$key` — not offered. " *
            "Offered: $(sort(collect(keys(by_name)))).")
        b = by_name[key]
        by_name[key] = BRMOperation(b.name, String(t), b.inputs, b.outputs, b.origin, b.run)
    end
    BRMOperation[by_name[k] for k in order]
end

# ---- accessors --------------------------------------------------------------

"""
    brm_columns(d::BRMDescriptor) -> Tuple{Vararg{Symbol}}

The dataframe columns this declaration reads — the schema a form should
collect for a `:replay` / `:reprocess`. Equivalent to `d.columns`.
"""
brm_columns(d::BRMDescriptor) = d.columns

"""
    brm_outputs(d::BRMDescriptor; logical=nothing, role=nothing, kind=nothing)
        -> Vector{BRMOutput}

Every emitted output matching the given semantic filters, in descriptor order.
Each filter accepts a `Symbol` or any collection of them, and an omitted filter
does not constrain. This is the **discovery** query — it can legitimately return
zero, one, or many; use [`brm_output`](@ref) when exactly one is required.

```julia
brm_outputs(d; role=:posterior_predictive)          # every predictive carrier
brm_outputs(d; logical=:pk_conc)                    # every carrier of one target
brm_outputs(d; role=(:parameter, :random_effect))   # either role
```

Filtering here is on BRM meaning (`logical` / `role`) or Stan representation
(`kind`) — never on the emitted NAME, which the compiler owns.
"""
function brm_outputs(d::BRMDescriptor; logical=nothing, role=nothing, kind=nothing)
    BRMOutput[o for o in d.outputs
              if _brm_matches(o.logical, logical) &&
                 _brm_matches(o.role, role) &&
                 _brm_matches(o.kind, kind)]
end

_brm_matches(_value, ::Nothing) = true
_brm_matches(value, wanted::Symbol) = value === wanted
_brm_matches(value, wanted) = any(w -> value === w, wanted)

"""
    brm_output(d::BRMDescriptor, logical::Symbol; role=nothing) -> BRMOutput

Return the unique emitted output that physically carries declaration target
`logical`. This resolves BRM meaning to Stan representation without parsing an
emitter-owned name. For example, a ragged `loc ~ kernel(...)` result can resolve
to an output named `loc__pl_mem_1` while retaining `logical == :loc`.

**One logical target legitimately has several carriers, and `role` is how you
pick one.** An observation `pk_conc ~ Normal(loc, sigma)` emits a
posterior-predictive twin *and* a pointwise-log-likelihood twin, both carrying
`logical === :pk_conc`. Ask for the one you mean:

```julia
brm_output(d, :pk_conc; role=:posterior_predictive)   # the *_gen carrier
brm_output(d, :pk_conc; role=:pointwise_loglik)       # the per-element loglik
```

The emitted names (`pk_conc_gen`, …) are StanBlocks' to choose and are NOT part
of this contract; that is the whole point of asking by role.

Fails loudly when the target has no emitted carrier or still maps to several;
the caller must never guess from descriptor order in either case. The
several-carriers message lists each candidate WITH its role, so the fix is the
`role=` to add rather than a name to hardcode.
"""
function brm_output(d::BRMDescriptor, logical::Symbol; role=nothing)
    found = brm_outputs(d; logical, role)
    length(found) == 1 && return only(found)
    qualifier = isnothing(role) ? "" : " with role `$role`"
    if isempty(found)
        available = unique!(Symbol[o.role for o in d.outputs if o.logical === logical])
        hint = isempty(available) ? "" :
            " That target does emit carriers with role(s) $(Tuple(available))."
        error("brm_descriptor: logical output `$logical`$qualifier has no emitted " *
              "posterior carrier in model `$(d.name)`.$hint")
    end
    error("brm_descriptor: logical output `$logical`$qualifier spans several emitted " *
          "carriers $(Tuple((o.name, o.role) for o in found)); pass `role=` to select " *
          "one, or use `brm_outputs` to take them all.")
end

"""
    brm_output_coordinates(d::BRMDescriptor, logical::Symbol, constrained_names;
                           role=nothing) -> Vector{Int}

Resolve a logical BRM output to its columns in BridgeStan's constrained
`param_names` (or an equivalent posterior name vector). Resolution first uses
[`brm_output`](@ref) to obtain the descriptor's emitted name, then matches only
that exact name or its documented container coordinates (`name.1`,
`name.1.1`, …). It never parses a compiler-owned plate suffix.

`role` disambiguates a target with several carriers, exactly as on
[`brm_output`](@ref) — a posterior-predictive slice of an observation is

```julia
brm_output_coordinates(d, :pk_conc, param_names; role=:posterior_predictive)
```

The returned integers index `constrained_names` in their existing order. For a
ragged carrier they compose with that output's `segments`: group `g` is
`coordinates[segments[g-1]+1 : segments[g]]`.

A missing carrier is an error, which catches descriptor/artifact drift instead
of returning an empty posterior slice.
"""
function brm_output_coordinates(d::BRMDescriptor, logical::Symbol,
                                constrained_names; role=nothing)
    output = brm_output(d, logical; role)
    stem = String(output.name)
    prefix = stem * "."
    coordinates = Int[]
    for (i, name) in enumerate(constrained_names)
        rendered = string(name)
        (rendered == stem || startswith(rendered, prefix)) && push!(coordinates, i)
    end
    isempty(coordinates) && error(
        "brm_descriptor: logical output `$logical` resolves to emitted " *
        "`$(output.name)`, but that carrier is absent from the supplied constrained " *
        "names. Re-reflect the model that produced the posterior draws.")
    coordinates
end

"""
    brm_operation(d::BRMDescriptor, name::Symbol) -> BRMOperation

The named operation. **Fails closed**: an operation this model does not offer
errors and names the ones it does, so the discovery never moves into the
consumer.
"""
function brm_operation(d::BRMDescriptor, name::Symbol)
    for op in d.operations
        op.name === name && return op
    end
    error("brm_descriptor: model `$(d.name)` does not offer operation `$name`. " *
          "Offered: $([op.name for op in d.operations]).")
end

"""
    brm_execute(d::BRMDescriptor, name::Symbol, args...; kwargs...)

Run a derived operation.

```julia
brm_execute(d, :transpile)                       # the Stan source
prob = brm_execute(d, :fit)                      # a BridgeStan-backed StanProblem
brm_execute(d, :predict; problem=prob, draws=theta_unc, seed=1234)
brm_execute(d, :replay, new_df)                  # a NEW BRMDescriptor
```

`:stan`-origin operations forward to StanBlocks' `stan_execute` (data keywords
re-bind inputs; `:predict` requires `draws` and `seed`). `:brm`-origin
operations take the new dataframe positionally and return a new descriptor.
Unknown names fail closed via [`brm_operation`](@ref).
"""
brm_execute(d::BRMDescriptor, name::Symbol, args...; kwargs...) =
    brm_operation(d, name).run(d, args...; kwargs...)

# ---- display ----------------------------------------------------------------

Base.show(io::IO, d::BRMDescriptor) = begin
    print(io, "BRMDescriptor `", d.name, "` (id ", d.id, ")\n")
    print(io, "  formula:    ", replace(strip(d.formula), "\n" => "\n              "), "\n")
    print(io, "  columns:    ", join(d.columns, ", "), "\n")
    print(io, "  inputs:     ", join(required_brm_inputs(d), ", "), "\n")
    print(io, "  operations: ", join((op.name for op in d.operations), ", "), "\n")
    isempty(d.highlights) ||
        print(io, "  highlights: ", join((h.name for h in d.highlights), ", "), "\n")
    isempty(d.unpredictable) ||
        print(io, "  no predictive draw for: ", join(d.unpredictable, ", "), "\n")
    for o in d.outputs
        print(io, "  ", o.name, " :: ", o.role, " (", o.kind, "/", o.generative, ")")
        isnothing(o.logical) || o.name === o.logical || print(io, " => ", o.logical)
        isnothing(o.labels) || print(io, " [", join(o.labels, ", "), "]")
        print(io, "\n")
    end
end

"""
    required_brm_inputs(d::BRMDescriptor) -> Vector{Symbol}

The Stan data keys a consumer must actually supply — the
neither-`derived`-nor-`inlined` subset of `d.inputs`. The BRM analogue of
StanBlocks' `required_inputs`; for the *dataframe* schema use
[`brm_columns`](@ref).
"""
required_brm_inputs(d::BRMDescriptor) =
    Symbol[i.name for i in d.inputs if !i.derived && !i.inlined]
