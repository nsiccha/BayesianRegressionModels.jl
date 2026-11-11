# Downstream grouped-term extension recipes

### Downstream grouped terms: the reusable-term pattern

A parametric shape with group-varying parameters — a single-peak transient, a
saturating sigmoid — belongs in the downstream package whose science needs it,
not in BRM core. BRM core promotes such a term only once two or more
downstream packages have shipped the same curves; until then the package ships
its own term through the supported group-block seams, first-class rather than
as an escape hatch. The worked example below is a `transient` /
`saturating` pair.

The recipe, for a term used in nested predictor position
(`mu ~ 1 + transient(logt; series)`), where it runs on both backends:

1. Declare the marker in the downstream module (`function transient end`).
2. Declare one structured-latent field per parameter group with the fields
   form of `_sb_term_group_block`. The `group` spec names the grouping
   column (`(; kwarg=:series)` reads it from the call's `series=` keyword);
   `prior=:correlated_normal` draws the per-group parameters through BRM's
   non-centered LKJ block on both backends.
3. Route the nested summand to your emitter:
   `_sb_is_direct_term(::typeof(transient)) = true`.
4. Prepare once, replay frozen: `_brm_prepares_term` is true;
   `_brm_prepare_term` delegates to `_brm_prepare_structured_term` and bakes
   the axis values the native effect needs into state; `_brm_replay_term`
   refreshes indices and axis values against the fitted levels.
5. Assemble per-row values natively in
   `_brm_native_structured_effect` — pure Julia math over the block row.
6. Emit the Stan contribution in `_sb_predictor_term!`, reusing StanBlocks
   builtins (e.g. `biomarker_time_response`) and threading the preallocated
   block via `_sb_find_group_block`, so the downstream module ships no
   custom Stan code. (Extend, never shadow:
   `import BayesianRegressionModels: _sb_term_group_block, _sb_is_direct_term,
   _brm_prepares_term, _brm_prepare_term, _brm_replay_term,
   _brm_native_structured_effect, _sb_predictor_term!`.)

```julia
module DownstreamTerms
using StanBlocks
import BayesianRegressionModels
const BRM = BayesianRegressionModels
import BayesianRegressionModels: _sb_term_group_block, _sb_is_direct_term,
    _brm_prepares_term, _brm_prepare_term, _brm_replay_term,
    _brm_native_structured_effect, _sb_predictor_term!

function transient end
_sb_is_direct_term(::typeof(transient)) = true
_sb_term_group_block(::typeof(transient)) = (; fields=[
    (; name=:transient, n_per_group=3, group=(; kwarg=:series),
       prior=:correlated_normal),
])

# Independent Julia reference math, shared by the native effect below.
bump_math(x, loc, ls, mag) = begin
    xi = (x - loc) * exp(ls)
    s = 1 / (1 + exp(-xi))
    sm = 1 / (1 + exp(xi))
    exp(log(s) + log(sm)) * mag
end

_brm_prepares_term(::BRM.ExprColumn{typeof(transient)}) = true
function _brm_prepare_term(term::BRM.ExprColumn{typeof(transient)}, target,
                           context)
    base = BRM._brm_prepare_structured_term(term, target, context,
        _sb_term_group_block(transient, term))
    xkey, xraw = BRM._brm_term_data(
        :transient, only(BRM.getargs(term)), context)
    BRM._BRMPreparedTerm(base.callable, base.source,
        merge(base.state, (; xname=xkey, x=collect(Float64, xraw))),
        base.dependencies)
end
function _brm_replay_term(::typeof(transient), training, fresh, context)
    fields = map(training.state.fields) do field
        raw = context.data[field.source]
        merge(field, (; idx=BRM._brm_apply_levels(field.levels, raw)))
    end
    xraw = context.data[training.state.xname]
    BRM._BRMPreparedTerm(training.callable, training.source,
        merge(training.state,
            (; fields=Tuple(fields), x=collect(Float64, xraw))),
        training.dependencies)
end

function _brm_native_structured_effect(
        term::BRM._BRMPreparedTerm{typeof(transient)}, block, field)
    st = term.state
    [bump_math(st.x[i], block[field.idx[i], 1], block[field.idx[i], 2],
               block[field.idx[i], 3]) for i in eachindex(field.idx)]
end

function _sb_predictor_term!(stmts, data, ::typeof(transient), t;
                             target::Symbol, group_block_lookup=Dict(),
                             kwargs...)
    xname, xraw = BRM._sb_inner_data(:transient, only(BRM.getargs(t)))
    data[xname] = collect(Float64, xraw)
    info = BRM._sb_find_group_block(transient, t, group_block_lookup)
    isnothing(info) && error("sbimpl: `transient` found no allocated block")
    (; block_name, idx_name) = info
    loc = Symbol(:transient_, target, :_loc)
    ls = Symbol(:transient_, target, :_ls)
    mag = Symbol(:transient_, target, :_mag)
    col = Symbol(:transient_, target, :_, xname)
    push!(stmts, :($loc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :($ls = $(block_name)[$(idx_name), 2]))
    push!(stmts, :($mag = $(block_name)[$(idx_name), 3]))
    push!(stmts, :($col = biomarker_time_response($xname, $loc, $ls, $mag)))
    col
end
end
```

7. Fit through `SBBRMI(brmi; mod=DownstreamTerms)`. The Stan path lowers to
   stanc-clean code, and the native effect's row math is bit-exact —
   `test/downstream_group_block_term.jl` verifies both against independent
   Julia math.

A second shape follows the same seven steps. A saturating 0-to-1 dose
multiplier is two per-group parameters (`loc`, `log_slope`) whose Stan emit
exps the log-sigmoid builtin and whose native effect is one line of Julia:

```julia
function saturating end
_sb_is_direct_term(::typeof(saturating)) = true
_sb_term_group_block(::typeof(saturating)) = (; fields=[
    (; name=:saturating, n_per_group=2, group=(; kwarg=:series),
       prior=:correlated_normal),
])
sigmoid_math(x, loc, ls) = 1 / (1 + exp(-((x - loc) * exp(ls))))
# ... same prep/replay shape as above, with `saturating` for `transient` ...
function _brm_native_structured_effect(
        term::BRM._BRMPreparedTerm{typeof(saturating)}, block, field)
    st = term.state
    [sigmoid_math(st.x[i], block[field.idx[i], 1], block[field.idx[i], 2])
     for i in eachindex(field.idx)]
end
function _sb_predictor_term!(stmts, data, ::typeof(saturating), t;
                             target::Symbol, group_block_lookup=Dict(),
                             kwargs...)
    xname, xraw = BRM._sb_inner_data(:saturating, only(BRM.getargs(t)))
    data[xname] = collect(Float64, xraw)
    info = BRM._sb_find_group_block(saturating, t, group_block_lookup)
    isnothing(info) && error("sbimpl: `saturating` found no allocated block")
    (; block_name, idx_name) = info
    loc = Symbol(:saturating_, target, :_loc)
    ls = Symbol(:saturating_, target, :_ls)
    col = Symbol(:saturating_, target, :_, xname)
    push!(stmts, :($loc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :($ls = $(block_name)[$(idx_name), 2]))
    push!(stmts, :($col = exp(biomarker_dose_response($xname, $loc, $ls))))
    col
end
```

Shapes compose two ways. Additively, as ordinary predictor summands:

```julia
mu ~ 1 + transient(logt; series) + saturating(logd; series)
```

Multiplicatively — baseline plus bump times response —
through intercept-free named predictors combined by assignment (write `*`
or `.*`: the formula layer is element-wise by intent and sbimpl dots it —
the dotted spellings `.* ./ .+ .- .^` are exact aliases of the plain
operators):

```text
brmi = (@brm df begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    bump ~ 0 + transient(logt; series)
    resp ~ 0 + saturating(logd; series)
    mu = base + bump * resp
    y ~ Normal(mu, sigma)
end)
sb = SBBRMI(brmi; mod=DownstreamTerms)
```

Each shape owns its own per-group hierarchy (two LKJ blocks here):
parameters correlate within a shape, not across shapes. A model that needs
the bump and the sigmoid parameters jointly correlated (all six in one
covariance) wants one joint term with `n_per_group=6` instead, or the
shared-bucket + `@plate for` composition; which one fits is decided per
model. Term-internal
prior statements (`sd(mu, transient(logt))`) are not addressed yet: the
hierarchical scales keep their shared defaults, and naming a new term in a
prior address needs core registration alongside `_TERM_HEADS`.

Verified contract (`test/downstream_group_block_term.jl` pins all of it):
stanc-clean Stan lowering, native row math bit-exact versus independent Julia
math, `brm_descriptor`, and
`reprocess` / `restan_data` on fitted group levels. Replaying a group level
the fit never saw stays refused by the shared ranef floor — the fitted model
has no coordinate for it, on any backend, for core grouped terms alike.
