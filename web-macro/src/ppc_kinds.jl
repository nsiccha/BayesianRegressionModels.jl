# ---- PPC kinds: dispatch-based (one type per kind, methods per kind) -----
#
# Replaces the earlier kind=:Symbol switch + `if-elseif` cascade with a
# small `abstract type PPCKind` hierarchy and a handful of single-dispatch
# methods. Adding a new kind = a new struct + the methods it needs.
#
# Every kind carries the common fields (`response`, `loc`, `family`,
# `link_fn`, `n_trials`, `covariates`) so renderers don't have to thread
# them through. `covariates` is the WIDE picker set: every data column
# this LP transitively depends on (DAG-traced via `dependencies`), so
# users can re-channel any of them onto color / row / column / detail.

abstract type PPCKind end

# `is_primary=false` means this kind belongs to a *secondary* LP of a
# distributional family (e.g. `err` in `Normal(loc, err)`). Such an LP is
# NOT a likelihood location, so we must not overlay the observed response
# nor frame the panel as a "predictive check of <response>". Renderers
# branch on this flag.
struct ScalarPPC <: PPCKind
    response::Symbol
    loc::Symbol
    family
    link_fn::Function
    n_trials::Union{Nothing,Symbol}
    covariates::Vector{Symbol}
    is_primary::Bool
end

struct ScalarRePPC <: PPCKind
    response::Symbol
    loc::Symbol
    family
    link_fn::Function
    n_trials::Union{Nothing,Symbol}
    covariates::Vector{Symbol}
    group::Symbol
    is_primary::Bool
end

struct LinearPPC <: PPCKind
    response::Symbol
    loc::Symbol
    family
    link_fn::Function
    n_trials::Union{Nothing,Symbol}
    covariates::Vector{Symbol}
    predictor::Symbol
    is_primary::Bool
end

struct LinearRePPC <: PPCKind
    response::Symbol
    loc::Symbol
    family
    link_fn::Function
    n_trials::Union{Nothing,Symbol}
    covariates::Vector{Symbol}
    predictor::Symbol
    group::Symbol
    is_primary::Bool
end

struct CategoricalPPC <: PPCKind
    response::Symbol
    loc::Symbol
    family
    link_fn::Function
    n_trials::Union{Nothing,Symbol}
    covariates::Vector{Symbol}
    predictor::Symbol
    is_primary::Bool
end

struct MultiContinuousPPC <: PPCKind
    response::Symbol
    loc::Symbol
    family
    link_fn::Function
    n_trials::Union{Nothing,Symbol}
    covariates::Vector{Symbol}
    predictors::Vector{Symbol}
    is_primary::Bool
end

# ---- detection: enumerate all PPCKinds for a BRMI -------------------------

# One PPCKind per (likelihood × linear-predictor) -- distributional
# `Normal(loc, err)` with `loc ~ ...` and `log(err) ~ ...` produces TWO
# kinds (one per LP), each with its own picker. Returns an empty Vector
# when nothing matches.
function _ppc_kinds(brmi::BRMI)
    out = PPCKind[]
    for o in outcomes(brmi)
        o.family in (Normal, Poisson, Bernoulli, BernoulliLogit,
                     Binomial, BinomialLogit, OrderedLogistic, Ordinal,
                     ZeroInflatedPoisson, NegativeBinomial2) || continue
        # Trials column ONLY for Binomial / BinomialLogit -- other
        # families that happen to take a data-arg (none today, but e.g.
        # an exposure offset someday) shouldn't be misread as trials.
        n_trials = if o.family === Binomial || o.family === BinomialLogit
            dargs = data_args(o)
            isempty(dargs) ? nothing : first(dargs).name
        else
            nothing
        end
        primary = primary_lp(o)
        for lp in linear_predictor_args(o)
            kind = _kind_for_lp(brmi, o, lp, n_trials, lp === primary)
            isnothing(kind) || push!(out, kind)
        end
    end
    out
end

# Family-natural inverse link. For *Logit families the LP is sampled
# on the logit scale and the family applies `inv_logit` internally;
# the PPC needs to do the same to land predicted draws on the same
# (probability) scale as the observed response. Only kicks in for
# primary LPs whose user-facing link_fn is `identity` -- if the user
# wrote `logit(p) ~ ...` they've already declared the scale.
_family_inverse_link(family) =
    (family === BernoulliLogit || family === BinomialLogit) ? logistic :
    identity

# Build the PPCKind for one outcome × LP pair. Returns `nothing` if the
# LP's predictor pattern doesn't match a supported shape.
function _kind_for_lp(brmi::BRMI, o, lp, n_trials, is_primary::Bool)
    pred = predictors(brmi, lp.link_lp)
    isnothing(pred) && return nothing

    # Flatten RE-internal predictors so `(1 + a | g)` contributes `a`
    # as a bare predictor (with `g` carried out as the group).
    cont = copy(pred.continuous)
    cat  = copy(pred.categorical)
    group = nothing
    for re in pred.re_terms
        group = re.group  # last RE wins; multi-RE coalescing is future work
        append!(cont, re.inner.continuous)
        append!(cat,  re.inner.categorical)
    end

    # Wide picker covariates: every data column this LP's DAG touches.
    # Includes the predictor itself + the group + any deeper deps.
    covs = copy(dependencies(brmi, lp.link_lp).data)

    n_cont = length(cont)
    n_cat  = length(cat)

    link_fn = (is_primary && lp.link_fn === identity) ?
        _family_inverse_link(o.family) : lp.link_fn

    # Pick the most-specific kind that fits. Extra predictors beyond the
    # one(s) the kind nails into structural slots are NOT silently
    # dropped -- they're already in `covs` (DAG-traced), so the picker
    # exposes them as remappable channels (color / row / column).
    # Order matters: try the kinds with the richest structural axes
    # first, fall through to scalar.
    if n_cont >= 2 && isnothing(group)
        MultiContinuousPPC(o.response, lp.link_lp, o.family, link_fn, n_trials, covs, cont, is_primary)
    elseif n_cont == 1 && isnothing(group)
        LinearPPC(o.response, lp.link_lp, o.family, link_fn, n_trials, covs, cont[1], is_primary)
    elseif n_cont == 1 && !isnothing(group)
        LinearRePPC(o.response, lp.link_lp, o.family, link_fn, n_trials, covs, cont[1], group, is_primary)
    elseif n_cont == 0 && n_cat >= 1
        CategoricalPPC(o.response, lp.link_lp, o.family, link_fn, n_trials, covs, cat[1], is_primary)
    elseif n_cont == 0 && n_cat == 0 && isnothing(group)
        ScalarPPC(o.response, lp.link_lp, o.family, link_fn, n_trials, covs, is_primary)
    elseif n_cont == 0 && n_cat == 0 && !isnothing(group)
        ScalarRePPC(o.response, lp.link_lp, o.family, link_fn, n_trials, covs, group, is_primary)
    else
        nothing
    end
end

# ---- per-kind table helpers (dispatch on PPCKind subtype) ----------------

_loc_long(long::DataFrame, loc::Symbol) = begin
    rows = filter(:param => ==(string(loc)), long)
    isempty(rows) ? nothing : rows
end

# Cap the number of distinct draws plotted as separate ECDF curves --
# above this, the panel becomes a hairball and downstream Vega slows
# to a crawl. Subsampling is deterministic so the picture is stable
# across reloads.
MAX_PPC_DRAW_LINES = 50

# Filter `rows` to a deterministic random subset of distinct draws so
# at most `max_n` curves are emitted. Returns `rows` unchanged when the
# `:draw` column is missing or already small enough.
function _subsample_draws(rows::DataFrame; max_n::Int=MAX_PPC_DRAW_LINES)
    hasproperty(rows, :draw) || return rows
    draws = unique(rows.draw)
    length(draws) <= max_n && return rows
    rng = MersenneTwister(20260425)
    keep = Set(draws[randperm(rng, length(draws))[1:max_n]])
    filter(:draw => in(keep), rows)
end

# Join every covariate column from `df` into `tab`. Pred tables index
# `df` by `tab.index` (one row per (i, draw)); obs tables take the
# whole column (one row per obs). Caller controls which by passing the
# index vector or `:` -- skips columns already present in `tab` (the
# kind's structural cols like :group / :level / :predictor).
function _enrich_covariates!(tab::DataFrame, df::DataFrame, covariates, by)
    for c in covariates
        c in propertynames(tab) && continue
        hasproperty(df, c) || continue
        tab[!, c] = by === Colon() ? df[!, c] : df[!, c][by]
    end
    tab
end

# ---- pred_table: the predicted (model) data table per kind ---------------

pred_table(::PPCKind, long, df) = nothing  # fallback

pred_table(p::ScalarPPC, long, df) = begin
    rows = _loc_long(long, p.loc); isnothing(rows) && return nothing
    rows = _subsample_draws(rows)
    DataFrame(y = p.link_fn.(rows.value), draw = rows.draw)
end

pred_table(p::ScalarRePPC, long, df) = begin
    rows = _loc_long(long, p.loc); isnothing(rows) && return nothing
    rows = _subsample_draws(rows)
    tab = DataFrame(
        group = df[!, p.group][rows.index],
        y     = p.link_fn.(rows.value),
        draw  = rows.draw,
        index = rows.index,
    )
    _enrich_covariates!(tab, df, p.covariates, rows.index)
end

pred_table(p::LinearPPC, long, df) = begin
    rows = _loc_long(long, p.loc); isnothing(rows) && return nothing
    tab = DataFrame(
        x     = df[!, p.predictor][rows.index],
        y     = p.link_fn.(rows.value),
        draw  = rows.draw,
        index = rows.index,
    )
    _enrich_covariates!(tab, df, p.covariates, rows.index)
end

pred_table(p::LinearRePPC, long, df) = begin
    rows = _loc_long(long, p.loc); isnothing(rows) && return nothing
    tab = DataFrame(
        x     = df[!, p.predictor][rows.index],
        y     = p.link_fn.(rows.value),
        draw  = rows.draw,
        index = rows.index,
        group = df[!, p.group][rows.index],
    )
    _enrich_covariates!(tab, df, p.covariates, rows.index)
end

pred_table(p::CategoricalPPC, long, df) = begin
    rows = _loc_long(long, p.loc); isnothing(rows) && return nothing
    rows = _subsample_draws(rows)
    tab = DataFrame(
        level = df[!, p.predictor][rows.index],
        y     = p.link_fn.(rows.value),
        draw  = rows.draw,
        index = rows.index,
    )
    _enrich_covariates!(tab, df, p.covariates, rows.index)
end

# Build the stacked `(predictor, x, y, draw, index)` table once: y / draw
# / index / covariates are the SAME across predictor blocks (they only
# depend on the loc draws), so we slice them once and `repeat(... outer)`
# instead of slicing per predictor-block as the naive vcat-of-blocks did.
pred_table(p::MultiContinuousPPC, long, df) = begin
    rows = _loc_long(long, p.loc); isnothing(rows) && return nothing
    n = nrow(rows)
    k = length(p.predictors)
    y_link = p.link_fn.(rows.value)
    out = DataFrame(
        predictor = repeat(string.(p.predictors), inner=n),
        x         = vcat((df[!, q][rows.index] for q in p.predictors)...),
        y         = repeat(y_link, outer=k),
        draw      = repeat(rows.draw, outer=k),
        index     = repeat(rows.index, outer=k),
    )
    for c in p.covariates
        c in propertynames(out) && continue
        hasproperty(df, c) || continue
        out[!, c] = repeat(df[!, c][rows.index], outer=k)
    end
    out
end

# ---- obs_table: the observed-y table per kind (posterior side only) ------

obs_table(::PPCKind, df, obs_y) = nothing
obs_table(p::ScalarPPC, df, obs_y)     = DataFrame(y = obs_y)
obs_table(p::ScalarRePPC, df, obs_y)   = _enrich_covariates!(
    DataFrame(group = df[!, p.group], y = obs_y), df, p.covariates, :)
obs_table(p::LinearPPC, df, obs_y)     = _enrich_covariates!(
    DataFrame(x = df[!, p.predictor], y = obs_y), df, p.covariates, :)
obs_table(p::LinearRePPC, df, obs_y)   = _enrich_covariates!(
    DataFrame(x = df[!, p.predictor], y = obs_y, group = df[!, p.group]),
    df, p.covariates, :)
obs_table(p::CategoricalPPC, df, obs_y) = _enrich_covariates!(
    DataFrame(level = df[!, p.predictor], y = obs_y), df, p.covariates, :)
obs_table(p::MultiContinuousPPC, df, obs_y) = begin
    n = length(obs_y)
    k = length(p.predictors)
    out = DataFrame(
        predictor = repeat(string.(p.predictors), inner=n),
        x         = vcat((df[!, q] for q in p.predictors)...),
        y         = repeat(obs_y, outer=k),
    )
    for c in p.covariates
        c in propertynames(out) && continue
        hasproperty(df, c) || continue
        out[!, c] = repeat(df[!, c], outer=k)
    end
    out
end

# ---- prior_spec / posterior_spec: per-kind plot specs --------------------

# Prior-predictive: NO observation overlay. Posterior-predictive: predicted
# layer + observed overlay (a black ECDF / scatter).
prior_spec(::PPCKind, long, df; title) = nothing
posterior_spec(::PPCKind, long, df, obs_y; title) = nothing

function prior_spec(p::ScalarPPC, long, df; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    AoG.data(pred) * AoG.mapping(:y, group=:draw => nonnumeric) *
        AoG.visual(ECDFPlot; opacity=0.3) *
        config(title=title)
end
function posterior_spec(p::ScalarPPC, long, df, obs_y; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    pred_layer = AoG.data(pred) *
        AoG.mapping(:y, group=:draw => nonnumeric) *
        AoG.visual(ECDFPlot; opacity=0.3)
    spec = p.is_primary ?
        pred_layer +
            AoG.data(obs_table(p, df, obs_y)) * AoG.mapping(:y) *
                AoG.visual(ECDFPlot; color="black", strokeWidth=2) :
        pred_layer
    spec * config(title=title)
end

function prior_spec(p::ScalarRePPC, long, df; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    AoG.data(pred) *
        AoG.mapping(:y, row=:group => nonnumeric, group=:draw => nonnumeric) *
        AoG.visual(ECDFPlot; opacity=0.3) *
        config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end
function posterior_spec(p::ScalarRePPC, long, df, obs_y; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    pred_layer = AoG.data(pred) *
        AoG.mapping(:y, row=:group => nonnumeric, group=:draw => nonnumeric) *
        AoG.visual(ECDFPlot; opacity=0.3)
    spec = p.is_primary ?
        pred_layer +
            AoG.data(obs_table(p, df, obs_y)) *
                AoG.mapping(:y, row=:group => nonnumeric) *
                AoG.visual(ECDFPlot; color="black", strokeWidth=2) :
        pred_layer
    spec * config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end

function prior_spec(p::LinearPPC, long, df; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    AoG.data(pred) * AoG.mapping(:x, :y, group=:draw) * lineribbon() *
        config(title=title)
end
function posterior_spec(p::LinearPPC, long, df, obs_y; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    p.is_primary || return AoG.data(pred) *
        AoG.mapping(:x, :y, group=:draw) * lineribbon() *
        config(title=title)
    obs = obs_table(p, df, obs_y)
    ppc_overlay(obs, pred; x=:x, y=:y, group=:draw) * config(title=title)
end

function prior_spec(p::LinearRePPC, long, df; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    AoG.data(pred) *
        AoG.mapping(:x, :y, group=:draw, row=:group => nonnumeric) *
        lineribbon() *
        config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end
function posterior_spec(p::LinearRePPC, long, df, obs_y; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    p.is_primary || return AoG.data(pred) *
        AoG.mapping(:x, :y, group=:draw, row=:group => nonnumeric) *
        lineribbon() *
        config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
    obs = obs_table(p, df, obs_y)
    ppc_overlay(obs, pred; x=:x, y=:y, group=:draw, row=:group) *
        config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end

function prior_spec(p::CategoricalPPC, long, df; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    AoG.data(pred) *
        AoG.mapping(:y, row=:level => nonnumeric, group=:draw => nonnumeric) *
        AoG.visual(ECDFPlot; opacity=0.3) *
        config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end
function posterior_spec(p::CategoricalPPC, long, df, obs_y; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    pred_layer = AoG.data(pred) *
        AoG.mapping(:y, row=:level => nonnumeric, group=:draw => nonnumeric) *
        AoG.visual(ECDFPlot; opacity=0.3)
    spec = p.is_primary ?
        pred_layer +
            AoG.data(obs_table(p, df, obs_y)) *
                AoG.mapping(:y, row=:level => nonnumeric) *
                AoG.visual(ECDFPlot; color="black", strokeWidth=2) :
        pred_layer
    spec * config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end

function prior_spec(p::MultiContinuousPPC, long, df; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    AoG.data(pred) * AoG.mapping(:x, :y, group=:draw, row=:predictor) *
        lineribbon() *
        config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end
function posterior_spec(p::MultiContinuousPPC, long, df, obs_y; title)
    pred = pred_table(p, long, df); isnothing(pred) && return nothing
    pred_layer = AoG.data(pred) *
        AoG.mapping(:x, :y, group=:draw, row=:predictor) * lineribbon()
    spec = p.is_primary ?
        pred_layer +
            AoG.data(obs_table(p, df, obs_y)) *
                AoG.mapping(:x, :y, row=:predictor) *
                AoG.visual(Scatter; color="black") :
        pred_layer
    spec * config(title=title, facet=(; linkxaxes=:none, linkyaxes=:none))
end

# ---- picker_dims: structural + wide covariates ---------------------------

# Structural channels (the "natural" facet/color candidate per kind).
# Default empty; per-kind overrides extend it.
structural_dims(::PPCKind) = Pair{String,String}[]
structural_dims(p::ScalarRePPC) = ["group" => "Group ($(p.group))"]
structural_dims(p::LinearRePPC) = ["group" => "Group ($(p.group))"]
structural_dims(p::CategoricalPPC) = ["level" => "Level of $(p.predictor)"]
structural_dims(p::MultiContinuousPPC) = ["predictor" => "Predictor"]

# Reserved column names a kind's pred / obs tables already use for
# their plot axes / structural facets -- the picker shouldn't expose
# them as remappable extras (they'd collide with the spec's mapping).
reserved_columns(::PPCKind)            = Set{String}(["y"])
reserved_columns(::ScalarPPC)          = Set{String}(["y"])
reserved_columns(::ScalarRePPC)        = Set{String}(["y", "group"])
reserved_columns(::LinearPPC)          = Set{String}(["x", "y"])
reserved_columns(::LinearRePPC)        = Set{String}(["x", "y", "group"])
reserved_columns(::CategoricalPPC)     = Set{String}(["y", "level"])
reserved_columns(::MultiContinuousPPC) = Set{String}(["x", "y", "predictor"])

# Raw data column names a kind already consumes as a structural
# slot (x axis, group facet, level facet). These appear in
# `p.covariates` because the DAG reaches them, but re-exposing them
# in the picker would let the user (or auto-remap) place a column
# already on the plot onto another channel -- e.g. faceting `y vs a`
# by `a` itself.
consumed_data_columns(::PPCKind) = Symbol[]
consumed_data_columns(p::ScalarRePPC)        = [p.group]
consumed_data_columns(p::LinearPPC)          = [p.predictor]
consumed_data_columns(p::LinearRePPC)        = [p.predictor, p.group]
consumed_data_columns(p::CategoricalPPC)     = [p.predictor]
consumed_data_columns(p::MultiContinuousPPC) = copy(p.predictors)

# Wide picker: structural + every covariate the LP's DAG touches.
# Skip covariates whose name collides with a reserved table column
# (the structural mapping would clash), with a structural-channel
# label, or with a raw data column already consumed as an axis /
# facet.
function picker_dims(p::PPCKind)
    structural = structural_dims(p)
    structural_keys = Set(string(first(d)) for d in structural)
    reserved = reserved_columns(p)
    consumed = Set(string(c) for c in consumed_data_columns(p))
    extras = [string(c) => "Covariate $c" for c in p.covariates
              if !(string(c) in structural_keys) &&
                 !(string(c) in reserved) &&
                 !(string(c) in consumed)]
    isempty(structural) && isempty(extras) ? nothing :
        Pair{String,String}[structural..., extras...]
end

# ---- caption / heading helpers ------------------------------------------

predictor_label(::PPCKind) = ""
predictor_label(p::LinearPPC) = " vs $(p.predictor)"
predictor_label(p::LinearRePPC) = " vs $(p.predictor)"
predictor_label(p::CategoricalPPC) = " vs $(p.predictor)"
predictor_label(p::MultiContinuousPPC) = " vs " * join(string.(p.predictors), ", ")

# Stable machine identifier (used in plot ids / log lines) -- kept as
# the bare struct name. User-facing copy goes through `display_name`.
kind_tag(::T) where {T<:PPCKind} = string(nameof(T))

display_name(::PPCKind)             = "scalar"
display_name(::ScalarPPC)           = "scalar"
display_name(p::ScalarRePPC)        = "by $(p.group)"
display_name(::LinearPPC)           = "linear"
display_name(p::LinearRePPC)        = "linear by $(p.group)"
display_name(p::CategoricalPPC)     = "per $(p.predictor) level"
display_name(::MultiContinuousPPC)  = "multi-predictor"
