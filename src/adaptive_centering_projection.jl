# src/adaptive_centering_projection.jl — the projected part of a random-effect
# cell's hierarchical location (snag adaptive-centrin-647119ba, decision
# `BayesianRegressionModels:snag.adaptive-centrin-647119ba/decisions/2026-11-22T11-56-21-218-10krhre`).
#
# A cell of level `j`, term `k` is centred on everything else in its predictor
# that the level's own rows pin, so its `c = 1` coordinate is the level's total
# as the data see it. Write `Z_j` for the level's random-effect design (every
# random-effect column of the grouping on that predictor, `||` blocks included,
# on the level's rows `I_j`). A summand `x * theta` of the predictor whose
# coefficient `theta` no later cell rewrites enters cell `(j, k)`'s location
# with weight `(Z_j \ x[I_j])[k]`, the least-squares fit of the column on the
# level's own design:
#
# - a column constant within the level enters the intercept whole, and a column
#   equal to the cell's own design column enters its own term with weight 1;
# - a partly varying column enters by its projection (snag: `lprev`, 97.7 % of
#   whose centred sum of squares lies in the within-trip projection on `1 + pos`).
#
# The summands read are the predictor's categorical population terms (one
# indicator column per level code), its numeric population columns, and its
# `rw(...)` walks, whose row values are `sigma * W z` for a fixed `W`. Leaving
# any of them out leaves the ridge `(theta + d, levels - d)` to the prior: on
# the reporter's MBTA split, covariates without the walk level left R-hat at
# 1.10 to 2.87, the walk level without covariates at 2.52, both at 1.04.
#
# The rules of decisions `1xdxiv1` (matched coefficient) and `1wxltaq` (factor
# levels, coarser groupings) are special cases: a matched coefficient's column
# is its own design column (`coefficient`, unchanged), and an indicator constant
# within a level with an intercept term enters that intercept with weight 1,
# exactly as before. Coarser groupings stay `_adaptive_containers`.
#
# A zero-start walk's innovations are adaptive cells of their own (`:rw_walk`
# blocks, location 0 and the walk's SD as scale, the `cdar` cells' design), mapped
# before the random-effect cells, which read them at their model values like a
# containing cell (`walks`). A sum-to-zero walk's standardized path is not a
# cell: its level `sigma * phi[t]` is read directly (`factor_scales`).

# One `rw(...)` term of a predictor, as adaptive centring reads it. Row `i`'s
# walk value is `sigma * Σ_t _adaptive_walk_weight(w, i, t) * x[path[t]]`, with
# `sigma = exp(x[sigma])` (the walk's SD is declared `lower = 0`).
struct _AdaptiveWalkTerm
    predictor::Symbol
    term::Symbol
    order::Int
    sum_to_zero::Bool
    sigma::Int
    # Zero-start walks: the `n_steps - 1` innovations; sum-to-zero walks: the
    # `n_steps` standardized path values.
    path::Vector{Int}
    # The step each of the predictor's rows reads, `1:n_steps`.
    time_idx::Vector{Int}
    # The walk's cells (zero-start walks), `nothing` for a sum-to-zero walk.
    block::Union{Nothing,AdaptiveCenteringBlock}
end

# `W[i, t]`: the zero-start walk of order 1 is `x[1] = 0`, `x[s + 1] = x[s] +
# sigma z[s]`, so step `s` holds `sigma Σ_{t<s} z[t]`; of order 2,
# `x[s + 1] = 2x[s] - x[s - 1] + sigma z[s]` from `x[1] = 0`, so step `s` holds
# `sigma Σ_{t<s} (s - t) z[t]` (`random_walk_rows` / `random_walk2_rows`,
# `src/sbimpl.jl`); a sum-to-zero walk's step `s` holds `sigma phi[s]`.
function _adaptive_walk_weight(w::_AdaptiveWalkTerm, i::Int, t::Int)
    s = w.time_idx[i]
    w.sum_to_zero && return Float64(s == t)
    w.order == 1 ? Float64(t < s) : Float64(max(0, s - t))
end

# A random-effect block of a zero-start walk's innovations: one scalar cell per
# innovation, its scale the walk's SD `exp(x[sigma])`, its compiled frame the
# emitted standard-normal `z` (`target_c = 0`), located at 0.
function _adaptive_walk_block(term::Symbol, source::Symbol, steps, path, sigma)
    n = length(path)
    ranef = RanefBlock(term, :rw_walk, source, nothing, nothing,
                       collect(Any, steps[2:n + 1]), 1, n, Symbol(term, :_z),
                       true, false)
    AdaptiveCenteringBlock(ranef, 0.0, reshape(collect(Int, path), 1, n), Int[],
                           [sigma])
end

# A zero-start walk's path is its innovations, a sum-to-zero walk's its
# standardized values (`:structured`, as a `car` field's `phi`).
_adaptive_walk_role(entry) =
    _brm_rw_options(getkwargs(entry.value)).sum_to_zero ? :structured : :innovations

_adaptive_term_generated(::typeof(rw), descriptor, logical, entry) =
    _brm_term_parameter_output(descriptor, logical, entry,
                               _adaptive_walk_role(entry)).kind === :generated_quantity

"""
    _adaptive_walk_terms(model, unc_names) -> Vector{_AdaptiveWalkTerm}

Every `rw(...)` term of an `SBBRMI` or `GenerativePlan` that adaptive centring
reads: zero-start walks of order 1 or 2 (their innovations become `:rw_walk`
cells) and sum-to-zero walks (read in locations only). A walk is left out, and
keeps its emitted frame, when its SD has a declared upper bound or a lower bound
other than 0 (its coordinate is then not `log sigma`), when it is emitted in
generated quantities, or when a replay appends steps past the fitted grid
(`n_new_steps > 0`). A walk whose coordinates or row steps disagree with its
emitted declaration raises.
"""
function _adaptive_walk_terms(model, unc_names)
    descriptor = model isa BRMDescriptor ? model : brm_descriptor(model)
    plan = descriptor.plan
    brmi = plan.parent
    out = _AdaptiveWalkTerm[]
    context = nothing
    for predictor in linear_predictors(brmi)
        logical = predictor.name
        for entry in _brm_term_coordinate_entries(brmi, logical)
            getf(entry.value) === rw || continue
            _adaptive_term_generated(rw, descriptor, logical, entry) && continue
            path = brm_term_coordinates(descriptor, logical, unc_names;
                                        term=entry.term,
                                        parameter=_adaptive_walk_role(entry))
            sigma = brm_term_coordinates(descriptor, logical, unc_names;
                                         term=entry.term, parameter=:sd)
            owner = path.output.declaration
            isnothing(owner) && error(
                "BRM adaptive centering: rw `$(entry.term)` innovations have no " *
                "compiler declaration owner.")
            new_key = get(owner.keywords, :n_new_steps, nothing)
            n_new = new_key isa Symbol ? get(plan.data, new_key, 0) : 0
            n_new == 0 || continue
            length(sigma.coordinates) == 1 || error(
                "BRM adaptive centering: rw `$(entry.term)` must have one SD coordinate.")
            lower, upper = only.(_adaptive_bounds(descriptor, sigma.output, 1,
                                                  entry.term, "innovation SD"))
            lower == 0 && upper == Inf || continue
            isnothing(context) && (context = _brm_backend_context(brmi))
            prepared = _brm_prepare_term(entry.value, logical, context)
            st = prepared.state
            n_path = st.sum_to_zero ? st.n_steps : st.n_steps - 1
            length(path.coordinates) == n_path || error(
                "BRM adaptive centering: rw `$(entry.term)` owns " *
                "$(length(path.coordinates)) path coordinates for $(st.n_steps) steps.")
            idx_key = get(owner.keywords, :time_idx, nothing)
            emitted = idx_key isa Symbol ? get(plan.data, idx_key, nothing) : nothing
            isnothing(emitted) || emitted == st.time_idx || error(
                "BRM adaptive centering: rw `$(entry.term)` emits row steps that " *
                "differ from its prepared time index.")
            block = st.sum_to_zero || n_path == 0 ? nothing :
                _adaptive_walk_block(entry.term, prepared.source, st.steps,
                                     path.coordinates, only(sigma.coordinates))
            push!(out, _AdaptiveWalkTerm(logical, entry.term, st.order, st.sum_to_zero,
                only(sigma.coordinates), collect(Int, path.coordinates),
                collect(Int, st.time_idx), block))
        end
    end
    out
end

_adaptive_walk_blocks(walks) =
    AdaptiveCenteringBlock[w.block for w in walks if !isnothing(w.block)]

# ---- The level design and its projection -------------------------------------

# A numeric summand `values * x[coordinate]` of a predictor.
struct _AdaptiveNumericColumn
    coordinate::Int
    values::Vector{Float64}
end

# A categorical summand: row `i` reads `x[coordinates[codes[i]]]` (0: a level
# without a coordinate, such as a treatment-coded reference).
struct _AdaptiveCategoricalColumns
    codes::Vector{Int}
    coordinates::Vector{Int}
end

# What a predictor's rows read, per grouping: the grouping's random-effect
# design columns by coefficient label, and the summands its cells are centred
# on. `nothing` entries mean the shared geometry cannot be read for it.
struct _AdaptivePredictorRows
    n_rows::Int
    # grouping column => (labels, design matrix n_rows × M, row level codes)
    designs::Dict{Symbol,Tuple{Vector{Symbol},Matrix{Float64},Vector{Int}}}
    numeric::Vector{_AdaptiveNumericColumn}
    categorical::Vector{_AdaptiveCategoricalColumns}
    walks::Vector{_AdaptiveWalkTerm}
end

# The random-effect design of each grouping of `predictor`, from BRM's shared
# geometry (`_brm_simple_random_effect_plans`): per grouping column, the
# coefficient labels of its random-effect columns in plan order, their values on
# the predictor's rows, and each row's level code. A grouping whose plans
# disagree on the rows, or repeat a label, has no design.
function _adaptive_grouping_designs(brmi, context, predictor)
    out = Dict{Symbol,Tuple{Vector{Symbol},Matrix{Float64},Vector{Int}}}()
    plans = _brm_simple_random_effect_plans(brmi, predictor, context)
    isnothing(plans) && return out
    groups = Dict{Symbol,Vector{Any}}()
    for p in plans
        p isa _BRMRandomEffectPlan && p.group isa Symbol && isnothing(p.by) || continue
        push!(get!(groups, p.group, Any[]), p)
    end
    for (group, ps) in groups
        n = length(first(ps).indices)
        all(p -> length(p.indices) == n && p.indices == first(ps).indices, ps) || continue
        labels = Symbol[c.label for p in ps for c in p.columns]
        allunique(labels) || continue
        all(p -> all(c -> length(c.values) == n, p.columns), ps) || continue
        Z = reduce(hcat, [Float64.(c.values) for p in ps for c in p.columns])
        out[group] = (labels, reshape(Z, n, length(labels)), collect(Int, first(ps).indices))
    end
    out
end

# The numeric population columns of `predictor` its cells read: every column of
# the shared population design whose label is a population coefficient that
# `coordinate(label)` resolves to a sampled, unbounded coordinate (0 otherwise),
# on `n` rows. Matched columns (a label of a random-effect design column) are
# left to the cell's `coefficient`.
function _adaptive_numeric_columns(brmi, context, predictor, n, matched, coordinate)
    out = _AdaptiveNumericColumn[]
    op = linear_predictor_op(brmi, predictor)
    isnothing(op) && return out
    _, rhs = getargs(op, 2)
    terms = Tuple(t for t in _brm_additive_terms(rhs) if !_brm_prepares_term(t))
    design = _brm_population_design(predictor, terms, context.data,
                                    get(context.target_obs, predictor, nothing))
    isnothing(design) && return out
    names = popcoefnames(brmi, predictor)
    for column in design.columns
        label = column.label
        label in names && !(label in matched) || continue
        length(column.values) == n || continue
        j = coordinate(label)
        j == 0 || push!(out, _AdaptiveNumericColumn(j, Float64.(column.values)))
    end
    out
end

# The columns of `Z` (rows `I`) that are not linear combinations of the ones
# before them, in order: the level's own identified design.
function _adaptive_kept_columns(Z::AbstractMatrix)
    kept = Int[]
    Q = Vector{Vector{Float64}}()
    for m in axes(Z, 2)
        r = Vector{Float64}(Z[:, m])
        for q in Q
            r .-= dot(q, r) .* q
        end
        nr = norm(r)
        nr > 1e-10 * max(1.0, norm(view(Z, :, m))) || continue
        push!(kept, m)
        push!(Q, r ./ nr)
    end
    kept
end

# The least-squares weights of the column `x` (on one level's rows) on that
# level's design `Z` with identified columns `kept` and their QR factor `F`:
# exact for a column constant on the rows (whole on an all-ones column) or equal
# to a design column, the QR fit otherwise, with negligible weights dropped.
# Returns `(term, weight)` pairs.
function _adaptive_level_weights(x::AbstractVector, Z::AbstractMatrix, kept, ones_column,
                                 F)
    out = Tuple{Int,Float64}[]
    all(iszero, x) && return out
    if allequal(x) && ones_column != 0
        push!(out, (ones_column, Float64(first(x))))
        return out
    end
    for m in kept
        view(Z, :, m) == x && (push!(out, (m, 1.0)); return out)
    end
    beta = F \ Vector{Float64}(x)
    tol = 1e-12 * max(1.0, maximum(abs, x))
    for (i, m) in enumerate(kept)
        abs(beta[i]) > tol && push!(out, (m, beta[i]))
    end
    out
end

# One grouping's projected location entries for every level and every design
# column: `factors[m, g]` (coordinate, weight, scale coordinate) and
# `walks[m, g]` (walk block, step, weight), `m` the design column.
struct _AdaptiveProjectedEntries
    labels::Vector{Symbol}
    factors::Matrix{Vector{Tuple{Int,Float64,Int}}}
    walks::Matrix{Vector{Tuple{Any,Int,Float64}}}
end

function _adaptive_project_grouping(rows::_AdaptivePredictorRows, group::Symbol,
                                    G::Int)
    labels, Z, codes = rows.designs[group]
    M = length(labels)
    factors = [Tuple{Int,Float64,Int}[] for _ in 1:M, _ in 1:G]
    walks = [Tuple{Any,Int,Float64}[] for _ in 1:M, _ in 1:G]
    members = [Int[] for _ in 1:G]
    for (i, g) in enumerate(codes)
        1 <= g <= G && push!(members[g], i)
    end
    for g in 1:G
        I = members[g]
        isempty(I) && continue
        Zg = Z[I, :]
        kept = _adaptive_kept_columns(Zg)
        isempty(kept) && continue
        ones_column = something(findfirst(m -> all(isone, view(Zg, :, m)), kept), 0)
        ones_column = ones_column == 0 ? 0 : kept[ones_column]
        F = qr(Zg[:, kept])
        add!(x, f) = for (m, w) in _adaptive_level_weights(x, Zg, kept, ones_column, F)
            f(m, w)
        end
        for cat in rows.categorical
            level_codes = cat.codes[I]
            for code in unique(level_codes)
                j = 1 <= code <= length(cat.coordinates) ? cat.coordinates[code] : 0
                j == 0 && continue
                add!(Float64.(level_codes .== code),
                     (m, w) -> push!(factors[m, g], (j, w, 0)))
            end
        end
        for col in rows.numeric
            add!(view(col.values, I), (m, w) -> push!(factors[m, g], (col.coordinate, w, 0)))
        end
        for walk in rows.walks
            last_step = maximum(walk.time_idx[i] for i in I)
            for t in eachindex(walk.path)
                walk.sum_to_zero || t < last_step || break
                x = [_adaptive_walk_weight(walk, i, t) for i in I]
                if walk.sum_to_zero
                    add!(x, (m, w) -> push!(factors[m, g], (walk.path[t], w, walk.sigma)))
                elseif !isnothing(walk.block)
                    add!(x, (m, w) -> push!(walks[m, g], (walk.block, t, w)))
                end
            end
        end
    end
    _AdaptiveProjectedEntries(labels, factors, walks)
end

# The rows of `predictor` adaptive centring projects on, or `nothing` when its
# shared geometry has no grouping design.
function _adaptive_predictor_rows(brmi, context, predictor, walks, coordinate,
                                  categorical)
    designs = _adaptive_grouping_designs(brmi, context, predictor)
    isempty(designs) && return nothing
    n = length(last(first(values(designs))))
    all(d -> length(last(d)) == n, values(designs)) || return nothing
    matched = Set{Symbol}(l for d in values(designs) for l in first(d))
    numeric = _adaptive_numeric_columns(brmi, context, predictor, n, matched, coordinate)
    cats = filter(c -> length(c.codes) == n, categorical)
    ws = filter(w -> w.predictor === predictor && length(w.time_idx) == n, walks)
    _AdaptivePredictorRows(n, designs, numeric, cats, ws)
end

# Projected entries by `(predictor, grouping)`, computed on first use.
struct _AdaptiveProjector
    brmi::Any
    context::Base.RefValue{Any}
    walks::Vector{_AdaptiveWalkTerm}
    coordinate::Any       # (predictor, label) -> coordinate, 0 for none
    categorical::Any      # predictor -> Vector{_AdaptiveCategoricalColumns}
    rows::Dict{Symbol,Any}
    entries::Dict{Tuple{Symbol,Symbol},Any}
end

_AdaptiveProjector(brmi, walks, coordinate, categorical) =
    _AdaptiveProjector(brmi, Ref{Any}(nothing), walks, coordinate, categorical,
                       Dict{Symbol,Any}(), Dict{Tuple{Symbol,Symbol},Any}())

function _adaptive_projector_context(p::_AdaptiveProjector)
    isnothing(p.context[]) && (p.context[] = _brm_backend_context(p.brmi))
    p.context[]
end

# Cell `(term coefficient, level g)` of a block grouped by `group` on
# `predictor` with row codes `block_rows`: the projected entries of every level,
# as `(factors, walks)` vectors indexed by level, or `nothing` when the grouping
# cannot be projected (its geometry is unreadable, its rows do not match the
# block's, or the term is not one of its design columns).
function _adaptive_projected_term(p::_AdaptiveProjector, predictor::Symbol, group::Symbol,
                                  coefficient::Symbol, block_rows::AbstractVector, G::Int)
    entries = get!(p.entries, (predictor, group)) do
        rows = get!(p.rows, predictor) do
            context = _adaptive_projector_context(p)
            _adaptive_predictor_rows(p.brmi, context, predictor, p.walks,
                                     label -> p.coordinate(predictor, label),
                                     p.categorical(predictor))
        end
        isnothing(rows) || !haskey(rows.designs, group) ? nothing :
            (; projected=_adaptive_project_grouping(rows, group, G),
               codes=rows.designs[group][3])
    end
    isnothing(entries) && return nothing
    entries.codes == block_rows || return nothing
    m = findfirst(==(coefficient), entries.projected.labels)
    isnothing(m) && return nothing
    size(entries.projected.factors, 2) == G || return nothing
    (entries.projected.factors[m, :], entries.projected.walks[m, :])
end

# The walks the wrapper maps for `model`, as `_adaptive_cell_blocks` reads them.
_adaptive_cell_walks(model, names) = _adaptive_walk_terms(model, names)
_adaptive_cell_walks(::RKBRMI, names) = _AdaptiveWalkTerm[]
