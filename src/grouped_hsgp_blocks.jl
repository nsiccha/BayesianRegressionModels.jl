# ==============================================================================
# grouped_hsgp_blocks.jl — a grouped HSGP's per-level weights for the
# population and transport routes.
#
# `hsgp(x...; by = g)` draws its standardized basis weights as one plain
# `zflat_<suffix> ~ std_normal()` vector (`_sb_term_group_block`, prior
# `:iid_normal`), reshaped to the levels × basis block `b_<suffix>`, and a
# per-level hyper-predictor (`log(sd(hsgp(x))) ~ 1 + (1 | g)`) draws its level
# effects `<term>_z_sigma` / `<term>_z_rho` inside the term's submodel. Neither
# is a `ranef_*` submodel, so `ranef_blocks` does not list them; they are
# still `g`'s effects. `population_draws`, `population_model` and
# `transport_draws` read them from here (snag `brm-use-silent-o-0039d181`,
# decision `0wlazld`).
#
# Everything is read from the emitter's own block plan (`_sb_group_block_plan`,
# the one the emitter, the descriptor and adaptive centering read) and from the
# traced program's placement, never from a parsed coordinate name.
# ==============================================================================

# One grouped HSGP term's weights and its per-level hyper effects.
#   block      — the levels × basis weights binding `b_<suffix>`
#   weights    — the standardized draw `zflat_<suffix>`, index `(g - 1) * n_basis + j`
#   term       — the term statement's binding (the declaration owning the hypers)
#   group      — the `by=` column
#   levels     — its labels, in the emitter's index order
#   n_basis    — basis functions per row of the block
#   per_level  — one row per level (false: `cor = multigroup((x, y), …)`, whose
#                rows are an inducing grid shared by every level)
#   generated  — the weights are drawn in generated quantities
#   hypers     — `(; hyper, z, generated)` per hyper-predictor with level effects
const _GroupedHSGPBlock = NamedTuple{
    (:block, :weights, :term, :group, :levels, :n_basis, :per_level, :generated, :hypers)}

_grouped_hsgp_blocks(model) = _grouped_hsgp_blocks(_ranef_plan(model), nothing)
_grouped_hsgp_blocks(d::BRMDescriptor{<:Any,<:StanBlocks.ModelDescriptor}) =
    _grouped_hsgp_blocks(d.plan, d.stan)

function _grouped_hsgp_blocks(plan, stan)
    brmi = plan.parent
    entries = [p for p in _sb_group_block_plan(_sb_collect_group_block_terms(brmi))
               if p.f === hsgp]
    isempty(entries) && return _GroupedHSGPBlock[]
    kinds = _ranef_output_kinds(plan, stan)
    body = plan.model.model
    out = _GroupedHSGPBlock[]
    seen = Set{Symbol}()
    for (; field, rhs_e, suffix) in entries
        suffix in seen && continue
        push!(seen, suffix)
        block = Symbol(:b_, suffix)
        weights = Symbol(:zflat_, suffix)
        group = _sb_group_name(field.group, rhs_e)
        kind = get(kinds, weights, nothing)
        kind in (:parameter, :generated_quantity) || error(
            "BRM prediction: grouped HSGP weights `$weights` have descriptor kind ",
            "`$kind`; expected a sampled parameter or generated quantity.")
        term = _grouped_hsgp_term_binding(body, block)
        hypers = NamedTuple[]
        for (hyper, local_z) in ((:sd, :z_sigma), (:length_scale, :z_rho))
            z = Symbol(term, :_, local_z)
            zk = get(kinds, z, nothing)
            isnothing(zk) && continue
            zk in (:parameter, :generated_quantity) || error(
                "BRM prediction: hyper-predictor effects `$z` have descriptor kind ",
                "`$zk`; expected a sampled parameter or generated quantity.")
            push!(hypers, (; hyper, z, generated=zk === :generated_quantity))
        end
        push!(out, (; block, weights, term, group,
                    levels=_grouped_hsgp_levels(plan, brmi, group),
                    n_basis=Int(field.n_per_group),
                    per_level=!haskey(field, :n_rows),
                    generated=kind === :generated_quantity,
                    hypers=Tuple(hypers)))
    end
    out
end

# The term statement that reads the weights block as its `beta=`: the
# declaration under whose name the hyper-predictor effects are emitted.
function _grouped_hsgp_term_binding(body::Expr, block::Symbol)
    hits = Symbol[]
    for stmt in body.args
        Meta.isexpr(stmt, :call, 3) && stmt.args[1] === :~ || continue
        rhs = stmt.args[3]
        Meta.isexpr(rhs, :call) && length(rhs.args) >= 2 &&
            Meta.isexpr(rhs.args[2], :parameters) || continue
        any(kw -> Meta.isexpr(kw, :kw) && kw.args[1] === :beta && kw.args[2] === block,
            rhs.args[2].args) || continue
        push!(hits, _sb_plan_lhs_name(stmt.args[2]))
    end
    length(hits) == 1 || error(
        "BRM prediction: grouped HSGP weights block `$block` is read by ",
        "$(length(hits)) term statements; expected exactly one.")
    only(hits)
end

# The labels the grouping's index was coded through: the `:group_index`
# record (fitted labels on a frozen replay, the new rows' labels for a
# resampled factor), else the column's own levels.
function _grouped_hsgp_levels(plan, brmi, group::Symbol)
    entry = get(plan.preproc, Symbol(group, :_idx), nothing)
    entry isa PreprocEntry && entry.kind === :group_index &&
        return collect(entry.const_.levels)
    raw = column_data(brmi, group)
    isnothing(raw) && error(
        "BRM prediction: grouped HSGP grouping `$group` is neither a data column ",
        "nor a data-only assignment.")
    collect(_sb_fit_levels(raw))
end

_grouped_hsgp_matches(b, group::Symbol) = b.group === group

# The weights' coordinates by Stan name: an `n_rows × n_basis` matrix of
# positions into `unc_names` (`n_rows` the levels, or the inducing grid).
function _grouped_hsgp_weight_coordinates(b, unc_names)
    pos = _ranef_name_positions(unc_names)
    prefix = string(b.weights, ".")
    n = count(nm -> startswith(String(nm), prefix), unc_names)
    n > 0 && n % b.n_basis == 0 || error(
        "BRM prediction: grouped HSGP weights `$(b.weights)` have $n coordinates, ",
        "not a whole number of rows of $(b.n_basis) basis functions.")
    rows = n ÷ b.n_basis
    !b.per_level || rows == length(b.levels) || error(
        "BRM prediction: grouped HSGP weights `$(b.weights)` hold $rows rows but ",
        "`$(b.group)` has $(length(b.levels)) levels.")
    [let nm = string(b.weights, ".", (r - 1) * b.n_basis + j)
         haskey(pos, nm) ? pos[nm] : error("BRM prediction: coordinate `$nm` is missing.")
     end for r in 1:rows, j in 1:b.n_basis]
end

# A hyper-predictor's per-level effects: one position per level.
function _grouped_hsgp_hyper_coordinates(b, h, unc_names)
    pos = _ranef_name_positions(unc_names)
    [let nm = string(h.z, ".", g)
         haskey(pos, nm) ? pos[nm] : error("BRM prediction: coordinate `$nm` is missing.")
     end for g in eachindex(b.levels)]
end

# Every sampled coordinate of the selected groupings' grouped HSGPs.
function _grouped_hsgp_population_coordinates(blocks, unc_names)
    out = Int[]
    for b in blocks
        b.generated || append!(out, vec(_grouped_hsgp_weight_coordinates(b, unc_names)))
        for h in b.hypers
            h.generated || append!(out, _grouped_hsgp_hyper_coordinates(b, h, unc_names))
        end
    end
    out
end

# Copy a grouped HSGP's weights and per-level hyper effects from `bf` onto
# `bt` by level label, recording the source coordinate of each target one in
# `plan_idx` and claiming it. A level the source lacks (or a `resample`d
# grouping) needs fresh weights, which come from StanBlocks' generated-quantity
# re-draw (`reprocess(…; resample_groups)`), not from Julia-side randomness.
# Weights on an inducing grid are shared by every level and copy row for row.
function _grouped_hsgp_transport!(plan_idx, claimed, bf, bt, unc_from, unc_to, resample_set)
    fresh = bt.group in resample_set
    level_pos = Dict(l => g for (g, l) in enumerate(bf.levels))
    new_levels = [l for l in bt.levels if !haskey(level_pos, l)]
    sampled_per_level = (bt.per_level && !bt.generated) || any(h -> !h.generated, bt.hypers)
    if sampled_per_level && (fresh || !isempty(new_levels))
        shown = join(repr.(first(new_levels, 5)), ", ")
        error("BRM prediction: grouped HSGP `$(bt.block)` (group `$(bt.group)`) needs ",
              "fresh per-level draws for ",
              fresh ? "every level (`resample` names `$(bt.group)`)" :
                  "level(s) $shown$(length(new_levels) > 5 ? " … ($(length(new_levels)) total)" : "")",
              ", which the source model does not have. Build the target with ",
              "`reprocess(fit, new_df; resample_groups=[$(repr(bt.group))])` so StanBlocks ",
              "draws the new levels in generated quantities, then `transport_draws` ",
              "onto that artifact.")
    end
    source_row(r) = bt.per_level ? level_pos[bt.levels[r]] : r
    if !bt.generated
        bf.generated && error(
            "BRM prediction: the target samples grouped HSGP weights `$(bt.weights)`, ",
            "but the source draws them in generated quantities, so the fitted draws ",
            "carry no weights to transport onto it.")
        wt = _grouped_hsgp_weight_coordinates(bt, unc_to)
        wf = _grouped_hsgp_weight_coordinates(bf, unc_from)
        bt.per_level || size(wt) == size(wf) || error(
            "BRM prediction: grouped HSGP `$(bt.block)` has a $(size(wt, 1))-point ",
            "inducing grid in the target and $(size(wf, 1)) in the source.")
        for r in axes(wt, 1), j in axes(wt, 2)
            claimed[wt[r, j]] = true
            plan_idx[wt[r, j]] = wf[source_row(r), j]
        end
    end
    for h in bt.hypers
        h.generated && continue
        hf = findfirst(x -> x.hyper === h.hyper, bf.hypers)
        isnothing(hf) && error(
            "BRM prediction: grouped HSGP `$(bt.block)` predicts its $(h.hyper) per level ",
            "in the target but not in the source.")
        bf.hypers[hf].generated && error(
            "BRM prediction: the target samples hyper-predictor effects `$(h.z)`, but ",
            "the source draws them in generated quantities.")
        ct = _grouped_hsgp_hyper_coordinates(bt, h, unc_to)
        cf = _grouped_hsgp_hyper_coordinates(bf, bf.hypers[hf], unc_from)
        for g in eachindex(ct)
            claimed[ct[g]] = true
            plan_idx[ct[g]] = cf[level_pos[bt.levels[g]]]
        end
    end
    nothing
end
