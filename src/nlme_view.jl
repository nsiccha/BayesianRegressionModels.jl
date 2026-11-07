include("nlme_selection.jl")

# src/nlme_view.jl — the nonlinear-mixed-effects reading of an RK-lowered model.
#
# NLME estimators (posthoc/EBE, FOCEI, SAEM; the separate NLMEEstimation
# package) see a population model as: population coordinates θ, subject-level
# covariance coordinates Ω, other scalars σ, and one compact random-effect
# vector η_i per subject, with subjects conditionally independent given
# (θ, Ω, σ). Shared sampled latents may be estimated in θ or fixed explicitly.
# The RK
# emitter already files one semantic record per sampled declaration it can
# address (`_rk_coordinate_record!`, shared with the cross-backend coordinate
# transport), so the partition is read from those records, never from
# coordinate positions or name guessing. Every packed coordinate must be claimed
# by exactly one record; anything unclaimed or outside the per-subject structure
# is refused by name.

"""
    BRMNLMEViewError <: Exception

A model that has no nonlinear-mixed-effects reading in [`brm_nlme_view`](@ref):
no subject-level random effects, a grouping named in neither `nested_groups`
nor `shared_groups`, a unit-owned grouping that couples its levels
(`gr(...; cor=...)`), an `mm(...)` block, or a coordinate the semantic
inventory does not cover. A stratified `gr(g, by=b)` block is read as one
covariance family per stratum. `message` names the offending declaration or coordinates.
"""
struct BRMNLMEViewError <: Exception
    message::String
end
Base.showerror(io::IO, err::BRMNLMEViewError) = print(io, err.message)
_brm_nlme_error(message) = throw(BRMNLMEViewError(message))

"""
    BRMNLMEView

The nonlinear-mixed-effects partition of one [`RKBRMI`](@ref), returned by
[`brm_nlme_view`](@ref). All positions index the backend's packed
unconstrained coordinate vector (`coordinates`, in packed order).

- `group` — the subject grouping column; `levels[i]` is subject `i`'s label,
  in the RK level order.
- `role[c]` — `:population` (population coefficients, categorical contrasts,
  monotonic simplexes, ordinal thresholds and vector parameters),
  `:shared_latent` (sampled coordinates shared by the units: HSGP components,
  missing-value completions, submodels, `car(...)` fields (only their separator
  draws, scale and mixing share when lowered with `car_units`), and every
  coordinate of a random-effect block over a grouping in `shared_groups`; a
  grouped HSGP or `cdar` walk whose levels are the units shares only its
  hyperparameters),
  `:scalar` (other sampled scalars such as residual scales),
  `:subject_scale` / `:subject_correlation` (the random-effect scales and
  Cholesky correlation factors, i.e. Ω), or `:subject_effect` /
  `:subject_mixing` (subject `subject[c]`'s standardized draw or Student-t
  mixing weight).
- `subject_coordinates[i]` — subject `i`'s compact block of standardized
  draws. Flat models share one order; nested models add one instance per
  nested level owned by this subject.
- `block_margins[k]` — `(; block, predictor, coefficient)` for position `k` of
  each random-effect family (`coefficient === :mixing` for a mixing weight).
  `block` is `(; group, id, predictor)`: the grouping column, the `|ID|`, and,
  for a block without an ID, the predictor that owns it (`nothing` otherwise).
- `family_sizes[f]` and `subject_families[i]` — each covariance family's
  dimension and the ordered family IDs for subject `i`'s block instances.
- `families[f]` — `(; block, record, stratum, size, scales, factor, fixed)`:
  family `f`'s block identity, its coordinate record, its stratum (the
  position of the `by` level for a stratified `gr(g, by=b)` block, `nothing`
  otherwise), its dimension, the packed positions of its scales (margin
  order) and of its LKJ correlation factor (empty for `K = 1`), and its fixed
  covariance (`nothing` when estimated). A stratified block has one family
  per stratum; each level's block belongs to its stratum's family, and its
  `block` identity carries `stratum` (the `by` level). A non-centered
  completion (`RKBRMI(brmi; completions=:noncentered)`) whose rows lie on the
  grouping axis or read it through data-indexed gathers is one family of
  size 1 with covariance fixed at 1: each
  missing cell is an instance of it in the subject owning its row, with role
  `:subject_effect` (its block identity carries `completion=true`). The
  grouped entries of a non-centered `mi(x)::shares(total, rows)` completion
  are one more such family (its block identity also carries `shares=true`):
  each coordinate of a group, `K - 1` for a group of `K` missing entries, is an
  instance in the subject owning the group's rows, and the group's observed
  term (`RKBRMI`'s `completions`) belongs to that subject. Bounded free
  entries use the same Gaussian family and observe their transformed law
  minus the Gaussian density, preserving the exact full density. A grouped
  HSGP `hsgp(x; by=g)` without `cor=` whose levels are the units (`g` is the
  subject grouping, a nested grouping, or a column whose labels are subjects)
  is one family whose size is its number of basis functions, with covariance
  fixed at the identity: each level's standardized weights are an instance of
  it in the subject owning the level (its block identity carries `hsgp`, the
  term's declaration). Each `(1 | g)` effect of a hyper-predicted amplitude or
  length scale (`log(sd(hsgp(x))) ~ 1 + (1 | g)`) is an estimated family of
  size 1 over the same units. A walk `cdar(step; by=g, cor=C)` whose groups
  are units in the same sense, and whose `cor=` couples only groups of one
  subject, is one family whose size is its number of steps, with covariance
  fixed at the identity: each group's standardized innovations are an
  instance of it in the subject owning the group (its block identity carries
  `cdar`, the term's declaration; the margins are `cdar(step)[s]` for each
  step value `s`). Its marginal SD and persistence are `:shared_latent`. A
  `car(...)` field lowered with `RKBRMI(brmi; car_units)` over the subject
  labels has a family of size 1 fixed at 1 for its interior draws, each draw an
  instance in the subject that is its unit, and for a BYM2 field one more for
  each area's iid part (block identity `car`, the term's declaration, and
  `part`, `:draw` or `:iid`; margin `car(area)[part]`, level the draw's or
  area's position).
- `subject_margins[i]` — the compact coordinates' family margin and grouping
  level labels for subject `i`.
- `mu_references` — one `(; predictor, coefficient, link, population, subject,
  design)` per population coefficient that can be absorbed into a subject
  margin. `population` is the coefficient's packed position, `subject` the
  margin's position in a flat subject block, and `design[i]` its multiplier
  for subject `i`. Nested models currently expose no such references.
- `rows[i]` — the rows of the grouping column that belong to subject `i`.
- `nested_groups`, `shared_groups` — the groupings the caller named as owned
  by a unit and as crossed with the units.
- `unit_partition` — `:subjects` (default) or `:connected`. In connected mode,
  `levels[i]` is the tuple of scientific subject labels in inference unit `i`;
  `subject_coordinates`, `subject_families`, `subject_margins` and `rows` are
  indexed by inference unit. Every original covariance-family instance remains.
- `subject_levels`, `subject_unit`, `unit_members`, `subject_ranges` — original
  scientific labels, their unit IDs, each unit's original subject indices, and
  each original subject's range inside its unit's compact effect vector.
  `coordinate_subject[c]` retains the original owner of packed coordinate `c`;
  `subject[c]` is its inference-unit owner. Zero means shared in both maps.
- `factor_supports` — in connected mode, structural packed-coordinate supports
  per observation cell, keyed by observation name. A support is a vector of
  indices or `:whole`; it includes dormant branches and zero coefficients.
  These supports determine connectivity and retain individual-cell sparsity.

Subject draws are standardized (`η = diag(scale) L z`, `z ~ N(0, I)`): the
subject block holds `z`, and Ω is read through the scale and correlation
coordinates. Because the map from `z` to `η` is linear given Ω, Laplace and
FOCE objectives agree in either coordinate system.
"""
struct BRMNLMEView
    group::Symbol
    levels::Vector{Any}
    coordinates::Vector{Symbol}
    role::Vector{Symbol}
    subject::Vector{Int}
    subject_coordinates::Vector{Vector{Int}}
    block_margins::Vector{NamedTuple}
    mu_references::Vector{NamedTuple}
    rows::Vector{Vector{Int}}
    family_sizes::Vector{Int}
    subject_families::Vector{Vector{Int}}
    subject_margins::Vector{Vector{NamedTuple}}
    nested_groups::NamedTuple
    shared_groups::Tuple{Vararg{Symbol}}
    families::Vector{NamedTuple}
    ragged::Bool
    unit_partition::Symbol
    subject_levels::Vector{Any}
    subject_unit::Vector{Int}
    unit_members::Vector{Vector{Int}}
    subject_ranges::Vector{UnitRange{Int}}
    coordinate_subject::Vector{Int}
    factor_supports::Dict{Symbol,Vector{Union{Symbol,Vector{Int}}}}
end

# Subjects whose η blocks differ in shape (nested groupings, stratified
# blocks) use NLMEEstimation's ragged layout.
_brm_nlme_ragged(view::BRMNLMEView) = view.ragged

Base.show(io::IO, v::BRMNLMEView) = print(io, "BRMNLMEView(",
    length(v.levels), " subjects of `", v.group, "`, ",
    _brm_nlme_ragged(v) ? string("η lengths ", length.(v.subject_coordinates)) :
        string(length(v.block_margins), " coordinates per subject"), ", ",
    count(==(:population), v.role), " population, ",
    count(==(:shared_latent), v.role), " shared latent, ",
    count(r -> r === :subject_scale || r === :subject_correlation, v.role),
    " Ω and ", count(==(:scalar), v.role), " scalar coordinates)")

# The coordinate records this view reads. The old panel-only kernel plan has
# no semantic inventory. Closure-style `kernel(...) do` cells instead use the
# value plan, like `@plate for` cells.
_brm_nlme_records(plan::Union{_RKStructuralPlan,_RKValuePlan,_RKHeldOutPlan}) =
    _rk_coordinate_records(plan)
_brm_nlme_records(plan) = _brm_nlme_error(
    "brm_nlme_view: an RK plan of kind `$(nameof(typeof(plan)))` has no " *
    "semantic coordinate inventory, so its NLME partition cannot be read. " *
    "The old panel-only `_RKKernelPlan` route is not covered; " *
    "`kernel(...) do` and `@plate for` cells lower to value plans, which are.")

_brm_nlme_owned(name::Symbol, declaration::Symbol) = name === declaration ||
    startswith(String(name), string(declaration, "."))

# The packed coordinates a declaration owns. A declaration the layout holds at
# zero length (`empty`), such as a single-level treatment-coded factor's
# coefficient block, owns none and has nothing to partition. Any other
# declaration without a coordinate is a defect in the inventory.
function _brm_nlme_declaration_coordinates(names, declaration, empty)
    hits = [i for (i, name) in enumerate(names) if _brm_nlme_owned(name, declaration)]
    isempty(hits) && declaration ∉ empty && error("brm_nlme_view: internal: " *
        "declaration `$declaration` owns no packed coordinate")
    hits
end

# A random-effect block's identity: its grouping and `|ID|`. A block without an
# ID belongs to one predictor, and several predictors can each own one over the
# same grouping, so its owning predictor completes the identity.
_brm_nlme_block_identity(record) = (; group=record.group, id=record.id,
    predictor=record.id === nothing ? first(first(record.margins)) : nothing)

_brm_nlme_regression(plan::_RKStructuralPlan) = plan
_brm_nlme_regression(plan::_RKValuePlan) = plan.regression
_brm_nlme_regression(plan::_RKHeldOutPlan) = _brm_nlme_regression(plan.parent)

# Read the same scalar design columns that the RK affine emitter multiplies by
# its population coefficients. Other coefficient families need a separate
# coordinate-to-column record and are deliberately not inferred from names.
function _brm_nlme_population_column(plan::_RKStructuralPlan, predictor, coefficient, nrows)
    priors = [p for p in plan.population_priors if p.predictor === predictor &&
        p.addressee === coefficient]
    # MuReferencing shifts unconstrained θ additively. Bounded/positive priors
    # transform that coordinate before the predictor sees it, so their
    # population design cannot certify this invariance.
    length(priors) == 1 && only(priors).family in
        (:Normal, :Cauchy, :Laplace, :Logistic, :StudentT, :Flat) || return nothing
    specs = [term for p in plan.predictors if p.name === predictor
        for term in p.terms if term.addressee === coefficient &&
        term.kind in (:intercept, :continuous)]
    length(specs) == 1 || return nothing
    term = only(specs)
    term.kind === :intercept && return ones(Float64, nrows)
    # A completed covariate is a live model value, not a frozen design column;
    # it cannot certify a mu-reference from the prepared column table.
    values = get(plan.columns, only(term.columns), nothing)
    values === nothing && return nothing
    length(values) == nrows && all(x -> x isa Real && isfinite(x), values) || return nothing
    Float64.(values)
end

# A population column is mu-referenced through a random slope when its actual
# emitted Z column is identical. A subject-constant column can instead be
# absorbed through an intercept, including a covariate slope without a random
# slope. Both cases preserve the whole predictor row by row.
function _brm_nlme_mu_references(backend, block_margins, population, rows, nrows)
    plan = _brm_nlme_regression(backend.plan)
    references = NamedTuple[]
    for ((predictor, coefficient), position) in sort!(collect(population); by=x -> last(x))
        column = _brm_nlme_population_column(plan, predictor, coefficient, nrows)
        column === nothing && continue
        link = _rk_predictor_link(parent(backend), predictor)
        # A matching random slope takes precedence, since it also works when
        # the observed covariate varies among a subject's rows.
        matched = false
        for (k, margin) in enumerate(block_margins)
            margin.predictor === predictor && margin.coefficient === coefficient || continue
            recipes = [r.z for bucket in plan.ranef_buckets
                if bucket.group === margin.block.group && bucket.id === margin.block.id
                for r in bucket.margins if r.predictor === predictor &&
                    r.coefficient === coefficient]
            length(recipes) == 1 || continue
            recipe = only(recipes)
            z = recipe.kind === :ones ? ones(Float64, nrows) :
                recipe.kind === :column ? get(plan.columns, recipe.column, nothing) : nothing
            z === nothing && continue
            length(z) == nrows && all(isequal.(column, z)) || continue
            push!(references, (; predictor, coefficient, link, population=position,
                subject=k, design=ones(Float64, length(rows))))
            matched = true
            break
        end
        matched && continue
        for (k, margin) in enumerate(block_margins)
            margin.predictor === predictor && margin.coefficient === :Intercept || continue
            values = Float64[]
            for subject_rows in rows
                isempty(subject_rows) && break
                value = column[first(subject_rows)]
                all(r -> isequal(column[r], value), subject_rows) || break
                push!(values, value)
            end
            length(values) == length(rows) || continue
            push!(references, (; predictor, coefficient, link, population=position,
                subject=k, design=values))
            break
        end
    end
    references
end

# The subject owning each level `levels` of a term grouped by `by`, or
# `nothing` when the term is shared by the subjects: a `by` column the caller
# named in `shared_groups`, or one whose levels name no subject. The levels are
# units when `by` is a nested grouping (each level owned by its subject) or
# when every level names a subject: the subject grouping itself, or another
# column with its labels, joined by label as one `|ID|` joins grouping
# columns. `term` names the term and `states` its per-level values in the
# refusal of a `by` mixing both kinds of label.
function _brm_nlme_level_units(by, levels, term, states, group, subject_index, owner,
        group_levels, nested_groups, shared_groups)
    by in shared_groups && return nothing
    if haskey(nested_groups, by)
        index = Dict(_brm_transport_level(level) => g
            for (g, level) in enumerate(group_levels[by]))
        return Int[owner[by][get(index, _brm_transport_level(level)) do
                error("brm_nlme_view: internal: level $(repr(level)) of `$by` " *
                    "has no row")
            end] for level in levels]
    end
    units = Int[get(subject_index, _brm_transport_level(level), 0) for level in levels]
    all(iszero, units) && return nothing
    unmatched = levels[iszero.(units)]
    isempty(unmatched) || _brm_nlme_error(
        "brm_nlme_view: $term has levels that are subjects of `$group` and levels " *
        "that are not ($(join(repr.(unmatched), ", "))), so its $states are " *
        "neither one subject's each nor shared by all. Group it by the subjects, " *
        "by a grouping nested in them (`nested_groups`), or by labels that name " *
        "no subject.")
    units
end

# The subject owning each level of a grouped HSGP whose levels are units, or
# `nothing` when the term is shared by the subjects: an ungrouped term, a
# `cor=` that couples its levels, or levels that are not units
# (`_brm_nlme_level_units`).
function _brm_nlme_hsgp_units(record, group, subject_index, owner, group_levels,
        nested_groups, shared_groups)
    record.grouped && record.group !== nothing && !record.level_correlation ||
        return nothing
    _brm_nlme_level_units(record.group, record.levels,
        "`hsgp($(join(record.axes, ", ")); by=$(record.group))` in `$(record.predictor)`",
        "weights", group, subject_index, owner, group_levels, nested_groups, shared_groups)
end

# The subject owning each group of a `cdar(step; by=g, cor=C)` walk whose groups
# are units (`_brm_nlme_level_units`), or `nothing` when the walk is shared by
# the subjects: groups that are not units, or a `cor=` coupling groups of
# different subjects. Each group's standardized innovations then enter only its
# own subject's rows.
function _brm_nlme_cdar_units(record, group, subject_index, owner, group_levels,
        nested_groups, shared_groups)
    units = _brm_nlme_level_units(record.group, record.levels,
        "`cdar($(record.step); by=$(record.group))` in `$(record.predictor)`",
        "innovations", group, subject_index, owner, group_levels, nested_groups,
        shared_groups)
    units === nothing && return nothing
    all(c -> iszero(record.cor_factor[c]) || units[c[1]] == units[c[2]],
        CartesianIndices(record.cor_factor)) || return nothing
    units
end

# A hyper-predictor's `(1 | g)` effects of a grouped HSGP, as a random-effect
# record: one margin, the log-scale intercept of the predicted
# hyperparameter (`log(sd(<predictor>, hsgp(x)))`), its scale and its
# standardized draws `<z>.<level>`.
function _brm_nlme_hsgp_hyper_record(record, hyper, effects)
    address = Symbol("log(", hyper, "(", record.predictor, ", hsgp(",
        join(record.axes, ", "), ")))")
    (; kind=:hsgp_hyper, group=record.group, id=nothing,
        margins=((address, :Intercept),),
        scale=Symbol(record.declaration, ".", effects.scale), scales=nothing, L=nothing,
        z=Symbol(record.declaration, ".", effects.z))
end

# The stratum position (into `strata`, the `by` column's levels) of each level
# of a stratified block's grouping column. Every level lies in one stratum;
# `fail` throws the refusal of one that does not, prefixed by `caller`.
function _brm_level_strata(groups, by, levels, strata, record;
        caller="brm_nlme_view", fail=_brm_nlme_error)
    index = Dict(_brm_transport_level(level) => s for (s, level) in enumerate(strata))
    stratum = Dict{Any,Int}()
    for (g, b) in zip(groups, by)
        key, s = _brm_transport_level(g), index[_brm_transport_level(b)]
        get!(stratum, key, s) == s || fail(
            "$caller: level $(repr(g)) of `$(record.group)` lies in more than " *
            "one stratum of `$(record.by)`, so the stratified block " *
            "`gr($(record.group), by=$(record.by))` has no single stratum for it")
    end
    Int[get(stratum, _brm_transport_level(level)) do
            error("$caller: internal: level $(repr(level)) of " *
                "`$(record.group)` has no row")
        end for level in levels]
end

# The component declaration owning every coordinate of a random-effect block
# (`<draws>.z`, `<draws>.tau`, `<draws>.L`, `<draws>.w`).
_brm_nlme_ranef_declaration(record) = Symbol(chopsuffix(String(record.z), ".z"))

# `shared_groups` as a tuple of distinct grouping names.
function _brm_nlme_shared_groups(groups)
    groups isa Union{Tuple,AbstractVector} && all(g -> g isa Symbol, groups) ||
        throw(ArgumentError("shared_groups must be a tuple of grouping column " *
            "names, such as `shared_groups = (:class,)`; got $(repr(groups))"))
    Tuple(Symbol[unique(groups)...])
end

# The groupings a caller may name: those of random-effect blocks and of the
# terms grouped by a column of their own, grouped HSGPs and `cdar` walks.
_brm_nlme_term_groupings(records) = unique(Symbol[r.group for r in records
    if (r.kind === :hsgp && r.grouped && r.group !== nothing) || r.kind === :cdar])
_brm_nlme_groupings(records) = unique(Symbol[
    [r.group for r in records if r.kind in (:ranef, :stratified_ranef)];
    _brm_nlme_term_groupings(records)])

_brm_nlme_ticked(names) = join(string.("`", names, "`"), ", ")

# A nested grouping's parent labels, one per row of the grouping: a column the
# model reads, by name, or the labels themselves. The backend keeps only the
# columns its formulas read, so a column no formula reads is passed as labels.
_brm_nlme_parent_labels(columns, nested, parent::Symbol) = get(columns, parent) do
    _brm_nlme_error("brm_nlme_view: the parent column `$parent` of `$nested` " *
        "is not a column the model reads, and the backend keeps no other data. " *
        "Pass its labels instead, row-aligned with `$nested`: " *
        "`nested_groups = ($nested = data.$parent,)`")
end
_brm_nlme_parent_labels(columns, nested, parent::AbstractVector) = parent
_brm_nlme_parent_labels(columns, nested, parent) = _brm_nlme_error(
    "brm_nlme_view: the parent of `$nested` must be a column name (a Symbol) " *
    "or a vector of subject labels, got $(typeof(parent))")
_brm_nlme_parent_name(parent::Symbol) = " (`$parent`)"
_brm_nlme_parent_name(parent) = ""

function _brm_nlme_check_shared_groups(records, shared_groups, subject_group,
        nested_groups)
    known = _brm_nlme_groupings(records)
    for g in shared_groups
        g in known || _brm_nlme_error(
            "brm_nlme_view: shared_groups names `$g`, which no random-effect block, " *
            "grouped `hsgp(...; by = ...)` term or `cdar` walk groups by; the model's groupings " *
            "are $(_brm_nlme_ticked(known))")
        g === subject_group && _brm_nlme_error(
            "brm_nlme_view: `$g` is the subject grouping, so it cannot also be " *
            "crossed with the subjects (shared_groups)")
        haskey(nested_groups, g) && _brm_nlme_error(
            "brm_nlme_view: `$g` is named in both nested_groups (owned by one " *
            "subject) and shared_groups (crossed with the subjects); name it in one")
    end
end

# Every subject-owned grouping except the subject grouping is a key of
# `nested_groups`, and every key is such a grouping. The refusal names the
# groupings named nowhere and the keys that are not groupings. A joined `|ID|`
# block over several grouping columns is ONE grouping, named for its first
# column, so its other columns are no keys.
function _brm_nlme_check_nested_groups(records, group, groups, nested_groups,
        shared_groups)
    expected = filter(!=(group), groups)
    missing_groups = filter(g -> !haskey(nested_groups, g), expected)
    # A term's own grouping (`cdar(...; by = g)`, `hsgp(...; by = g)`) may be
    # named too, so that its levels are units.
    terms = _brm_nlme_term_groupings(records)
    extra = filter(k -> k ∉ expected && k ∉ terms, collect(keys(nested_groups)))
    isempty(missing_groups) && isempty(extra) && return nothing
    joined(k) = [r for r in records if r.kind in (:ranef, :stratified_ranef) &&
        k in get(r, :joint_columns, ()) && r.group !== k]
    notes = map(extra) do k
        hits = joined(k)
        isempty(hits) ? "`$k`" : string("`$k` (a column of the joined `|",
            first(hits).id, "|` block over ", _brm_nlme_ticked(first(hits).joint_columns),
            ", which is one grouping, `", first(hits).group, "`)")
    end
    _brm_nlme_error(string(
        "brm_nlme_view: every grouping except the subject grouping `$group` must be ",
        "named once, in nested_groups (each level owned by one subject: ",
        "`<grouping> = :<subject column>` or `<grouping> = <subject labels>`, ",
        "row-aligned with the grouping) or in shared_groups (crossed ",
        "with the subjects, read as shared latents).",
        isempty(missing_groups) ? "" :
            " Named in neither: $(_brm_nlme_ticked(missing_groups)).",
        isempty(extra) ? "" :
            " nested_groups names $(join(notes, ", ")), which is no subject-owned " *
            "grouping of a random-effect block, `cdar` walk or grouped `hsgp` term.",
        " The subject-owned groupings are $(_brm_nlme_ticked(groups))",
        isempty(shared_groups) ? "." :
            "; the shared ones are $(_brm_nlme_ticked(shared_groups))."))
end

"""
    brm_nlme_view(backend::RKBRMI; subject_group=nothing,
                  nested_groups=NamedTuple(), shared_groups=(),
                  unit_partition=:subjects) -> BRMNLMEView

Read the nonlinear-mixed-effects partition of an RK-lowered model: population
coordinates, subject-level covariance coordinates, other scalars, and one
compact random-effect block per subject. For several grouping columns, supply
the top-level `subject_group` and name every other grouping once: in
`nested_groups`, a NamedTuple mapping each nested grouping to its parent
subject labels, row-aligned with the grouping (every nested level must belong
to exactly one subject), or in `shared_groups`, a tuple of groupings crossed
with the subjects. The parent labels are the name of a column the model reads
(`nested_groups = (g = :region,)`) or the labels themselves, a vector
(`nested_groups = (g = data.g_region,)`): the backend keeps only the columns
its formulas read, so a column no formula reads is passed as a vector. Every
coordinate of a random-effect block over a shared grouping (its standardized
draws, scales, correlation factor and Student-t mixing weights, for any block
form) is a `:shared_latent`, and so is a grouped `hsgp(x; by = g)` or a
`cdar(step; by = g)` walk with `g` in `shared_groups`.
`nested_groups` may also name the `by` grouping of a `cdar` or grouped `hsgp`
term that no random effect uses, so that its levels are units. A `car(...)`
field is shared, since its edges couple its areas, unless the backend was
lowered by the field's unit border separator with the subject labels
(`RKBRMI(brmi; car_units = :subject)`): then only its separator draws, scale
and mixing share are shared, and each subject's interior draws (and a BYM2
field's iid part of each of its areas) are that subject's unit draws. The
grouped entries of
`mi(x)::shares(total, rows)` are unit draws of the subject owning each
group's rows when the backend is non-centered
(`RKBRMI(brmi; completions = :noncentered)`); centred they are refused, since
their density reads the law and with it the units' draws. A declaration of zero
length, such as the coefficient block of a single-level treatment-coded factor,
owns no coordinate and is not a refusal. See [`BRMNLMEView`](@ref).

`unit_partition=:subjects` keeps the scientific subjects as inference units.
With `:connected`, structural likelihood-cell supports join subjects that a
cell reads together, transitively. Shared coordinates do not join units.
The model, priors and scientific subject labels are preserved; estimator
indices refer to whole connected units. Merged units use compact ragged
effects and retain every original covariance-family instance. This policy
does not expand the coordinate inventory or remove the structural refusals
below.

Refuses with [`BRMNLMEViewError`](@ref) when the model has no such reading:
no random effects over the subjects, a grouping named in neither
`nested_groups` nor `shared_groups` (the message names it), a subject-owned
grouping that couples its levels, a `car(...)` field lowered with `car_units`
that are not the subjects, a centred `mi(x)::shares(total, rows)`
completion or one with a group whose missing entries lie on rows of several
subjects, or any coordinate the semantic inventory does not cover.
"""
function brm_nlme_view(backend::RKBRMI; subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), unit_partition=:subjects)
    unit_partition in (:subjects, :connected) || throw(ArgumentError(
        "unit_partition must be :subjects or :connected"))
    names = Vector{Symbol}(_rk_layout_coordinate_names(backend))
    position = Dict{Symbol,Int}(n => i for (i, n) in enumerate(names))
    records = _brm_nlme_records(backend.plan)
    nested_groups isa NamedTuple || throw(ArgumentError(
        "nested_groups must be a NamedTuple mapping each nested grouping to its parent-label column"))
    shared_groups = _brm_nlme_shared_groups(shared_groups)
    _brm_nlme_check_shared_groups(records, shared_groups, subject_group, nested_groups)
    # Blocks over a grouping crossed with the subjects are shared latents; the
    # others are owned by the subjects.
    ranefs = [r for r in records if r.kind in (:ranef, :stratified_ranef) &&
        r.group ∉ shared_groups]
    isempty(ranefs) && _brm_nlme_error(
        "brm_nlme_view: the model has no subject-level random effects; an NLME " *
        "reading needs at least one `(... | group)` block" *
        (isempty(shared_groups) ? "" : " over a grouping not in `shared_groups`"))
    # NLMEEstimation can hold a family fixed (`FixedOmega`), but known-value
    # blocks are not handed to it yet (todo 2026-10-16T18-16-39-760-0840h34).
    for r in ranefs
        r.kind === :ranef || continue
        # Correlated levels (`gr(g; cor=...)`) couple the subjects, which an
        # NLME reading treats as independent.
        haskey(r, :level_correlation) && _brm_nlme_error(
            "brm_nlme_view: random-effect block `|$(r.id)|` correlates the levels " *
            "of `$(r.group)` with each other (`gr($(r.group); cor=$(r.level_correlation))`), " *
            "so its subjects are not independent and the model has no NLME " *
            "reading. Fit it with the Stan or RK sampler.")
        haskey(r, :known_scales) && _brm_nlme_error(
            "brm_nlme_view: random-effect block `|$(r.id)|` (group `$(r.group)`) " *
            "holds a known scale or correlation (`sd(...) = value` / " *
            "`cor(:, ID) = R`). BRM does not yet hand known-value blocks to " *
            "NLMEEstimation's fixed covariance families (BRM todo " *
            "2026-10-16T18-16-39-760-0840h34), so this model has no NLME reading. " *
            "Estimate the block, or fit it with the Stan or RK sampler.")
    end
    groups = unique(r.group for r in ranefs)
    if length(groups) == 1
        group = only(groups)
        subject_group === nothing || subject_group === group || _brm_nlme_error(
            "brm_nlme_view: subject_group `$subject_group` is not `$group`")
        _brm_nlme_check_nested_groups(records, group, groups, nested_groups, shared_groups)
    else
        subject_group isa Symbol && subject_group in groups || _brm_nlme_error(
            "brm_nlme_view: random effects use $(length(groups)) grouping columns " *
            "($(join(string.(groups), ", "))); supply subject_group, and name every " *
            "other grouping in nested_groups (owned by one subject) or shared_groups " *
            "(crossed with the subjects), to establish independent subject ownership")
        group = subject_group
        _brm_nlme_check_nested_groups(records, group, groups, nested_groups, shared_groups)
    end
    levels = Any[_rk_grouping_levels(backend.plan.columns[group])...]
    n = length(levels)
    owner = Dict{Symbol,Vector{Int}}(group => collect(1:n))
    group_levels = Dict{Symbol,Vector{Any}}(group => levels)
    subject_index = Dict(_brm_transport_level(level) => i for (i, level) in enumerate(levels))
    for (nested, labels) in pairs(nested_groups)
        children = get(backend.plan.columns, nested) do
            _brm_nlme_error("brm_nlme_view: nested group `$nested` is not a " *
                "grouping column of the model")
        end
        parents = _brm_nlme_parent_labels(backend.plan.columns, nested, labels)
        length(children) == length(parents) || _brm_nlme_error(
            "brm_nlme_view: `$nested` has $(length(children)) rows but its parent " *
            "labels$(_brm_nlme_parent_name(labels)) have $(length(parents))")
        nested_levels = Any[_rk_grouping_levels(children)...]
        parent_of = Dict{Any,Int}()
        for (child, parent) in zip(children, parents)
            c = _brm_transport_level(child)
            p = _brm_transport_level(parent)
            i = get(subject_index, p, 0)
            i == 0 && _brm_nlme_error(
                "brm_nlme_view: nested level $(repr(child)) names unknown subject $(repr(parent))")
            if haskey(parent_of, c)
                parent_of[c] == i || _brm_nlme_error(
                    "brm_nlme_view: nested level $(repr(child)) belongs to multiple subjects")
            else
                parent_of[c] = i
            end
        end
        group_levels[nested] = nested_levels
        owner[nested] = map(nested_levels) do level
            get(parent_of, _brm_transport_level(level)) do
                _brm_nlme_error("brm_nlme_view: nested level $(repr(level)) has no parent row")
            end
        end
    end
    # The subject owning each row of the grouping column.
    row_subject = Int[subject_index[_brm_transport_level(label)]
        for label in backend.plan.columns[group]]
    completion_subjects = _brm_nlme_row_subjects(backend.plan, row_subject).completions
    role = fill(:unclaimed, length(names))
    subject = zeros(Int, length(names))
    claim!(i, r) = (role[i] === :unclaimed || error(
        "brm_nlme_view: internal: coordinate `$(names[i])` is claimed twice " *
        "($(role[i]) and $r)"); role[i] = r; nothing)
    empty_declarations = _rk_layout_empty_declarations(backend)
    function claim_declaration!(declaration, r)
        hits = _brm_nlme_declaration_coordinates(names, declaration, empty_declarations)
        foreach(i -> claim!(i, r), hits)
        hits
    end
    coordinate(name) = get(position, name) do
        error("brm_nlme_view: internal: coordinate `$name` is not packed")
    end
    blocks = [Int[] for _ in 1:n]
    block_margins = NamedTuple[]
    subject_margins = [NamedTuple[] for _ in 1:n]
    subject_families = [Int[] for _ in 1:n]
    families = NamedTuple[]
    population = Dict{Tuple{Symbol,Symbol},Int}()
    for record in records
        kind = record.kind
        if kind in (:ranef, :stratified_ranef) && record.group in shared_groups
            # A block over a grouping crossed with the subjects: every
            # coordinate of its component (standardized draws, scales,
            # correlation factor, mixing weights) is shared by the subjects.
            claim_declaration!(_brm_nlme_ranef_declaration(record), :shared_latent)
        elseif kind === :ranef
            family = length(families) + 1
            block = _brm_nlme_block_identity(record)
            scales = reduce(vcat, (claim_declaration!(declaration, :subject_scale)
                for declaration in (isnothing(record.scales) ? (record.scale,) : record.scales)))
            factor = isnothing(record.L) ? Int[] :
                claim_declaration!(record.L, :subject_correlation)
            length(scales) == length(record.margins) || error(
                "brm_nlme_view: internal: block `|$(record.id)|` has $(length(scales)) " *
                "scale coordinates for $(length(record.margins)) margins")
            push!(families, (; block, record, stratum=nothing, size=length(scales),
                scales, factor, fixed=nothing))
            mixing = get(record, :mixing, nothing)
            for (k, (predictor, coefficient)) in enumerate(record.margins)
                push!(block_margins, (; block, predictor, coefficient))
                for g in eachindex(group_levels[record.group])
                    i = coordinate(Symbol(record.z, ".", g, ".", k))
                    claim!(i, :subject_effect)
                    subject[i] = owner[record.group][g]
                end
            end
            for g in eachindex(group_levels[record.group])
                i = owner[record.group][g]
                push!(subject_families[i], family)
                for (k, (predictor, coefficient)) in enumerate(record.margins)
                    push!(blocks[i], coordinate(Symbol(record.z, ".", g, ".", k)))
                    push!(subject_margins[i], (; block, predictor, coefficient,
                        level=group_levels[record.group][g]))
                end
            end
            if !isnothing(mixing)
                record.group === group || _brm_nlme_error(
                    "brm_nlme_view: nested Student-t random effects are not covered")
                push!(block_margins, (; block, predictor=:none, coefficient=:mixing))
                for g in 1:n
                    i = coordinate(Symbol(mixing, ".", g))
                    claim!(i, :subject_mixing)
                    subject[i] = g
                    push!(blocks[g], i)
                    push!(subject_margins[g], (; block, predictor=:none,
                        coefficient=:mixing, level=levels[g]))
                end
            end
        elseif kind === :stratified_ranef
            # One covariance family per stratum of `by`; each level's block
            # belongs to its stratum's family.
            K = length(record.margins)
            strata = Any[_rk_grouping_levels(backend.plan.columns[record.by])...]
            stratum_of = _brm_level_strata(backend.plan.columns[record.group],
                backend.plan.columns[record.by], group_levels[record.group], strata, record)
            first_family = length(families) + 1
            for (s, level) in enumerate(strata)
                block = (; _brm_nlme_block_identity(record)..., stratum=level)
                scales = Int[coordinate(isnothing(record.scales) ?
                        Symbol(record.scale, ".", s, ".", k) : Symbol(record.scales[k], ".", s))
                    for k in 1:K]
                foreach(i -> claim!(i, :subject_scale), scales)
                # RK packs every stratum's factor as one `K × K × levels(by)`
                # value, coordinates `L.<m>.<s>` (`m` in LKJ packing order).
                factor = isnothing(record.L) ? Int[] :
                    Int[coordinate(Symbol(record.L, ".", m, ".", s)) for m in 1:(K * (K - 1)) ÷ 2]
                foreach(i -> claim!(i, :subject_correlation), factor)
                push!(families, (; block, record, stratum=s, size=K, scales, factor,
                    fixed=nothing))
                for (predictor, coefficient) in record.margins
                    push!(block_margins, (; block, predictor, coefficient))
                end
            end
            for g in eachindex(group_levels[record.group])
                i = owner[record.group][g]
                s = stratum_of[g]
                push!(subject_families[i], first_family + s - 1)
                for (k, (predictor, coefficient)) in enumerate(record.margins)
                    c = coordinate(Symbol(record.z, ".", g, ".", k))
                    claim!(c, :subject_effect)
                    subject[c] = i
                    push!(blocks[i], c)
                    push!(subject_margins[i], (; block=families[first_family + s - 1].block,
                        predictor, coefficient, level=group_levels[record.group][g]))
                end
            end
        elseif kind === :population
            index = get(record, :index, ())
            if isempty(index)
                hits = claim_declaration!(record.declaration, :population)
                length(hits) == 1 &&
                    (population[(record.predictor, record.coefficient)] = only(hits))
            else
                i = coordinate(Symbol(record.declaration, ".", join(index, ".")))
                claim!(i, :population)
                population[(record.predictor, record.coefficient)] = i
            end
        elseif kind === :monotonic && get(record, :column, nothing) !== nothing
            # One predictor's column of a joint monotonic block: its simplex
            # column `<block>.simplex_incr.<j>.<t>` and its coefficient entry.
            t = string(record.column)
            prefix = string(record.declaration, ".simplex_incr.")
            for (i, name) in enumerate(names)
                parts = split(String(name), '.')
                startswith(String(name), prefix) && length(parts) == 4 &&
                    parts[end] == t && claim!(i, :population)
            end
            # The coefficient entry exists whether or not it pairs with Stan.
            beta = get(record, :beta_declaration, nothing)
            if beta !== nothing
                index = get(record, :beta_index, ())
                claim!(coordinate(isempty(index) ? beta :
                    Symbol(beta, ".", join(index, "."))), :population)
            end
        elseif kind in (:population_block, :monotonic, :threshold_coefficients, :vector)
            claim_declaration!(record.declaration, :population)
        elseif kind === :scalar
            claim_declaration!(record.declaration, :scalar)
        elseif kind in (:gmrf, :gmrf_footprint, :gmrf_multiresolution)
            # A model-scope field's states and hyperparameters: shared, like a
            # shared HSGP's. Per-cell fields of a `@plate` declare no such record.
            claim_declaration!(record.declaration, :shared_latent)
        elseif kind === :lkj_cholesky
            # A declared level correlation (`Omega ~ LKJCholesky(K, eta)`),
            # which only a grouped HSGP's `cor=` can read here (a random-effect
            # block's was refused above): shared, like that term's weights.
            claim_declaration!(record.declaration, :shared_latent)
        elseif kind === :hsgp
            units = _brm_nlme_hsgp_units(record, group, subject_index, owner,
                group_levels, nested_groups, shared_groups)
            if units === nothing
                # The whole HSGP component, including its scale, length and
                # basis weights, is shared across subjects and remains in the
                # original unconstrained layout.
                claim_declaration!(record.declaration, :shared_latent)
            else
                # Grouped by the units: each level's standardized weights are
                # unit draws of the subject owning the level, one family fixed
                # at the identity (the spectral scaling stays in the model).
                # A hyper-predictor's `(1 | g)` effects are one estimated
                # family each. The shared hyperparameters stay shared.
                for (hyper, effects) in pairs(record.locals.hypers)
                    effects === nothing && continue
                    hyper_record = _brm_nlme_hsgp_hyper_record(record, hyper, effects)
                    family = length(families) + 1
                    block = _brm_nlme_block_identity(hyper_record)
                    scale = coordinate(hyper_record.scale)
                    claim!(scale, :subject_scale)
                    push!(families, (; block, record=hyper_record, stratum=nothing,
                        size=1, scales=[scale], factor=Int[], fixed=nothing))
                    margin = only(hyper_record.margins)
                    push!(block_margins, (; block, predictor=first(margin),
                        coefficient=last(margin)))
                    for (g, i) in enumerate(units)
                        c = coordinate(Symbol(hyper_record.z, ".", g))
                        claim!(c, :subject_effect)
                        subject[c] = i
                        push!(blocks[i], c)
                        push!(subject_families[i], family)
                        push!(subject_margins[i], (; block, predictor=first(margin),
                            coefficient=last(margin), level=record.levels[g]))
                    end
                end
                weights = Symbol(record.declaration, ".", record.locals.weights)
                M, rest = divrem(count(name -> _brm_nlme_owned(name, weights), names),
                    length(units))
                M > 0 && rest == 0 || error("brm_nlme_view: internal: HSGP weights " *
                    "`$weights` do not hold one row per level of `$(record.group)`")
                family = length(families) + 1
                block = (; group=record.group, id=nothing, predictor=record.predictor,
                    hsgp=record.declaration)
                push!(families, (; block, record, stratum=nothing, size=M,
                    scales=Int[], factor=Int[], fixed=Matrix{Float64}(I, M, M)))
                coefficients = [Symbol("hsgp(", join(record.axes, ", "), ")[", j, "]")
                    for j in 1:M]
                for coefficient in coefficients
                    push!(block_margins, (; block, predictor=record.predictor, coefficient))
                end
                for (g, i) in enumerate(units)
                    push!(subject_families[i], family)
                    for (j, coefficient) in enumerate(coefficients)
                        c = coordinate(Symbol(weights, ".", g, ".", j))
                        claim!(c, :subject_effect)
                        subject[c] = i
                        push!(blocks[i], c)
                        push!(subject_margins[i], (; block, predictor=record.predictor,
                            coefficient, level=record.levels[g]))
                    end
                end
                # Hyperparameters and hyper-predictor intercepts.
                for (i, name) in enumerate(names)
                    role[i] === :unclaimed && _brm_nlme_owned(name, record.declaration) &&
                        claim!(i, :shared_latent)
                end
            end
        elseif kind === :cdar
            units = _brm_nlme_cdar_units(record, group, subject_index, owner,
                group_levels, nested_groups, shared_groups)
            if units === nothing
                # Shared by the subjects: the whole walk, scales and innovations.
                claim_declaration!(record.declaration, :shared_latent)
            else
                # Each group's standardized innovations are unit draws of the
                # subject owning the group, one family of size W fixed at the
                # identity (the walk's scale and persistence stay in the
                # model). The marginal SD and persistence stay shared.
                eta = Symbol(record.declaration, ".", record.locals.eta)
                W = length(record.steps)
                family = length(families) + 1
                block = (; group=record.group, id=nothing, predictor=record.predictor,
                    cdar=record.declaration)
                push!(families, (; block, record, stratum=nothing, size=W,
                    scales=Int[], factor=Int[], fixed=Matrix{Float64}(I, W, W)))
                coefficients = [Symbol("cdar(", record.step, ")[", step, "]")
                    for step in record.steps]
                for coefficient in coefficients
                    push!(block_margins, (; block, predictor=record.predictor, coefficient))
                end
                for (g, i) in enumerate(units)
                    push!(subject_families[i], family)
                    for (w, coefficient) in enumerate(coefficients)
                        c = coordinate(Symbol(eta, ".", g, ".", w))
                        claim!(c, :subject_effect)
                        subject[c] = i
                        push!(blocks[i], c)
                        push!(subject_margins[i], (; block, predictor=record.predictor,
                            coefficient, level=record.levels[g]))
                    end
                end
                for (i, name) in enumerate(names)
                    role[i] === :unclaimed && _brm_nlme_owned(name, record.declaration) &&
                        claim!(i, :shared_latent)
                end
            end
        elseif kind === :missing_value && get(record, :bijection, nothing) === :shares
            # The grouped entries of `mi(x)::shares(total, rows)`. Non-centered,
            # each coordinate is a standard-normal draw whose group observes
            # the rest of its density, so it is a unit draw of the subject
            # owning the group's rows, in a family fixed at 1. Centred, the
            # joint latent's density is a prior term reading the law, whose
            # arguments may read the units' draws: the population prior
            # evaluates it with every draw at zero, so no shared reading is
            # exact.
            record.noncentered || _brm_nlme_error(
                "brm_nlme_view: the grouped entries of `mi($(record.target))::shares(…)` " *
                "are centred, so their density, the law at the entries, is a prior " *
                "term; the NLME paths evaluate the population prior with every unit " *
                "draw at zero, while the law's arguments may read those draws. Lower " *
                "the model with `RKBRMI(brmi; completions = :noncentered)`, the same " *
                "model, whose grouped coordinates are unit draws.")
            owners = get(completion_subjects, record.target, nothing)
            owners === nothing && _brm_nlme_error(
                "brm_nlme_view: cannot trace the rows of `mi($(record.target))::shares(…)` " *
                "to subjects of `$group` through its law's predictors and gathers")
            groups_of = _brm_nlme_completion_groups(backend.plan, record.target, length(owners))
            family = length(families) + 1
            block = (; group=record.target, id=nothing, predictor=record.target,
                completion=true, shares=true)
            push!(families, (; block, record, stratum=nothing, size=1,
                scales=Int[], factor=Int[], fixed=fill(1.0, 1, 1)))
            push!(block_margins, (; block, predictor=record.target, coefficient=:shares))
            c = 0
            for (g, rows) in enumerate(groups_of)
                i = _brm_nlme_group_subject(owners, rows, record.target, g, group)
                for _ in 1:(length(rows) - 1)
                    c += 1
                    p = coordinate(Symbol(record.declaration, ".", c))
                    claim!(p, :subject_effect)
                    subject[p] = i
                    push!(blocks[i], p)
                    push!(subject_families[i], family)
                    push!(subject_margins[i], (; block, predictor=record.target,
                        coefficient=:shares, level=g))
                end
            end
        elseif kind === :missing_value
            # Only the completion's sampled carrier is a parameter. A
            # non-centered completion whose rows trace to the grouping axis is
            # one standard-normal draw per missing cell: a unit draw of the
            # subject owning the cell's row, in a family whose variance is
            # fixed at 1. Every other completion is shared by all subjects.
            owners = get(completion_subjects, record.target, nothing)
            missing_rows = record.noncentered && owners !== nothing ?
                _brm_nlme_completion_missing_rows(backend.plan, record.target, length(owners)) :
                nothing
            if missing_rows === nothing
                claim_declaration!(record.declaration, :shared_latent)
            else
                family = length(families) + 1
                block = (; group=record.target, id=nothing, predictor=record.target,
                    completion=true)
                push!(families, (; block, record, stratum=nothing, size=1,
                    scales=Int[], factor=Int[], fixed=fill(1.0, 1, 1)))
                push!(block_margins, (; block, predictor=record.target, coefficient=:missing))
                for (k, r) in enumerate(missing_rows)
                    c = coordinate(Symbol(record.declaration, ".", k))
                    claim!(c, :subject_effect)
                    i = owners[r]
                    subject[c] = i
                    push!(blocks[i], c)
                    push!(subject_families[i], family)
                    push!(subject_margins[i], (; block, predictor=record.target,
                        coefficient=:missing, level=r))
                end
            end
        elseif kind === :submodel
            # A provider may be deterministic and own no packed coordinates.
            # Its sampled declarations are scoped under its target.
            for (i, name) in enumerate(names)
                _brm_nlme_owned(name, record.target) && claim!(i, :shared_latent)
            end
        elseif kind === :car && get(record, :separation, nothing) === nothing
            # An areal field couples its areas through its edges, so the
            # whole field is shared by the subjects, whatever its area column.
            claim_declaration!(record.declaration, :shared_latent)
        elseif kind === :car
            # A field lowered by its unit border separator
            # (`RKBRMI(brmi; car_units)`): each unit's interior draws, and each
            # area's BYM2 iid part, are unit draws of the subject the unit is,
            # in size-1 families fixed at 1; the unit's terms (its edges and
            # its draws' cancelled standard normals) are its observations
            # (`_brm_nlme_car_term_units`). The shared coordinates (separator
            # values, interior means, component totals), the scale and the
            # mixing share are shared.
            separation = record.separation
            unit_subject(label) = get(subject_index, _brm_transport_level(label)) do
                _brm_nlme_error("brm_nlme_view: `car($(record.area))` was lowered with " *
                    "`car_units` whose unit $(repr(label)) is not a level of the subject " *
                    "grouping `$group`; lower it with the subject labels as `car_units`")
            end
            # Each unit's interior draws `<field>.<z_u>.<j>`, then each area's
            # BYM2 iid part `<field>.<theta>.<area>`.
            draws = [(Symbol(record.declaration, ".", z, ".", j), b.label)
                for (b, z) in zip(separation.blocks, record.locals.z) if z !== nothing
                for j in 1:b.ndraws]
            parts = Any[(draws, :draw)]
            record.type === :bym2 && push!(parts,
                ([(Symbol(record.declaration, ".", record.locals.theta, ".", k), label)
                    for (k, label) in enumerate(separation.node_units)], :iid))
            for (coordinates, part) in parts
                isempty(coordinates) && continue
                family = length(families) + 1
                block = (; group=record.area, id=nothing, predictor=record.predictor,
                    car=record.declaration, part)
                push!(families, (; block, record, stratum=nothing, size=1,
                    scales=Int[], factor=Int[], fixed=fill(1.0, 1, 1)))
                coefficient = Symbol("car(", record.area, ")[", part, "]")
                push!(block_margins, (; block, predictor=record.predictor, coefficient))
                for (k, (name, label)) in enumerate(coordinates)
                    label === nothing && continue
                    i = unit_subject(label)
                    c = coordinate(name)
                    claim!(c, :subject_effect)
                    subject[c] = i
                    push!(blocks[i], c)
                    push!(subject_families[i], family)
                    push!(subject_margins[i], (; block, predictor=record.predictor,
                        coefficient, level=k))
                end
            end
            for (i, name) in enumerate(names)
                role[i] === :unclaimed && _brm_nlme_owned(name, record.declaration) &&
                    claim!(i, :shared_latent)
            end
        else
            _brm_nlme_error("brm_nlme_view: coordinate record kind `$kind` has no " *
                "NLME role")
        end
    end
    unclaimed = names[role .=== :unclaimed]
    isempty(unclaimed) || _brm_nlme_error(
        "brm_nlme_view: $(length(unclaimed)) coordinate(s) of " *
        join(("`$root`" for root in unique(_rk_record_root.(unclaimed))), ", ") *
        " are not covered by the semantic inventory: " *
        "$(join(string.(unclaimed), ", ")). The view reads population " *
        "coefficients, scalars and vectors, random-effect blocks on a plain " *
        "`(... | group)` or stratified `gr(group, by=stratum)` grouping, HSGP " *
        "terms, `cdar` walks, `car(...)` fields, missing-value completions and " *
        "submodels; other terms, such as `mm(...)` blocks, file no coordinate record.")
    labels = map(_brm_transport_level, backend.plan.columns[group])
    # Build row ownership once, rather than scanning every row for each subject.
    row_index = Dict(_brm_transport_level(level) => i for (i, level) in enumerate(levels))
    rows = [Int[] for _ in levels]
    for (row, label) in enumerate(labels)
        push!(rows[row_index[label]], row)
    end
    # Subjects of a stratified block, or with completion draws, hold different
    # families, so their blocks differ in shape, as nested ones do; so do
    # subjects without a level of a grouped HSGP.
    ragged = !isempty(nested_groups) ||
        any(f -> f.stratum !== nothing || get(f.block, :completion, false), families) ||
        !allequal(subject_families)
    mu_references = ragged ? NamedTuple[] :
        _brm_nlme_mu_references(backend, block_margins, population,
            rows, length(labels))
    view = BRMNLMEView(group, levels, names, role, subject, blocks, block_margins,
        mu_references, rows, Int[f.size for f in families],
        subject_families, subject_margins, nested_groups, shared_groups, families,
        ragged, :subjects, copy(levels), collect(eachindex(levels)),
        [[i] for i in eachindex(levels)], [1:length(block) for block in blocks],
        copy(subject), Dict{Symbol,Vector{Union{Symbol,Vector{Int}}}}())
    unit_partition === :subjects ? view : _brm_nlme_connected_view(backend, view)
end

# ── Evaluation for NLME estimators ───────────────────────────────────────────
#
# Subjects are conditionally independent given (θ, Ω, σ). The lockstep path
# serves all subjects in one RK evaluation. Observation cells that each belong
# to one subject (plate cells, or rows of an elementwise observation) also
# permit cell queries that evaluate just one subject at a time.

# The packed unconstrained point at which every random-effect block has unit
# scales and an identity correlation factor, so that each block's standardized
# draws ARE the natural-scale random effects η_i. Verified on the constrained
# values rather than assumed from the transforms.
function _brm_nlme_unit_point(backend::RKBRMI, view::BRMNLMEView;
        shared_point=nothing)
    u = zeros(Float64, length(view.coordinates))
    shared = findall(==(:shared_latent), view.role)
    if !isempty(shared) && shared_point !== nothing
        length(shared_point) == length(u) || throw(DimensionMismatch(
            "shared_point has length $(length(shared_point)), expected $(length(u))"))
        u[shared] .= shared_point[shared]
        all(isfinite, Base.view(u, shared)) || _brm_nlme_error(
            "brm_nlme_model: shared_point contains a nonfinite shared latent coordinate")
    end
    values = _rk_constrained_values(backend, u)
    for record in unique(f.record for f in view.families if f.fixed === nothing)
        for declaration in (isnothing(record.scales) ? (record.scale,) : record.scales)
            scale = _brm_rk_declaration_value(values, declaration)
            all(isequal(1), scale) || _brm_nlme_error(
                "brm_nlme_model: random-effect scale `$declaration` is $(scale), not 1, " *
                "at the zero unconstrained point; its transform has no unit point here")
        end
        isnothing(record.L) && continue
        _brm_nlme_identity_factor(_brm_rk_declaration_value(values, record.L)) ||
            _brm_nlme_error("brm_nlme_model: correlation factor `$(record.L)` is " *
                "not the identity at the zero unconstrained point")
    end
    u
end

# A block's factor, or a stratified block's `K × K × levels(by)` factors.
_brm_nlme_identity_factor(L::AbstractMatrix) = L == I
_brm_nlme_identity_factor(L::AbstractArray{<:Any,3}) =
    all(s -> _brm_nlme_identity_factor(L[:, :, s]), axes(L, 3))

_brm_nlme_gaussian_plan(plan::_RKHeldOutPlan) =
    isempty(plan.held_out) && _brm_nlme_gaussian_plan(plan.parent)
_brm_nlme_gaussian_plan(plan::_RKStructuralPlan) =
    !isempty(plan.responses) && all(o -> o.family === :gaussian && o.weights === nothing &&
        o.interval === nothing && o.mi_jobs === nothing, plan.responses)
_brm_nlme_gaussian_plan(plan::_RKValuePlan) =
    !isempty(plan.observations) && all(_brm_nlme_gaussian_observation, plan.observations)
_brm_nlme_gaussian_plan(plan) = false
_brm_nlme_gaussian_observation(o) =
    o.distribution.callable in (Normal, StanBlocks.normal) &&
        o.modifier === nothing && o.weight === nothing && o.missing_response === nothing
# A `@scan` observes its series jointly (`AR1Normal`'s AR(1) residuals), not
# as independent Normal entries.
_brm_nlme_gaussian_observation(::_RKScanObservation) = false

function _brm_nlme_observed(backend::RKBRMI, attribution, nsubjects)
    observed = [Float64[] for _ in 1:nsubjects]
    scalar_attribution = Pair{Symbol,Vector{Int}}[]
    for (name, subjects) in attribution
        # Value plans have already gathered ragged response values into the
        # pointwise query's order in their bound `columns[name]`.
        values = backend.plan.columns[name]
        length(values) == length(subjects) || error(
            "brm_nlme_model: response `$name` has $(length(values)) values for " *
            "$(length(subjects)) attributed pointwise entries")
        expanded = Int[]
        for (value, i) in zip(values, subjects)
            items = value isa AbstractArray ? vec(value) : (value,)
            for y in items
                y isa Real && isfinite(y) || _brm_nlme_error(
                    "brm_nlme_model: Gaussian response `$name` contains a non-finite " *
                    "or non-numeric observation $(repr(y))")
                push!(observed[i], Float64(y))
                push!(expanded, i)
            end
        end
        push!(scalar_attribution, name => expanded)
    end
    observed, scalar_attribution
end

function _brm_nlme_observation_argument(value, sizes, name, field)
    n = sum(sizes)
    values = if value isa Number
        fill(Float64(value), n)
    elseif value isa AbstractVector && length(value) == length(sizes)
        out = Float64[]
        sizehint!(out, n)
        for (part, size) in zip(value, sizes)
            if part isa Number
                append!(out, Iterators.repeated(Float64(part), size))
            elseif part isa AbstractArray
                length(part) == size || throw(DimensionMismatch(
                    "brm_nlme_model: `$name.$field` has $(length(part)) entries " *
                    "for a plate cell with $size observations"))
                append!(out, Float64.(vec(part)))
            else
                _brm_nlme_error(
                    "brm_nlme_model: `$name.$field` is neither a scalar nor an array " *
                    "in a plate cell")
            end
        end
        out
    else
        Float64.(brm_flatten_cells(value))
    end
    length(values) == n || throw(DimensionMismatch(
        "brm_nlme_model: `$name.$field` has $(length(values)) entries for $n observations"))
    values
end

function _brm_nlme_moments(m, arguments, tangents=())
    nsubjects = length(m.view.levels)
    means = [Float64[] for _ in 1:nsubjects]
    variances = [Float64[] for _ in 1:nsubjects]
    mean_rows = [Vector{Float64}[] for _ in 1:nsubjects]
    variance_rows = [Vector{Float64}[] for _ in 1:nsubjects]
    for (name, subjects) in m.observation_attribution
        n = length(subjects)
        sizes = [value isa AbstractArray ? length(value) : 1
            for value in m.backend.plan.columns[name]]
        sum(sizes) == n || error(
            "brm_nlme_model: response `$name` has $(sum(sizes)) observed values " *
            "for $n attributed pointwise entries")
        field = getproperty(arguments, name)
        locations = _brm_nlme_observation_argument(field.location, sizes, name, :location)
        scales = _brm_nlme_observation_argument(field.scale, sizes, name, :scale)
        all(s -> isfinite(s) && s > 0, scales) || _brm_nlme_error(
            "brm_nlme_model: Gaussian response `$name` has a non-finite or " *
            "non-positive scale")
        dlocations = [_brm_nlme_observation_argument(getproperty(t, name).location,
            sizes, name, :location) for t in tangents]
        dscales = [_brm_nlme_observation_argument(getproperty(t, name).scale,
            sizes, name, :scale) for t in tangents]
        for (k, i) in enumerate(subjects)
            push!(means[i], locations[k])
            push!(variances[i], abs2(scales[k]))
            if !isempty(tangents)
                push!(mean_rows[i], [d[k] for d in dlocations])
                push!(variance_rows[i], [2 * scales[k] * d[k] for d in dscales])
            end
        end
    end
    for i in 1:nsubjects
        length(means[i]) == length(m.observed[i]) || error(
            "brm_nlme_model: subject $i has $(length(means[i])) observation moments " *
            "for $(length(m.observed[i])) observed values")
    end
    matrix(rows, nη) = isempty(rows) ? zeros(Float64, 0, nη) :
        reduce(vcat, (permutedims(row[1:nη]) for row in rows))
    isempty(tangents) ? collect(zip(means, variances)) :
        [(means[i], variances[i],
            matrix(mean_rows[i], length(m.view.subject_coordinates[i])),
            matrix(variance_rows[i], length(m.view.subject_coordinates[i])))
            for i in 1:nsubjects]
end

# Trace the subject labels carried by model values through pointwise
# assignments and data-indexed gathers. A completion can have fewer rows, more
# rows, or a permutation of the subject column's rows. Cache by authored name
# so shared expression paths are visited once; never allocate row-by-row
# dependency matrices or probe one coordinate at a time.
_brm_nlme_row_subjects(plan::_RKHeldOutPlan, subjects) =
    _brm_nlme_row_subjects(plan.parent, subjects)
_brm_nlme_row_subjects(_plan, _subjects) =
    (; completions=Dict{Symbol,Any}(), observations=Dict{Symbol,Any}())
function _brm_nlme_row_subjects(plan::_RKValuePlan, subjects)
    assignments = Dict(a.name => a.expression for a in plan.assignments
        if a isa _BRMPreparedAssignment)
    predictors = Dict(p.name => p for p in plan.regression.predictors)
    completions = Dict(c.source => c for c in plan.completions)
    cache = Dict{Symbol,Any}()
    active = Set{Symbol}()
    function combine(parts, target)
        known = filter(!isnothing, parts)
        isempty(known) && return nothing
        broadcast(known...) do owners...
            all(==(first(owners)), owners) || _brm_nlme_error(
                "brm_nlme_view: the law of `$target` combines rows of different " *
                "subjects; its rows cannot belong to one subject each")
            first(owners)
        end
    end
    function indices(value)
        value isa _BRMPreparedRef && return get(plan.columns, value.name, nothing)
        value isa _BRMPreparedExpr || return value
        args = map(indices, value.args)
        kwargs = map(indices, value.kwargs)
        any(isnothing, args) || any(isnothing, values(kwargs)) ? nothing :
            value.callable(args...; kwargs...)
    end
    function rows(value, target)
        if value isa _BRMPreparedRef
            return named(value.name, target)
        elseif value isa _BRMPreparedExpr
            if value.callable === getindex && length(value.args) == 2
                source = rows(first(value.args), target)
                source === nothing && return nothing
                index = indices(last(value.args))
                index === nothing && return nothing
                return source[index]
            elseif haskey(_RK_DERIVED_BINOPS, value.callable) ||
                    haskey(_RK_DERIVED_MATH, value.callable) ||
                    haskey(_RK_DERIVED_CMP, value.callable) ||
                    value.callable === _brm_effective_lower
                return combine(map(arg -> rows(arg, target), value.args), target)
            end
        end
        nothing
    end
    function named(name, target)
        haskey(cache, name) && return cache[name]
        name in active && return nothing
        push!(active, name)
        result = if haskey(assignments, name)
            rows(assignments[name], target)
        elseif haskey(predictors, name)
            p = predictors[name]
            length(get(plan.columns, p.row_source, ())) == length(subjects) ? subjects : nothing
        elseif haskey(completions, name)
            c = completions[name]
            owned = combine(map(arg -> rows(arg, name), c.distribution.args), name)
            n = length(plan.columns[c.observed_rows]) + length(plan.columns[c.missing_rows])
            if owned === nothing
                n == length(subjects) ? subjects : nothing
            elseif owned isa Integer
                fill(owned, n)
            else
                length(owned) == n || _brm_nlme_error(
                    "brm_nlme_view: the law of `mi($name)` traces $(length(owned)) " *
                    "subject rows for $n completion rows")
                collect(Int, owned)
            end
        else
            nothing
        end
        delete!(active, name)
        cache[name] = result
    end
    completed = Dict(c.source => named(c.source, c.source) for c in plan.completions)
    # Consumers of a gathered completion retain that order too. In
    # particular, equal lengths do not imply equal axes: assigning a
    # permuted response by position would silently change its unit density.
    observed = Dict{Symbol,Any}()
    for o in plan.observations
        o isa _BRMPreparedObservation && o.modifier === nothing || continue
        d = o.distribution
        d isa _BRMPreparedExpr || continue
        owned = combine(map(arg -> rows(arg, o.name), (d.args..., o.weight)), o.name)
        owned === nothing || (observed[o.name] = owned)
    end
    (; completions=completed, observations=observed)
end

# The grouping-axis rows of a completion's missing cells, in the order of its
# sampled coordinates, when its observed and missing rows cover exactly the
# `nrows` rows of the grouping column; `nothing` otherwise.
_brm_nlme_completion_missing_rows(plan::_RKHeldOutPlan, target, nrows) =
    _brm_nlme_completion_missing_rows(plan.parent, target, nrows)
_brm_nlme_completion_missing_rows(_plan, _target, _nrows) = nothing
function _brm_nlme_completion_missing_rows(plan::_RKValuePlan, target, nrows)
    c = _brm_nlme_axis_completion(plan, target, nrows)
    c === nothing && return nothing
    # With group totals, the free entries' rows.
    collect(Int, plan.columns[c.totals === nothing ? c.missing_rows : c.totals.free_rows])
end

# The completion of `target` when its rows are the grouping column's, else
# `nothing`.
function _brm_nlme_axis_completion(plan::_RKValuePlan, target, nrows)
    matches = [c for c in plan.completions if c.source === target]
    length(matches) == 1 || return nothing
    c = only(matches)
    observed = get(plan.columns, c.observed_rows, nothing)
    missing = get(plan.columns, c.missing_rows, nothing)
    (observed === nothing || missing === nothing) && return nothing
    sort(vcat(observed, missing)) == collect(1:nrows) || return nothing
    c
end

# The rows of each group's missing entries of `mi(target)::shares(total, rows)`,
# in the completion's group order, or `nothing` when its rows are not the
# grouping column's.
_brm_nlme_completion_groups(plan::_RKHeldOutPlan, target, nrows) =
    _brm_nlme_completion_groups(plan.parent, target, nrows)
_brm_nlme_completion_groups(_plan, _target, _nrows) = nothing
function _brm_nlme_completion_groups(plan::_RKValuePlan, target, nrows)
    c = _brm_nlme_axis_completion(plan, target, nrows)
    (c === nothing || c.totals === nothing) && return nothing
    grouped = plan.columns[c.totals.grouped_rows]
    ends = plan.columns[c.totals.ends]
    [collect(Int, grouped[(k == 1 ? 1 : ends[k - 1] + 1):stop]) for (k, stop) in enumerate(ends)]
end

# The groups of the non-centered completion whose group terms the observation
# `name` holds (`_rk_mi_grouped_arguments!`), or `nothing`.
_brm_nlme_completion_weights(plan::_RKHeldOutPlan, name, subjects) =
    _brm_nlme_completion_weights(plan.parent, name, subjects)
_brm_nlme_completion_weights(_plan, _name, _subjects) = nothing
function _brm_nlme_completion_weights(plan::_RKValuePlan, name, subjects)
    matches = [c for c in plan.completions if c.noncentered && c.totals !== nothing &&
        c.totals.weights === name]
    isempty(matches) && return nothing
    c = only(matches)
    owners = get(subjects, c.source, nothing)
    owners === nothing && return nothing
    groups = _brm_nlme_completion_groups(plan, c.source, length(owners))
    (; target=c.source, groups, owners)
end

# The unit label of each of a separated `car` field's unit terms, when `name`
# is the placeholder they observe (`_rk_car_separated_options`), else `nothing`.
function _brm_nlme_car_term_units(plan, name)
    for predictor in _rk_replay_predictors(plan), term in predictor.terms
        term.kind === :car || continue
        separation = get(term.options, :separation, nothing)
        separation === nothing || separation.terms !== name || return separation.term_units
    end
    nothing
end

# The subject owning every row of a completion's group `g`. The group's shares
# couple its entries, so a group spanning subjects has no NLME reading.
function _brm_nlme_group_subject(row_subject, rows, target, g, group)
    owners = unique(row_subject[rows])
    length(owners) == 1 || _brm_nlme_error(
        "brm_nlme_view: group $g of `mi($target)::shares(…)` has missing entries on " *
        "rows of $(length(owners)) subjects of `$group`; its shares couple them, so " *
        "its subjects are not independent and the model has no NLME reading")
    only(owners)
end

_brm_nlme_completion_rows(_plan, _name, _subjects) = nothing
_brm_nlme_completion_rows(plan::_RKHeldOutPlan, name, subjects) =
    _brm_nlme_completion_rows(plan.parent, name, subjects)
function _brm_nlme_completion_rows(plan::_RKValuePlan, name, subjects)
    matches = [c for c in plan.completions if c.observed === name ||
        _rk_mi_free_weight_name(c) === name]
    length(matches) == 1 || return nothing
    completion = only(matches)
    rows = completion.observed === name ? completion.observed_rows :
        completion.totals === nothing ? completion.missing_rows : completion.totals.free_rows
    observed = get(plan.columns, rows, nothing)
    observed === nothing && return nothing
    owners = get(subjects, completion.source, nothing)
    owners === nothing ? nothing : owners[observed]
end

# The grouping-axis rows each cell of a plate observation reads, through the
# ragged joins of the plate that observes `name` (`groups(g => cells)`,
# `ragged(x, g => cells)`, and each group's rows in a `groups(g)` plate,
# including the series a `@scan` observes in place): per cell, the union over
# the joins whose group column is the subject grouping `group` or another
# column on its `nrows`-row axis. `nothing` when no such join exists.
_brm_nlme_cell_rows(_plan, _name, _group, _nrows) = nothing
function _brm_nlme_cell_rows(plan::_RKValuePlan, name, group, nrows)
    for a in plan.assignments
        a isa _RKPreparedKernelAssignment || continue
        scanned = [o for o in a.scan_observed if o.source === name]
        any(o -> o.source === name, a.observations) || !isempty(scanned) || continue
        on_axis(g) = g === group ||
            length(get(a.columns, g, get(plan.columns, g, ()))) == nrows
        joins = Any[a.columns[input.rows] for input in a.inputs
            if input.kind === :gather && on_axis(input.group)]
        append!(joins, (o.rows for o in scanned
            if get(o, :group, nothing) !== nothing && on_axis(o.group)))
        isempty(joins) && return nothing
        ncells = length(first(joins))
        all(rows -> length(rows) == ncells, joins) || return nothing
        return [sort!(unique!(reduce(vcat, (rows[k] for rows in joins))))
            for k in 1:ncells]
    end
    nothing
end

# Each pointwise density's subject, and the observations whose subjects were
# measured (`_brm_nlme_measured_subjects`) rather than read from their rows.
function _brm_nlme_attribution(backend::RKBRMI, view::BRMNLMEView, query, unit)
    view.unit_partition === :connected && return _brm_nlme_connected_attribution(
        view, keys(_rk_pointwise_values(query, unit)))
    fields = _rk_pointwise_values(query, unit)
    L = sum(length, view.rows)
    row_subject = zeros(Int, L)
    for (i, rows) in enumerate(view.rows), r in rows
        row_subject[r] = i
    end
    any(iszero, row_subject) && error(
        "brm_nlme_model: internal: a row of `$(view.group)` belongs to no subject")
    traced = _brm_nlme_row_subjects(backend.plan, row_subject)
    completion_subjects = traced.completions
    attribution = Pair{Symbol,Vector{Int}}[]
    measured = Symbol[]
    probe = nothing
    for (name, values) in pairs(fields)
        m = length(values)
        # A `car_units` field's unit terms (its edges that read a unit's draws,
        # and their cancelled standard normals): each belongs to its unit.
        units = _brm_nlme_car_term_units(backend.plan, name)
        if units !== nothing
            length(units) == m || error("brm_nlme_model: internal: `$name` has $m " *
                "densities for $(length(units)) unit terms")
            push!(attribution, name => map(units) do label
                i = findfirst(l -> isequal(_brm_transport_level(l),
                    _brm_transport_level(label)), view.levels)
                i === nothing && _brm_nlme_error("brm_nlme_model: a `car_units` unit " *
                    "$(repr(label)) is not a level of the subject grouping `$(view.group)`")
                i
            end)
            continue
        end
        # A non-centered `mi(x)::shares(total, rows)` completion's group terms:
        # each belongs to the subject owning its group's rows.
        weights = _brm_nlme_completion_weights(backend.plan, name, completion_subjects)
        if weights !== nothing
            length(weights.groups) == m || error(
                "brm_nlme_model: internal: `$name` has $m group terms for " *
                "$(length(weights.groups)) groups of `mi($(weights.target))`")
            push!(attribution, name => [_brm_nlme_group_subject(weights.owners, rows,
                weights.target, g, view.group) for (g, rows) in enumerate(weights.groups)])
            continue
        end
        completion_rows = _brm_nlme_completion_rows(backend.plan, name, completion_subjects)
        if completion_rows !== nothing
            length(completion_rows) == m || _brm_nlme_error(
                "brm_nlme_model: completed-covariate observation `$name` has $m " *
                "densities for $(length(completion_rows)) observed rows")
            push!(attribution, name => completion_rows)
            continue
        end
        # One density per cell of a ragged join over the grouping axis (a
        # district total over its rows): the subject owning every one of the
        # cell's rows.
        cell_rows = _brm_nlme_cell_rows(backend.plan, name, view.group, L)
        if cell_rows !== nothing && length(cell_rows) == m
            push!(attribution, name => map(enumerate(cell_rows)) do (k, rows)
                owners = unique(row_subject[rows])
                length(owners) == 1 || _brm_nlme_error(
                    "brm_nlme_model: cell $k of observation `$name` reads rows of " *
                    "$(length(owners)) subjects of `$(view.group)`, so its density " *
                    "belongs to no single subject")
                only(owners)
            end)
            continue
        end
        owners = get(traced.observations, name, nothing)
        if owners !== nothing
            owners isa Integer && (owners = fill(owners, m))
            length(owners) == m || _brm_nlme_error(
                "brm_nlme_model: observation `$name` traces $(length(owners)) " *
                "subject rows for $m densities")
            push!(attribution, name => owners)
            continue
        end
        # One density element per row of the grouping column: flat rows on the
        # grouping axis, or one in-cell array per plate cell.
        if m == L
            push!(attribution, name => row_subject)
            continue
        end
        # Densities on rows of their own (a fragment's rows, read through an
        # index or joined by its own plate): measured.
        probe === nothing && (probe = _brm_nlme_dependence_probe(query, unit, view))
        push!(attribution, name => _brm_nlme_measured_subjects(probe, view, name))
        push!(measured, name)
    end
    attribution, measured
end

# Which subjects' η each pointwise density reads, measured on the prepared
# `:pointwise` query. Bit b of a subject's index (from 0) splits the subjects
# in two halves; moving the η blocks of one half, then of the other, a density
# that reads one subject's η changes in exactly the evaluation that moved that
# subject, so its changes over the bits spell the subject's index: 2⌈log₂ n⌉
# evaluations for every density at once. The probe's base point moves every
# coordinate of the unit point too, by fixed random amounts, so that no
# coefficient sits at zero there and hides a dependence (the unit point's
# unconstrained coordinates are all zero); the η moves are fixed random
# amounts as well, so that two η a density reads do not cancel.
function _brm_nlme_dependence_probe(query, unit, view::BRMNLMEView)
    rng = Random.Xoshiro(20261029)
    amounts(lo) = (lo .+ lo .* rand(rng, length(unit))) .* rand(rng, (-1, 1), length(unit))
    base = unit .+ amounts(0.1)
    shift = amounts(0.2)
    n = length(view.levels)
    moved(subjects) = _brm_nlme_moved_values(query, base, shift, view, subjects)
    bits = n == 1 ? 0 : ndigits(n - 1; base=2)
    halves = [(moved(filter(i -> isodd((i - 1) >> b), 1:n)),
        moved(filter(i -> iseven((i - 1) >> b), 1:n))) for b in 0:(bits - 1)]
    (; query, base, shift, values=_rk_pointwise_values(query, base), halves)
end

function _brm_nlme_moved_values(query, base, shift, view::BRMNLMEView, subjects)
    u = copy(base)
    for i in subjects, c in view.subject_coordinates[i]
        u[c] += shift[c]
    end
    _rk_pointwise_values(query, u)
end

function _brm_nlme_measured_subjects(probe, view::BRMNLMEView, name)
    base = getproperty(probe.values, name)
    halves = [(getproperty(odd, name), getproperty(even, name)) for (odd, even) in probe.halves]
    Int[_brm_nlme_measured_subject(probe, view, name, base, halves, k)
        for k in eachindex(base)]
end

function _brm_nlme_measured_subject(probe, view::BRMNLMEView, name, base, halves, k)
    all(isfinite, base[k]) || _brm_nlme_error(
        "brm_nlme_model: density $k of observation `$name` is $(base[k]) at the " *
        "point that measures which subject of `$(view.group)` it belongs to " *
        "(its rows are not those of `$(view.group)`)")
    subject, still = 1, 0
    for (b, (odd, even)) in enumerate(halves)
        in_odd, in_even = !isequal(odd[k], base[k]), !isequal(even[k], base[k])
        if in_odd && in_even
            read = _brm_nlme_subjects_read(probe, view, name, k,
                collect(eachindex(view.levels)))
            _brm_nlme_error("brm_nlme_model: density $k of observation `$name` " *
                "reads the random effects of $(length(read)) subjects of " *
                "`$(view.group)` (" * join(repr.(view.levels[read]), ", ") *
                "), so the subjects are not independent and the model has no " *
                "NLME reading")
        elseif in_odd
            subject += 1 << (b - 1)
        elseif !in_even
            still += 1
        end
    end
    # A density that reads no subject's η adds the same term to the summed
    # objective wherever it counts: it counts toward the first subject.
    still == length(halves) && return 1
    still == 0 && subject <= length(view.levels) || _brm_nlme_error(
        "brm_nlme_model: density $k of observation `$name` changed when the " *
        "random effects of some halves of the subjects of `$(view.group)` moved " *
        "but not others, so its subject cannot be measured")
    subject
end

# The subjects whose η density `k` of `name` reads, among `candidates`, by
# moving halves of them and following those that move it (a refusal's message).
function _brm_nlme_subjects_read(probe, view::BRMNLMEView, name, k, candidates)
    length(candidates) == 1 && return candidates
    base = getproperty(probe.values, name)[k]
    half = length(candidates) ÷ 2
    reduce(vcat, (_brm_nlme_subjects_read(probe, view, name, k, part)
        for part in (candidates[1:half], candidates[(half + 1):end])
        if !isequal(getproperty(_brm_nlme_moved_values(probe.query, probe.base,
            probe.shift, view, part), name)[k], base)); init=Int[])
end

# Implemented by `BayesianRegressionModelsReactiveKernelsExt`: the prepared
# `:pointwise` query of a backend and its evaluation at a packed point, and the
# sampler query that retains the pointwise densities of its reverse sweep, with
# `(value, gradient, pointwise)` from one evaluation.
function _rk_pointwise_query end
function _rk_pointwise_values end
function _rk_retained_pointwise_sampler end
function _rk_value_gradient_and_pointwise! end
function _rk_eta_hvp_query end
function _rk_eta_hvp end
function _rk_observation_query end
function _rk_observation_pushforward end
function _rk_observation_values end
function _rk_observation_value_and_pushforward end
function _rk_nlme_cell_query end
function _rk_nlme_cell_value end
function _rk_nlme_partitioned end
function _rk_nlme_partition_context end
function _rk_nlme_partition_query end
function _rk_nlme_is_partition_separability_error end
function _rk_nlme_partition_labels end
function _rk_nlme_partition_value end
function _rk_nlme_partition_query_footprint end
function _rk_nlme_partition_sampler end
function _rk_nlme_partition_footprint end
function _rk_nlme_partition_value_and_gradient! end
function _rk_nlme_partition_local_footprint end
function _rk_nlme_partition_value_and_local_gradient! end
function _rk_nlme_cell_footprint_query end
function _rk_nlme_cell_footprint end
function _rk_nlme_raw_cell_footprint end
function _rk_nlme_population_prior end
function _rk_nlme_population_prior_value end
function _rk_nlme_population_prior_value_gradient end
# Implemented by `BayesianRegressionModelsReactiveKernelsEnzymeExt`: an Enzyme
# `mode` carrying the runtime-activity setting of the caller's `ad_backend`.
function _brm_enzyme_mode end
_brm_nlme_forward_backend(ad_backend) = nothing

# A cell query still addresses the original packed coordinate space. Overlay
# only the shared coordinates and the selected subject's η onto the read-only
# unit point; constructing a packed Vector would make each call scale with the
# number of other subjects. A read is a binary search: the θ and σ positions
# are sorted (the model's `findall`), and the subject's η positions, in block
# order, are kept sorted beside it (`eta_sorted`, their places in `eta` in
# `eta_order`), so a cell's reads cost a logarithm of the block each instead of
# a scan of it.
struct _BRMNLMEPoint{T,S,E} <: AbstractVector{Float64}
    unit::Vector{Float64}
    theta_positions::Vector{Int}
    theta::T
    sigma_positions::Vector{Int}
    sigma::S
    eta_positions::Vector{Int}
    eta::E
    eta_sorted::Vector{Int}
    eta_order::Vector{Int}
end
function _BRMNLMEPoint(unit, theta_positions, theta, sigma_positions, sigma,
        eta_positions, eta)
    order = sortperm(eta_positions)
    _BRMNLMEPoint(unit, theta_positions, theta, sigma_positions, sigma,
        eta_positions, eta, eta_positions[order], order)
end
Base.size(p::_BRMNLMEPoint) = size(p.unit)
Base.IndexStyle(::Type{<:_BRMNLMEPoint}) = IndexLinear()
# Where packed position `j` sits in the sorted `positions`, or 0.
function _brm_sorted_slot(positions, j)
    k = searchsortedfirst(positions, j)
    k <= length(positions) && positions[k] == j ? k : 0
end
function Base.getindex(p::_BRMNLMEPoint, j::Int)
    k = _brm_sorted_slot(p.theta_positions, j)
    k == 0 || return p.theta[k]
    k = _brm_sorted_slot(p.sigma_positions, j)
    k == 0 || return p.sigma[k]
    k = _brm_sorted_slot(p.eta_sorted, j)
    k == 0 || return p.eta[p.eta_order[k]]
    p.unit[j]
end

# A subject's conditional density is a sum of RK query cells, in groups that
# share cell positions. The subject plate group holds the observations with one
# cell per subject row: in-cell observations and formula observations of kernel
# output observed per subject (`_RKNestedObservation`). Every other observation
# is observed elementwise with one density per row it was observed on: a flat
# grouped response (one per row of the grouping column) or a completed
# covariate's observed values. Each density is then a cell of its own, owned by
# the subject of its row, and the observation forms one group. So does an
# observation on rows of its own (`measured`), whose cells, plate iterations
# included, belong to the subjects the attribution measured. Returns
# `(; names, positions, elementwise)` per group, `positions[i]` being subject
# `i`'s cells, or `nothing`.
function _brm_nlme_subject_cells(backend::RKBRMI, view::BRMNLMEView, attribution,
        measured)
    plan = backend.plan
    isempty(attribution) && return nothing
    _brm_nlme_cell_plan(plan) || return nothing
    if view.unit_partition === :connected
        # Every stored support is one complete cell, including a vector-valued
        # observation. Build buckets in one pass; never split such a cell.
        return Tuple(map(attribution) do (name, owners)
            positions = [Int[] for _ in view.levels]
            for (cell, owner) in enumerate(owners)
                push!(positions[owner], cell)
            end
            (; names=(name,), positions, elementwise=true)
        end)
    end
    plate_observations = setdiff(_brm_nlme_plate_observations(plan), measured)
    n = length(view.levels)
    L = sum(length, view.rows)
    plate = [entry for entry in attribution if first(entry) in plate_observations]
    groups = Any[]
    if !isempty(plate)
        positions = if all(rows -> length(rows) == 1, view.rows)
            # One grouping row per subject: the plate's cells are the subjects.
            rows = [only(group_rows) for group_rows in view.rows]
            all(subjects -> length(subjects) == n &&
                all(i -> subjects[rows[i]] == i, 1:n), last.(plate)) || return nothing
            [[row] for row in rows]
        else
            # Several rows per subject (long-format data): every plate
            # observation has one density per cell of a plate over the
            # subjects' rows, such as `groups(g; order = t)`, and attributes
            # its cells alike.
            cells = last(first(plate))
            all(plate) do (name, subjects)
                cell_rows = _brm_nlme_cell_rows(plan, name, view.group, L)
                subjects == cells && cell_rows !== nothing &&
                    length(cell_rows) == length(cells)
            end || return nothing
            [findall(==(i), cells) for i in 1:n]
        end
        push!(groups, (; names=Tuple(first.(plate)), positions, elementwise=false))
    end
    for (name, subjects) in attribution
        name in plate_observations && continue
        push!(groups, (; names=(name,),
            positions=[findall(==(i), subjects) for i in 1:n], elementwise=true))
    end
    Tuple(groups)
end

# Plans whose observations RK evaluates cell by cell.
_brm_nlme_cell_plan(::Union{_RKStructuralPlan,_RKValuePlan}) = true
_brm_nlme_cell_plan(_plan) = false

# The observations a plate observes, one cell per plate iteration: those of
# authored `@plate for` cells (with the series their `@scan`s observe) and
# formula observations of kernel output observed per subject.
_brm_nlme_plate_observations(_plan) = Set{Symbol}()
function _brm_nlme_plate_observations(plan::_RKValuePlan)
    names = Set(o.source for a in plan.assignments
        if a isa _RKPreparedKernelAssignment for o in (a.observations..., a.scan_observed...))
    union!(names, (o.name for o in plan.observations if o isa _RKNestedObservation))
end

# One prepared cell query per group of `_brm_nlme_subject_cells`, counted under
# `progress`: each one prepares a program of its own (on ArbeitsplatzAtlas's
# model, 32 groups at 48–127 s each).
_brm_nlme_cell_queries(backend::RKBRMI, cells, partitioned, unit, progress=nothing) =
    _brm_with_counter(progress, length(cells); description="Cell queries") do node
        map(ntuple(identity, length(cells)), cells) do g, cell
            query = _rk_nlme_cell_query(backend, cell.names, partitioned, unit)
            _brm_update_progress!(node, g)
            (; cell.names, cell.positions, cell.elementwise, query)
        end
    end

# A point that notes each packed position read through it: evaluating a cell's
# value at it records the coordinates that cell reads, for the cost of that one
# evaluation. A query prepared with `partitioned` also reads every shared
# coordinate on each call to compare it with its cache, and those reads are
# recorded too.
struct _BRMNLMERecordingPoint{P<:AbstractVector{Float64}} <: AbstractVector{Float64}
    point::P
    seen::Vector{Int}
end
Base.size(p::_BRMNLMERecordingPoint) = size(p.point)
Base.IndexStyle(::Type{<:_BRMNLMERecordingPoint}) = IndexLinear()
Base.getindex(p::_BRMNLMERecordingPoint, j::Int) = (push!(p.seen, j); p.point[j])


# Each observation's cell positions map to exactly one NLME unit. These data
# columns drive the compiled partition; they are not sampled coordinates.
function _brm_nlme_partition_keys(cells)
    map(cells) do cell
        labels = zeros(Int, sum(length, cell.positions))
        for (i, positions) in enumerate(cell.positions), p in positions
            1 <= p <= length(labels) && iszero(labels[p]) || _brm_nlme_error(
                "brm_nlme_model: observation cells $(cell.names) do not form a " *
                "partition at cell $p")
            labels[p] = i
        end
        all(!iszero, labels) || _brm_nlme_error(
            "brm_nlme_model: observation cells $(cell.names) leave cells unattributed")
        labels
    end
end

# BRM owns the packed-coordinate attribution, including nested/ragged blocks.
# Structural reads may include shared values and this unit's effects only.
function _brm_nlme_check_partition_ownership(view, i, footprint)
    foreign = [j for j in footprint if view.subject[j] != 0 && view.subject[j] != i]
    isempty(foreign) && return nothing
    names = join(string.(view.coordinates[foreign]), ", ")
    _brm_nlme_error("brm_nlme_model: subject $i's observation partition reads " *
        "$(length(foreign)) effect coordinates owned by another subject: $names")
end

# ── Concurrent one-unit calls ────────────────────────────────────────────────
#
# A one-unit call writes θ, σ and η into a dense work point and runs a prepared
# RK partition query or sampler. Both are mutable state that RK allows one
# caller at a time (rkppl-use §14). An evaluator owns one value query, its work
# point and, from its first gradient, one gradient plan with its own sampler,
# gradients and work point. The model keeps a pool of evaluators. A call takes
# an idle one, prepares a new one while the pool has fewer than one per Julia
# thread, or waits until one comes back. So one-unit calls from several tasks
# run in parallel, as NLMEEstimation's `WarmupHMCNUTS(threads = true)` and
# `check_protocol(...; threads = true)` make them. A process whose calls never
# overlap prepares only the first evaluator. Every evaluator runs the same
# compiled program, and a call restores its point to `m.unit`, so a unit's
# result does not depend on which evaluator ran it.
#
# A task returning its evaluator can wait for the pool's lock and so let its
# thread start another task, which is how holders come to outnumber threads.
# `Threads.maxthreadid()` is not the bound: it counts the GC threads too, 18 for
# 13 threads, and a contended pool reached it. A task past the bound only waits
# as long as a holder takes to return, far less than preparing an evaluator.
_brm_nlme_max_evaluators() = Threads.nthreads(:default) + Threads.nthreads(:interactive)
mutable struct _BRMNLMEEvaluator
    const primary::Bool  # holds the model's own query, point and plan
    const query::Any
    const point::Vector{Float64}
    plan::Any            # `nothing` until the first gradient
end

_brm_nlme_evaluator_pool(query, point) = (; condition=Threads.Condition(),
    idle=[_BRMNLMEEvaluator(true, query, point, nothing)], count=Ref(1))

function _brm_nlme_with_evaluator(f, m)
    pool = m.cell_gradient.evaluators
    evaluator = _brm_nlme_take_evaluator!(m, pool)
    try
        f(evaluator)
    finally
        lock(pool.condition) do
            push!(pool.idle, evaluator)
            notify(pool.condition; all=false)
        end
    end
end

function _brm_nlme_take_evaluator!(m, pool)
    evaluator = lock(pool.condition) do
        while isempty(pool.idle) && pool.count[] >= _brm_nlme_max_evaluators()
            wait(pool.condition)
        end
        isempty(pool.idle) || return pop!(pool.idle)
        pool.count[] += 1
        nothing
    end
    evaluator === nothing || return evaluator
    # Prepared outside the pool's lock, so the other evaluators stay usable.
    try
        _brm_nlme_new_evaluator(m)
    catch
        lock(pool.condition) do
            pool.count[] -= 1
            notify(pool.condition; all=false)
        end
        rethrow()
    end
end

# A second evaluator means calls overlap: reverse gradients then run beside
# other threads, which needs Enzyme's concurrent-GC repair (enzyme-use, "Reverse
# gradients beside other threads"). NLMEEstimation checks `ad_backend(model)`
# before a threaded fit; BRM checks again here for every other caller.
function _brm_nlme_new_evaluator(m)
    _brm_concurrent_gradients(nothing) || _brm_nlme_error(
        "brm_nlme_model: one-unit calls overlap on several tasks, so reverse " *
        "gradients run beside other threads, and this Enzyme lacks the " *
        "concurrent-GC repair, so a gradient can use freed memory; use Enzyme's " *
        "dogfood line (from 72d5d25) or call the model from one task at a time")
    _BRMNLMEEvaluator(false, _rk_nlme_partition_query(m.cell_gradient.context, m.unit),
        copy(m.unit), nothing)
end

# The evaluator's gradient plan: the model's own for the first evaluator, a
# worker copy of it (own sampler, gradients and point) for the others.
function _brm_nlme_evaluator_plan!(m, evaluator::_BRMNLMEEvaluator)
    evaluator.plan === nothing || return evaluator.plan
    plan = _brm_nlme_cell_state(m).plan
    evaluator.plan = evaluator.primary ? plan :
        _brm_nlme_worker_plan(plan, m.cell_gradient, m.unit)
end

# Evaluate a unit on an evaluator's plan. A plan's type is settled by its first
# gradient, so the evaluator holds it untyped: this is the one dynamic call a
# one-unit gradient makes on it, and everything inside it is compiled for the plan.
function _brm_nlme_on_plan!(evaluate!, gradient, plan, m, i, θ, σ, η)
    _brm_nlme_write_unit!(plan.point, m, i, θ, σ, η)
    try
        evaluate!(gradient, plan, i)
    finally
        _brm_nlme_restore_unit!(plan.point, m, i)
    end
end

function _brm_nlme_partition_queries(backend, cells, view, unit, partitioned, progress=nothing)
    _brm_with_phases(progress, (context="Bind partition keys",
            query="Prepare partition kernel", footprints="Check coordinate ownership")) do phases
        context = _brm_run_phase(phases.context) do _
            _rk_nlme_partition_context(backend, cells, _brm_nlme_partition_keys(cells), partitioned)
        end
        query = _brm_run_phase(phases.query) do _
            try
                _rk_nlme_partition_query(context, unit)
            catch error
                _rk_nlme_is_partition_separability_error(error) || rethrow()
                # RK reports packed positions. On refusal only, use the same
                # structural cell reads as sparse curvature to name BRM's
                # foreign coordinates, including dormant branches.
                _brm_nlme_structural_reads(cells, unit, view)
                _brm_nlme_error("brm_nlme_model: observation partition reads effect " *
                    "coordinates owned by another subject: " * sprint(showerror, error))
            end
        end
        labels = Set(_rk_nlme_partition_labels(query))
        footprints = _brm_run_phase(phases.footprints) do phase
            _brm_with_counter(phase, length(view.subject_coordinates);
                    description="Unit footprints") do node
                map(eachindex(view.subject_coordinates)) do i
                    footprint = if i in labels
                        read = _rk_nlme_partition_query_footprint(query, unit, i)
                        _brm_nlme_check_partition_ownership(view, i, read)
                        read
                    else
                        Int[]
                    end
                    _brm_update_progress!(node, i)
                    footprint
                end
            end
        end
        point = copy(unit)
        (; context, query, labels, footprints, state=Ref{Any}(nothing),
            lock=ReentrantLock(), point, evaluators=_brm_nlme_evaluator_pool(query, point))
    end
end

# A unit is evaluated at a dense packed point owned by its query or plan, not
# at a lazy overlay: RK's partition gradient copies any other AbstractVector
# into a Vector on every call, which costs the whole packed length (224k
# coordinates on Atlas) per unit, and the overlay's reads are binary searches.
# Only the positions the caller sets are written, and `_brm_nlme_restore!`
# puts back `unit` there, so outside a call the point equals `unit`. The
# caller owns the point for the call: an evaluator taken from the model's pool
# (`_brm_nlme_with_evaluator`), or an SVI plan its task owns.
function _brm_nlme_restore!(point, unit, positions)
    for j in positions
        point[j] = unit[j]
    end
    point
end
function _brm_nlme_write!(point, positions, values)
    for (k, j) in enumerate(positions)
        point[j] = values[k]
    end
    point
end
_brm_nlme_unit_positions(m, i) = (m.theta, m.sigma, m.view.subject_coordinates[i])
function _brm_nlme_write_unit!(point, m, i, θ, σ, η)
    _brm_nlme_write!(point, m.theta, θ)
    _brm_nlme_write!(point, m.sigma, σ)
    _brm_nlme_write!(point, m.view.subject_coordinates[i], η)
    point
end
function _brm_nlme_restore_unit!(point, m, i)
    for positions in _brm_nlme_unit_positions(m, i)
        _brm_nlme_restore!(point, m.unit, positions)
    end
    point
end

# One compact sampler per owner of mutable AD workspace (model, SVI target or
# worker). Its structural footprint also includes fixed population coordinates;
# slots maps only the globals and unit effects requested by this consumer.
function _brm_nlme_partition_plan(partition, unit, globals, view;
        units=view.subject_coordinates)
    sampler = _rk_nlme_partition_sampler(partition.context, unit)
    indices = [_brm_nlme_unit_selection(globals, block) for block in units]
    footprints = map(eachindex(indices)) do i
        i in partition.labels || return Int[]
        footprint = _rk_nlme_partition_footprint(sampler, i)
        _brm_nlme_check_partition_ownership(view, i, footprint)
        footprint == partition.footprints[i] || _brm_nlme_error(
            "brm_nlme_model: subject $i's value and gradient partitions " *
            "have different structural footprints")
        footprint
    end
    slots = [[_brm_sorted_slot(indices[i], j) for j in footprint]
        for (i, footprint) in enumerate(footprints)]
    gradients = [zeros(length(footprint)) for footprint in footprints]
    # The unit's own effects only (RK `shared=false`): an η gradient at fixed
    # population coordinates neither differentiates nor pulls back through the
    # shared values. `eta_slots[i][k]` is the place in the unit's block of the
    # k-th local footprint position; positions RK declares local are the
    # unit's effects (`_rk_nlme_partitioned`), which ownership checks confirm.
    local_footprints = map(eachindex(indices)) do i
        i in partition.labels || return Int[]
        _rk_nlme_partition_local_footprint(sampler, i)
    end
    eta_slots = map(eachindex(local_footprints)) do i
        block = view.subject_coordinates[i]
        place = Dict(j => k for (k, j) in enumerate(block))
        map(local_footprints[i]) do j
            k = get(place, j, nothing)
            k === nothing && _brm_nlme_error("brm_nlme_model: subject $i's " *
                "local partition gradient reads coordinate " *
                "$(view.coordinates[j]), which is not one of its effects")
            k
        end
    end
    local_gradients = [zeros(length(footprint)) for footprint in local_footprints]
    (; sampler, indices, footprints, slots, gradients, local_footprints, eta_slots,
        local_gradients, labels=partition.labels, point=copy(unit))
end

# A plan for another task: its own AD sampler, gradients and work point (RK's
# one-per-caller contract), sharing the read-only tables, which scale with the
# units.
function _brm_nlme_worker_plan(plan, partition, unit)
    merge(plan, (; sampler=_rk_nlme_partition_sampler(partition.context, unit),
        gradients=[zeros(length(g)) for g in plan.gradients],
        local_gradients=[zeros(length(g)) for g in plan.local_gradients],
        point=copy(unit)))
end

# Evaluate a unit in one compiled pass at the plan's dense point, whose
# `indices[i]` positions the caller has written; `gradient` follows
# `indices[i]`. The caller owns this plan's workspace or holds its model's lock.
function _brm_nlme_partition_evaluate!(gradient, plan, i)
    fill!(gradient, 0.0)
    i in plan.labels || return 0.0
    value = first(_rk_nlme_partition_value_and_gradient!(plan.sampler,
        plan.gradients[i], plan.point, i))
    for (k, slot) in enumerate(plan.slots[i])
        slot == 0 || (gradient[slot] = plan.gradients[i][k])
    end
    value
end

# The unit's value and its η gradient in block order, at the plan's point.
function _brm_nlme_partition_eta!(gradient, plan, i)
    fill!(gradient, 0.0)
    i in plan.labels || return 0.0
    g = plan.local_gradients[i]
    value = first(_rk_nlme_partition_value_and_local_gradient!(plan.sampler,
        g, plan.point, i))
    for (k, slot) in enumerate(plan.eta_slots[i])
        gradient[slot] = g[k]
    end
    value
end

# Individual cell supports preserve the sparse eta-Hessian pattern. One
# footprint kernel per observation group is reused for all its cells, without
# AD, numeric probes or a retained partition plan per cell. Keep block order.
function _brm_nlme_structural_reads(cells, unit, view; progress=nothing)
    queries = view.unit_partition === :connected ? nothing :
        map(cell -> _rk_nlme_cell_footprint_query(cell.query), cells)
    place = zeros(Int, length(unit))
    map(eachindex(view.subject_coordinates)) do i
        block = view.subject_coordinates[i]
        place[block] .= eachindex(block)
        reads = map(ntuple(identity, length(cells)), cells) do g, cell
            group = map(cell.positions[i]) do p
                footprint = if view.unit_partition === :connected
                    support = view.factor_supports[only(cell.names)][p]
                    support === :whole ? eachindex(unit) : support
                else
                    _rk_nlme_cell_footprint(queries[g], unit, p)
                end
                _brm_nlme_check_partition_ownership(view, i, footprint)
                block[sort!([place[j] for j in footprint if place[j] != 0])]
            end
            _brm_update_progress!(progress, (i - 1) * length(cells) + g)
            group
        end
        place[block] .= 0
        reads
    end
end

"""
    BRMNLMEModel

An RK-lowered population model prepared for NLME estimators, returned by
[`brm_nlme_model`](@ref). Its coordinates follow the NLME convention:

- `θ` — the population coordinates and, unless fixed at `shared_point`, shared
  sampled coordinates, in packed order;
- `σ` — the other sampled scalars (`view.role .=== :scalar`), in packed order;
- `η_i` — subject `i`'s random effects on the natural scale. A flat model uses
  column `i` of a `neta × nsubjects` matrix; a nested model uses compact vector
  `H[i]` in `view.subject_families[i]` order.

All three are in the model's unconstrained coordinates. The random-effect
covariance Ω is not a model coordinate here: estimators own it, and the model
evaluates the conditional likelihood `log p(y_i | θ, σ, η_i)` without the
random-effect prior.

Shared sampled coordinates have role `:shared_latent` and enter θ by default.
They retain the original RK layout's unconstrained values and transforms.
Supplying `shared_point` instead fixes them at that full RK packed point. The
conditional likelihood excludes all population priors; the NLMEEstimation
population-prior protocol includes the authored priors once in the fit objective.

When every fitted response is an independent, unweighted Normal observation
supported by RK's `:observations` preset, `observed` holds subject observation
vectors in the same order as the prepared query. `observations_pushforward`
uses one colored direction per random-effect margin across all subjects.
"""
# What SVI prepared on a model, kept for the model's next fit or continuation in
# this process (`_brm_svi_checkout!`, src/nlme_svi.jl). It is workspace, not part
# of the model's value, so a copy or a serialized model starts with an empty slot.
mutable struct _BRMSVITargetSlot
    prepared::Any
    lock::ReentrantLock
    _BRMSVITargetSlot() = new(nothing, ReentrantLock())
end
Base.deepcopy_internal(::_BRMSVITargetSlot, ::IdDict) = _BRMSVITargetSlot()
Serialization.serialize(s::Serialization.AbstractSerializer, ::_BRMSVITargetSlot) =
    Serialization.serialize_type(s, _BRMSVITargetSlot)
Serialization.deserialize(::Serialization.AbstractSerializer, ::Type{_BRMSVITargetSlot}) =
    _BRMSVITargetSlot()

# A heap object whose fields never change, not an immutable struct: it holds its
# prepared fields inline, so each dynamic call that passed an immutable one copied
# the whole model to the heap (8.5 KB on an 80-subject PK model).
mutable struct BRMNLMEModel{B<:RKBRMI,S,Q,V,O,J,C,G,P}
    const view::BRMNLMEView
    const backend::B
    const sampler::S
    const pointwise::Q
    const eta_hvp::V
    const attribution::Vector{Pair{Symbol,Vector{Int}}}
    const theta::Vector{Int}
    const sigma::Vector{Int}
    const eta_blocks::Vector{Int}
    const unit::Vector{Float64}
    const observations_query::O
    const observations_pushforward::J
    const observed::Vector{Vector{Float64}}
    const observation_attribution::Vector{Pair{Symbol,Vector{Int}}}
    const cell_query::C
    const cell_gradient::G
    const population_prior::P
    # The population prior's coordinate structure (which packed coordinates
    # hold each covariance block's scales and correlation factor), built by
    # the NLMEEstimation extension on the first prior evaluation.
    const prior_structure::Base.RefValue{Any}
    # The sparse Laplace problems of the integrated families, one per set of
    # families, built by the NLMEEstimation extension on first use.
    const laplace::Dict{Vector{Int},Any}
    const svi_target::_BRMSVITargetSlot
end

Base.show(io::IO, m::BRMNLMEModel) = print(io, "BRMNLMEModel(",
    length(m.view.levels), " subjects, θ: ", length(m.theta), ", σ: ",
    length(m.sigma), ", η: ",
    _brm_nlme_ragged(m.view) ? string(length.(m.view.subject_coordinates)) :
        string(sum(m.eta_blocks)), " in blocks ", m.eta_blocks, ")")

"""
    brm_nlme_model(backend::RKBRMI; ad_backend, observation_ad_backend=nothing,
        shared_point=nothing, subject_group=nothing,
        nested_groups=NamedTuple(), shared_groups=(), unit_partition=:subjects,
        progress=nothing) -> BRMNLMEModel

Prepare an RK-lowered population model for NLME estimators: its
[`brm_nlme_view`](@ref) partition, the backend's sampler query retaining its
pointwise densities (first-order, reverse-mode Enzyme: `ad_backend` must be
`AutoEnzyme(; mode=Enzyme.Reverse)`), a reverse-over-forward sampler HVP query,
its pointwise densities, and the subject attribution of every observed density.

The HVP query is prepared the first time it is used: by the dense η Hessian
blocks (`NLMEEstimation.conditional_loglikelihood_hessian_blocks`) and by the
per-unit Laplace. With ReactiveKernels before 0.1.12 its preparation holds 8n²
bytes for the n packed coordinates (n one-hot basis vectors of length n), 200
GB at 158,343 coordinates; from 0.1.12 it is linear in n. No step of the
preparation copies the model per thread.

`observation_ad_backend` selects the forward-mode backend for Gaussian moment
Jacobians. With the NLMEEstimation/Enzyme extension loaded, an `AutoEnzyme`
reverse backend automatically selects Enzyme forward mode. Other callers can
pass a forward backend explicitly.

Shared sampled coordinates enter θ by default and can therefore be estimated
jointly with ordinary population parameters. To hold them fixed, pass
`shared_point`, a full RK packed unconstrained point. Only its shared entries
are used. The model's θ positions and names make the chosen fit mode explicit.

Pass the same `subject_group`, `nested_groups`, `shared_groups` and
`unit_partition` as for
`brm_nlme_view` when the model has several groupings. With nested effects its
NLME layout has one shared covariance family per random-effect declaration and
one compact η vector per subject; the blocks over `shared_groups` are shared
latents in θ (or fixed by `shared_point`).

With `unit_partition=:connected`, individual structural cell supports join
likelihood-linked subjects before attribution. Every likelihood, gradient and
subset then uses complete inference units; the view retains original subject
labels and slices. The attribution description below applies to the default
`:subjects` policy. Both policies retain strict locality validation.

Each pointwise density belongs to one subject. A density on the subject
column's rows (a flat grouped response, a completed covariate's observed
values, a group term of `mi(x)::shares`, a plate cell joining those rows)
belongs to the subject of its rows. Any other observation, on rows of its own
(a fragment's rows read through an index, `exp(log_f[f_obs])`, or joined by a
plate over the fragment's own groups), has its densities' subjects measured on
the pointwise query: moving the η blocks of halves of the subjects,
2⌈log₂ n⌉ evaluations for n subjects. A density that reads no subject's η
counts toward the first subject.

When every observed density belongs to one subject's cells, the model prepares
one compiled observation partition per unit label. Each unit value or gradient
runs all its cells in one pass. RK's strict locality check refuses a value that
cannot be computed at the requested rows, naming its producer. BRM checks the
structural packed-coordinate footprints against its unit ownership before
returning the model, including for value-only use.

One compact partition sampler is prepared on first gradient use. For nested
models, value-only cell footprint kernels supply the individual supports whose
cliques form each unit's sparse η Hessian pattern. These supports include both
arms of branches and zero-weight coordinates; no numerical probes or gradient
pruning determine the structure. Retained conditional captures can still have
conservative supports; a structural read of another unit's effect is refused.

**Calls from several tasks.** One-unit calls can come from several tasks at
once: [`brm_nlme_loglikelihood`](@ref),
[`brm_nlme_loglikelihood_and_gradient`](@ref), the NLMEEstimation protocol's
per-subject methods, and [`brm_nlme_unit_problem`](@ref) densities. NLMEEstimation's
`WarmupHMCNUTS(threads = true)` makes such calls. Each call takes an evaluator
of its own. The first evaluator is the model's own partition query and sampler.
Further ones are prepared when calls first overlap, at most one per Julia
thread, each with its own partition query, compact sampler and work points.
Overlapping calls therefore run in parallel instead of waiting for each other.
A unit's value and gradient are the same bit for bit whichever evaluator
computes them. Overlapping calls need Enzyme's concurrent-GC repair (Enzyme
dogfood from `72d5d25`). Without it the first call that would need a second
evaluator raises a `BRMNLMEViewError`. Batched calls run on the calling task.

`progress` reports the preparation through Treebars: pass a Treebars node and
the call builds a `Prepare NLME model` node under it, with one phase per step
(subject partition, pointwise sampler, observation attribution, η
Hessian-vector product, Gaussian observations, subject cells, population
prior; a step the model does not need is skipped), or a backend symbol such as
`:state` to start a new root. An interrupt requested on that node or an
ancestor (`Treebars.request_interrupt!`) raises `Treebars.ProgressInterrupted`
when the next step starts. `nothing` (the default) reports nothing.

Refuses ([`BRMNLMEViewError`](@ref)) everything `brm_nlme_view` refuses, Student-t
random-effect blocks (the estimators own a Gaussian η prior), and observed
densities that cannot be attributed under the selected policy. In default
`:subjects` mode, a plate cell joining rows of
several subjects, and a measured density that reads several subjects' η (the
message names them), is not finite at the measuring point, or changes with
some of the moves but not all.

Numerical attribution alone cannot detect every piecewise-constant dependence.
For models with subject cells, the subsequent structural ownership check also
includes dormant branches and refuses a partition that reads another subject's
effects before returning the model.
"""
function brm_nlme_model(backend::RKBRMI; ad_backend,
        observation_ad_backend=nothing, shared_point=nothing,
        subject_group=nothing, nested_groups=NamedTuple(), shared_groups=(),
        unit_partition=:subjects, progress=nothing)
    _brm_with_progress(progress; description="Prepare NLME model") do node
        _brm_nlme_model(backend, node; ad_backend, observation_ad_backend,
            shared_point, subject_group, nested_groups, shared_groups, unit_partition)
    end
end

# The preparation steps of `brm_nlme_model`, each a phase of `node`. A step the
# model does not need (Gaussian observations, subject cells) is left skipped.
const _BRM_NLME_MODEL_PHASES = (view="Subject partition",
    sampler="Pointwise sampler", attribution="Observation attribution",
    eta_hvp="η Hessian-vector product", observations="Gaussian observations",
    cells="Subject cells", prior="Population prior")

function _brm_nlme_model(backend::RKBRMI, node; ad_backend, observation_ad_backend,
        shared_point, subject_group, nested_groups, shared_groups=(), unit_partition=:subjects)
    _brm_with_phases(node, _BRM_NLME_MODEL_PHASES) do phases
        view, unit = _brm_run_phase(phases.view) do _
            v = brm_nlme_view(backend; subject_group, nested_groups, shared_groups, unit_partition)
            any(==(:subject_mixing), v.role) && _brm_nlme_error(
                "brm_nlme_model: Student-t random-effect blocks are not covered; NLME " *
                "estimators own a Gaussian random-effect prior")
            v, _brm_nlme_unit_point(backend, v; shared_point)
        end
        sampler = _brm_run_phase(phases.sampler) do _
            _rk_retained_pointwise_sampler(backend; ad_backend)
        end
        pointwise, attribution, measured = _brm_run_phase(phases.attribution) do _
            query = _rk_pointwise_query(backend)
            query, _brm_nlme_attribution(backend, view, query, unit)...
        end
        eta_hvp = _brm_run_phase(phases.eta_hvp) do _
            _rk_eta_hvp_query(backend, unit, ad_backend)
        end
        eta_blocks = Int[]
        previous = nothing
        for margin in view.block_margins
            margin.block == previous ? (eta_blocks[end] += 1) : push!(eta_blocks, 1)
            previous = margin.block
        end
        # NLMEEstimation's Gaussian-observation protocol currently serves FOCE(I),
        # which does not accept RaggedNLMELayout. Keep the ragged conditional and
        # Hessian paths active without advertising an unusable observation model.
        gaussian = !_brm_nlme_ragged(view) && _brm_nlme_gaussian_plan(backend.plan)
        observations_query, observations_pushforward, observed, observation_attribution =
            gaussian ? _brm_run_phase(phases.observations) do _
                forward_backend = observation_ad_backend === nothing ?
                    _brm_nlme_forward_backend(ad_backend) : observation_ad_backend
                (_rk_observation_query(backend),
                 forward_backend === nothing ? nothing :
                    _rk_observation_pushforward(backend, view, unit, forward_backend),
                 _brm_nlme_observed(backend, attribution, length(view.levels))...)
            end : (nothing, nothing, Vector{Vector{Float64}}(), Pair{Symbol,Vector{Int}}[])
        theta = findall(r -> r === :population ||
            (shared_point === nothing && r === :shared_latent), view.role)
        sigma = findall(==(:scalar), view.role)
        cells = _brm_nlme_subject_cells(backend, view, attribution, measured)
        cell_query, cell_gradient = cells === nothing ? (nothing, nothing) :
            _brm_run_phase(phases.cells) do phase
                partitioned = _rk_nlme_partitioned(backend, view)
                queries = _brm_nlme_cell_queries(backend, cells, partitioned, unit, phase)
                partition = _brm_with_progress(phase; description="Value partition") do node
                    _brm_nlme_partition_queries(backend, queries, view, unit, partitioned, node)
                end
                queries, partition
            end
        population_prior = _brm_run_phase(phases.prior) do _
            _rk_nlme_population_prior(backend, unit, ad_backend)
        end
        BRMNLMEModel(view, backend, sampler, pointwise, eta_hvp, attribution,
            theta, sigma,
            eta_blocks, unit, observations_query, observations_pushforward,
            observed, observation_attribution, cell_query, cell_gradient,
            population_prior, Ref{Any}(nothing), Dict{Vector{Int},Any}(), _BRMSVITargetSlot())
    end
end

_brm_nlme_cell_point(m::BRMNLMEModel, i::Integer, θ, σ, η) =
    _BRMNLMEPoint(m.unit, m.theta, Vector{Float64}(θ), m.sigma,
        Vector{Float64}(σ), _brm_nlme_check_unit(m, i, θ, σ, η),
        Vector{Float64}(η))

# The argument checks of every one-unit evaluation; returns the unit's block.
function _brm_nlme_check_unit(m::BRMNLMEModel, i::Integer, θ, σ, η)
    m.cell_query === nothing && _brm_nlme_error(
        "brm_nlme_loglikelihood: per-subject evaluation needs observation " *
        "cells that each belong to one subject, and this model's observations " *
        "have none; use the lockstep methods")
    1 <= i <= length(m.view.levels) || throw(BoundsError(m.view.levels, i))
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "θ has length $(length(θ)), expected $(length(m.theta))"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "σ has length $(length(σ)), expected $(length(m.sigma))"))
    block = m.view.subject_coordinates[i]
    length(η) == length(block) || throw(DimensionMismatch(
        "η has length $(length(η)), expected $(length(block))"))
    block
end

"""
    brm_nlme_loglikelihood(m::BRMNLMEModel, i, θ, σ, η_i) -> Float64

One subject's conditional log-likelihood from its observation cells: its plate
cells, its rows of flat grouped responses and its observed values of completed
covariates. The work depends on those cells' observations, not on other
subjects. Models whose observations have no subject cells use
[`brm_nlme_loglikelihoods`](@ref) instead.
"""
function brm_nlme_loglikelihood(m::BRMNLMEModel, i::Integer, θ, σ, η)
    if _brm_nlme_ragged(m.view) && m.cell_query === nothing
        1 <= i <= length(m.view.levels) || throw(BoundsError(m.view.levels, i))
        H = [zeros(Float64, length(block)) for block in m.view.subject_coordinates]
        length(η) == length(H[i]) || throw(DimensionMismatch(
            "η has length $(length(η)), expected $(length(H[i]))"))
        H[i] = Vector{Float64}(η)
        return brm_nlme_loglikelihoods(m, θ, σ, H)[i]
    end
    _brm_nlme_check_unit(m, i, θ, σ, η)
    i in m.cell_gradient.labels || return 0.0
    _brm_nlme_with_evaluator(m) do evaluator
        point = _brm_nlme_write_unit!(evaluator.point, m, i, θ, σ, η)
        try
            _rk_nlme_partition_value(evaluator.query, point, i)
        finally
            _brm_nlme_restore_unit!(point, m, i)
        end
    end
end

# One unit-partition sampler, built on first gradient use under its lock.
# Nested models also derive sparse Hessian patterns from each cell's structural
# support; a whole-unit footprint is never turned into a clique.
function _brm_nlme_cell_state(m::BRMNLMEModel)
    g = m.cell_gradient
    g === nothing && _brm_nlme_error(
        "brm_nlme_model: this model's observations have no subject cells")
    lock(g.lock) do
        g.state[] === nothing || return g.state[]
        _brm_with_progress(_brm_cell_progress();
                description="Unit partition gradients") do node
            globals = sort!(vcat(m.theta, m.sigma))
            plan = _brm_with_counter(node, 1; description="Partition sampler") do c
                p = _brm_nlme_partition_plan(g, m.unit, globals, m.view)
                _brm_update_progress!(c, 1)
                p
            end
            reads = if _brm_nlme_ragged(m.view)
                steps = length(m.view.subject_coordinates) * length(m.cell_query)
                _brm_with_counter(node, steps; description="Cell structural footprints") do c
                    _brm_nlme_structural_reads(m.cell_query, m.unit, m.view; progress=c)
                end
            else
                nothing
            end
            patterns = reads === nothing ? nothing :
                _brm_nlme_eta_patterns(m.view.subject_coordinates, reads)
            g.state[] = (; reads, plan, patterns)
        end
    end
end

# Each subject's η Hessian pattern for NLMEEstimation's `eta_hessian_sparsity`:
# a nested model with subject cells declares one; others declare none.
_brm_nlme_eta_sparsity(m::BRMNLMEModel) =
    m.cell_gradient === nothing || !_brm_nlme_ragged(m.view) ? nothing :
        _brm_nlme_cell_state(m).patterns

# Which of subject `i`'s η can enter one observation together, for each subject
# of a nested model: entry (j, k) is true when η_j and η_k are both read by one
# of the subject's observation cells (`_brm_nlme_structural_reads`). The subject's
# conditional log-likelihood is the sum of its cells, so this covers every
# nonzero of its η Hessian. NLMEEstimation computes the Hessian's values from
# these patterns (`eta_hessian_sparsity`); a nested subject's units couple only
# through the effects above them, so the pattern stays sparse where its dense
# block would grow with the square of its size.
function _brm_nlme_eta_patterns(blocks, reads)
    map(eachindex(blocks)) do i
        block = blocks[i]
        n = length(block)
        local_index = Dict(j => k for (k, j) in enumerate(block))
        # Repeated observations do not add curvature edges. Keep one column
        # per distinct cell support, then form their incidence product. Sparse
        # multiplication accumulates each output column without storing every
        # duplicate pair from every cell (billions of pairs on Atlas).
        supports = Set{Vector{Int}}()
        for group in reads[i], read in group
            isempty(read) && continue
            push!(supports, sort!(unique!([local_index[j] for j in read])))
        end
        rows, pointers = Int[], Int[1]
        for support in supports
            append!(rows, support)
            push!(pointers, length(rows) + 1)
        end
        incidence = SparseMatrixCSC(n, length(supports), pointers, rows,
            fill(true, length(rows)))
        # Boolean sparse multiplication counts common cells. Only whether a
        # pair shares any cell matters; no whole-unit clique is introduced.
        (incidence * transpose(incidence)) .!= 0
    end
end

# The point's values at sorted, unique packed `indices`: θ, σ and η at their
# positions and the unit point elsewhere, in one pass over each.
function _brm_nlme_point_values(p::_BRMNLMEPoint, indices::Vector{Int})
    values = p.unit[indices]
    for (positions, xs) in ((p.theta_positions, p.theta), (p.sigma_positions, p.sigma),
            (p.eta_positions, p.eta))
        for (j, x) in zip(positions, xs)
            k = searchsortedfirst(indices, j)
            k <= length(indices) && indices[k] == j && (values[k] = x)
        end
    end
    values
end

# θ, σ and η at the unit's packed positions (the plan's `indices[i]`: the
# model's globals are θ and σ), and the gradient there.
function _brm_nlme_cell_value_and_selected_gradient(m::BRMNLMEModel, i::Integer, θ, σ, η)
    m.cell_gradient === nothing && _brm_nlme_error(
        "brm_nlme_model: this subject cell has no prepared compact gradient")
    block = _brm_nlme_check_unit(m, i, θ, σ, η)
    # Every evaluator's plan shares these read-only tables.
    indices = _brm_nlme_cell_state(m).plan.indices[i]
    selected_gradient = zeros(length(indices))
    value = _brm_nlme_with_evaluator(m) do evaluator
        _brm_nlme_on_plan!(_brm_nlme_partition_evaluate!, selected_gradient,
            _brm_nlme_evaluator_plan!(m, evaluator), m, i, θ, σ, η)
    end
    value, copy(indices), selected_gradient, block
end

"""
    brm_nlme_loglikelihood_and_gradient(m::BRMNLMEModel, i, θ, σ, η_i)

Return one subject's conditional log-likelihood and its η_i gradient. RK's
compact partition sampler evaluates all that subject's cells in one fused
call and differentiates only the subject's own effects: θ and σ are held
fixed, so no derivative flows through the values they share with other units.
"""
function brm_nlme_loglikelihood_and_gradient(m::BRMNLMEModel, i::Integer, θ, σ, η)
    # A nested model without subject cells evaluates the whole model.
    if m.cell_gradient === nothing && _brm_nlme_ragged(m.view)
        1 <= i <= length(m.view.levels) || throw(BoundsError(m.view.levels, i))
        H = [zeros(Float64, length(block)) for block in m.view.subject_coordinates]
        length(η) == length(H[i]) || throw(DimensionMismatch(
            "η has length $(length(η)), expected $(length(H[i]))"))
        H[i] = Vector{Float64}(η)
        values, gradients = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
        return values[i], gradients[i]
    end
    m.cell_gradient === nothing && _brm_nlme_error(
        "brm_nlme_model: this subject cell has no prepared compact gradient")
    block = _brm_nlme_check_unit(m, i, θ, σ, η)
    gradient = zeros(length(block))
    value = _brm_nlme_with_evaluator(m) do evaluator
        _brm_nlme_on_plan!(_brm_nlme_partition_eta!, gradient,
            _brm_nlme_evaluator_plan!(m, evaluator), m, i, θ, σ, η)
    end
    value, gradient
end

# Subject `i`'s η: column `i` of a flat model's matrix, or entry `i` of a
# nested model's vector of compact subject vectors.
_brm_nlme_subject_eta(H::AbstractMatrix, i) = view(H, :, i)
_brm_nlme_subject_eta(H::AbstractVector{<:AbstractVector}, i) = H[i]
_brm_nlme_eta_eltype(H::AbstractMatrix) = eltype(H)
_brm_nlme_eta_eltype(H::AbstractVector{<:AbstractVector}) = eltype(eltype(H))

function _brm_nlme_pack_point(unit, theta_indices, sigma_indices,
        subject_coordinates, θ, σ, H)
    u = Vector{promote_type(Float64, eltype(θ), eltype(σ),
        _brm_nlme_eta_eltype(H))}(unit)
    u[theta_indices] .= θ
    u[sigma_indices] .= σ
    for (i, block) in enumerate(subject_coordinates)
        u[block] .= _brm_nlme_subject_eta(H, i)
    end
    u
end

function _brm_nlme_point(m::BRMNLMEModel, θ, σ, H::AbstractMatrix)
    _brm_nlme_ragged(m.view) && throw(DimensionMismatch(
        "ragged subjects need one compact η vector per subject, not a matrix"))
    n = length(m.view.levels)
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "θ has length $(length(θ)), the model has $(length(m.theta)) population coordinates"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "σ has length $(length(σ)), the model has $(length(m.sigma)) scalar coordinates"))
    size(H) == (sum(m.eta_blocks), n) || throw(DimensionMismatch(
        "η matrix has size $(size(H)), the model needs ($(sum(m.eta_blocks)), $n)"))
    _brm_nlme_pack_point(m.unit, m.theta, m.sigma,
        m.view.subject_coordinates, θ, σ, H)
end

function _brm_nlme_point(m::BRMNLMEModel, θ, σ,
        H::AbstractVector{<:AbstractVector})
    n = length(m.view.levels)
    length(θ) == length(m.theta) || throw(DimensionMismatch(
        "θ has length $(length(θ)), expected $(length(m.theta))"))
    length(σ) == length(m.sigma) || throw(DimensionMismatch(
        "σ has length $(length(σ)), expected $(length(m.sigma))"))
    length(H) == n || throw(DimensionMismatch(
        "η has $(length(H)) subject vectors, expected $n"))
    for (i, block) in enumerate(m.view.subject_coordinates)
        length(H[i]) == length(block) || throw(DimensionMismatch(
            "subject $i η has length $(length(H[i])), expected $(length(block))"))
    end
    _brm_nlme_pack_point(m.unit, m.theta, m.sigma,
        m.view.subject_coordinates, θ, σ, H)
end

@inline function _brm_nlme_sum_fields!(out, attribution,
        fields::NamedTuple{names}, ::Val{K}) where {names,K}
    K > length(names) && return out
    name, subjects = attribution[K]
    name === names[K] || error("NLME pointwise field order differs from its attribution")
    values = getfield(fields, K)
    for (k, i) in enumerate(subjects)
        out[i] += sum(values[k])
    end
    _brm_nlme_sum_fields!(out, attribution, fields, Val(K + 1))
end

function _brm_nlme_sum_by_subject(attribution, nsubjects, fields::NamedTuple)
    length(attribution) == length(fields) || error(
        "NLME pointwise field count differs from its attribution")
    out = zeros(Float64, nsubjects)
    # The attribution is built in pointwise NamedTuple order. A static field
    # index keeps each field's (possibly nested) value type concrete for Enzyme.
    _brm_nlme_sum_fields!(out, attribution, fields, Val(1))
end

_brm_nlme_sum_by_subject(m::BRMNLMEModel, fields::NamedTuple) =
    _brm_nlme_sum_by_subject(m.attribution, length(m.view.levels), fields)

"""
    brm_nlme_loglikelihoods(m::BRMNLMEModel, θ, σ, H) -> Vector{Float64}

Every subject's conditional log-likelihood `log p(y_i | θ, σ, η_i)`, with
`η_i = H[:, i]` for a flat model or `η_i = H[i]` for a nested model, from one
evaluation of the model. See [`BRMNLMEModel`](@ref) for the coordinates.
"""
brm_nlme_loglikelihoods(m::BRMNLMEModel, θ, σ, H) =
    _brm_nlme_sum_by_subject(m, _rk_pointwise_values(m.pointwise, _brm_nlme_point(m, θ, σ, H)))

"""
    brm_nlme_loglikelihoods_and_gradients(m::BRMNLMEModel, θ, σ, H) -> (values, G)

[`brm_nlme_loglikelihoods`](@ref) together with each subject's η gradient,
in the same matrix or vector-of-vectors shape as `H`, from one reverse sweep
that also retains the pointwise densities. Subjects are
conditionally independent, so the joint gradient restricted to subject `i`'s
block is its conditional gradient once the standard-normal draw prior is removed.
"""
function brm_nlme_loglikelihoods_and_gradients(m::BRMNLMEModel, θ, σ, H)
    u = _brm_nlme_point(m, θ, σ, H)
    g = Vector{Float64}(undef, length(u))
    _, g, pointwise = _rk_value_gradient_and_pointwise!(m.sampler, g, u)
    values = _brm_nlme_sum_by_subject(m, pointwise)
    G = if H isa AbstractMatrix
        out = Matrix{Float64}(undef, size(H))
        for (i, block) in enumerate(m.view.subject_coordinates)
            out[:, i] .= view(g, block) .+ view(H, :, i)
        end
        out
    else
        [g[block] .+ H[i] for (i, block) in enumerate(m.view.subject_coordinates)]
    end
    values, G
end

# One direction per position within a subject block colors that position in
# every subject. Conditional independence makes its HVP the corresponding
# column of each subject's Hessian of the sampler density at the packed point
# `u`, which includes the standardized draws' N(0, I) prior.
function _brm_nlme_sampler_hessian_blocks(m::BRMNLMEModel, u::Vector{Float64})
    nη = maximum(length, m.view.subject_coordinates)
    blocks = [Matrix{Float64}(undef, length(c), length(c))
        for c in m.view.subject_coordinates]
    for k in 1:nη
        direction = zeros(Float64, length(u))
        for coordinates in m.view.subject_coordinates
            k <= length(coordinates) && (direction[coordinates[k]] = 1.0)
        end
        hv = _rk_eta_hvp(m.eta_hvp, direction, u)
        for (i, coordinates) in enumerate(m.view.subject_coordinates)
            k <= length(coordinates) || continue
            blocks[i][:, k] .= view(hv, coordinates)
        end
    end
    blocks
end

# The conditional likelihood's η Hessian blocks: at the unit point z = η, so
# adding I removes the standardized draws' prior from the sampler's blocks.
function _brm_nlme_hessian_blocks(m::BRMNLMEModel, θ, σ, H)
    blocks = _brm_nlme_sampler_hessian_blocks(m, _brm_nlme_point(m, θ, σ, H))
    for block in blocks, k in axes(block, 1)
        block[k, k] += 1.0
    end
    blocks
end
