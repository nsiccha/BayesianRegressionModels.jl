module BayesianRegressionModelsTreeArraysExt

# Labelled containers for julianic postprocessing draws (user directive
# 2026-08-04: "for Postprocessing, we don't need to be so strict, and should
# probably use TreeArrays.jl").
#
# The core draws functions already stack each site draw-major
# (`ndraws × site-shape…`). What they cannot supply is what those axes MEAN, and
# that is precisely TreeArrays' job: the axis carries its labels once, as
# metadata, instead of being smeared across rows the way a long dataframe does.
# So this is a thin naming layer over the same arrays — no restructuring, no
# copy — which is also why it belongs in a weak-dep extension rather than the
# core: a consumer who only wants the density never pays for it.

using BayesianRegressionModels
using TreeArrays: TreeArrays, TreeData

const BRM = BayesianRegressionModels
const NP = BRM.NativePPL

# The axis names for ONE site's stacked draws. The draw axis always comes first
# (that is the stacking convention); the site's own axes follow, unlabelled —
# their coordinates are positions within the site, and inventing names like
# `:index` for them would be metadata nobody asserted. A caller who knows what
# the trailing axes mean can relabel, but the container should not guess.
_julianic_draw_dims(store::AbstractVector, name::Symbol) = (:draw,)
_julianic_draw_dims(store::AbstractArray, name::Symbol) =
    (:draw, ntuple(i -> Symbol(name, :_, i), ndims(store) - 1)...)

_julianic_tree(store::AbstractArray, name::Symbol) =
    TreeData(store, _julianic_draw_dims(store, name)...)

# A record of per-site trees: sites have different shapes (a scalar `sigma`, a
# length-k `tau`, a K×K correlation factor), which is exactly the ragged-record
# shape TreeArrays models — as against forcing them into one rectangular block.
_julianic_tree_record(draws::NamedTuple) = TreeData((;
    (name => _julianic_tree(getfield(draws, name), name)
     for name in propertynames(draws))...))

"""
    NativePPL.jconstrained_tree(prepared, positions) -> TreeData

`jconstrained_draws` as a labelled `TreeData` record — one field per latent
site, each with a leading `:draw` axis. The arrays are the same ones
`jconstrained_draws` returns; only the axis labels are added.
"""
NP.jconstrained_tree(prepared, positions::AbstractMatrix) =
    _julianic_tree_record(NP.jconstrained_draws(prepared, positions))

"""
    NativePPL.jpointwise_tree(prepared, positions) -> TreeData

`jpointwise_draws` as a labelled `TreeData` record — one field per observation
site, `:draw` × the site's observation axis.
"""
NP.jpointwise_tree(prepared, positions::AbstractMatrix) =
    _julianic_tree_record(NP.jpointwise_draws(prepared, positions))

"""
    NativePPL.jpredict_tree([rng], prepared, positions) -> TreeData

`jpredict_draws` as a labelled `TreeData` record.
"""
NP.jpredict_tree(rng, prepared, positions::AbstractMatrix) =
    _julianic_tree_record(NP.jpredict_draws(rng, prepared, positions))
NP.jpredict_tree(prepared, positions::AbstractMatrix) =
    _julianic_tree_record(NP.jpredict_draws(prepared, positions))

end # module
