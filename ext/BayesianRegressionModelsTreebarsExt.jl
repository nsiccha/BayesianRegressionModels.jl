module BayesianRegressionModelsTreebarsExt

# BRM's progress hooks (src/progress.jl) for Treebars nodes and backend symbols.
# The Treebars 0.1.1 compatibility floor includes `reuse_progress!` and the
# cooperative interrupt flag used by these hooks.

using BayesianRegressionModels
import Treebars
using Treebars: ProgressNode
const BRM = BayesianRegressionModels

BRM._brm_with_progress(f, progress::Union{ProgressNode,Symbol}; description) =
    Treebars.with_progress(f, progress; description)
BRM._brm_with_counter(f, parent::ProgressNode, n::Integer; description) =
    Treebars.with_progress(f, parent, n; description)
BRM._brm_with_phases(f, node::ProgressNode, labels::NamedTuple) =
    Treebars.with_prepared_phases(f, node, labels)
BRM._brm_enter_phase(f, phase::ProgressNode) = Treebars.with_prepared_progress(f, phase)
BRM._brm_throw_if_interrupted(node::ProgressNode) = Treebars.throw_if_interrupted(node)
BRM._brm_interrupt_requested(node::ProgressNode) = Treebars.interrupt_requested(node)
BRM._brm_reuse_phase!(phase::ProgressNode) = (Treebars.reuse_progress!(phase); nothing)
BRM._brm_update_progress!(node::ProgressNode, message) =
    (Treebars.update_progress!(node, message); nothing)

end
