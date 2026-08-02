module BayesianRegressionModelsEnzymeExt

using BayesianRegressionModels
using Enzyme

const BRM = BayesianRegressionModels

function BRM._native_ppl_logdensity_and_gradient!(
    workspace::BRM.NativePPLWorkspace{T},
    prepared::BRM.NativePPLPrepared,
    position::AbstractVector{T},
) where {T<:AbstractFloat}
    BRM._native_ppl_check_execution(workspace, prepared, position)
    fill!(workspace.gradient, zero(T))
    fill!(workspace.adjoint.location, zero(T))
    fill!(workspace.adjoint.pointwise_loglikelihood, zero(T))

    result = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        BRM._native_ppl_logdensity_kernel,
        Enzyme.Active,
        Enzyme.Duplicated(position, workspace.gradient),
        Enzyme.Const(prepared),
        Enzyme.Duplicated(workspace.primal, workspace.adjoint),
    )
    last(result), workspace.gradient
end

end
