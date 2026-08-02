module BayesianRegressionModelsDifferentiationInterfaceExt

using BayesianRegressionModels
import DifferentiationInterface as DI

const BRM = BayesianRegressionModels

struct NativePPLDIState{B,P}
    backend::B
    preparation::P
end

function BRM._native_ppl_workspace(
    prepared::BRM.NativePPLPrepared,
    ::Type{T},
    backend::DI.AbstractADType,
) where {T<:AbstractFloat}
    workspace = BRM.NativePPLWorkspace(prepared, T)
    position = zeros(T, BRM.LogDensityProblems.dimension(prepared))
    preparation = DI.prepare_gradient(
        BRM._native_ppl_logdensity_kernel,
        backend,
        position,
        DI.Constant(prepared),
        DI.Cache(workspace.primal),
    )
    derivative = NativePPLDIState(backend, preparation)
    BRM.NativePPLWorkspace{
        T,typeof(workspace.primal),typeof(workspace.gradient),typeof(derivative),
    }(workspace.primal, workspace.gradient, derivative)
end

function BRM._native_ppl_logdensity_and_gradient!(
    workspace::BRM.NativePPLWorkspace{T,B,G,<:NativePPLDIState},
    prepared::BRM.NativePPLPrepared,
    position::AbstractVector{T},
) where {T<:AbstractFloat,B,G}
    BRM._native_ppl_check_execution(workspace, prepared, position)
    state = workspace.derivative
    DI.value_and_gradient!(
        BRM._native_ppl_logdensity_kernel,
        workspace.gradient,
        state.preparation,
        state.backend,
        position,
        DI.Constant(prepared),
        DI.Cache(workspace.primal),
    )
end

end
