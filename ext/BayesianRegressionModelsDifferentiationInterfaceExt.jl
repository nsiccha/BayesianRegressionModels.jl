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
    BRM._native_ppl_require_response(
        prepared, "DifferentiationInterface gradient preparation")
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
    position::AbstractVector,
) where {T<:AbstractFloat,B,G}
    BRM._native_ppl_check_execution(workspace, prepared, position)
    position isa Vector{T} || throw(ArgumentError(
        "native PPL DifferentiationInterface gradients require a Vector{$T} " *
        "position; got $(typeof(position))"))
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

function BRM.NativePPL._factor_workspace(
    prepared::BRM.NativePPL.FactorPrepared,
    ::Type{T},
    backend::DI.AbstractADType,
) where {T<:AbstractFloat}
    workspace = BRM.NativePPL.FactorWorkspace(prepared, T)
    position = zeros(T, BRM.LogDensityProblems.dimension(prepared))
    preparation = DI.prepare_gradient(
        BRM.NativePPL._factor_logdensity_kernel,
        backend,
        position,
        DI.Constant(prepared),
        DI.Cache(workspace.primal),
    )
    derivative = NativePPLDIState(backend, preparation)
    BRM.NativePPL.FactorWorkspace{
        T,typeof(workspace.primal),typeof(workspace.gradient),typeof(derivative),
    }(workspace.primal, workspace.gradient, derivative)
end

function BRM.NativePPL._factor_logdensity_and_gradient!(
    workspace::BRM.NativePPL.FactorWorkspace{T,B,G,<:NativePPLDIState},
    prepared::BRM.NativePPL.FactorPrepared,
    position::AbstractVector,
) where {T<:AbstractFloat,B,G}
    BRM.NativePPL._factor_check_execution(workspace, prepared, position)
    position isa Vector{T} || throw(ArgumentError(
        "native PPL DifferentiationInterface factor gradients require a " *
        "Vector{$T} position; got $(typeof(position))"))
    state = workspace.derivative
    DI.value_and_gradient!(
        BRM.NativePPL._factor_logdensity_kernel,
        workspace.gradient,
        state.preparation,
        state.backend,
        position,
        DI.Constant(prepared),
        DI.Cache(workspace.primal),
    )
end

# --- Julianic "run-the-body" stack ---------------------------------------
# Enzyme straight through the run-the-body primal kernel `θ -> Real`. The
# prepared model (its lowered body closure + conditioned data) is a Constant;
# `theta` is the sole active input. No DI.Cache: the primal allocates its own
# context per call.

function BRM.NativePPL._julianic_prepare_gradient(
    prepared::BRM.NativePPL.JulianicPrepared,
    backend::DI.AbstractADType,
    position::AbstractVector,
)
    preparation = DI.prepare_gradient(
        BRM.NativePPL._julianic_kernel,
        backend,
        position,
        DI.Constant(prepared),
    )
    NativePPLDIState(backend, preparation)
end

function BRM.NativePPL._julianic_value_and_gradient!(
    prepared::BRM.NativePPL.JulianicPrepared,
    state::NativePPLDIState,
    gradient::AbstractVector,
    theta::AbstractVector,
)
    return DI.value_and_gradient!(
        BRM.NativePPL._julianic_kernel,
        gradient,
        state.preparation,
        state.backend,
        theta,
        DI.Constant(prepared),
    )
end

end
