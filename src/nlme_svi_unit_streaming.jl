# Data-only local factors: optimizer moments never enter a carried prior.
function _brm_svi_saved_unit(keys, M, Λ::AbstractMatrix, U::AbstractMatrix)
    deepcopy((; keys, M, Λ, U))
end
function _brm_svi_saved_unit(keys, M, Λ::_BRMSVIUnitPrecision, U::_BRMSVIUnitRoot)
    f = Λ.factor
    deepcopy((; keys, M, family=:diagonal_lowrank, f.ρ, f.Β))
end

_brm_svi_saved_unit_draw(rng, unit) =
    _brm_svi_saved_unit_draw(rng, unit, Val(get(unit, :family, :dense)))
_brm_svi_saved_unit_draw(rng, unit, ::Val{:dense}) =
    unit.M + LowerTriangular(unit.U)' \ randn(rng, length(unit.M))
_brm_svi_saved_unit_draw(rng, unit, ::Val{:diagonal_lowrank}) =
    unit.M + exp.(unit.ρ) .* randn(rng, length(unit.M)) +
        unit.Β * randn(rng, size(unit.Β, 2))

_brm_svi_restore_unit!(q, i, old, perm) =
    _brm_svi_restore_unit!(q, i, old, perm, Val(get(old, :family, :dense)))
function _brm_svi_restore_unit!(q, i, old, perm, ::Val{:dense})
    q.M[i] .= old.M[perm]
    q.Λ[i] = Matrix{Float64}(old.Λ[perm, perm])
    q.U[i] = Matrix(cholesky(Symmetric(q.Λ[i])).L)
    q.unit_opt[i] = nothing
    nothing
end
function _brm_svi_restore_unit!(q, i, old, perm, ::Val{:diagonal_lowrank})
    q.M[i] .= old.M[perm]
    f = _BRMSVILowRankUnit(copy(old.ρ[perm]), copy(old.Β[perm, :]))
    q.Λ[i], q.U[i] = _BRMSVIUnitPrecision(f), _BRMSVIUnitRoot(f)
    q.unit_opt[i] = _BRMSVIAdam(length(perm) * (2 + size(f.Β, 2)), 0.1)
    nothing
end

_brm_svi_state_format(units) =
    any(u -> get(u, :family, :dense) === :diagonal_lowrank, values(units)) ? 2 : 1
