module BayesianRegressionModelsReactiveKernelsEnzymeExt

using BayesianRegressionModels
using DifferentiationInterface: AutoEnzyme, SecondOrder
using ReactiveKernels
using ReactiveKernelsPPL
import Enzyme

const BRM = BayesianRegressionModels

# A program whose reverse gradient needs runtime activity needs it in BRM's
# other Enzyme modes of the same program too (the η Hessian's
# reverse-over-forward, Gaussian moment pushforwards), so those modes take the
# runtime-activity setting of the caller's `ad_backend`.
BRM._brm_enzyme_mode(mode, ad_backend::AutoEnzyme) = ad_backend.mode === nothing ?
    mode : Enzyme.set_runtime_activity(mode, ad_backend.mode)

function BRM._rk_eta_hvp_query(backend::BRM.RKBRMI, unit::Vector{Float64}, ad_backend)
    rkext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    second_order = SecondOrder(
        AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Reverse, ad_backend)),
        AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Forward, ad_backend)))
    prepare_query_ad(prepare_ad_hvp, backend.model,
        rkext._rk_translated(backend), :sampler,
        second_order, (zeros(Float64, length(unit)),), unit)
end

BRM._brm_svi_default_ad(::BRM.RKBRMI) = AutoEnzyme(; mode=Enzyme.Reverse)

BRM._rk_eta_hvp(query, direction::Vector{Float64}, u::Vector{Float64}) =
    only(ad_hvp(query, (direction,), u))

# The sampler density's third-order contractions `D³f(u)[v, v, ·] = ∇(vᵀ H v)`,
# reverse over forward over forward, for the per-unit Laplace gradient
# (src/nlme_laplace.jl). A direction summed over units returns the sum of every
# unit's own contraction.
function BRM._rk_laplace_third_query(backend::BRM.RKBRMI, unit::Vector{Float64}, ad_backend)
    rkext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    forward = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Forward, ad_backend))
    third_order = SecondOrder(SecondOrder(
        AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Reverse, ad_backend)), forward),
        forward)
    prepare_query_ad(prepare_ad_third, backend.model,
        rkext._rk_translated(backend), :sampler,
        third_order, (zeros(Float64, length(unit)),), unit)
end

BRM._rk_laplace_third(query, direction::Vector{Float64}, u::Vector{Float64}) =
    only(ad_third(query, (direction,), u))

function BRM._rk_nlme_compact_cell_sampler(backend::BRM.RKBRMI,
        observations::Tuple, base::Vector{Float64}, selected::Vector{Int})
    rkext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    prepare_compact_cell_sampler(backend.model,
        rkext._rk_translated(backend), observations, base, selected;
        backend=AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)))
end

BRM._rk_nlme_compact_cell_value_and_gradient!(sampler::CompactCellSampler,
        gradient::Vector{Float64}, base::Vector{Float64},
        active::Vector{Float64}, selected::Vector{Int}, row::Int) =
    cell_value_and_gradient!(sampler, gradient, base, active, selected, row)

end
