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

BRM._brm_rk_default_ad(caller, ::BRM.RKBRMI) = AutoEnzyme(; mode=Enzyme.Reverse)

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

# The row-structured Laplace (src/nlme_laplace_structured.jl): the likelihood
# parameters `ports` (natural scale, one value per row or a scalar) at a
# packed point, one reverse pullback of them all, the random-effect draws `effects`
# (one levels × K matrix per block) with their forward derivatives along the
# `omega` coordinates, and the population prior.
_rk_slap_tuple(value, n) = n == 1 ? (value,) : Tuple(value)

function BRM._rk_slap_kernels(backend::BRM.RKBRMI, ports::Tuple, effects::Tuple,
        omega::Vector{Int}, u0::Vector{Float64}, ad_backend)
    bound = BRM.rk_bound_data(backend)
    spec = backend.model.spec
    have = (:unconstrained, keys(bound)...)
    prepare_want(want) = Base.invokelatest(ReactiveKernels.prepare, spec;
        have, want, bound)
    values = prepare_want(ports)
    v0 = _rk_slap_tuple(Base.invokelatest(values, u0), length(ports))
    # One reverse pullback of every parameter, prepared on first use: a model
    # whose parameters are affine in the population coordinates never needs
    # it. Its seed holds one cotangent per port (untupled for a single port).
    reverse = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Reverse, ad_backend))
    pullback = Ref{Any}(nothing)
    seeds = map(value -> value isa Number ? 0.0 : zeros(Float64, length(value)), v0)
    prepare_pullback() = Base.invokelatest(prepare_ad_pullback, values, reverse,
        _rk_slap_seed(seeds), u0; active=:unconstrained)
    # One draws kernel per block, differentiated along the `omega` coordinates
    # that block owns (`<block>.…`): each block's own directions only, which
    # measured faster than one pushforward of every block along all of them
    # (ReactiveKernels snag `ad-pullback-push-6f576925`).
    names = BRM._rk_layout_coordinate_names(backend)
    forward = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Forward, ad_backend))
    blocks = map(effects) do effect
        kernel = prepare_want(effect)
        owned = findall(k -> startswith(String(names[k]), string(effect, ".")), omega)
        directions = Tuple(begin
            v = zeros(Float64, length(u0))
            v[omega[j]] = 1.0
            v
        end for j in owned)
        pushforward = isempty(directions) ? nothing :
            Base.invokelatest(prepare_ad_pushforward, kernel, forward, directions, u0;
                active=:unconstrained)
        (; kernel, owned, directions, pushforward)
    end
    prior = BRM._rk_nlme_population_prior(backend, u0, ad_backend)
    pointwise = BRM._rk_pointwise_query(backend)
    (; values, nports=length(ports), pullback, prepare_pullback, blocks,
        nomega=length(omega), prior, pointwise)
end

BRM._rk_slap_values(k, u::Vector{Float64}) =
    _rk_slap_tuple(Base.invokelatest(k.values, u), k.nports)

_rk_slap_seed(cotangents) = length(cotangents) == 1 ? only(cotangents) : Tuple(cotangents)

function BRM._rk_slap_pullback(k, u::Vector{Float64}, cotangents)
    k.pullback[] === nothing && (k.pullback[] = k.prepare_pullback())
    Base.invokelatest(ad_pullback, k.pullback[], _rk_slap_seed(cotangents), u)
end

# Each block's draws, and per `omega` coordinate the tangent of every block's
# draws (zero for the blocks it does not own).
function BRM._rk_slap_effects(k, u::Vector{Float64})
    values = Vector{Any}(undef, length(k.blocks))
    tangents = [Vector{Any}(undef, length(k.blocks)) for _ in 1:k.nomega]
    for (b, block) in enumerate(k.blocks)
        if block.pushforward === nothing
            values[b] = Base.invokelatest(block.kernel, u)
        else
            values[b], t = Base.invokelatest(ad_value_and_pushforward,
                block.pushforward, block.directions, u)
            for (j, tj) in zip(block.owned, t)
                tangents[j][b] = tj
            end
        end
        for j in 1:k.nomega
            isassigned(tangents[j], b) || (tangents[j][b] = zero(values[b]))
        end
    end
    Tuple(values), map(Tuple, tangents)
end

BRM._rk_slap_pointwise(k, u::Vector{Float64}) = BRM._rk_pointwise_values(k.pointwise, u)

BRM._rk_slap_prior(k, u::Vector{Float64}) =
    BRM._rk_nlme_population_prior_value(k.prior, u)
BRM._rk_slap_prior_gradient(k, u::Vector{Float64}) =
    BRM._rk_nlme_population_prior_value_gradient(k.prior, u)

end
