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

# The sampler density's Hessian-vector products, prepared on first use. Only
# dense η Hessian blocks (`_brm_nlme_sampler_hessian_blocks`) and the per-unit
# Laplace read them, so an NLME fit whose estimators take the η curvature from
# gradients never compiles or holds them. Before ReactiveKernels 0.1.12 the
# preparation held 8n² bytes for n packed coordinates, 200 GB for a model of
# 157,808 unit effects: DifferentiationInterface's reverse-over-forward
# preparation also prepares a forward-mode gradient, whose basis is n one-hot
# vectors of length n. From 0.1.12 RK prepares it in memory linear in n.
BRM._rk_eta_hvp_query(backend::BRM.RKBRMI, unit::Vector{Float64}, ad_backend) =
    (; backend, unit, ad_backend, cached=Ref{Any}(nothing), lock=ReentrantLock())

function _rk_eta_hvp_prepared(state)
    lock(state.lock) do
        state.cached[] === nothing || return state.cached[]
        backend, ad_backend = state.backend, state.ad_backend
        rkext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
        second_order = SecondOrder(
            AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Reverse, ad_backend)),
            AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Forward, ad_backend)))
        state.cached[] = prepare_query_ad(prepare_ad_hvp, backend.model,
            rkext._rk_translated(backend), :sampler,
            second_order, (zeros(Float64, length(state.unit)),), state.unit)
    end
end

BRM._brm_rk_default_ad(caller, ::BRM.RKBRMI) = AutoEnzyme(; mode=Enzyme.Reverse)

BRM._rk_eta_hvp(state, direction::Vector{Float64}, u::Vector{Float64}) =
    only(ad_hvp(_rk_eta_hvp_prepared(state), (direction,), u))

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
    # A model whose observations are all `@plate` cells has no row parameters.
    values = isempty(ports) ? nothing : prepare_want(ports)
    v0 = values === nothing ? () :
        _rk_slap_tuple(Base.invokelatest(values, u0), length(ports))
    # One reverse pullback of every parameter, prepared on first use: a model
    # whose parameters are affine in the population coordinates never needs
    # it. Its seed holds one cotangent per port (untupled for a single port).
    reverse = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Reverse, ad_backend))
    pullback = Ref{Any}(nothing)
    seeds = map(value -> value isa Number ? 0.0 : zeros(Float64, size(value)), v0)
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
    # The parameters' forward derivatives along chosen `omega` coordinates (the
    # sparse Laplace's C, probed through the parameters themselves), one
    # preparation per set of coordinates, each on first use: a block's probes
    # need its own coordinates' directions only.
    omega_directions = map(o -> (v = zeros(Float64, length(u0)); v[o] = 1.0; v), omega)
    pushforwards = Dict{Vector{Int},Any}()
    prepare_pushforward(js) = Base.invokelatest(prepare_ad_pushforward, values, forward,
        Tuple(omega_directions[js]), u0; active=:unconstrained)
    (; values, nports=length(ports), pullback, prepare_pullback, blocks,
        nomega=length(omega), prior, pointwise, omega_directions, pushforwards,
        prepare_pushforward)
end

# The parameters at a packed point and their forward derivative there along
# each of the `omega` coordinates `js` (one tuple of ports per coordinate).
function BRM._rk_slap_values_and_tangents(k, u::Vector{Float64}, js::Vector{Int})
    (isempty(js) || k.values === nothing) &&
        return BRM._rk_slap_values(k, u), ntuple(_ -> (), length(js))
    prepared = get!(() -> k.prepare_pushforward(js), k.pushforwards, js)
    value, tangents = Base.invokelatest(ad_value_and_pushforward, prepared,
        Tuple(k.omega_directions[js]), u)
    _rk_slap_tuple(value, k.nports), map(t -> _rk_slap_tuple(t, k.nports), tangents)
end

BRM._rk_slap_values(k, u::Vector{Float64}) = k.values === nothing ? () :
    _rk_slap_tuple(Base.invokelatest(k.values, u), k.nports)

_rk_slap_seed(cotangents) = length(cotangents) == 1 ? only(cotangents) : Tuple(cotangents)

# The coordinate transport's standardized completions
# (src/coordinate_transport.jl): one reverse pullback of the named values at a
# packed point, taking one cotangent per name shaped as its value. A value the
# bound data alone produce (a law of data columns) is a constant whose
# pullback is zero, so only the values that read the point are differentiated.
function BRM._rk_value_pullback(backend::BRM.RKBRMI, names::Tuple{Vararg{Symbol}},
        u0::Vector{Float64})
    bound = BRM.rk_bound_data(backend)
    live = findall(name -> _rk_reads_point(backend, bound, name), collect(names))
    isempty(live) && return (u::Vector{Float64}, cotangents::Tuple) -> zeros(Float64, length(u))
    values = Base.invokelatest(ReactiveKernels.prepare, backend.model.spec;
        have=(:unconstrained, keys(bound)...), want=names[live], bound)
    v0 = _rk_slap_tuple(Base.invokelatest(values, u0), length(live))
    seeds = map(value -> value isa Number ? 0.0 : zeros(Float64, size(value)), v0)
    reverse = AutoEnzyme(; mode=Enzyme.Reverse)
    prepared = Base.invokelatest(prepare_ad_pullback, values, reverse,
        _rk_slap_seed(seeds), u0; active=:unconstrained)
    function pullback(u::Vector{Float64}, cotangents::Tuple)
        seed = map(i -> (c = cotangents[i]; c isa Number ? Float64(c) : Array{Float64}(c)),
            Tuple(live))
        Base.invokelatest(ad_pullback, prepared, _rk_slap_seed(seed), u)
    end
end

# Whether a graph value reads the packed point: planned from the bound data
# alone, a value that needs the point is not producible (`PlanningError`); the
# same value plans with the point, so no other cause remains.
function _rk_reads_point(backend::BRM.RKBRMI, bound, name::Symbol)
    try
        Base.invokelatest(ReactiveKernels.plan, backend.model.spec;
            have=Tuple(keys(bound)), want=(name,))
        false
    catch err
        err isa PlanningError || rethrow()
        true
    end
end

function BRM._rk_slap_pullback(k, u::Vector{Float64}, cotangents)
    # no row parameters (every observation an `@plate` cell): nothing to pull back
    k.values === nothing && return zeros(length(u))
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

# The sparse Laplace's `@plate` cells (src/nlme_laplace_sparse.jl): one cell of
# `observations` differentiated over a compact selection of partitioned
# coordinates, conditional on the shared ones. RK gives every compact operator
# its own shared cache. Newton's HVP therefore runs in its own pass, and the
# third-order operator owns the additive accumulators, whose
# shared pullback restores the mixed population derivatives. Reference identity
# keeps the nested RK records out of every cell group.
mutable struct _SPLCellAD{H,N,A,P}
    const hvp::H
    const newton::N
    const accumulators::A
    const pullback::P
end

# Newton's HVP order for an ordinary cell. Forward over reverse allocates less,
# but under `partitioned` Enzyme can refuse it at call time.
_spl_newton_order(forward, reverse) = SecondOrder(reverse, forward)

function BRM._rk_spl_cell_ad(backend::BRM.RKBRMI, observations::Tuple,
        base0::Vector{Float64}, indices0::Vector{Int}, ad_backend, K::Int; partitioned)
    rkext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    built, plan = backend.model, rkext._rk_translated(backend)
    m = length(indices0)
    forward = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Forward, ad_backend))
    reverse = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Reverse, ad_backend))
    second = _spl_newton_order(forward, reverse)
    hvp = Base.invokelatest(prepare_compact_cell_ad, prepare_ad_hvp, built, plan,
        observations, base0, indices0, second,
        ntuple(_ -> zeros(Float64, m), K); partitioned, locality=:error)
    # RK's accumulators take reverse over forward for order 1 and the supported
    # reverse over forward over forward for order 2.
    second = SecondOrder(reverse, forward)
    third = SecondOrder(second, forward)
    owner = Base.invokelatest(prepare_compact_cell_ad, prepare_ad_third, built, plan,
        observations, base0, indices0, third, (zeros(Float64, m),);
        partitioned, locality=:error)
    accumulators = (
        Base.invokelatest(prepare_cell_accumulator, owner, reverse, base0; order=0),
        Base.invokelatest(prepare_cell_accumulator, owner, second, base0; order=1),
        Base.invokelatest(prepare_cell_accumulator, owner, third, base0; order=2),
    )
    _SPLCellAD(hvp, Base.invokelatest(prepare_cell_pullback, hvp, reverse, base0),
        accumulators, Base.invokelatest(prepare_cell_pullback, owner, reverse, base0))
end

function BRM._rk_spl_cell_newton(f, ad::_SPLCellAD, base)
    result = nothing
    cell_pullback!(zeros(length(base)), ad.newton, base) do batch
        result = f(batch)
    end
    result
end

BRM._rk_spl_cell_block(ad::_SPLCellAD, directions::Tuple, base::Vector{Float64},
    indices::Vector{Int}, i::Int; context) =
    ad_gradient_and_hvp(ad.hvp, directions, context, base[indices], indices, i)

function BRM._rk_spl_cell_pass(f, ad::_SPLCellAD, base)
    packed = zeros(length(base))
    result = nothing
    cell_pullback!(packed, ad.pullback, base) do batch
        result = f(batch)
    end
    result, packed
end

BRM._rk_spl_cell_accumulate(batch, ad::_SPLCellAD, order, direction, weight, base,
        indices, cell) =
    accumulate_cell!(batch, ad.accumulators[order + 1], direction, weight,
        base[indices], indices, cell)

# Diagnostics perturb the compact active values inside one pass and discard its
# accumulated gradient. Order zero's weighted contraction is the cell's value.
BRM._rk_spl_cell_value(ad::_SPLCellAD, base, indices, cell; context) =
    first(BRM._rk_spl_cell_accumulate(context, ad, 0, nothing, 1.0, base, indices, cell))

# A factor's complete derivative block is small: the outer sums and direct
# reads, or one summand's own reads. Operators are shared only across factors
# with the same selection length, never across concurrent problem copies.
# Give the shared preparation reference identity. Its nested immutable RK
# records otherwise get embedded into every term and repeatedly boxed when
# walking heterogeneous cell states (over 3 KB per record in the sum fixture).
mutable struct _SPLFactorAD{H,T,D,G,A,P}
    const hvp::H
    const third::T
    const dirs::D
    const r::Int
    const outer::Bool
    const gradient::G
    const accumulators::A
    const pullback::P
end

function BRM._rk_spl_factor_ad(q, part, base, indices, ad_backend; cell0=1, term0=1, K=nothing)
    r = part === :outer ? length(q.terms) : 0
    m = length(indices)
    shape(v) = part === :outer ? (v[1:r], v[(r + 1):end]) : v
    K = K === nothing ? r + m : K
    dirs = ntuple(_ -> shape(zeros(r + m)), K)
    forward = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Forward, ad_backend))
    reverse = AutoEnzyme(; mode=BRM._brm_enzyme_mode(Enzyme.Reverse, ad_backend))
    # An outer factor's active point is the `(sums, active)` tuple, and a
    # forward-over-reverse HVP over a tuple selector fails inside
    # DifferentiationInterface (reactivekernels-use §7a2 Boundaries).
    second = SecondOrder(reverse, forward)
    hvp = K == 0 ? nothing : Base.invokelatest(prepare_cell_factor_ad, prepare_ad_hvp, q.factors,
        part, base, indices, second, dirs; cell0, term0)
    third = Base.invokelatest(prepare_cell_factor_ad, prepare_ad_third, q.factors,
        part, base, indices, SecondOrder(second, forward), (shape(zeros(r + m)),);
        cell0, term0)
    gradient = Base.invokelatest(prepare_cell_factor_ad, prepare_ad, q.factors,
        part, base, indices, reverse; cell0, term0)
    cached = hasproperty(q, :cached) && q.cached
    accumulators = cached ? (
        Base.invokelatest(prepare_cell_accumulator, gradient, reverse, base; order=0),
        Base.invokelatest(prepare_cell_accumulator, third, second, base; order=1),
        Base.invokelatest(prepare_cell_accumulator, third, SecondOrder(second, forward), base; order=2),
    ) : nothing
    pullback = cached ? Base.invokelatest(prepare_cell_pullback, q.factors, reverse, base) : nothing
    _SPLFactorAD(hvp, third, dirs, r, part === :outer, gradient, accumulators, pullback)
end

_spl_factor_flat(v::Tuple) = vcat(v...)
_spl_factor_flat(v::AbstractVector) = v
_spl_factor_point(ad, base, indices, cell, sums, term, context=base) = ad.outer ?
    (context, sums, base[indices], indices, cell) :
    (context, base[indices], indices, cell, term)

function BRM._rk_spl_factor_block(ad, base, indices, cell; sums=nothing, term=1,
        columns=collect(1:(ad.r + length(indices))), context=base)
    point = _spl_factor_point(ad, base, indices, cell, sums, term, context)
    if isempty(columns)
        _, g = ad_value_and_gradient(ad.gradient, point...)
        g = _spl_factor_flat(g)
        return g, zeros(length(g), 0)
    end
    function set!(v, k, value)
        if ad.outer
            k <= ad.r ? (v[1][k] = value) : (v[2][k - ad.r] = value)
        else
            v[k] = value
        end
    end
    for (v, k) in zip(ad.dirs, columns)
        set!(v, k, 1.0)
    end
    try
        g, cols = ad_gradient_and_hvp(ad.hvp, ad.dirs, point...)
        # A reused plan can restrict the integrated set without preparing AD
        # again. Its original direction tuple then has a zero unused suffix.
        _spl_factor_flat(g), hcat(map(_spl_factor_flat, cols[1:length(columns)])...)
    finally
        for (v, k) in zip(ad.dirs, columns)
            set!(v, k, 0.0)
        end
    end
end

function BRM._rk_spl_factor_third(ad, direction, base, indices, cell;
        sums=nothing, term=1, context=base)
    d = ad.outer ? (direction[1:ad.r], direction[(ad.r + 1):end]) : direction
    point = _spl_factor_point(ad, base, indices, cell, sums, term, context)
    g, (hv,), (tv,) = ad_gradient_hvp_and_third(ad.third, (d,), point...)
    _spl_factor_flat(g), _spl_factor_flat(hv), _spl_factor_flat(tv)
end

# A whole observation shares one cache. Enter it once per sweep, then pass the
# batch to every local operation, so validation never scans shared coordinates
# once per cell. The returned packed cotangent includes the prefix pullback.
function BRM._rk_spl_cell_pass(f, ad::_SPLFactorAD, base)
    packed = zeros(length(base))
    result = nothing
    cell_pullback!(packed, ad.pullback, base) do batch
        result = f(batch)
    end
    result, packed
end

function BRM._rk_spl_factor_accumulate(batch, ad::_SPLFactorAD, order, direction,
        weight, base, indices, cell; sums=nothing, term=1)
    d = direction === nothing ? nothing : ad.outer ?
        (direction[1:ad.r], direction[(ad.r + 1):end]) : direction
    tail = ad.outer ? (sums, base[indices], indices, cell) :
        (base[indices], indices, cell, term)
    value, local_gradient = accumulate_cell!(batch, ad.accumulators[order + 1],
        d, weight, tail...)
    value, _spl_factor_flat(local_gradient)
end

function BRM._rk_spl_factor_value(ad::_SPLFactorAD, base, indices, cell;
        sums=nothing, term=1, context=base)
    value, _ = ad_value_and_gradient(ad.gradient,
        _spl_factor_point(ad, base, indices, cell, sums, term, context)...)
    value
end

include("rk_svi_laplace.jl")

end
