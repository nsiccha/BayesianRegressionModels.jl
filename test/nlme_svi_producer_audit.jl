# Producer-owned acceptance helper. BRM internals stay inside its validation
# harness; the returned artifact contains data only, never prepared queries.
using BayesianRegressionModels, LogDensityProblems, LinearAlgebra

function svi_producer_audit(t, points; atol=2e-8, rtol=2e-9)
    brm = BayesianRegressionModels
    m = t.m
    estimated = sort!(vcat(t.globals, reduce(vcat, t.units; init=Int[])))
    @assert allunique(estimated)
    @assert length(t.ownership.owner) == length(m.unit)
    factors = t.ownership.factors
    factor_inventory = map(factors.events, factors.owned, factors.support) do event, owned, support
        owners = unique(filter(!iszero, t.ownership.owner[union(owned, support)]))
        @assert length(owners) <= 1
        (; kind=event.kind, site=event.site, entry=event.entry,
            owned=copy(owned), support=copy(support), unit=isempty(owners) ? 0 : only(owners))
    end
    likelihood_inventory = NamedTuple[]
    for (name, owners) in t.ownership.attribution
        for (cell, unit) in enumerate(owners)
            support = m.view.factor_supports[name][cell]
            support = support === :whole ? :whole : copy(support)
            reads = support === :whole ? eachindex(m.unit) : support
            @assert all(j -> t.ownership.owner[j] in (0, unit), reads)
            push!(likelihood_inventory, (; name, cell, unit, support))
        end
    end
    sampler = brm._brm_svi_unit_sampler(:work, t.ownership.work)
    probabilities = sampler.probabilities
    full = rk_logdensity_problem(m.backend; ad_backend=m.population_prior.ad_backend)
    checks = map(points) do supplied
        @assert length(supplied) == length(m.unit)
        point = copy(m.unit)
        point[estimated] .= supplied[estimated]
        g = point[t.globals]
        value, global_gradient = brm._brm_svi_global_term(t, g)
        gradient = zeros(length(point))
        gradient[t.globals] .= global_gradient
        weighted_value, weighted_gradient = value, copy(gradient)
        terms = zeros(length(t.units))
        for (i, block) in enumerate(t.units)
            gg, gz = zeros(length(g)), zeros(length(block))
            term = brm._brm_svi_unit_term!(gg, gz, t, i, g, point[block])
            terms[i] = term
            value += term
            gradient[t.globals] .+= gg
            gradient[block] .+= gz
            # Exact expectation of one replacement draw, including its 1/p
            # objective/global-gradient weight. Local gradients are unweighted.
            p = probabilities[i]
            weighted_value += p * (term / p)
            weighted_gradient[t.globals] .+= p .* (gg ./ p)
            weighted_gradient[block] .+= gz
        end
        expected, native = LogDensityProblems.logdensity_and_gradient(full, point)
        @assert isapprox(value, expected; atol, rtol)
        @assert isapprox(gradient[estimated], native[estimated]; atol, rtol)
        @assert isapprox(weighted_value, expected; atol, rtol)
        @assert isapprox(weighted_gradient[estimated], native[estimated]; atol, rtol)
        (; point, native_value=expected, split_value=value, weighted_value,
            gradient_max_error=maximum(abs, gradient[estimated] - native[estimated]; init=0.0),
            weighted_gradient_max_error=maximum(abs,
                weighted_gradient[estimated] - native[estimated]; init=0.0), terms)
    end
    (; coordinates=copy(m.view.coordinates), globals=copy(t.globals),
        units=copy.(t.units), owner=copy(t.ownership.owner),
        unit_levels=deepcopy(t.ownership.levels), subject_unit=copy(t.ownership.subject_unit),
        separator=deepcopy(t.ownership.separator), work=copy(t.ownership.work),
        probabilities=copy(probabilities), factor_inventory, likelihood_inventory, checks,
        prior_retained_bytes=Base.summarysize(t.prior),
        target_retained_bytes=Base.summarysize(t), atol, rtol)
end
