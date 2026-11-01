# A carried Gaussian replaces the authored prior on the entire sampler
# coordinate vector. Its ordering must already match the current model.
# Coordinate identity/replay is the caller's responsibility, not inferred here.
struct _BRMSVILaplaceCarryPrior{C,F}
    carry::C
    conditional::F
    latent::Vector{Int}
    globals::Vector{Int}
    lower_bound::Float64
end

function _brm_svi_laplace_carry_prior(certificate,carry)
    latent,globals=certificate.latent,certificate.globals
    sort(vcat(latent,globals)) == collect(eachindex(carry.mu)) ||
        _brm_svi_carry_error("prior coordinates must match the full sampler partition")
    isempty(latent) && _brm_svi_carry_error("prior needs a nonempty conditional latent block")
    conditional=_brm_svi_carry_conditional(carry,latent,globals)
    lower=_brm_svi_carry_precision_lower_bound(conditional.field_state)
    _BRMSVILaplaceCarryPrior(carry,conditional,copy(latent),copy(globals),lower)
end

# Reuses the prepared query workspace, so the old/new targets must not execute
# concurrently. Counters and numerical diagnostics belong to the new target.
function _brm_svi_laplace_with_carry(t::_BRMSVILaplaceTarget,carry)
    length(carry.mu) == length(t.m.unit) || throw(DimensionMismatch("carried sampler coordinates"))
    prior=_brm_svi_laplace_carry_prior(t.certificate,carry)
    _BRMSVILaplaceTarget(t.m,t.certificate,t.ad,prior,t.options,
        copy(t.globals),deepcopy(t.units),zeros(Int,5),Dict{Symbol,Float64}())
end

function _brm_svi_laplace_prior_value_gradient(prior::_BRMSVILaplaceCarryPrior,ad,u)
    value,gradient=_rk_svi_laplace_value_gradient(ad,u,:likelihood)
    carried=_brm_svi_carry_density_gradient(prior.carry,u)
    value+carried.value,gradient+carried.gradient
end

function _brm_svi_laplace_carry_direction(prior::_BRMSVILaplaceCarryPrior,t,v)
    prior.latent == t.certificate.latent && prior.globals == t.globals ||
        _brm_svi_carry_error("prior and target coordinate partitions differ")
    length(v) == length(prior.latent) || throw(DimensionMismatch("carried latent direction"))
    direction=zeros(length(prior.carry.mu))
    direction[prior.latent]=v
    direction
end
function _brm_svi_laplace_prior_precision(prior::_BRMSVILaplaceCarryPrior,t,v)
    direction=_brm_svi_laplace_carry_direction(prior,t,v)
    _brm_svi_carry_precision(prior.carry,direction)[prior.latent]
end
function _brm_svi_laplace_joint_hvp(prior::_BRMSVILaplaceCarryPrior,t,u,v)
    direction=_brm_svi_laplace_carry_direction(prior,t,v)
    # Keep all output rows: the mixed global/latent prior curvature is needed
    # by implicit differentiation. Gaussian prior third derivatives vanish.
    _brm_svi_laplace_likelihood_hvp(t,u,v)-_brm_svi_carry_precision(prior.carry,direction)
end
_brm_svi_laplace_lower_bound(prior::_BRMSVILaplaceCarryPrior,t)=prior.lower_bound

# Initial unit priors are the literal Gaussian sampler priors certified by the
# plan. Small blocks only; never construct a covariance for all selected units.
function _brm_svi_laplace_initial_unit_priors(t)
    t.prior === nothing || _brm_svi_carry_error("a correlated carry requires auxiliary-conditional unit priors")
    c=t.certificate
    position=Dict(i=>j for (j,i) in enumerate(c.latent))
    map(c.unit_blocks) do indices
        1 <= length(indices) <= 16 || _brm_svi_carry_error("unit block exceeds fixed cap")
        all(i->haskey(position,i),indices) || _brm_svi_carry_error("unit coordinate is not integrated")
        p=[position[i] for i in indices]
        _brm_svi_carry_block(c.prior_mean[p],zeros(length(p),0),
            Matrix(Diagonal(inv.(c.prior_precision[p]))))
    end
end

"""Integrate small Gaussian unit blocks conditional on physical core and eta.

`priors` is ordered exactly as `t.certificate.unit_blocks`; each entry is an
auxiliary block `(a,B,F,...)` from the SAME carried prior, conditional on the
provided auxiliary `eta`. For an initial target, omitting both uses its literal
Gaussian unit priors. A correlated marginal prior must never be used as a local
unit prior. Conditional independence and quadratic likelihood are certified by
the model plan; this helper does not infer them from numerical Hessians.

Returns a full vector with unit entries replaced by conditional means, a vector
of small conditional covariance matrices, and counts of native derivative calls.
"""
function _brm_svi_laplace_unit_moments(t,u;priors=nothing,eta=nothing)
    length(u) == length(t.m.unit) || throw(DimensionMismatch("unit moment sampler point"))
    all(isfinite,u) || _brm_svi_carry_error("nonfinite unit moment point")
    units=t.certificate.unit_blocks
    if priors === nothing
        eta === nothing || _brm_svi_carry_error("auxiliary core needs explicit conditional priors")
        priors=_brm_svi_laplace_initial_unit_priors(t)
        eta=Float64[]
    else
        eta === nothing && _brm_svi_carry_error("conditional priors need an auxiliary core value")
    end
    length(priors) == length(units) || throw(DimensionMismatch("unit conditional prior count"))
    length(eta) <= 64 && all(isfinite,eta) || _brm_svi_carry_error("invalid auxiliary core value")
    isempty(units) && return (;means=copy(u),covariances=Matrix{Float64}[],
        calls=(;likelihood_gradients=0,colored_likelihood_hvps=0))
    width=maximum(length,units)
    1 <= width <= 16 || _brm_svi_carry_error("unit block exceeds fixed cap")
    flat=reduce(vcat,units;init=Int[])
    integrated=Set(t.certificate.latent)
    length(unique(flat)) == length(flat) && all(in(integrated),flat) ||
        _brm_svi_carry_error("unit blocks must be disjoint subsets of the integrated block")
    for (indices,b) in zip(units,priors)
        !isempty(indices) && length(b.a) == length(indices) &&
            size(b.B) == (length(indices),length(eta)) && size(b.F) == (length(indices),length(indices)) ||
            throw(DimensionMismatch("auxiliary-conditional unit block"))
    end
    t.counts[1]+=1
    _,gradient=_rk_svi_laplace_value_gradient(t.ad,u,:likelihood)
    hessians=[zeros(length(indices),length(indices)) for indices in units]
    for k in 1:width
        direction=zeros(length(u))
        for indices in units
            k <= length(indices) && (direction[indices[k]]=1.)
        end
        t.counts[2]+=1
        hv=_rk_svi_laplace_hvp(t.ad,u,direction)
        for (indices,H) in zip(units,hessians)
            k <= length(indices) && (H[:,k]=hv[indices])
        end
    end
    means=copy(u);covariances=Matrix{Float64}[]
    for (indices,H,b) in zip(units,hessians,priors)
        isapprox(H,H';atol=1e-10,rtol=1e-10) || _brm_svi_carry_error("unit likelihood Hessian is not symmetric")
        Q=b.F\Matrix{Float64}(I,length(indices),length(indices))-H
        F=cholesky(Symmetric(Q))
        score=gradient[indices]-b.F\(u[indices]-b.a-b.B*eta)
        means[indices]=u[indices]+F\score
        push!(covariances,Matrix(inv(F)))
    end
    all(isfinite,means) && all(C->all(isfinite,C),covariances) ||
        _brm_svi_carry_error("nonfinite conditional unit moments")
    (;means,covariances,calls=(;likelihood_gradients=1,colored_likelihood_hvps=width))
end
