# The examples' sampling call: WarmupHMC's quorum sampler over independent
# adaptive chains. `quorum_fit(rng, lpdf; n_draws, kwargs...)` runs `n_chains`
# chains of `lpdf` (WarmupHMC deep-copies it per chain), seeded from `rng`, and
# stops once `div(3 * n_chains, 4)` of them hold their draws (Bruno's 75% rule:
# 16 chains -> 12, 4 -> 3). A chain still warming up then, such as one whose step
# size collapsed in its first window, finishes its current window and is left
# out instead of holding the run or entering the draws. WarmupHMC warns on every
# call that keeping the faster chains can bias inference; the outcome names the
# chains it left out (`completion`).
#
# Returns the fields the examples read from a single-chain fit:
# `posterior_position` (coordinates x draws, the admitted chains side by side,
# model frame) and `n_divergent_samples` (their sum), plus `completion`.
using Random: AbstractRNG, Xoshiro
using WarmupHMC: completion_warmup_mcmc

function quorum_fit(rng::AbstractRNG, lpdf; n_chains=4, init=missing, kwargs...)
    rngs = [Xoshiro(rand(rng, UInt64)) for _ in 1:n_chains]
    # A position vector is one start for every chain (WarmupHMC reads any vector
    # `init` as one entry per chain).
    inits = init isa AbstractVector{<:Real} ? [copy(init) for _ in 1:n_chains] : init
    out = completion_warmup_mcmc(rngs, lpdf; init=inits,
                                 min_completed=max(1, div(3 * n_chains, 4)), kwargs...)
    omitted = out.completion.omitted_chain_indices
    isempty(omitted) || println("QUORUM: left out chain(s) ", omitted, " of ", n_chains)
    (; posterior_position=reduce(hcat, (r.posterior_position for r in out.results)),
       n_divergent_samples=sum(r.n_divergent_samples for r in out.results),
       completion=out.completion)
end
