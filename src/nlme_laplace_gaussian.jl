# Independent Gaussian vectors stay in their authored packed coordinates.
# RK's prior at z=0 retains the normalization; the sparse engine adds
#   log p(z|phi) - log p(0|phi) = sum(P * z * (mu - z/2)).
# Thus nonzero means and outer-dependent scales need no coordinate transport.
function _rk_spl_value_reads end

function _brm_spl_gaussian_prior_reads(backend, u0)
    records = _rk_coordinate_records(backend.plan)
    ports = Tuple(s for r in records if r.kind === :vector && get(r, :gaussian, nothing) !== nothing
        for s in (r.gaussian.mean, r.gaussian.scale))
    isempty(ports) ? Int[] : _rk_spl_value_reads(backend, ports, u0)
end

# A Gaussian law's parameters are conditional outer quantities. If a declared
# law uses another candidate's draws, retain that supplier as sampled, even
# when the dependent vector itself is sampled. This check is structural and
# precedes derivative preparation, like the observation-width policy.
function _brm_spl_gaussian_outer(backend, candidates)
    isempty(candidates) && return Set{Symbol}()
    layout = _brm_spl_layout(backend, unique(t.grouping for t in candidates);
        declarations=Set(t.declaration for t in candidates))
    reads = Set(_brm_spl_gaussian_prior_reads(backend, zeros(length(layout.names))))
    Set(b.declaration for b in layout.blocks if any(i -> layout.integrated[i] in reads, b.z))
end

function _brm_spl_gaussian_prior(p, records, u0)
    reads = _brm_spl_gaussian_prior_reads(p.backend, u0)
    isempty(intersect(reads, p.integrated)) || _brm_nlme_error(
        "brm_laplace_problem: declared Gaussian means and scales must depend only " *
        "on outer coordinates; sample the terms that supply their prior parameters")
    vectors = [r for r in records if r.kind === :vector]
    isempty(vectors) && return nothing
    ports = Tuple(s for r in vectors for s in (r.gaussian.mean, r.gaussian.scale))
    blocks = [(; z=vec(only(b for b in p.blocks if b.declaration === r.declaration).z),
        meanport=2j-1, scaleport=2j) for (j, r) in enumerate(vectors)]
    read = _rk_value_reader(p.backend, ports)
    pullback = _rk_value_pullback(p.backend, ports, u0)
    _BRMSPLGaussianPrior(blocks, ports, read, pullback, zeros(length(p.integrated)),
        ones(length(p.integrated)), Ref{Any}(nothing))
end

_brm_spl_gaussian_element(x::Number, i) = x
_brm_spl_gaussian_element(x, i) = x[i]

function _brm_spl_gaussian_parameters!(p, u)
    g = p.gaussian
    g === nothing && return nothing
    values = Tuple(g.read(u))
    g.values[] = values
    for b in g.blocks
        mu, sigma = values[b.meanport], values[b.scaleport]
        _brm_spl_gaussian_parameters!(g.mean, g.precision, b.z, mu, sigma)
    end
    nothing
end

function _brm_spl_gaussian_parameters!(means, precision, z, mu, sigma)
    for x in (mu, sigma)
        x isa Number || length(x) == length(z) || throw(DimensionMismatch(
            "declared Gaussian prior argument has $(length(x)) values for $(length(z)) draws"))
    end
    for (j, i) in enumerate(z)
        means[i] = _brm_spl_gaussian_element(mu, j)
        sd = _brm_spl_gaussian_element(sigma, j)
        precision[i] = sd > 0 ? inv(sd^2) : NaN
    end
    nothing
end

@inline _brm_spl_gaussian_parameters(p, i) = p.gaussian === nothing ? (0.0, 1.0) :
    (p.gaussian.mean[i], p.gaussian.precision[i])

# Differentiate the prior difference, the diagonal log determinant and the
# mode's implicit derivative. S is the selected inverse of the full precision;
# v = H^-1 * (1/2 trace(S * likelihood''')) is shared with the likelihood path.
# Scalar prior arguments accumulate all coordinates before one RK pullback.
function _brm_spl_gaussian_gradient!(gradient, p, state, S, v, active)
    g = p.gaussian
    g === nothing && return gradient
    values = g.values[]
    seeds = Any[x isa Number ? 0.0 : zeros(length(x)) for x in values]
    components = active === nothing ? nothing : Set(p.term_component[t]
        for t in eachindex(active) if active[t])
    for b in g.blocks
        dm = zeros(length(b.z)); ds = zeros(length(b.z))
        for (j, i) in enumerate(b.z)
            components === nothing || p.component[i] in components || continue
            mu, precision = g.mean[i], g.precision[i]
            z = state.z[i]
            dm[j] = precision * (z + v[i])
            ds[j] = (z^2 - 2mu*z + S[p.diagonal[i]] + 2(z-mu)*v[i]) * precision * sqrt(precision)
        end
        seeds[b.meanport] = values[b.meanport] isa Number ? sum(dm) : dm
        seeds[b.scaleport] = values[b.scaleport] isa Number ? sum(ds) : ds
    end
    gradient .+= g.pullback(state.uA, Tuple(seeds))[p.population]
    gradient
end
