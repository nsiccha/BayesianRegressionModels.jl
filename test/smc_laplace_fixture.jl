using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, ADTypes, DifferentiationInterface, SequentialMonteCarlo, WarmupHMC
using Random, LogDensityProblems, LinearAlgebra, Distributions, Serialization

const BRM = BayesianRegressionModels
const LSMC_AD = AutoEnzyme(; mode=Enzyme.Reverse)
const LSMC_DATA = (; g=repeat(["a", "b", "c"], inner=2),
    y=[0.3, 0.9, -0.5, 0.1, 1.1, 0.7])
const LSMC_OPTIONS = SMCOptions(; moves=2, stepsize=0.45, max_tree_depth=4,
    history_limit=32)

lsmc_model(data) = @brm data begin
    mu ~ 1 + (1 | g)
    effect(mu, Intercept) ~ Normal(0, 1)
    sd(mu, g) = 0.7
    y ~ Normal(mu, 0.8)
end

lsmc_scale_model(data) = @brm data begin
    mu ~ 1 + (1 | g)
    effect(mu, Intercept) ~ Normal(0, 1)
    sd(mu, g) ~ LogNormal(log(0.7), 0.2)
    y ~ Normal(mu, 0.8)
end

function lsmc_problem(kind=:sparse; data=LSMC_DATA,
        rk=RKBRMI(kind in (:generic, :structured) ? lsmc_scale_model(data) : lsmc_model(data)))
    kind === :generic && return brm_laplace_problem(rk; ad_backend=LSMC_AD, structured=false)
    kind === :structured && return brm_laplace_problem(rk; ad_backend=LSMC_AD, structured=true)
    kind === :sparse && return brm_laplace_problem(rk; ad_backend=LSMC_AD, integrate=(:g,))
    kind === :sampled || error("unknown fixture kind")
    plan = brm_laplace_plan(rk; ad_backend=LSMC_AD, integrate=(), sample=(:g,), draws=8)
    brm_laplace_problem(rk; ad_backend=LSMC_AD, integrate=plan)
end

# Independent finite Gaussian integrals, in the model's packed coordinate order.
# The design is obtained by evaluating the authored predictor at basis points;
# neither the Laplace solver nor the SMC output enters this oracle.
function lsmc_oracle(rk, data=LSMC_DATA; quadrature=64)
    d = length(BRM._rk_bound_coordinate_names(rk))
    values = rk_draw_values(rk, hcat(zeros(d), Matrix{Float64}(I,d,d)), (:mu,)).mu
    offset = first(values)
    X = hcat([value - offset for value in values[2:end]]...)
    scale = findfirst(n -> occursin(".tau", String(n)), BRM._rk_bound_coordinate_names(rk))
    if scale !== nothing
        # Conditional on log(scale), the intercept and standardized effects
        # have an exact Gaussian posterior. Integrate the remaining scalar
        # against its Normal prior by independent Gauss-Hermite quadrature.
        indices = setdiff(1:d, [scale])
        beta = only(filter(j -> all(==(1.0), X[:,j]), indices))
        effects = setdiff(indices, [beta])
        eigenrule = eigen(SymTridiagonal(zeros(quadrature), sqrt.(1:quadrature-1)))
        nodes = log(0.7) .+ 0.2 .* eigenrule.values
        prior_weights = eigenrule.vectors[1,:].^2
        means = Vector{Vector{Float64}}()
        seconds = Vector{Matrix{Float64}}()
        weights = Float64[]
        for (t,w) in zip(nodes, prior_weights)
            A = copy(X[:,indices])
            A[:, findall(in(effects), indices)] .*= exp(t)
            C = inv(Symmetric(I + A'A / 0.8^2))
            m = C * A' * data.y / 0.8^2
            fullmean = zeros(d); fullmean[indices] = m; fullmean[scale] = t
            second = fullmean * fullmean'; second[indices,indices] += C
            push!(means, fullmean); push!(seconds, second)
            push!(weights, w * pdf(MvNormal(zeros(length(data.y)),
                Symmetric(0.8^2 * I + A*A')), data.y))
        end
        evidence = sum(weights); weights ./= evidence
        m = sum(means .* weights)
        C = sum(seconds .* weights) - m*m'
        marginal = function (population, phi)
            u = zeros(d); u[population] = phi
            covariance = Symmetric(0.8^2 * I + exp(2u[scale]) * X[:,effects] * X[:,effects]')
            logpdf(Normal(log(0.7), 0.2), u[scale]) + logpdf(Normal(), u[beta]) +
                logpdf(MvNormal(fill(u[beta], length(data.y)), covariance), data.y)
        end
        return (; mean=m, covariance=C, logz=log(evidence), marginal)
    end
    C = inv(Symmetric(I + X'X / 0.8^2))
    m = C * X' * (data.y - offset) / 0.8^2
    logz = logpdf(MvNormal(offset, Symmetric(0.8^2 * I + X * X')), data.y)
    marginal = (population, phi) -> logz + logpdf(MvNormal(m[population], C[population,population]), phi)
    (; mean=m, covariance=C, logz, marginal)
end

function lsmc_roundtrip(x)
    io = IOBuffer(); serialize(io, x); seekstart(io); deserialize(io)
end

# Checkpoint records include immutable wrappers around arrays, for which the
# default isequal compares array identity. Compare every serialized data field
# instead, including particle arrays, RNG state, tuning and diagnostic history.
function lsmc_equal(a, b)
    typeof(a) === typeof(b) || return false
    if a isa AbstractArray
        return axes(a) == axes(b) && all(lsmc_equal(x,y) for (x,y) in zip(a,b))
    elseif a isa AbstractDict
        return keys(a) == keys(b) && all(lsmc_equal(a[k], b[k]) for k in keys(a))
    elseif a isa Union{Number,AbstractString,Symbol,Type,Module,Function,Nothing,Missing}
        return isequal(a, b)
    end
    all(lsmc_equal(getfield(a,n), getfield(b,n)) for n in fieldnames(typeof(a)))
end

function lsmc_mismatch(a, b, path="checkpoint")
    lsmc_equal(a,b) && return nothing
    typeof(a) === typeof(b) || return (; path, expected_type=typeof(a), actual_type=typeof(b))
    if a isa AbstractArray
        axes(a) == axes(b) || return (; path, expected_axes=axes(a), actual_axes=axes(b))
        for i in eachindex(a,b)
            result = lsmc_mismatch(a[i],b[i],"$path[$i]")
            result === nothing || return result
        end
    elseif a isa AbstractDict
        keys(a) == keys(b) || return (; path, expected_keys=keys(a), actual_keys=keys(b))
        for k in keys(a)
            result = lsmc_mismatch(a[k],b[k],"$path[$k]")
            result === nothing || return result
        end
    elseif a isa Union{Number,AbstractString,Symbol,Type,Module,Function,Nothing,Missing}
        return (; path, expected=a, actual=b)
    else
        for n in fieldnames(typeof(a))
            result = lsmc_mismatch(getfield(a,n),getfield(b,n),"$path.$n")
            result === nothing || return result
        end
    end
    nothing
end
