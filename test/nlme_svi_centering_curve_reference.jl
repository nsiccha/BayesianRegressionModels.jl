# Independent Gaussian integration for svi_centering_curve_model on its [-1, 1]
# domain. Given its four packed scales (length, amplitude, noise and intercept
# scale) the model is Gaussian in its 73 coefficients, so their conditional is
# exact and only the scales are integrated numerically. The result is the
# posterior mean and SD of every coordinate, in `packed_names`' order.
module CenteringCurveReference

include(joinpath(@__DIR__, "fitting_packages.jl"))
using LinearAlgebra
using LogDensityProblems: dimension, logdensity_and_gradient
using LogExpFunctions: logsumexp
using QuadGK: gauss

function subject_blocks(data)
    frequency = (pi / 2) .* (1:5)
    basis = sin.((data.t .+ 1) .* frequency')
    subjects = unique(data.subject)
    blocks = map(subjects) do subject
        rows = findall(==(subject), data.subject)
        B, y = basis[rows, :], data.y[rows]
        # Weight-space sufficient statistics of the subject's rows.
        (; basis=B, y, gram=B' * B, ones=vec(sum(B; dims=1)), cross=B' * y,
           n=length(rows), total=sum(y))
    end
    return (; frequency, subjects, blocks)
end

# BRM's packed coordinate names in `posterior_moments`' order: the four scales
# (log(rho - floor), log amplitude, log noise, log tau), the population
# intercept, then per subject its five standard-normal HSGP weights and its
# standard-normal intercept offset (intercept tau * offset). BRM numbers the
# grouping's levels in sorted label order.
function packed_names(panel)
    level = Dict(subject => j for (j, subject) in enumerate(sort(panel.subjects)))
    names = ["hsgp_t.rho", "hsgp_t.sigma", "sigma", "b_subject.tau.1", "mu_Intercept"]
    for subject in panel.subjects
        j = level[subject]
        append!(names, ["hsgp_t.z.$j.$k" for k in eachindex(panel.frequency)])
        push!(names, "b_subject.z.$j.1")
    end
    return names
end

# The weights' spectral variances at x = log(rho - floor), and the restricted
# LogNormal(0,1) density of rho with its Jacobian for x.
function spectral_variance(panel, x)
    floor = (4 / pi) * sqrt(log(100) / (length(panel.frequency)^2 - 1))
    rho = floor + exp(x)
    variance = (rho * sqrt(2pi)) .* exp.(-rho^2 .* panel.frequency.^2 ./ 2)
    return variance, x - log(rho) - log(rho)^2 / 2
end

function spectrum(panel, x)
    variance, logprior = spectral_variance(panel, x)
    subjects = map(panel.blocks) do block
        # The five standard-normal weights give covariance B * Diagonal(variance) * B'.
        eig = eigen(Symmetric(block.basis * Diagonal(variance) * block.basis'))
        (; values=eig.values, intercept=vec(sum(eig.vectors; dims=1)), y=eig.vectors' * block.y)
    end
    return subjects, logprior
end

function subject_moments(subjects, amplitude, noise)
    return map(subjects) do subject
        # In this basis the within-subject covariance V is diagonal. Retain
        # log|V| and the three entries of [1 y]' inv(V) [1 y].
        logdet_v = a = b = c = 0.0
        for i in eachindex(subject.values)
            v = noise^2 + amplitude^2 * subject.values[i]
            one_i, y_i = subject.intercept[i], subject.y[i]
            logdet_v += log(v)
            a += one_i^2 / v
            b += one_i * y_i / v
            c += y_i^2 / v
        end
        (; logdet_v, a, b, c)
    end
end

function log_likelihood(moments, tau)
    # Integrate each N(0,tau²) subject intercept as a rank-one update, then
    # complete the square for the shared N(0,1) population intercept.
    precision, linear, residual, logdet_v = 1.0, 0.0, 0.0, 0.0
    for m in moments
        adjustment = 1 + tau^2 * m.a
        precision += m.a / adjustment
        linear += m.b / adjustment
        residual += m.c - tau^2 * m.b^2 / adjustment
        logdet_v += m.logdet_v + log(adjustment)
    end
    return -(logdet_v + log(precision) + residual - linear^2 / precision) / 2
end

# The three other scale priors, on their packed log coordinates, are
# LogNormal(0,1) for amplitude and tau, and Exponential(1) for noise.
other_logprior(log_amp, log_noise, log_tau) =
    -log_amp^2 / 2 + log_noise - exp(log_noise) - log_tau^2 / 2

function log_marginal(panel, scales)
    x, log_amp, log_noise, log_tau = scales
    subjects, prior = spectrum(panel, x)
    moments = subject_moments(subjects, exp(log_amp), exp(log_noise))
    return prior + other_logprior(log_amp, log_noise, log_tau) + log_likelihood(moments, exp(log_tau))
end

# The coefficients' Gaussian conditional at fixed scales, in weight space. A
# subject's rows are y ~ N(intercept + A u, noise²) with u its six
# standard-normal coefficients and A = [B * Diagonal(amplitude * sqrt.(variance))
# tau * 1]. Given the intercept, u has precision Q = I + A'A / noise² and mean
# a - intercept * c; the intercept's own N(0,1) conditional completes the
# square. Returns the conditional means and variances in `packed_names`' order
# after its four scales.
function coefficient_conditional(panel, scales)
    x, log_amp, log_noise, log_tau = scales
    variance, _ = spectral_variance(panel, x)
    d = exp(log_amp) .* sqrt.(variance)
    tau, noise2 = exp(log_tau), exp(2log_noise)
    K = length(d)
    parts = map(panel.blocks) do block
        Q = Matrix{Float64}(I, K + 1, K + 1)
        Q[1:K, 1:K] .+= (d .* block.gram .* d') ./ noise2
        Q[1:K, K+1] .= tau .* d .* block.ones ./ noise2
        Q[K+1, 1:K] .= view(Q, 1:K, K + 1)
        Q[K+1, K+1] += tau^2 * block.n / noise2
        factor = cholesky(Symmetric(Q))
        by = [d .* block.cross; tau * block.total] ./ noise2
        b1 = [d .* block.ones; tau * block.n] ./ noise2
        a, c = factor \ by, factor \ b1
        (; a, c, variance=diag(inv(factor)),
           precision=block.n / noise2 - dot(b1, c), linear=block.total / noise2 - dot(b1, a))
    end
    precision = 1 + sum(p -> p.precision, parts)
    linear = sum(p -> p.linear, parts)
    intercept = linear / precision
    mean = [intercept; reduce(vcat, [p.a .- intercept .* p.c for p in parts])]
    var = [1 / precision; reduce(vcat, [p.variance .+ p.c .^ 2 ./ precision for p in parts])]
    return (; mean, var)
end

const DEFAULT_BOUNDS = ((-34.0, 1.5), (-2.8, 2.4), (-1.9, -0.37), (-5.2, 1.8))

# Posterior mean and SD of every coordinate in `packed_names`' order. The
# scales are integrated with Gauss-Legendre quadrature on `bounds`, weighted by
# `log_marginal`; at each node the coefficients' conditional is exact
# (`coefficient_conditional`). Nodes more than `cutoff` nats below the largest
# carry at most exp(-cutoff) of its mass each and are left out of the
# coefficients' moments only.
function posterior_moments(panel; order=48, bounds=DEFAULT_BOUNDS, cutoff=30.0)
    x, wx = gauss(2order, bounds[1]...)
    amp, wa = gauss(order, bounds[2]...)
    noise, wn = gauss(order, bounds[3]...)
    tau, wt = gauss(order, bounds[4]...)
    tau_scale = exp.(tau)
    logmass = Array{Float64}(undef, length(x), length(amp), length(noise), length(tau))
    for i in eachindex(x)
        subjects, logprior = spectrum(panel, x[i])
        for j in eachindex(amp), k in eachindex(noise)
            moments = subject_moments(subjects, exp(amp[j]), exp(noise[k]))
            for l in eachindex(tau)
                logmass[i, j, k, l] = log(wx[i] * wa[j] * wn[k] * wt[l]) + logprior +
                    other_logprior(amp[j], noise[k], tau[l]) + log_likelihood(moments, tau_scale[l])
            end
        end
    end
    total, top = logsumexp(logmass), maximum(logmass)
    m1 = zeros(length(packed_names(panel)))
    m2 = zero(m1)
    for index in CartesianIndices(logmass)
        i, j, k, l = Tuple(index)
        w = exp(logmass[index] - total)
        scales = (x[i], amp[j], noise[k], tau[l])
        for (m, s) in enumerate(scales)
            m1[m] += w * s
            m2[m] += w * s^2
        end
        logmass[index] < top - cutoff && continue
        conditional = coefficient_conditional(panel, scales)
        @. m1[5:end] += w * conditional.mean
        @. m2[5:end] += w * (conditional.var + conditional.mean^2)
    end
    return (; mean=m1, sd=sqrt.(max.(m2 .- m1 .^ 2, 0.0)))
end

# Independently collapse BRM's emitted joint at fixed scales. The conditional
# log density is quadratic in all other coordinates, so unit gradient
# differences give its precision exactly (up to rounding). Returns the log
# marginal and the conditional means and variances of the coefficients, the
# packed coordinates `coefficients`.
function rk_conditional(problem, scale_indices, scales)
    u = zeros(dimension(problem))
    u[scale_indices] .= scales
    coefficients = setdiff(eachindex(u), scale_indices)
    lp, gradient = logdensity_and_gradient(problem, u)
    g = gradient[coefficients]
    precision = map(coefficients) do j
        shifted = copy(u)
        shifted[j] = 1
        g - last(logdensity_and_gradient(problem, shifted))[coefficients]
    end
    factor = cholesky(Symmetric(stack(precision)))
    mean = factor \ g
    return (; log_marginal=lp + (dot(g, mean) - logdet(factor)) / 2, coefficients,
        mean, var=diag(inv(factor)))
end

end
