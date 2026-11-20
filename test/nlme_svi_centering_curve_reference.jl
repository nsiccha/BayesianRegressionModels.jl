# Independent Gaussian integration for svi_centering_curve_model on its [-1, 1]
# domain. Only the packed length, amplitude, noise and intercept scales remain.
module CenteringCurveReference

include(joinpath(@__DIR__, "fitting_packages.jl"))
using LinearAlgebra
using LogDensityProblems: dimension, logdensity_and_gradient
using LogExpFunctions: logsumexp
using QuadGK: gauss

function subject_blocks(data)
    frequency = (pi / 2) .* (1:5)
    basis = sin.((data.t .+ 1) .* frequency')
    blocks = map(unique(data.subject)) do subject
        rows = findall(==(subject), data.subject)
        (; basis=basis[rows, :], y=data.y[rows])
    end
    return (; frequency, blocks)
end

function spectrum(panel, x)
    floor = (4 / pi) * sqrt(log(100) / (length(panel.frequency)^2 - 1))
    rho = floor + exp(x)
    variance = (rho * sqrt(2pi)) .* exp.(-rho^2 .* panel.frequency.^2 ./ 2)
    subjects = map(panel.blocks) do block
        # The five standard-normal weights give covariance B * Diagonal(variance) * B'.
        eig = eigen(Symmetric(block.basis * Diagonal(variance) * block.basis'))
        (; values=eig.values, intercept=vec(sum(eig.vectors; dims=1)), y=eig.vectors' * block.y)
    end
    # Restricted LogNormal(0,1) density and Jacobian for x = log(rho - floor).
    return subjects, x - log(rho) - log(rho)^2 / 2
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

function length_moments(panel; order=48, bounds=((-34.0, 1.5), (-2.8, 2.4), (-1.9, -0.37), (-5.2, 1.8)))
    x, wx = gauss(2order, bounds[1]...)
    amp, wa = gauss(order, bounds[2]...)
    noise, wn = gauss(order, bounds[3]...)
    tau, wt = gauss(order, bounds[4]...)
    tau_scale = exp.(tau)
    logmass = map(eachindex(x)) do i
        subjects, logprior = spectrum(panel, x[i])
        logweights = map(Iterators.product(eachindex(amp), eachindex(noise))) do (j, k)
            moments = subject_moments(subjects, exp(amp[j]), exp(noise[k]))
            conditional = logsumexp(log(wt[l]) + other_logprior(amp[j], noise[k], tau[l]) +
                log_likelihood(moments, tau_scale[l]) for l in eachindex(tau))
            log(wa[j] * wn[k]) + conditional
        end
        log(wx[i]) + logprior + logsumexp(logweights)
    end
    mass = exp.(logmass .- logsumexp(logmass))
    mean_x = dot(mass, x)
    return (; mean=mean_x, sd=sqrt(dot(mass, (x .- mean_x).^2)))
end

# Independently collapse BRM's emitted joint at fixed scales. The conditional
# log density is quadratic in all other coordinates, so unit gradient
# differences give its precision exactly (up to rounding).
function rk_log_marginal(problem, scale_indices, scales)
    u = zeros(dimension(problem))
    u[scale_indices] .= scales
    coeff = setdiff(eachindex(u), scale_indices)
    lp, gradient = logdensity_and_gradient(problem, u)
    g = gradient[coeff]
    precision = map(coeff) do j
        shifted = copy(u)
        shifted[j] = 1
        g - last(logdensity_and_gradient(problem, shifted))[coeff]
    end
    factor = cholesky(Symmetric(stack(precision)))
    return lp + (dot(g, factor \ g) - logdet(factor)) / 2
end

end
