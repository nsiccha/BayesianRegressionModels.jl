module StreamingOracles

# Independent analytic references for the streaming consumer tests. These are
# test-only, fixed two-coordinate Gaussian calculations, never sampler metrics.

function gaussian_summary(a, b, c, h1, h2, log_evidence, observations)
    determinant = a*c - b*b
    mean = ((c*h1 - b*h2)/determinant, (a*h2 - b*h1)/determinant)
    covariance = (c/determinant, -b/determinant, a/determinant)
    (; a, b, c, h1, h2, mean, covariance, log_evidence, observations)
end

gaussian_prior() = gaussian_summary(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0)

"""Assimilate only this window into a normalized two-coefficient Gaussian."""
function gaussian_window(previous, data; noise_sd=0.1)
    a, b, c, h1, h2 = previous.a, previous.b, previous.c, previous.h1, previous.h2
    precision = inv(noise_sd^2)
    square = 0.0
    for i in eachindex(data.y, data.x1, data.x2)
        x1, x2, y = data.x1[i], data.x2[i], data.y[i]
        a += precision*x1*x1
        b += precision*x1*x2
        c += precision*x2*x2
        h1 += precision*x1*y
        h2 += precision*x2*y
        square += precision*y*y
    end
    next = gaussian_summary(a, b, c, h1, h2, 0.0, previous.observations+length(data.y))
    old_quadratic = previous.h1*previous.mean[1] + previous.h2*previous.mean[2]
    new_quadratic = h1*next.mean[1] + h2*next.mean[2]
    increment = -length(data.y)*log(2pi*noise_sd^2)/2 +
        (log(previous.a*previous.c-previous.b^2)-log(a*c-b*b))/2 -
        (square+old_quadratic-new_quadratic)/2
    merge(next, (; log_evidence=previous.log_evidence+increment))
end

"""Joint static level and AR(1) latent state before the first observation."""
function kalman_prior(; level_mean=0.0, level_variance=1.0, state_variance=1.0)
    (; mean=(level_mean, 0.0), covariance=(level_variance, 0.0, state_variance),
       steps=0, log_evidence=0.0, predictive_logscore=0.0)
end

"""Exact filter for y[t] = level + state[t] + noise; missing y predicts only."""
function kalman_step(previous, y; phi=0.6, process_variance=1.0, noise_variance=1.0)
    level, state = previous.mean
    ll, ls, ss = previous.covariance
    if previous.steps > 0
        state *= phi
        ls *= phi
        ss = phi^2*ss + process_variance
    end
    score = 0.0
    if !ismissing(y)
        variance = ll + 2ls + ss + noise_variance
        residual = y - level - state
        score = -(log(2pi*variance) + residual^2/variance)/2
        level_covariance, state_covariance = ll+ls, ls+ss
        level += level_covariance*residual/variance
        state += state_covariance*residual/variance
        ll -= level_covariance^2/variance
        ls -= level_covariance*state_covariance/variance
        ss -= state_covariance^2/variance
    end
    (; mean=(level, state), covariance=(ll, ls, ss), steps=previous.steps+1,
       log_evidence=previous.log_evidence+score, predictive_logscore=score)
end

# Conditional scalar recursion, independent of the joint two-state update.
# Integrating this likelihood over the level's prior checks both its posterior
# and the joint oracle's normalizer without a dense observation covariance.
function kalman_loglikelihood(ys, level;
        phi=0.6, process_variance=1.0, noise_variance=1.0, state_variance=1.0)
    mean, variance, total = 0.0, state_variance, 0.0
    for (i, y) in enumerate(ys)
        if i > 1
            mean *= phi
            variance = phi^2*variance + process_variance
        end
        ismissing(y) && continue
        innovation_variance = variance+noise_variance
        residual = y-level-mean
        total -= (log(2pi*innovation_variance)+residual^2/innovation_variance)/2
        gain = variance/innovation_variance
        mean += gain*residual
        variance *= 1-gain
    end
    total
end

end
