# Generic scalar component laws for native mixture acceptance. The scientific
# parameterizations live with the caller; BRM only composes their distributions.
module SyntheticSkewComponent
using BayesianRegressionModels, StanBlocks, Distributions
import BayesianRegressionModels: _rk_observation_source!

StanBlocks.@deffun begin
    @lpxf centered_skew_lpdf(y::real, log_scale::real, alpha::real)::real = begin
        delta = alpha / hypot(1.0, alpha)
        center = 0.7978845608028654 * delta
        sd = sqrt(1.0 - square(center))
        z = y * exp(-log_scale) * sd + center
        skew_normal_lpdf_stable(z, 0.0, 1.0, alpha) + log(sd) - log_scale
    end
    centered_skew_lpdfs(y::real, log_scale::real, alpha::real)::real =
        centered_skew_lpdf(y, log_scale, alpha)
    centered_skew_rng(log_scale::real, alpha::real)::real = begin
        delta = alpha / hypot(1.0, alpha)
        center = 0.7978845608028654 * delta
        sd = sqrt(1.0 - square(center))
        exp(log_scale) * (skew_normal_rng(0.0, 1.0, alpha) - center) / sd
    end
end

function _rk_observation_source!(definitions, bindings, entry, ::typeof(centered_skew))
    density = Symbol(entry, :_density)
    push!(bindings, density => StanBlocks.skew_normal_lpdf_stable)
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, log_scale, alpha) = begin
        delta = alpha / hypot(1.0, alpha)
        center = 0.7978845608028654 * delta
        sd = sqrt(1.0 - center * center)
        inverse_scale = exp(-log_scale)
        log_normalizer = log(sd) - log_scale
        z = y * inverse_scale * sd + center
        result = $density(z, 0.0, 1.0, alpha) + log_normalizer
        return result
    end))
    :done
end
end

module SyntheticBoundedComponent
using BayesianRegressionModels, StanBlocks, Distributions

StanBlocks.@deffun begin
    @lpxf lower_shift_lpdf(y::real, lo::real)::real = exponential_lpdf(y - lo, 1.0)
    lower_shift_lpdfs(y::real, lo::real)::real = lower_shift_lpdf(y, lo)
    lower_shift_rng(lo::real)::real = lo + exponential_rng(1.0)
end

# Scalar real observations do not imply full-real support.
StanBlocks.autokwargs(call::StanBlocks.CanonicalExpr{typeof(lower_shift)}) =
    (; lower=call.args[1])
end

module SyntheticExcessComponent
using BayesianRegressionModels, StanBlocks, Distributions
import BayesianRegressionModels: _rk_observation_source!

StanBlocks.@deffun begin
    @lpxf logscale_excess_lpdf(y::real, location_ratio::real, log_scale::real,
            log_rate::real)::real = begin
        u = y * exp(-log_scale) - location_ratio
        exp_mod_normal_lpdf_standardized(u, log_scale, log_rate)
    end
    logscale_excess_lpdfs(y::real, location_ratio::real, log_scale::real,
            log_rate::real)::real =
        logscale_excess_lpdf(y, location_ratio, log_scale, log_rate)
    logscale_excess_rng(location_ratio::real, log_scale::real,
            log_rate::real)::real =
        exp(log_scale) * normal_rng(location_ratio, 1.0) + exponential_rng(exp(log_rate))
end

function _rk_observation_source!(definitions, bindings, entry, ::typeof(logscale_excess))
    density = Symbol(entry, :_density)
    push!(bindings, density => StanBlocks.exp_mod_normal_lpdf_standardized)
    push!(definitions, :(ReactiveKernels.@kernel $entry(y, location_ratio, log_scale,
            log_rate) = begin
        inverse_scale = exp(-log_scale)
        u = y * inverse_scale - location_ratio
        result = $density(u, log_scale, log_rate)
        return result
    end))
    :done
end
end
