# These equations run as ordinary Julia on the CPU. The RK extension passes
# the same definitions to @kernel so a device method can compose their graphs.
const _BRM_SVI_KERNEL_DEFINITIONS = (
    global_draw=:(function _svi_global_draw(mean, log_scale, loading, epsilon, eta)
        result = mean .+ exp.(log_scale) .* epsilon .+ loading * eta
        return result
    end),
    antithetic_pair=:(function _svi_antithetic_pair(mean, offset)
        result = (mean .+ offset, mean .- offset)
        return result
    end),
    work_weight=:(function _svi_work_weight(count, batch_size, probability)
        result = count ./ (batch_size .* probability)
        return result
    end),
    adam=:(function _svi_adam(x, moment, variance, gradient, beta1, beta2, correction1, correction2, rate)
        next_moment = beta1 .* moment .+ (1 - beta1) .* gradient
        next_variance = beta2 .* variance .+ (1 - beta2) .* gradient .^ 2
        next_x = x .+ rate .* (next_moment ./ correction1) ./ (sqrt.(next_variance ./ correction2) .+ 1e-8)
        result = (next_x, next_moment, next_variance)
        return result
    end),
)

for definition in _BRM_SVI_KERNEL_DEFINITIONS
    Core.eval(@__MODULE__, deepcopy(definition))
end
