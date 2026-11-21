# These equations run as ordinary Julia on the CPU. The RK extension passes
# the same definitions to @kernel so a device method can compose their graphs.
_svi_lower_cholesky(A) = cholesky(Symmetric(A, :L)).L

const _BRM_SVI_KERNEL_DEFINITIONS = (
    global_draw=:(function _svi_global_draw(mean, log_scale, loading, epsilon, eta)
        result = mean .+ exp.(log_scale) .* epsilon .+ loading * eta
        return result
    end),
    antithetic_pair=:(function _svi_antithetic_pair(mean, offset)
        result = (mean .+ offset, mean .- offset)
        return result
    end),
    unit_offset=:(function _svi_unit_offset(root, epsilon)
        result = transpose(LowerTriangular(root)) \ epsilon
        return result
    end),
    whitened_difference=:(function _svi_whitened_difference(root, difference)
        result = -(LowerTriangular(root) \ difference)
        return result
    end),
    work_weight=:(function _svi_work_weight(count, batch_size, probability)
        result = count ./ (batch_size .* probability)
        return result
    end),
    global_gradient=:(function _svi_global_gradient(w, epsilon, eta, scale, entropy_diagonal, entropy_loading)
        grad_scale = w .* epsilon .* scale .+ entropy_diagonal
        grad_loading = w * transpose(eta) .+ entropy_loading
        result = vcat(w, grad_scale, vec(grad_loading))
        return result
    end),
    global_entropy=:(function _svi_global_entropy(log_scale, loading)
        result = if size(loading, 2) == 0
            (ones(length(log_scale)), loading)
        else
            scale = exp.(log_scale)
            V = loading ./ scale
            eye = diagm(ones(size(loading, 2)))
            lower = _svi_lower_cholesky(transpose(V) * V .+ eye)
            # With L L' = I + V'V, the two derivatives are the diagonal
            # of I - V (L L')⁻¹ V' and diag(1/scale) V (L L')⁻¹.
            projection = lower \ transpose(V)
            diagonal = 1 .- vec(sum(projection .* projection; dims=1))
            (diagonal, (transpose(projection) * (lower \ eye)) ./ scale)
        end
        return result
    end),
    global_step_scale=:(function _svi_global_step_scale(log_scale, loading, mean_scale, rank_step)
        row_scale = exp.(log_scale)
        loading_scale = rank_step .* repeat(row_scale, size(loading, 2))
        result = vcat(mean_scale, ones(length(log_scale)), loading_scale)
        return result
    end),
    batch_clip=:(function _svi_batch_clip(weights, multiplier)
        result = multiplier * sqrt(maximum(weights))
        return result
    end),
    clip_gradient=:(function _svi_clip_gradient(gradient, variance, correction, clip)
        limit = clip .* sqrt.(variance ./ correction)
        result = ifelse.(variance .== 0, gradient, clamp.(gradient, .-limit, limit))
        return result
    end),
    sd_shrink=:(function _svi_sd_shrink(variance, bound)
        result = ifelse.(isfinite.(bound) .& (variance .> bound .^ 2), bound ./ sqrt.(variance), 1.0)
        return result
    end),
    bound_sd=:(function _svi_bound_sd(log_scale, loading, bound)
        variance = exp.(2 .* log_scale) .+ vec(sum(loading .* loading; dims=2))
        shrink = _svi_sd_shrink(variance, bound)
        next_log_scale = ifelse.(shrink .== 1, log_scale, log_scale .+ log.(shrink))
        result = (next_log_scale, loading .* shrink)
        return result
    end),
    # Both curvature designs use the same positive-diagonal root and cap the
    # mean's displacement in standard deviations of its old factor.
    precision_root=:(function _svi_precision_root(r, divisor)
        result = transpose(r) .* transpose(sign.(diag(r))) ./ divisor
        return result
    end),
    mean_trust=:(function _svi_mean_trust(root, delta, trust)
        result = min(1.0, trust / norm(transpose(root) * delta))
        return result
    end),
    curvature_axis=:(function _svi_curvature_axis(w, epsilon)
        a = norm(w)
        b = norm(epsilon)
        direction = w ./ ifelse(a == 0, 1.0, a) .+ epsilon ./ ifelse(b == 0, 1.0, b)
        n = norm(direction)
        active = (a != 0) & (b != 0) & (n != 0)
        lambda = ifelse(active, (dot(w, epsilon) + a * b) / 2, 0.0)
        axis = ifelse.(active, direction ./ ifelse(n == 0, 1.0, n), 0.0)
        result = (lambda, axis)
        return result
    end),
    pair_stack=:(function _svi_pair_stack(root, precision, lambda, axis, rate, trust)
        lower = LowerTriangular(root)
        v = lower * axis
        gap = Symmetric(lambda .* (v * transpose(v)) .- precision)
        radius = max(abs(lambda - 1), length(axis) > 1 ? 1.0 : 0.0)
        step = min(rate, trust / radius)
        stack = vcat(Matrix(transpose(lower)), lower \ (precision .+ step .* gap))
        result = (stack, step)
        return result
    end),
    orthogonal_stack=:(function _svi_orthogonal_stack(root, gap, radius, rate, trust)
        step = radius > 0 ? min(rate, trust / radius) : rate
        upper = Matrix(transpose(LowerTriangular(root)))
        stack = vcat(upper, (I(size(gap, 1)) + step .* gap) * upper)
        result = (stack, step)
        return result
    end),
    dense_update=:(function _svi_dense_update(mean, old_root, stack, gradient, rate, trust)
        factor = qr(stack).R
        root = LowerTriangular(_svi_precision_root(factor, sqrt(2)))
        delta = rate .* (transpose(root) \ (root \ gradient))
        next_mean = mean .+ _svi_mean_trust(old_root, delta, trust) .* delta
        result = (next_mean, root * transpose(root), Matrix(root))
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
