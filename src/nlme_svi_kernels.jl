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
    global_logdet=:(function _svi_global_logdet(log_scale, loading)
        scale = exp.(log_scale)
        diagonal = 2 * sum(log.(scale))
        result = if size(loading, 2) == 0
            diagonal
        else
            V = loading ./ scale
            gram = transpose(V) * V .+ diagm(ones(size(loading, 2)))
            lower = _svi_lower_cholesky(gram)
            diagonal + 2 * sum(log.(diag(lower)))
        end
        return result
    end),
    unit_entropy=:(function _svi_unit_entropy(root)
        result = -sum(log.(diag(root)))
        return result
    end),
    # Polishing stores covariances. Native precision matrices use their
    # upper triangle as in Symmetric(precision).
    unit_covariance=:(function _svi_unit_covariance(precision)
        result = if size(precision, 1) == 0
            precision
        else
            lower = _svi_lower_cholesky(transpose(precision))
            inverse_root = lower \ diagm(ones(size(precision, 1)))
            transpose(inverse_root) * inverse_root
        end
        return result
    end),
    weighted_average=:(function _svi_weighted_average(previous, current, weight)
        result = previous .+ weight .* (current .- previous)
        return result
    end),
    running_average=:(function _svi_running_average(previous, current, count)
        result = previous .+ (current .- previous) ./ count
        return result
    end),
    dense_logsd=:(function _svi_dense_logsd(precision)
        result = log.(diag(_svi_unit_covariance(precision))) ./ 2
        return result
    end),
    marginal_logsd=:(function _svi_marginal_logsd(log_scale, loading_row)
        result = log(hypot(exp(log_scale), norm(loading_row)))
        return result
    end),
    centering_control=:(function _svi_centering_control(x)
        result = if x >= 0
            inv(1 + exp(-x))
        else
            e = exp(x)
            e / (1 + e)
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
        # At opposite directions a fused normalization can leave a few ulps
        # in their sum. Its eigenvector is then undefined. The half-angle
        # identity also avoids subtracting equal eigenvalue terms nearby.
        active = (a != 0) & (b != 0) & !(n <= $(4 * eps(Float64)))
        half = n / 2
        lambda = ifelse(active, (a * half) * (b * half), 0.0)
        axis = ifelse.(active, direction ./ ifelse(n == 0, 1.0, n), 0.0)
        result = (lambda, axis)
        return result
    end),
    unit_design=:(function _svi_unit_design(normals)
        factor = qr(normals)
        signs = ifelse.(diag(factor.R) .< 0, -1.0, 1.0)
        result = Matrix(factor.Q) .* transpose(sqrt(size(normals, 1)) .* signs)
        return result
    end),
    design_curvature=:(function _svi_design_curvature(whitened, design)
        result = (whitened * transpose(design) .+ design * transpose(whitened)) ./ (2 * size(design, 1))
        return result
    end),
    positive_curvature=:(function _svi_positive_curvature(curvature)
        factor = eigen(Symmetric(curvature))
        result = if first(factor.values) >= 0
            curvature
        else
            vectors = factor.vectors
            Matrix(Symmetric(vectors * Diagonal(max.(factor.values, 0.0)) * transpose(vectors)))
        end
        return result
    end),
    orthogonal_gap=:(function _svi_orthogonal_gap(curvature, positive)
        symmetric = (curvature .+ transpose(curvature)) ./ 2
        estimate = positive ? _svi_positive_curvature(symmetric) : symmetric
        gap = estimate - diagm(ones(size(curvature, 1)))
        result = (gap, maximum(abs.(eigvals(Symmetric(gap)))))
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
    coupling_noise=:(function _svi_coupling_noise(epsilon, columns, eta)
        result = vcat(epsilon[columns], eta)
        return result
    end),
    coupled_mean=:(function _svi_coupled_mean(mean_rows, loading, noise)
        result = mean_rows .+ loading * noise
        return result
    end),
    coupling_baseline=:(function _svi_coupling_baseline(value, gradient)
        valid = isfinite(value) & all(isfinite.(gradient))
        result = ifelse.(valid, gradient, 0.0)
        return result
    end),
    coupling_direction=:(function _svi_coupling_direction(root, gradient)
        lower = LowerTriangular(root)
        result = transpose(lower) \ (lower \ gradient)
        return result
    end),
    coupling_variances=:(function _svi_coupling_variances(root)
        inverse_root = LowerTriangular(root) \ diagm(ones(size(root, 1)))
        result = vec(sum(inverse_root .* inverse_root; dims=1))
        return result
    end),
    coupling_row_update=:(function _svi_coupling_row_update(noise, updates, direction, variance)
        scale = 1 ./ max.(1.0, noise)
        n = updates + 1
        next_noise = noise .+ (direction .^ 2 ./ variance .- noise) ./ n
        result = (scale, next_noise, n)
        return result
    end),
    coupling_metric=:(function _svi_coupling_metric(precision, direction)
        result = dot(direction, precision * direction)
        return result
    end),
    coupling_step=:(function _svi_coupling_step(loading, direction, noise, metric, rate, trust)
        den = 2 * length(noise)
        reach = sqrt(max(metric, 0.0)) * sqrt(sum(noise .* noise)) / den
        result = if reach > 0 && rate > 0
            amount = min(rate, trust / reach) / den
            loading .+ direction .* transpose(amount .* noise)
        else
            loading
        end
        return result
    end),
    structured_offset=:(function _svi_structured_offset(log_scale, loading, epsilon, eta)
        result = exp.(log_scale) .* epsilon .+ loading * eta
        return result
    end),
    structured_precision=:(function _svi_structured_precision(log_scale, loading, x)
        scale = exp.(log_scale)
        y = x ./ scale
        result = if size(loading, 2) == 0
            y ./ scale
        else
            V = loading ./ scale
            lower = _svi_lower_cholesky(transpose(V) * V .+ diagm(ones(size(V, 2))))
            z = transpose(lower) \ (lower \ (transpose(V) * y))
            (y .- V * z) ./ scale
        end
        return result
    end),
    structured_gradient=:(function _svi_structured_gradient(log_scale, loading, mean_gradient, difference, epsilon, eta)
        offset = _svi_structured_offset(log_scale, loading, epsilon, eta)
        score = difference .+ _svi_structured_precision(log_scale, loading, offset)
        result = vcat(mean_gradient, score .* exp.(log_scale) .* epsilon, vec(score * transpose(eta)))
        return result
    end),
    # With V = diag(1/s) B, I + W diag(a) W' is the square root of I + VV'.
    structured_sqrt_basis=:(function _svi_structured_sqrt_basis(V)
        result = if size(V, 2) == 0
            (V, zeros(0))
        else
            factor = eigen(Symmetric(transpose(V) * V))
            (V * factor.vectors, 1 ./ (sqrt.(1 .+ max.(factor.values, 0.0)) .+ 1))
        end
        return result
    end),
    structured_sqrt_action=:(function _svi_structured_sqrt_action(scale, W, a, x, transposed)
        result = if transposed
            y = scale .* x
            y .+ W * (a .* (transpose(W) * y))
        else
            scale .* (x .+ W * (a .* (transpose(W) * x)))
        end
        return result
    end),
    # Shorten the whole proposed move to one old-factor mean standard deviation
    # and at most one half in log-scale or diagonal-scaled loading displacement.
    structured_trust=:(function _svi_structured_trust(log_scale, loading, old, proposed)
        d = length(log_scale)
        delta = proposed .- old
        mean_move = delta[1:d]
        distance = sqrt(max(0.0, dot(mean_move, _svi_structured_precision(log_scale, loading, mean_move))))
        relative = d == 0 ? 0.0 : maximum(abs.(delta[d+1:2*d]))
        magnitude = if length(loading) == 0
            relative
        else
            load_move = reshape(delta[2*d+1:end], size(loading))
            max(relative, maximum(abs.(load_move ./ exp.(log_scale))))
        end
        fraction = min(1 / max(1.0, distance), 0.5 / max(0.5, magnitude))
        result = old .+ fraction .* delta
        return result
    end),
    structured_direction=:(function _svi_structured_direction(log_scale, loading, gradient)
        result = exp.(2 .* log_scale) .* gradient .+ loading * (transpose(loading) * gradient)
        return result
    end),
    structured_variances=:(function _svi_structured_variances(log_scale, loading)
        result = exp.(2 .* log_scale) .+ vec(sum(loading .* loading; dims=2))
        return result
    end),
    structured_metric=:(function _svi_structured_metric(log_scale, loading, direction)
        diagonal = exp.(2 .* log_scale)
        w = direction ./ diagonal
        result = if size(loading, 2) == 0
            dot(direction, w)
        else
            gram = diagm(ones(size(loading, 2))) .+ transpose(loading) * (loading ./ diagonal)
            lower = _svi_lower_cholesky(transpose(gram))
            z = lower \ (transpose(loading) * w)
            dot(direction, w) - sum(z .* z)
        end
        return result
    end),
    adam=:(function _svi_adam(x, moment, variance, gradient, beta1, beta2, correction1, correction2, rate)
        next_moment = beta1 .* moment .+ (1 - beta1) .* gradient
        next_variance = beta2 .* variance .+ (1 - beta2) .* gradient .^ 2
        next_x = x .+ rate .* (next_moment ./ correction1) ./ (sqrt.(next_variance ./ correction2) .+ 1e-8)
        result = (next_x, next_moment, next_variance)
        return result
    end),
    structured_adam=:(function _svi_structured_adam(log_scale, loading, x, gradient, moment, variance,
            count, beta1, beta2, rate, rank_step, clip)
        d = length(log_scale)
        scale = exp.(log_scale)
        basis = _svi_structured_sqrt_basis(loading ./ scale)
        W = basis[1]
        a = basis[2]
        tail_scale = vcat(ones(d), rank_step .* repeat(scale, size(loading, 2)))
        g = vcat(_svi_structured_sqrt_action(scale, W, a, gradient[1:d], true),
            gradient[d+1:end] .* tail_scale)
        bounded = if count > 0 && isfinite(clip)
            _svi_clip_gradient(g, variance, 1 - beta2^count, clip)
        else
            g
        end
        next_count = count + 1
        update = _svi_adam(zero(x), moment, variance, bounded,
            beta1, beta2, 1 - beta1^next_count, 1 - beta2^next_count, rate)
        step = update[1]
        move = vcat(_svi_structured_sqrt_action(scale, W, a, step[1:d], false),
            step[d+1:end] .* tail_scale)
        result = (x .+ move, update[2], update[3], next_count, any(bounded .!= g))
        return result
    end),
)

for definition in _BRM_SVI_KERNEL_DEFINITIONS
    Core.eval(@__MODULE__, deepcopy(definition))
end
