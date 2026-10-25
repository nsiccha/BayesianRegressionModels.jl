# One statistical source for the native catalogue and formula emission.
# They differ only in the `sigma`/`rho` positivity guards. The native
# catalogue (`rk_model(:gp_exp_quad_cov)`, `gp_exp_quad_cov`) keeps them: its
# caller passes any value, and the graph squares `sigma`. The formula path
# (`_rk_ast_gp_covariance!`) has none: `_rk_gp_hyper_prior` admits only
# positive-support priors, and jax_source cannot translate a throw on a
# sampled value (decision 2026-10-25T02-49-41-594-1dgw1m1).
_gp_scale_bodies(guarded::Bool) = guarded ?
    (:(BayesianRegressionModels.StatisticalPreparation._gp_positive(sigma, :sigma)),
     :(BayesianRegressionModels.StatisticalPreparation._gp_positive(rho, :rho))) :
    (:sigma, :rho)

function _gp_covariance_models(; guarded::Bool)
    scale_body, width_body = _gp_scale_bodies(guarded)
    return (
    gp_pair_locations = :(
gp_pair_locations(x) = begin
    indices = $_GP_INDICES_BODY
    locations = Float64.(x)
    left = locations
    right = reshape(locations, 1, :)
    row = indices
    col = reshape(indices, 1, :)
    return left, right, row, col
end
    ),
    gp_exp_quad_cov = :(
gp_exp_quad_cov_graph(x, sigma, rho, jitter) = begin
    left, right, row, col = gp_pair_locations(x)
    scale = $scale_body
    width = $width_body
    jit = $_GP_JITTER_BODY
    variance = scale^2
    denominator = 2 * width^2
    covariance = plate(left, right, row, col) do xi, xj, i, j
        distance = xi - xj
        squared_distance = distance * distance
        cell = variance * exp(-squared_distance / denominator)
        diagonal_jitter = ifelse(i == j, jit, 0.0)
        cell + diagonal_jitter
    end
    return covariance
end
    ),
    gp_periodic_cov = :(
gp_periodic_cov_graph(x, sigma, rho, period, jitter) = begin
    left, right, row, col = gp_pair_locations(x)
    scale = $scale_body
    width = $width_body
    per = $_GP_PERIOD_BODY
    jit = $_GP_JITTER_BODY
    variance = scale^2
    squared_width = width^2
    covariance = plate(left, right, row, col) do xi, xj, i, j
        distance = abs(xi - xj)
        sine = sin(pi * distance / per)
        cell = variance * exp(-2 * sine * sine / squared_width)
        diagonal_jitter = ifelse(i == j, jit, 0.0)
        cell + diagonal_jitter
    end
    return covariance
end
    ),
    )
end
const _GP_COVARIANCE_MODELS = _gp_covariance_models(; guarded=true)
const _GP_FORMULA_COVARIANCE_MODELS = _gp_covariance_models(; guarded=false)
