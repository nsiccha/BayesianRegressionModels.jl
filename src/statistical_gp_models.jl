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

# Axes are separate vector inputs, so their pairwise broadcasts stay visible
# in RK. Source size depends on the number of axes, never on the number of
# locations; the single covariance plate sums every axis's scaled distance.
function _gp_exp_quad_cov_model(dimensions::Int, iso::Bool; guarded::Bool)
    dimensions > 0 || throw(ArgumentError("GP covariance needs at least one axis"))
    axis_name(base, j) = dimensions == 1 ? base : Symbol(base, :_, j)
    xs = [axis_name(:x, j) for j in 1:dimensions]
    body = Expr(:block)
    operands, arguments, distances = Any[], Any[], Any[]
    for j in 1:dimensions
        left, right = axis_name(:left, j), axis_name(:right, j)
        row, col = axis_name(:row, j), axis_name(:col, j)
        xi, xj = axis_name(:xi, j), axis_name(:xj, j)
        width = iso ? :width : Symbol(:width_, j)
        denominator = iso ? :denominator : Symbol(:denominator_, j)
        push!(body.args, :(($left, $right, $row, $col) = gp_pair_locations($(xs[j]))))
        if j == 1 || !iso
            rho = iso ? :rho : :(rho[$j])
            value = guarded ? :(BayesianRegressionModels.StatisticalPreparation._gp_positive($rho, :rho)) : rho
            push!(body.args, :($width = $value), :($denominator = 2 * $width^2))
        end
        append!(operands, (left, right))
        append!(arguments, (xi, xj))
        push!(distances, :(($xi - $xj)^2 / $denominator))
    end
    scale = first(_gp_scale_bodies(guarded))
    push!(body.args, :(scale = $scale), :(jit = $_GP_JITTER_BODY), :(variance = scale^2))
    append!(operands, (axis_name(:row, 1), axis_name(:col, 1)))
    append!(arguments, (:i, :j))
    distance = foldl((a, b) -> :($a + $b), distances)
    cell = quote
        cell = variance * exp(-$distance)
        diagonal_jitter = ifelse(i == j, jit, 0.0)
        cell + diagonal_jitter
    end
    covariance = Expr(:do, Expr(:call, :plate, operands...),
        Expr(:->, Expr(:tuple, arguments...), cell))
    push!(body.args, :(covariance = $covariance), :(return covariance))
    Expr(:(=), Expr(:call, :gp_exp_quad_cov_graph, xs..., :sigma, :rho, :jitter), body)
end

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
    gp_exp_quad_cov = _gp_exp_quad_cov_model(1, true; guarded),
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
