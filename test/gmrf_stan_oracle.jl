import BridgeStan
using Statistics
using SpecialFunctions: logbeta
const BS = BridgeStan

# This oracle constructs a tiny dense covariance directly from FEM matrices
# and exp(-|t-s|/ell_t), independently of the emitted sparse LDL calculation.
function gmrf_dense_covariance(ell_s, ell_t, omega=data.Omega)
    mesh = BRM._brm_gmrf_mesh(data.xy, data.triangles)
    k2 = 2 / ell_s^2
    K = Matrix(mesh.stiffness) + k2 * Diagonal(mesh.mass)
    Qs = K * Diagonal(1 ./ mesh.mass) * K / (4pi * k2)
    times = [0.0, 0.3, 1.7]
    Ct = [exp(-abs(a-b)/ell_t) for a in times, b in times]
    kron(Ct, kron(omega, inv(Symmetric(Qs))))
end

function gmrf_dense_oracle(sb, point, selection; known, omega=data.Omega)
    state = point[selection.field]
    sigma = known ? 1.2 : exp(point[only(selection.sd)])
    ell_s = known ? 0.4 : exp(point[only(selection.spatial_length)])
    ell_t = known ? 1.7 : exp(point[only(selection.temporal_length)])
    covariance = gmrf_dense_covariance(ell_s, ell_t, omega)
    value = logpdf(MvNormal(zeros(length(state)), Symmetric(covariance)), state)
    stem = "gmrf_field_east_north_time_channel_"
    mu = zeros(length(data.y))
    for j in 1:3
        indices = sb.data[Symbol(stem, "index", j)]
        weights = sb.data[Symbol(stem, "weight", j)]
        mu .+= sigma .* weights .* state[indices]
    end
    value += sum(logpdf.(Normal.(mu, 0.2), data.y))
    if !known
        value += logpdf(Exponential(0.7), sigma) + log(sigma)
        value += logpdf(LogNormal(0.5, 0.8), ell_s) + log(ell_s)
        value += logpdf(LogNormal(0.7, 0.6), ell_t) + log(ell_t)
    end
    value, mu
end

