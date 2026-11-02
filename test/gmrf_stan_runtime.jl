include("gmrf_stan.jl")
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

@testset "GMRF sampled channel correlation retains its normalized joint law" begin
    input = (; (k => v for (k, v) in pairs(data) if k !== :Omega)...)
    model = @brm input begin
        Omega ~ LKJCholesky(2, 2.0)
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) = 1.2
        length_scale(field, gmrf((east, north), time, channel), space) = 0.4
        length_scale(field, gmrf((east, north), time, channel), time) = 1.7
        y ~ Normal(field, 0.2)
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    problem = BRM.stan_instantiate(sb)
    descriptor = brm_descriptor(sb; highlights=())
    # Stan's constrained correlation factor has four named entries but only
    # one free coordinate. Locate the identity field in unconstrained names.
    unames = BS.param_unc_names(problem.model)
    field = brm_term_coordinates(descriptor, :field, unames;
        term=:gmrf_field_east_north_time_channel, parameter=:field).coordinates
    correlation = only(setdiff(1:BS.param_unc_num(problem.model), field))
    point = zeros(BS.param_unc_num(problem.model))
    point[field] = 0.05 .* cos.(1:30)
    point[correlation] = atanh(0.35)
    oracle = function(q)
        rho = tanh(q[correlation])
        omega = [1.0 rho; rho 1.0]
        # LKJ(2,eta) is Beta(eta,eta) after mapping rho to (rho+1)/2.
        # Its rho-density plus tanh Jacobian is eta*log(1-rho²)-log B(1/2,eta).
        first(gmrf_dense_oracle(sb, q, (; field); known=true, omega)) +
            2log1p(-rho^2) - logbeta(0.5, 2.0)
    end
    value, gradient = BS.log_density_gradient(problem.model, point; propto=false, jacobian=true)
    @test value ≈ oracle(point) atol=2e-8 rtol=2e-10
    for j in eachindex(point)
        h = 1e-5
        plus, minus = copy(point), copy(point)
        plus[j] += h
        minus[j] -= h
        @test gradient[j] ≈ (oracle(plus)-oracle(minus))/(2h) atol=3e-5 rtol=3e-6
    end
end

@testset "GMRF compiled normalized density, gradient and local reads" begin
    for known in (false, true)
        sb = SBBRMI(gmrf_stan_model(; known); mod=@__MODULE__)
        problem = BRM.stan_instantiate(sb)
        descriptor = brm_descriptor(sb; highlights=())
        names = BS.param_names(problem.model; include_tp=false, include_gq=false)
        roles = known ? (:field,) : (:field, :sd, :spatial_length, :temporal_length)
        selection = NamedTuple{roles}(Tuple(brm_term_coordinates(descriptor, :field,
            names; term=:gmrf_field_east_north_time_channel, parameter=role).coordinates
            for role in roles))
        point = zeros(BS.param_unc_num(problem.model))
        point[selection.field] = 0.07 .* sin.(1:30)
        if !known
            point[only(selection.sd)] = log(0.9)
            point[only(selection.spatial_length)] = log(0.6)
            point[only(selection.temporal_length)] = log(1.1)
        end
        expected, mu = gmrf_dense_oracle(sb, point, selection; known)
        value, gradient = BS.log_density_gradient(problem.model, point;
            propto=false, jacobian=true)
        @test value ≈ expected atol=2e-8 rtol=2e-10
        for j in eachindex(point)
            h = 1e-5
            plus, minus = copy(point), copy(point)
            plus[j] += h
            minus[j] -= h
            finite_difference = (first(gmrf_dense_oracle(sb, plus, selection; known)) -
                first(gmrf_dense_oracle(sb, minus, selection; known))) / (2h)
            @test gradient[j] ≈ finite_difference atol=3e-5 rtol=3e-6
        end
        cnames = BS.param_names(problem.model; include_tp=true, include_gq=false)
        constrained = BS.param_constrain(problem.model, point; include_tp=true, include_gq=false)
        values = brm_output_coordinates(descriptor, :field, cnames)
        @test constrained[values] ≈ mu atol=1e-12 rtol=1e-12
        # At zero state, range derivatives still contain the learned normalizer.
        point[selection.field] .= 0
        _, atzero = BS.log_density_gradient(problem.model, point; propto=false, jacobian=true)
        @test all(isfinite, atzero)
        if !known
            @test abs(atzero[only(selection.spatial_length)]) > 1e-3
            @test abs(atzero[only(selection.temporal_length)]) > 1e-3
        end
    end
end
