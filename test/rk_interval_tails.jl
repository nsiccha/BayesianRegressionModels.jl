# Run: julia --startup-file=no --project=test test/rk_interval_tails.jl
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using LogDensityProblems, SpecialFunctions
using DifferentiationInterface: AutoEnzyme
import Enzyme

const INTERVAL_TAIL_MODEL = @brm begin
    mu ~ Normal(0.0, 2.0)
    log_sigma ~ Normal(0.0, 1.0)
    y ~ interval_censored(Normal(mu, exp(log_sigma)); upper=upper)
end

# Independent MPFR probabilities and analytic derivatives. Direct subtraction
# is safe at this precision/exponent range, including intervals at 400 sigma.
function interval_tail_reference(data, mu, log_sigma)
    setprecision(BigFloat, 256) do
        m, ls = BigFloat(mu), BigFloat(log_sigma)
        s = exp(ls)
        sqrt2 = sqrt(big(2))
        sqrt2pi = sqrt(2big(pi))
        q(z) = erfc(z / sqrt2) / 2
        phi(z) = exp(-z^2 / 2) / sqrt2pi
        rows = BigFloat[]
        dm = -m / 4
        dls = -ls
        for (lo, hi) in zip(data.y, data.upper)
            a, b = (BigFloat(lo) - m) / s, (BigFloat(hi) - m) / s
            mass = a >= 0 ? q(a) - q(b) : q(-b) - q(-a)
            push!(rows, log(mass))
            dm += (phi(a) - phi(b)) / (s * mass)
            dls += (a * phi(a) - b * phi(b)) / mass
        end
        prior = -log(2big(pi)) - log(big(2)) - m^2 / 8 - ls^2 / 2
        (; rows=Float64.(rows), value=sum(rows) + prior,
           dmu=Float64(dm), dlog_sigma=Float64(dls))
    end
end

@testset "RK Normal intervals retain values and native Reverse gradients" begin
    mu0, ls0 = 0.17, -1.2
    # Synthetic intervals span both tails and the centre, with rows bracketing
    # the Float64 probability-underflow boundary near 38 sigma.
    tails = [-402.0, -42.0, -40.5, -40.0, -32.0, -14.0, -5.0, -1.0,
             0.0, 3.0, 12.0, 30.0, 37.5, 38.0, 38.5, 40.0, 330.0, 400.0]
    controls = filter(z -> abs(z) <= 14, tails)
    @test Base.get_extension(BayesianRegressionModels,
        :BayesianRegressionModelsReactiveKernelsExt) !== nothing

    for (label, z) in (("moderate control", controls), ("both extreme tails", tails))
        @testset "$label" begin
            y = mu0 .+ exp(ls0) .* z
            data = (; y, upper=y .+ 2exp(ls0))
            saved_data = deepcopy(data)
            backend = RKBRMI(INTERVAL_TAIL_MODEL(data))
            problem = rk_logdensity_problem(backend;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            names = coordinate_names(backend.model.layout)
            imu = only(findall(==(:mu), names))
            ils = only(findall(==(:log_sigma), names))
            pointwise = prepare_query(backend.model, rk_bound_plan(backend), :pointwise)

            for (mu, ls) in ((mu0, ls0), (mu0 + 0.04, ls0 + 0.03))
                u = zeros(length(names))
                u[imu], u[ils] = mu, ls
                saved_u = copy(u)
                reference = interval_tail_reference(data, mu, ls)
                rows = collect(Base.invokelatest(pointwise, u).y)
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                expected_gradient = similar(u)
                expected_gradient[imu] = reference.dmu
                expected_gradient[ils] = reference.dlog_sigma

                # A second independent derivative check uses central differences
                # of the MPFR density, without differentiating the RK evaluator.
                h = 1e-5
                finite_difference = similar(u)
                finite_difference[imu] = (
                    interval_tail_reference(data, mu + h, ls).value -
                    interval_tail_reference(data, mu - h, ls).value) / (2h)
                finite_difference[ils] = (
                    interval_tail_reference(data, mu, ls + h).value -
                    interval_tail_reference(data, mu, ls - h).value) / (2h)

                @test all(isfinite, rows)
                @test rows ≈ reference.rows rtol=2e-12 atol=2e-10
                @test isfinite(value)
                @test value ≈ reference.value rtol=2e-12 atol=2e-9
                @test all(isfinite, gradient)
                @test gradient ≈ expected_gradient rtol=2e-9 atol=2e-7
                @test gradient ≈ finite_difference rtol=2e-9 atol=2e-7
                @test isequal(u, saved_u)
            end
            @test isequal(data, saved_data)
        end
    end
end
