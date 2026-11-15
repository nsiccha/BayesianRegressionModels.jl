# A point where a kernel cell's ODE solve fails is outside the model's support
# on RKBRMI, as on SBBRMI: Stan rejects it (`std::domain_error`), and RKBRMI's
# density answers -Inf there. ReactiveKernels throws a typed
# `ReactiveKernels.ODESolveFailure` (a `NumericalFailure`) when its solver
# cannot integrate from valid inputs, and RKPPL's sampler query answers -Inf
# for exactly that type (RK 0.1.251 / RKPPL 0.1.195), so
# `rk_logdensity_problem` rejects the point and every other exception still
# propagates. Needs the solver packages, which test/Project.toml does not
# carry: run it from a copy of the test environment that adds them, as
# `rk_kernel_ode_cell.jl` (test/README.md "ODE solvers in kernel cells").
# Run: julia --startup-file=no --project=<env> test/rk_ode_solve_failure.jl
using OrdinaryDiffEqLowOrderRK, SciMLBase, SciMLSensitivity
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# dy/dt = k y², whose solution from y(0) = 1 is 1 / (1 - k t): it blows up at
# t = 1 / k, so a solve past that time fails.
StanBlocks.@deffun @juliacompat blowup(t::real, y::vector[ny], k::real)::vector[ny] =
    k * (y .* y)

const BLOWUP_DATA = (; subject=["a", "b", "c"], weight=[0.1, -0.2, 0.3],
    t=[[0.5, 1.0, 2.0], [1.0, 3.0], [0.5, 2.0, 4.0]],
    dv=[[1.03, 1.05, 1.11], [1.05, 1.18], [1.03, 1.11, 1.25]])

# `k = exp(log_k) / 20`: 0.05 at the zero point, so every solve up to t = 4
# succeeds there, and a log_k intercept of 3 puts the blow-up at t = 1.
const BLOWUP_PREDICTIONS = (
    closed=:(1.0 ./ (1.0 .- k .* t[i])),
    rk45=:(to_vector(ode_rk45(blowup, rep_vector(1.0, 1), 0.0, to_array_1d(t[i]), k)[:, 1])),
    # A cell error that is not a solver failure: `sqrt` of a negative number
    # raises `DomainError` when log_k's intercept is below log(0.8).
    domain=:(to_vector(ode_rk45(blowup, rep_vector(sqrt(k - 0.04), 1), 0.0,
        to_array_1d(t[i]), k)[:, 1])),
)
for (name, prediction) in pairs(BLOWUP_PREDICTIONS)
    @eval $(Symbol(:blowup_, name))(data) = @brm data begin
        sigma ~ Exponential(1)
        log_k ~ 1 + weight + (1 | subject)
        @plate for i in eachindex(log_k)
            k = exp(log_k[i]) / 20
            pred[i] = $prediction
            dv[i] ~ normal(pred[i], sigma)
        end
    end
end

# The packed point with log_k's intercept (`pop_log_k.beta_pop.1`) at
# `intercept` and every other coordinate at `base`'s value.
function blowup_point(names, intercept; base=zeros(length(names)))
    u = copy(base)
    u[coordinate(names, Symbol("pop_log_k.beta_pop.1"))] = intercept
    u
end

@stestset "a failed ODE solve is -Inf on RKBRMI's density, and the next point is unaffected" begin
    backend, problem = consumer_problem(blowup_rk45(BLOWUP_DATA))
    _, closed = consumer_problem(blowup_closed(BLOWUP_DATA))
    names = coordinate_names(backend.model.layout)
    good = blowup_point(names, 0.0; base=[0.1 * sin(j) for j in eachindex(names)])
    bad = blowup_point(names, 3.0)

    # Positive control: the ODE solve is right where it succeeds.
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, good)
    expected, expected_gradient = LogDensityProblems.logdensity_and_gradient(closed, good)
    @test isfinite(value)
    @test value ≈ expected rtol=2e-6
    @test gradient ≈ expected_gradient rtol=2e-6

    # The point is a typed solver failure, not some other error: a query that
    # does not reject it raises ReactiveKernels' failure.
    likelihood = BRM._rk_likelihood_reader(backend)
    @test isfinite(likelihood(good))
    @test_throws ReactiveKernels.ODESolveFailure likelihood(bad)
    failure = try likelihood(bad); nothing catch err; err end
    @test failure isa ReactiveKernels.NumericalFailure
    @test failure.method == "native DP5"
    @test failure.retcode !== :Success
    @test 0 < failure.t < 4

    # Both density forms reject it, and leave the caller's point untouched.
    before = copy(bad)
    @test LogDensityProblems.logdensity(problem, bad) == -Inf
    bad_value, bad_gradient = LogDensityProblems.logdensity_and_gradient(problem, bad)
    @test bad_value == -Inf
    @test length(bad_gradient) == length(bad)
    @test all(isnan, bad_gradient)
    @test isequal(bad, before)

    # The rejection leaves no state behind: the good point evaluates
    # bit-identically afterwards.
    again_value, again_gradient = LogDensityProblems.logdensity_and_gradient(problem, good)
    @test isequal(again_value, value)
    @test isequal(again_gradient, gradient)
    @test isequal(LogDensityProblems.logdensity(problem, good), value)
end

@stestset "a cell error that is not a solver failure still propagates" begin
    backend, problem = consumer_problem(blowup_domain(BLOWUP_DATA))
    names = coordinate_names(backend.model.layout)
    @test isfinite(LogDensityProblems.logdensity(problem, zeros(length(names))))
    outside = blowup_point(names, -2.0)
    @test_throws DomainError LogDensityProblems.logdensity(problem, outside)
    @test_throws DomainError LogDensityProblems.logdensity_and_gradient(problem, outside)
end

@stestset "SBBRMI's compiled Stan rejects the same point" begin
    brmi = blowup_rk45(BLOWUP_DATA)
    backend, problem = consumer_problem(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "ode_blowup.stan"))
    transport = brm_coordinate_transport(backend, sb, BridgeStan.param_unc_names(stan.model))
    names = transport.rk_names
    good = blowup_point(names, 0.0)
    bad = blowup_point(names, 3.0)
    stan_value = BridgeStan.log_density(stan.model, brm_rk_point_to_stan(transport, good);
        propto=false, jacobian=true)
    @test LogDensityProblems.logdensity(problem, good) ≈
        stan_value + transport.logdensity_offset rtol=1e-6
    rejection = try
        BridgeStan.log_density(stan.model, brm_rk_point_to_stan(transport, bad);
            propto=false, jacobian=true)
        nothing
    catch err
        err
    end
    # Stan's RK45 returns a non-finite state past the blow-up, and the
    # observation's density rejects the point (`normal_lpdf: Location
    # parameter[2] is -nan, but must be finite!`).
    println("STAN_REJECTION ", rejection === nothing ? "none" : sprint(showerror, rejection))
    @test rejection isa ErrorException
    @test occursin("log_density() failed", sprint(showerror, rejection))
    @test occursin("must be finite", sprint(showerror, rejection))
    @test LogDensityProblems.logdensity(problem, bad) == -Inf
end
