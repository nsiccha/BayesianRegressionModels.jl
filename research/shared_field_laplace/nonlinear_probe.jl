# Non-Gaussian control exercising the field mode's implicit derivative.
# Run under one token: julia --project=test research/shared_field_laplace/nonlinear_probe.jl
include("trace_probe.jl")

# Deterministic Newton-CG for this log-concave-in-field Poisson fixture.
# It is not yet a general non-concave mode solver or a public entry point.
function shared_poisson_mode(op, point; tol=1e-10, maxiter=40)
    u = copy(point); u[op.field] .= 0.0
    operations = 0
    for iteration in 0:maxiter
        value, gradient = shared_value_gradient(op, u)
        isfinite(value) && all(isfinite, gradient) || error("nonfinite field point")
        g = gradient[op.field]
        norm(g) <= tol && return (; mode=u, gradient, value, iteration, operations)
        iteration == maxiter && break
        H(v) = shared_precision(op, u, v)
        step, info = KrylovKit.linsolve(H, g; issymmetric=true, isposdef=true,
            rtol=1e-12, atol=1e-13, krylovdim=128, maxiter=1)
        info.converged == 1 || error("mode CG did not converge: $(info.normres)")
        operations += info.numops
        decrement = dot(g, step)
        decrement > 0 || error("nonpositive Newton decrement")
        accepted = false
        for half in 0:40
            scale = 2.0^-half
            trial = copy(u); trial[op.field] .+= scale .* step
            trial_value, trial_gradient = shared_value_gradient(op, trial)
            if isfinite(trial_value) && all(isfinite, trial_gradient) &&
                    (decrement < 1e-12 || trial_value >= value + 1e-4*scale*decrement)
                u = trial
                accepted = true
                break
            end
        end
        accepted || error("field mode backtracking failed")
    end
    error("field mode did not reach tolerance $tol in $maxiter iterations")
end

@stestset "Poisson shared-field implicit gradient and conditional draw action" begin
    println((; stage="prepare Poisson control")); flush(stdout)
    (; rk, m, data) = shared_field_fixture(; poisson=true)
    op = shared_field_operator(m)
    u = first(BRM._brm_nlme_prior_median_point(m))
    u .+= 0.05 .* randn(Xoshiro(43), length(u))
    field, d = op.field, length(op.field)
    @test d == 8
    outer = setdiff(eachindex(u), field)
    bound = rk_bound_data(rk)
    query = Base.invokelatest(ReactiveKernels.prepare, rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(:mu,), bound)
    function predictor(point)
        value = Base.invokelatest(query, point)
        value isa Tuple ? only(value) : value
    end
    function affine(point)
        base = copy(point); base[field] .= 0.0
        b = predictor(base)
        A = hcat(map(1:d) do j
            e = copy(base); e[field[j]] = 1.0
            predictor(e) - b
        end...)
        (; base, b, A)
    end
    # An independent, bounded dense Newton reference from the observation law.
    # No native HVP or third-derivative result is used to find this mode.
    function oracle(point)
        (; base, b, A) = affine(point)
        z = zeros(d)
        objective(z) = sum(logpdf.(Poisson.(exp.(b + A*z)), data.y)) - sum(abs2,z)/2
        for iteration in 1:50
            rate = exp.(b + A*z)
            oracle_gradient = A' * (data.y - rate) - z
            oracle_precision = Matrix{Float64}(I, d, d) + A' * Diagonal(rate) * A
            norm(oracle_gradient) < 2e-12 && break
            step = oracle_precision \ oracle_gradient
            decrement = dot(oracle_gradient, step)
            before = objective(z)
            accepted = false
            for half in 0:40
                scale = 2.0^-half
                candidate = z + scale*step
                if decrement < 1e-12 || objective(candidate) >= before+1e-4*scale*decrement
                    z = candidate
                    accepted = true
                    break
                end
            end
            accepted || error("dense oracle line search failed")
            iteration == 50 && error("dense oracle did not converge")
        end
        oracle_precision = Matrix{Float64}(I, d, d) + A' * Diagonal(exp.(b+A*z)) * A
        prior0 = BRM._rk_nlme_population_prior_value(m.population_prior, base)
        value = prior0 + objective(z) + d*log(2π)/2 - logdet(Symmetric(oracle_precision))/2
        (; value, z, H=oracle_precision, A, rate=exp.(b+A*z))
    end
    expected = oracle(u)
    state = shared_poisson_mode(op, u)
    @test state.mode[field] ≈ expected.z atol=2e-10 rtol=2e-10
    @test shared_poisson_mode(op, u).mode == state.mode
    @test norm(state.gradient[field]) < 1e-10
    v = randn(Xoshiro(44), d)
    @test shared_precision(op, state.mode, v) ≈ expected.H*v rtol=2e-10
    native_third = shared_third(op, state.mode, v)
    @test native_third[field] ≈ -expected.A' *
        (expected.rate .* (expected.A*v).^2) rtol=2e-9 atol=1e-10
    @test norm(native_third[field]) > 1e-5

    probes = [sqrt(d) .* Matrix{Float64}(I,d,d)[:,j] for j in 1:d]
    trace = shared_trace(op, state.mode, probes).trace
    H(v) = shared_precision(op, state.mode, v)
    function solve(v)
        answer, info = KrylovKit.linsolve(H, v; issymmetric=true, isposdef=true,
            rtol=1e-12, atol=1e-13, krylovdim=128, maxiter=1)
        info.converged == 1 || error("conditional solve did not converge")
        answer
    end
    correction = shared_likelihood_hvp(op, state.mode, solve(trace[field]))
    native_gradient = state.gradient + (trace + correction)/2
    differences = map(outer) do j
        plus, minus = copy(u), copy(u)
        plus[j] += 2e-5; minus[j] -= 2e-5
        (oracle(plus).value-oracle(minus).value)/4e-5
    end
    @test native_gradient[outer] ≈ differences rtol=2e-6 atol=2e-7
    # A negative control observes the implicit correction, not just the direct
    # determinant derivative at a fixed mode.
    @test norm(correction[outer]) > 1e-4
    @test norm((state.gradient + trace/2)[outer] - differences) > 1e-5
    root = BRM._brm_spl_covariance_root(solve, v; rtol=1e-10, maxdim=16)
    E = eigen(Symmetric(expected.H))
    root_exact = E.vectors * ((E.vectors'*v) ./ sqrt.(E.values))
    @test root.draw ≈ root_exact rtol=2e-9 atol=1e-10
    println((; mode_iterations=state.iteration, mode_operations=state.operations,
        gradient_max_error=maximum(abs.(native_gradient[outer]-differences)),
        implicit_correction_norm=norm(correction[outer]),
        root_error=norm(root.draw-root_exact), root_iterations=root.iterations,
        scope="small nonlinear joint-Laplace control; no SVI/scaling acceptance"))
end
