# Synthetic models shared by the minibatch SVI tests (test/nlme_svi.jl,
# test/nlme_svi_optimizer.jl and its fresh-process script,
# test/nlme_svi_threads.jl).
using BayesianRegressionModels, Random
using LinearAlgebra: cholesky

# Whether two runs or results are the same to the last bit. The Adam moments
# are the run's gradients so far, so states compared at every iteration compare
# the gradient trajectory.
svi_same_adam(a, b) = all(n -> isequal(getfield(a, n), getfield(b, n)),
    fieldnames(typeof(a)))
svi_same_q(a, b) = all(n -> isequal(getfield(a, n), getfield(b, n)),
    (:family, :μ, :ρ, :Β, :M, :Λ, :U)) &&
    all(zip(a.unit_opt, b.unit_opt)) do (x, y)
        x === nothing ? y === nothing : y !== nothing && svi_same_adam(x, y)
    end
svi_same_run(a, b) = svi_same_adam(a.adam, b.adam) && svi_same_q(a.q, b.q) &&
    (a.average === nothing ? b.average === nothing :
        b.average !== nothing && svi_same_q(a.average, b.average)) &&
    all(n -> isequal(getfield(a.sampler, n), getfield(b.sampler, n)),
        fieldnames(typeof(a.sampler))) && a.rng == b.rng &&
    all(n -> isequal(getfield(a, n), getfield(b, n)),
        (:iteration, :trace, :windows, :polish_start, :converged, :sampling,
         :unit_gradient_calls, :averaged_through, :unit_entropy, :entropy_sum))
svi_same_state(a, b) = svi_same_run(a.run, b.run) && isequal(a.identity, b.identity) &&
    isequal(a.controls, b.controls)

function svi_same_result(a, b)
    da, db = a.diagnostics, b.diagnostics
    a.draws == b.draws && svi_same_q(da.q, db.q) && da.elbo == db.elbo &&
        da.elbo_windows == db.elbo_windows && da.iterations == db.iterations &&
        da.converged == db.converged && da.unit_gradient_calls == db.unit_gradient_calls
end

# Correlated random intercept and slope, `nobs` Normal observations per subject.
function svi_gauss_data(rng, N; nobs=5, a=1.0, b=0.5, sa=0.7, sb=0.3, rho=0.3,
        sigma=0.5)
    C = [sa^2 rho*sa*sb; rho*sa*sb sb^2]
    L = cholesky(C).L
    x = [randn(rng, nobs) for _ in 1:N]
    y = map(1:N) do i
        e = L * randn(rng, 2)
        a + e[1] .+ (b + e[2]) .* x[i] .+ sigma .* randn(rng, nobs)
    end
    (; subject=["s$i" for i in 1:N], x, y)
end

svi_gauss_model(data) = @brm data begin
    sigma ~ Exponential(1)
    a ~ 1 + (1 | g | subject)
    b ~ 1 + (1 | g | subject)
    @plate for i in eachindex(a)
        y[i] ~ Normal(a[i] .+ b[i] .* x[i], sigma)
    end
end

# `svi_gauss_model` calling a callable of the including script's scope, by the
# global name `shift`.
svi_callable_model(data, shift=:svi_shift) = BayesianRegressionModels.brm(quote
    sigma ~ Exponential(1)
    a ~ 1 + (1 | g | subject)
    b ~ 1 + (1 | g | subject)
    @plate for i in eachindex(a)
        y[i] ~ Normal($shift(a[i] .+ b[i] .* x[i]), sigma)
    end
end; mod=@__MODULE__)(data)

function svi_spacetime_data(rng; G=6, T=12)
    probe = String[]; day = Float64[]; c1 = Float64[]; trend = Float64[]
    for d in 1:T, g in 1:G
        push!(probe, "p$g"); push!(day, d); push!(c1, cos(2pi * d / 7)); push!(trend, d / T)
    end
    (; probe, day, c1, trend, y=3.0 .+ 0.5 .* c1 .+ 0.3 .* randn(rng, length(day)),
       probe_x=[100.0 * cos(g) for g in 1:G], probe_y=[100.0 * sin(g) for g in 1:G])
end

# Per-probe random effects, a grouped HSGP over days and AR(1) noise in a
# per-probe cell; `field` is the HSGP term and `extra` the declarations it reads.
svi_spacetime_model(field, extra...) = BayesianRegressionModels.brm(quote
    $(extra...)
    mu ~ 1 + c1 + (1 | lv | probe) + (0 + c1 | cc | probe) + $field
    tr ~ 0 + trend + (0 + trend | tt | probe)
    log(sigma) ~ 1 + (1 | ns | probe)
    phi ~ Uniform(-1.0, 1.0)
    @plate for p in groups(probe; order = day)
        m = mu[p] .+ tr[p]
        y[p] ~ AR1Normal(m, sigma[p], phi; time = day[p])
    end
end; mod=@__MODULE__)
