# test/hidden_markov.jl — `y[s] ~ HiddenMarkov(initial, transition, emissions)`
# in a `@plate` cell (snag `hidden-markov-ob-6b35403a`, design decision `0sl1z6o`).
#
# Run: julia --startup-file=no --project=test test/hidden_markov.jl [filter...]
#
# The Julia law is checked against the sum over every state path. Each model is
# then checked on both backends against that law:
#
# - SBBRMI: the BridgeStan density without its Jacobian is the log joint at the
#   constrained values, each sequence's `obs_likelihood` is its law's log
#   density, the gradient agrees with central differences, and `obs_gen` draws
#   sequences of the observed lengths in the emission support;
# - RKBRMI: each sequence's pointwise value at the restored values is its law's
#   log density, the native Enzyme gradient agrees with central differences,
#   and the printed program replays.
#
# Models whose parameters have the same coordinates on both backends also check
# density and gradient parity. The four models of the learnbayes.se notebook
# "Introduction to Hidden Markov models" (switching, softmax emissions, several
# sequences, hierarchical) are written for any number of states K and checked
# at K = 3 on seeded toy data.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra, Random

# ---- references ---------------------------------------------------------------

# The log density summed over every state path, by enumeration.
function path_logpdf(d::HiddenMarkov, y)
    K, T = length(d.initial), length(y)
    T == 0 && return 0.0
    weight(x) = d.log_weights ? x : log(x)
    at(A, idx...) = A[(size(A, i) == 1 ? 1 : idx[i] for i in 1:ndims(A))...]
    move(i, j, t) = ndims(d.transition) == 2 ? at(d.transition, i, j) :
        at(d.transition, i, j, t)
    emission(k, t) = ndims(d.emissions) == 1 ? d.emissions[k] : at(d.emissions, k, t)
    terms = Float64[]
    for path in Iterators.product(ntuple(_ -> 1:K, T)...)
        lp = weight(d.initial[path[1]]) + logpdf(emission(path[1], 1), y[1])
        for t in 2:T
            lp += weight(move(path[t - 1], path[t], t - 1)) + logpdf(emission(path[t], t), y[t])
        end
        push!(terms, lp)
    end
    logsumexp(terms)
end

column_softmax(D) = exp.(D) ./ sum(exp.(D); dims=1)
simplex(rng, K) = (w = rand(rng, K) .+ 0.2; w ./ sum(w))
stochastic_rows(rng, K) = permutedims(hcat((simplex(rng, K) for _ in 1:K)...))

function central_difference(f, u; step=1e-5)
    map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += step; minus[j] -= step
        (f(plus) - f(minus)) / (2step)
    end
end

points(n) = (zeros(n), fill(0.25, n), collect(range(-0.6, 0.6; length=n)))

# The constrained values of a BridgeStan point by name, transformed parameters
# and generated quantities included.
function stan_point(stan, u)
    names = BridgeStan.param_names(stan.model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(stan.model, u; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(stan.model, 7))
    Dict(zip(names, values))
end
# A reader of named values: `get(:mu, K)` a vector, `get(:G, m, n)` a matrix.
stan_reader(point) = (name, dims...) -> isempty(dims) ? point[string(name)] :
    length(dims) == 1 ? [point["$name.$i"] for i in 1:only(dims)] :
    [point["$name.$i.$j"] for i in 1:dims[1], j in 1:dims[2]]
function rk_reader(rk, u)
    restored = rk_restore_draws(rk, reshape(u, :, 1))
    (name, dims...) -> (v = getproperty(restored, name);
        v isa AbstractVector{<:AbstractMatrix} ? only(v) :
        v isa AbstractMatrix ? vec(v) : only(v))
end

# SBBRMI against the Julia law: `theta(get)` reads the model's values, `prior`
# and `laws` give the log prior and one `HiddenMarkov` per sequence.
function check_sb_hmm(stan, data, theta, prior, laws; response=:obs, support=nothing)
    sequences = getproperty(data, response)
    n = BridgeStan.param_unc_num(stan.model)
    for u in points(n)
        point = stan_point(stan, u)
        θ = theta(stan_reader(point))
        expected = [logpdf(d, y) for (d, y) in zip(laws(θ), sequences)]
        @test BridgeStan.log_density(stan.model, u; propto=false, jacobian=false) ≈
            prior(θ) + sum(expected) atol=1e-10 rtol=1e-12
        @test [point["$(response)_likelihood.$s"] for s in eachindex(sequences)] ≈
            expected atol=1e-10 rtol=1e-12
        _, gradient = BridgeStan.log_density_gradient(stan.model, u; propto=false, jacobian=true)
        @test gradient ≈ central_difference(
            v -> BridgeStan.log_density(stan.model, v; propto=false, jacobian=true), u) atol=1e-6 rtol=1e-6
        # One forward-simulated sequence of the observed length per cell.
        draws = [point["$(response)_gen.$i"] for i in 1:sum(length, sequences)]
        @test all(isfinite, draws)
        support === nothing || @test all(in(support), draws)
    end
end

# RKBRMI against the Julia law, and its printed program's replay.
function check_rk_hmm(rk, data, theta, laws; response=:obs)
    sequences = getproperty(data, response)
    ad = AutoEnzyme(; mode=Enzyme.Reverse)
    check_rk_source_roundtrip(rk; ad_backend=ad)
    problem = rk_logdensity_problem(rk; ad_backend=ad)
    query = BRM._rk_pointwise_query(rk)
    for u in points(LogDensityProblems.dimension(problem))
        θ = theta(rk_reader(rk, u))
        pointwise = getproperty(BRM._rk_pointwise_values(query, u), response)
        @test pointwise ≈ [logpdf(d, y) for (d, y) in zip(laws(θ), sequences)] atol=1e-10 rtol=1e-12
        _, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test gradient ≈ central_difference(v -> LogDensityProblems.logdensity(problem, v), u) atol=1e-6 rtol=1e-6
    end
end

# Density and gradient parity of two backends whose coordinates share names.
function check_hmm_parity(brmi, name)
    backend, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, name)
    names = ReactiveKernelsPPL.coordinate_names(backend.model.layout)
    mapping = [n => string(n) for n in names]
    for u in points(length(names))
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    backend, stan
end

# ---- the Julia law --------------------------------------------------------------

@stestset "the Julia law is the sum over state paths" begin
    rng = Xoshiro(20261113)
    K, M, T = 3, 4, 5
    init, G = simplex(rng, K), stochastic_rows(rng, K)
    phis = [simplex(rng, M) for _ in 1:K]
    y = rand(rng, 1:M, T)
    d = HiddenMarkov(init, G, Categorical.(phis))
    @test d isa Distribution{Multivariate,Discrete}
    @test logpdf(d, y) ≈ path_logpdf(d, y) atol=1e-12
    # Log weights: the same chain on the log scale.
    dl = HiddenMarkov(log.(init), log.(G), Categorical.(phis); log_weights=true)
    @test logpdf(dl, y) ≈ logpdf(d, y) atol=1e-12
    @test logpdf(dl, y) ≈ path_logpdf(dl, y) atol=1e-12
    # Emissions that change with the step, and transitions that change with it.
    x = randn(rng, T)
    E = [Normal(0.5k + 0.3x[t], 0.7 + 0.1k) for k in 1:K, t in 1:T]
    G3 = cat((stochastic_rows(rng, K) for _ in 1:(T - 1))...; dims=3)
    z = randn(rng, T)
    dt = HiddenMarkov(init, G3, E)
    @test dt isa Distribution{Multivariate,Continuous}
    @test logpdf(dt, z) ≈ path_logpdf(dt, z) atol=1e-12
    # A singleton step axis is shared by every step.
    @test logpdf(HiddenMarkov(init, reshape(G, K, K, 1), E), z) ≈
        logpdf(HiddenMarkov(init, G, E), z) atol=1e-12
    # Mixed families, one per state.
    dm = HiddenMarkov(init, G, [Normal(1.0, 0.5), Gamma(2.0, 1.5), Exponential(0.8)])
    w = rand(rng, T) .* 3 .+ 0.1
    @test logpdf(dm, w) ≈ path_logpdf(dm, w) atol=1e-12
    # Edge lengths.
    @test logpdf(d, Int[]) == 0.0
    @test logpdf(d, [2]) ≈ logsumexp(log.(init) .+ [logpdf(p, 2) for p in Categorical.(phis)]) atol=1e-14
    # A forward simulation draws a path and its observations.
    draw = BRM._brm_hidden_markov_draw(Xoshiro(1), d, 50)
    @test length(draw.y) == length(draw.z) == 50
    @test all(in(1:M), draw.y) && all(in(1:K), draw.z)
    @test_throws DimensionMismatch HiddenMarkov(init, G, Categorical.(phis[1:2]))
    @test_throws DimensionMismatch HiddenMarkov(init, ones(K, K + 1), Categorical.(phis))
end

# ---- the learnbayes.se models at general K --------------------------------------
#
# The notebook writes the transition Θ and the emission matrix Φ column-stochastic
# (column i is state i's next-state law, column k state k's symbol law). Its
# switching model draws every column from `Dirichlet(K, 1)`; the softmax models
# take Φ as the column softmax of `Diagonal(Φ_diag)`, `Φ_diag[k] ~ N(0, 1)` on
# [0, ∞); the hierarchical model gives individual j the transition whose column
# i is the softmax of `[μ[:, i] + τ[:, i] .* η_j[:, i]; 0]`.

const HMM_SWITCHING = @brm begin
    @plate for s in eachindex(obs)
        obs[s] ~ HiddenMarkov(init, permutedims(Theta), Categorical.(eachcol(Phi)))
    end
    init ~ Dirichlet(K, 1)
    eachcol(Theta[1:K, 1:K]) .~ Dirichlet(K, 1)
    eachcol(Phi[1:M, 1:K]) .~ Dirichlet(M, 1)
end

const HMM_SOFTMAX = @brm begin
    @plate for s in eachindex(obs)
        E = ones(K, K) - diagm(ones(K)) + diagm(exp.(Phi_diag))
        P = E * diagm(1 ./ (transpose(E) * ones(K)))
        obs[s] ~ HiddenMarkov(init, permutedims(Theta), Categorical.(eachcol(P)))
    end
    init ~ Dirichlet(K, 1)
    eachcol(Theta[1:K, 1:K]) .~ Dirichlet(K, 1)
    Phi_diag[1:K] .~ truncated.(Normal.(0, 1), 0, Inf)
end

const HMM_HIERARCHICAL = @brm begin
    @plate for s in eachindex(obs)
        Etheta = vcat(reshape(exp.(mu + tau .* eta[:, who[s]]), K - 1, K), ones(1, K))
        Theta = Etheta * diagm(1 ./ (transpose(Etheta) * ones(K)))
        E = ones(K, K) - diagm(ones(K)) + diagm(exp.(Phi_diag))
        P = E * diagm(1 ./ (transpose(E) * ones(K)))
        obs[s] ~ HiddenMarkov(init, permutedims(Theta), Categorical.(eachcol(P)))
    end
    init ~ Dirichlet(K, 1)
    Phi_diag[1:K] .~ truncated.(Normal.(0, 1), 0, Inf)
    mu[1:((K - 1) * K)] .~ Normal.(0, 1)
    tau[1:((K - 1) * K)] .~ truncated.(Cauchy.(0, 1), 0, Inf)
    eta[1:((K - 1) * K), 1:J] .~ Normal.(0, 1)
end

const HALF_NORMAL = truncated(Normal(0, 1), 0, Inf)
softmax_emissions(d) = column_softmax(Matrix(Diagonal(d)))
individual_transition(θ, K, j) =
    column_softmax(vcat(reshape(θ.mu .+ θ.tau .* θ.eta[:, j], K - 1, K), zeros(1, K)))

# Seeded toy sequences from a K-state chain (no notebook data).
function hmm_sequences(rng, K, M, lengths; who=nothing, J=1)
    init = simplex(rng, K)
    Gs = [stochastic_rows(rng, K) for _ in 1:J]
    phis = [simplex(rng, M) for _ in 1:K]
    individuals = who === nothing ? ones(Int, length(lengths)) : who
    [BRM._brm_hidden_markov_draw(rng, HiddenMarkov(init, Gs[j], Categorical.(phis)), T).y
     for (T, j) in zip(lengths, individuals)]
end

@stestset "switching: Dirichlet transition and emission columns" begin
    K, M = 3, 3
    data = (; obs=hmm_sequences(Xoshiro(1), K, M, [6, 4, 7]), K, M)
    @test sprint(show, HMM_SWITCHING) == """
        @plate for s in eachindex(obs)
            obs[s] ~ HiddenMarkov(init, permutedims(Theta), Categorical.(eachcol(Phi)))
        end
        init ~ Dirichlet(K, 1)
        eachcol(Theta[1:K, 1:K]) .~ Dirichlet(K, 1)
        eachcol(Phi[1:M, 1:K]) .~ Dirichlet(M, 1)"""
    theta(get) = (; init=get(:init, K), Theta=get(:Theta, K, K), Phi=get(:Phi, M, K))
    prior(θ) = logpdf(Dirichlet(K, 1.0), θ.init) +
        sum(logpdf(Dirichlet(K, 1.0), c) for c in eachcol(θ.Theta)) +
        sum(logpdf(Dirichlet(M, 1.0), c) for c in eachcol(θ.Phi))
    laws(θ) = fill(HiddenMarkov(θ.init, permutedims(θ.Theta),
        Categorical.(collect.(eachcol(θ.Phi)))), length(data.obs))
    brmi = HMM_SWITCHING(data)
    check_sb_hmm(consumer_stan(brmi, "hmm_switching"), data, theta, prior, laws; support=1:M)
    check_rk_hmm(RKBRMI(brmi), data, theta, laws)
end

@stestset "softmax emissions: one sequence and several" begin
    K = 3
    theta(get) = (; init=get(:init, K), Theta=get(:Theta, K, K), Phi_diag=get(:Phi_diag, K))
    prior(θ) = logpdf(Dirichlet(K, 1.0), θ.init) +
        sum(logpdf(Dirichlet(K, 1.0), c) for c in eachcol(θ.Theta)) +
        sum(logpdf.(HALF_NORMAL, θ.Phi_diag))
    for (label, lengths) in (("single", [9]), ("multi", [5, 3, 6, 4]))
        data = (; obs=hmm_sequences(Xoshiro(2), K, K, lengths), K)
        laws(θ) = fill(HiddenMarkov(θ.init, permutedims(θ.Theta),
            Categorical.(collect.(eachcol(softmax_emissions(θ.Phi_diag))))), length(data.obs))
        brmi = HMM_SOFTMAX(data)
        check_sb_hmm(consumer_stan(brmi, "hmm_softmax_$label"), data, theta, prior, laws;
            support=1:K)
        check_rk_hmm(RKBRMI(brmi), data, theta, laws)
    end
end

@stestset "hierarchical: per-individual transitions" begin
    K, J = 3, 2
    who = [1, 2, 1, 2, 2]
    data = (; obs=hmm_sequences(Xoshiro(3), K, K, [5, 4, 6, 3, 5]; who, J), who, K, J)
    L = (K - 1) * K
    theta(get) = (; init=get(:init, K), Phi_diag=get(:Phi_diag, K), mu=get(:mu, L),
        tau=get(:tau, L), eta=get(:eta, L, J))
    prior(θ) = logpdf(Dirichlet(K, 1.0), θ.init) + sum(logpdf.(HALF_NORMAL, θ.Phi_diag)) +
        sum(logpdf.(Normal(0, 1), θ.mu)) + sum(logpdf.(truncated(Cauchy(0, 1), 0, Inf), θ.tau)) +
        sum(logpdf.(Normal(0, 1), θ.eta))
    laws(θ) = [HiddenMarkov(θ.init, permutedims(individual_transition(θ, K, j)),
        Categorical.(collect.(eachcol(softmax_emissions(θ.Phi_diag))))) for j in who]
    brmi = HMM_HIERARCHICAL(data)
    check_sb_hmm(consumer_stan(brmi, "hmm_hierarchical"), data, theta, prior, laws; support=1:K)
    check_rk_hmm(RKBRMI(brmi), data, theta, laws)
end

# ---- emission spellings, parity of the two backends -------------------------------

const GAUSSIAN_DATA = let rng = Xoshiro(4)
    init0, G0 = [0.6, 0.4], [0.85 0.15; 0.25 0.75]
    lengths = [7, 1, 5]
    (; y=[randn(rng, T) .+ 1.0 for T in lengths], x=[randn(rng, T) for T in lengths],
       w=[rand(rng, T) .* 3 .+ 0.1 for T in lengths],
       init0, G0, log_init0=log.(init0), log_G0=log.(G0), K=2)
end

const HMM_BROADCAST = @brm begin
    @plate for s in eachindex(y)
        y[s] ~ HiddenMarkov(init0, G0, Normal.(mu, sigma))
    end
    mu[1:K] .~ Normal.(0, 2)
    sigma ~ LogNormal(0, 1)
end
const HMM_COMPREHENSION = @brm begin
    @plate for s in eachindex(y)
        y[s] ~ HiddenMarkov(init0, G0, [Normal(mu[k], sigma) for k in 1:K])
    end
    mu[1:K] .~ Normal.(0, 2)
    sigma ~ LogNormal(0, 1)
end
const HMM_LOG_WEIGHTS = @brm begin
    @plate for s in eachindex(y)
        y[s] ~ HiddenMarkov(log_init0, log_G0, Normal.(mu, sigma); log_weights = true)
    end
    mu[1:K] .~ Normal.(0, 2)
    sigma ~ LogNormal(0, 1)
end
const HMM_STEPS = @brm begin
    @plate for s in eachindex(y)
        y[s] ~ HiddenMarkov(init0, G0, Normal.(mu .+ beta .* transpose(x[s]), sigma))
    end
    mu[1:K] .~ Normal.(0, 2)
    beta ~ Normal(0, 1)
    sigma ~ LogNormal(0, 1)
end
const HMM_LISTED = @brm begin
    @plate for s in eachindex(w)
        w[s] ~ HiddenMarkov(init0, G0, [LogNormal(m, sigma), Gamma(2.0, theta)])
    end
    m ~ Normal(0, 1)
    sigma ~ LogNormal(0, 1)
    theta ~ LogNormal(0, 1)
end

@stestset "emission spellings agree on both backends" begin
    data = GAUSSIAN_DATA
    K = data.K
    normal_laws(θ) = fill(HiddenMarkov(data.init0, data.G0, Normal.(θ.mu, θ.sigma)), length(data.y))
    gaussian(get) = (; mu=get(:mu, K), sigma=get(:sigma))
    cases = (
        (HMM_BROADCAST, "hmm_broadcast", gaussian, normal_laws, :y),
        (HMM_COMPREHENSION, "hmm_comprehension", gaussian, normal_laws, :y),
        (HMM_LOG_WEIGHTS, "hmm_log_weights", gaussian, normal_laws, :y),
        (HMM_STEPS, "hmm_steps", get -> (; mu=get(:mu, K), beta=get(:beta), sigma=get(:sigma)),
            θ -> [HiddenMarkov(data.init0, data.G0,
                [Normal(θ.mu[k] + θ.beta * x[t], θ.sigma) for k in 1:K, t in eachindex(x)])
                for x in data.x], :y),
        (HMM_LISTED, "hmm_listed", get -> (; m=get(:m), sigma=get(:sigma), theta=get(:theta)),
            θ -> fill(HiddenMarkov(data.init0, data.G0,
                [LogNormal(θ.m, θ.sigma), Gamma(2.0, θ.theta)]), length(data.w)), :w),
    )
    for (model, name, theta, laws, response) in cases
        brmi = model(data)
        backend, stan = check_hmm_parity(brmi, name)
        point(u) = theta(rk_reader(backend, u))
        query = BRM._rk_pointwise_query(backend)
        for u in points(BridgeStan.param_unc_num(stan.model))
            θ = point(u)
            expected = [logpdf(d, y) for (d, y) in zip(laws(θ), getproperty(data, response))]
            @test getproperty(BRM._rk_pointwise_values(query, u), response) ≈ expected atol=1e-10 rtol=1e-12
            # Stan's own point and values (its coordinates are ordered differently).
            sb = stan_point(stan, u)
            stan_expected = [logpdf(d, y) for (d, y) in
                zip(laws(theta(stan_reader(sb))), getproperty(data, response))]
            @test [sb["$(response)_likelihood.$s"] for s in eachindex(expected)] ≈
                stan_expected atol=1e-10 rtol=1e-12
        end
    end
    # The two log-weight spellings are one law.
    a = consumer_stan(HMM_BROADCAST(data), "hmm_broadcast")
    b = consumer_stan(HMM_LOG_WEIGHTS(data), "hmm_log_weights")
    for u in points(3)
        @test BridgeStan.log_density(a.model, u; propto=false) ≈
            BridgeStan.log_density(b.model, u; propto=false) atol=1e-12
    end
end

@stestset "discrete emissions from a mapped and an unmapped family" begin
    rng = Xoshiro(5)
    data = (; c=[rand(rng, 0:6, T) for T in (6, 3)], init0=[0.5, 0.5], G0=[0.9 0.1; 0.3 0.7], K=2)
    model = @brm begin
        @plate for s in eachindex(c)
            c[s] ~ HiddenMarkov(init0, G0, Poisson.(lambda))
        end
        lambda[1:K] .~ LogNormal.(0, 1)
    end
    backend, stan = check_hmm_parity(model(data), "hmm_poisson")
    query = BRM._rk_pointwise_query(backend)
    for u in points(2)
        λ = rk_reader(backend, u)(:lambda, 2)
        d = HiddenMarkov(data.init0, data.G0, Poisson.(λ))
        @test BRM._rk_pointwise_values(query, u).c ≈ [path_logpdf(d, y) for y in data.c] atol=1e-10
    end
end

# ---- the RK emission plate builds each state's law once ---------------------------
#
# A law whose arguments are read whole per state (`eachcol(Phi)`, `eachrow(Phi)`,
# `Ref(x)`, a listed law) reads the state and not the step. RKBRMI then builds its
# column as a plate over the steps whose cell evaluates the law, so RK evaluates
# the law once per state, not once per (step, state) cell: a column slice and its
# simplex check per cell cost the categorical notebook models 2–3× (snag
# `rkbrmi-hiddenmar-7ebba16b`). A broadcast argument read per entry
# (`Normal.(mu, sigma)`) reads the step, so its law is built per cell anyway; it
# keeps one plate cell per (step, state), which costs less than nesting. The
# densities and gradients are checked above; this pins the shapes.

const RK_EXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)

# Every `plate(range) do var … end` in `ex`, outermost first.
function hmm_plates(ex, found=[])
    if Meta.isexpr(ex, :do)
        call, lambda = ex.args
        push!(found, (; range=call.args[2], var=only(lambda.args[1].args), body=lambda.args[2]))
    end
    ex isa Expr && foreach(a -> hmm_plates(a, found), ex.args)
    found
end
reads(ex, x) = ex == x || (ex isa Expr && any(a -> reads(a, x), ex.args))

function hmm_density_body(emission)
    law = BRM._brm_hmm_law(BRM._brm_hmm_emission(emission, name -> getfield(@__MODULE__, name)), false)
    F, M, B = law.parameters
    RK_EXT._rk_hmm_density_body(fieldtypes(F), M, B)
end

@stestset "RK builds each state's emission law once per sequence" begin
    # A broadcast: a plate over the states around a plate over the steps.
    inputs, body = hmm_density_body(:(Categorical.(eachcol(Phi))))
    @test inputs == (:value, :initial, :transition, :argument_1)
    states, steps = hmm_plates(body)
    @test (states.range, states.var) == (:(1:hmm_states), :hmm_k)
    @test (steps.range, steps.var) == (:(1:hmm_steps), :hmm_t)
    @test reads(steps.body, :(argument_1[:, hmm_k])) && reads(steps.body, :(value[hmm_t]))
    @test !reads(body, :hmm_cell)
    # A state's row, and a law per state written out: one step plate per law, no
    # branch on the state.
    _, rows = hmm_density_body(:(Categorical.(eachrow(Phi))))
    @test reads(last(hmm_plates(rows)).body, :(argument_1[hmm_k, :]))
    _, listed = hmm_density_body(:([LogNormal(m, sigma), Gamma(2.0, theta)]))
    plates = hmm_plates(listed)
    @test length(plates) == 2 && all(p -> (p.range, p.var) == (:(1:hmm_steps), :hmm_t), plates)
    @test !reads(listed, :hmm_k)
    # An argument read per entry: one plate cell per (step, state).
    _, entries = hmm_density_body(:(Normal.(mu, sigma)))
    cells = only(hmm_plates(entries))
    @test (cells.range, cells.var) == (:(1:(hmm_steps * hmm_states)), :hmm_cell)
end

# ---- what is refused, and how it is shown ----------------------------------------

@stestset "malformed HiddenMarkov observations are refused by name" begin
    data = GAUSSIAN_DATA
    two = @brm begin
        @plate for s in eachindex(y)
            y[s] ~ HiddenMarkov(init0, Normal.(mu, 1.0))
        end
        mu[1:K] .~ Normal.(0, 1)
    end
    keyword = @brm begin
        @plate for s in eachindex(y)
            y[s] ~ HiddenMarkov(init0, G0, Normal.(mu, 1.0); log_scale = true)
        end
        mu[1:K] .~ Normal.(0, 1)
    end
    whole = @brm begin
        @plate for s in eachindex(y)
            y[s] ~ HiddenMarkov(init0, G0, Normal(mu[1], 1.0))
        end
        mu[1:K] .~ Normal.(0, 1)
    end
    for (model, message) in ((two, "write `HiddenMarkov(initial, transition, emissions)`"),
            (keyword, "unknown keyword `log_scale`"),
            (whole, "the emission argument is one law per state"))
        @test_throws message SBBRMI(model(data); mod=@__MODULE__)
        @test_throws message RKBRMI(model(data))
    end
end

@stestset "a hidden Markov model's plan, descriptor and description" begin
    data = (; obs=hmm_sequences(Xoshiro(1), 3, 3, [6, 4]), K=3, M=3)
    plan = generative_plan(HMM_SWITCHING, data; mod=@__MODULE__)
    roles = Dict(d.target => d.role for d in plan.declarations)
    @test roles[:obs] === :observation
    @test all(roles[name] === :prior for name in (:init, :Theta_cols, :Phi_cols))
    descriptor = brm_descriptor(SBBRMI(HMM_SWITCHING(data); mod=@__MODULE__))
    shown = sprint(show, descriptor)
    @test occursin("obs_gen :: posterior_predictive", shown)
    @test occursin("obs_likelihood :: pointwise_loglik", shown)
    markdown = BRM.brm_description_markdown(BRM.brm_description(
        SBBRMI(HMM_SWITCHING(data); mod=@__MODULE__)))
    @test !occursin("Incomplete", markdown)
    @test occursin("forward algorithm", markdown)
end
