# test/matrix_parameters.jl — declared matrix parameters (snag
# `hidden-markov-ob-6b35403a`): `W[1:m, 1:n] .~ Family.(args...)`, and
# `eachrow(G[1:m, 1:n]) .~ Dirichlet(n, a)` / `eachcol(G[1:m, 1:n]) .~ Dirichlet(m, a)`.
#
# Run: julia --startup-file=no --project=test test/matrix_parameters.jl [filter...]
#
# Each declaration prints back as written. An elementwise matrix has the same
# coordinates on both backends, so its density and gradient must agree; on
# SBBRMI its density without Jacobian is the product of its elements' priors.
# A matrix of simplex rows or columns has each slice's Dirichlet density on
# SBBRMI, and on RKBRMI restores to rows (columns) that sum to one, read by the
# model as the matrix. Extents may be literals or computed from data.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

points(n) = (zeros(n), fill(0.25, n), collect(range(-0.6, 0.6; length=n)))

function stan_values(stan, u)
    names = BridgeStan.param_names(stan.model; include_tp=true)
    Dict(zip(names, BridgeStan.param_constrain(stan.model, u; include_tp=true)))
end
stan_matrix(values, name, m, n) = [values["$name.$i.$j"] for i in 1:m, j in 1:n]
function rk_matrix(rk, u, name)
    only(getproperty(rk_restore_draws(rk, reshape(u, :, 1)), name))
end

const MATRIX_NORMAL = @brm begin
    @plate for s in eachindex(y)
        y[s] ~ Normal(W[1, 2] - W[2, 3] + sum(W[2, :]), sigma)
    end
    W[1:m, 1:(m + 1)] .~ Normal.(0.5, 1.5)
    sigma ~ LogNormal(0, 1)
end

const MATRIX_ROWS = @brm begin
    @plate for s in eachindex(y)
        y[s] ~ Normal(G[1, 2] + 2 * G[2, 3], 0.5)
    end
    eachrow(G[1:2, 1:L]) .~ Dirichlet(L, 2)
end

const MATRIX_COLS = @brm begin
    @plate for s in eachindex(y)
        y[s] ~ Normal(G[1, 2] + 2 * G[3, 1], 0.5)
    end
    eachcol(G[1:L, 1:2]) .~ Dirichlet([1.0, 2.0, 3.0])
end

const DATA = (; y=[[0.3, -0.2], [0.9], [0.1, -0.5, 0.4]], m=2, L=3)

@stestset "declared matrices print as written" begin
    @test endswith(sprint(show, MATRIX_NORMAL), "W[1:m, 1:m + 1] .~ Normal.(0.5, 1.5)\nsigma ~ LogNormal(0, 1)")
    @test endswith(sprint(show, MATRIX_ROWS), "eachrow(G[1:2, 1:L]) .~ Dirichlet(L, 2)")
    @test endswith(sprint(show, MATRIX_COLS), "eachcol(G[1:L, 1:2]) .~ Dirichlet([1.0, 2.0, 3.0])")
end

@stestset "an elementwise matrix: SBBRMI and RKBRMI agree" begin
    brmi = MATRIX_NORMAL(DATA)
    backend, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "matrix_normal")
    names = ReactiveKernelsPPL.coordinate_names(backend.model.layout)
    # One coordinate per element, column-major, under the same names.
    @test sort(string.(names)) == sort(BridgeStan.param_unc_names(stan.model))
    @test filter(!=(:sigma), names) == Symbol.(["W.1.1", "W.2.1", "W.1.2", "W.2.2", "W.1.3", "W.2.3"])
    mapping = [n => string(n) for n in names]
    for u in points(length(names))
        check_consumer_stan(problem, stan, mapping, backend, u)
        values = stan_values(stan, u)
        W, sigma = stan_matrix(values, "W", 2, 3), values["sigma"]
        @test W == reshape(u[1:6], 2, 3)
        mu = W[1, 2] - W[2, 3] + sum(W[2, :])
        @test BridgeStan.log_density(stan.model, u; propto=false, jacobian=false) ≈
            sum(logpdf.(Normal(0.5, 1.5), W)) + logpdf(LogNormal(0, 1), sigma) +
            sum(logpdf(Normal(mu, sigma), v) for y in DATA.y for v in y) atol=1e-10
    end
    # The extents are data: another `m` sizes the same program.
    wide = consumer_stan(MATRIX_NORMAL((; DATA..., m=3)), "matrix_normal_3")
    @test BridgeStan.param_unc_num(wide.model) == 3 * 4 + 1
end

@stestset "a matrix of simplex rows or columns" begin
    for (model, name, slices, mean) in (
            (MATRIX_ROWS, "matrix_rows", :rows, G -> G[1, 2] + 2 * G[2, 3]),
            (MATRIX_COLS, "matrix_cols", :cols, G -> G[1, 2] + 2 * G[3, 1]))
        brmi = model(DATA)
        stan = consumer_stan(brmi, name)
        # Two simplexes of three entries: two free coordinates each.
        @test BridgeStan.param_unc_num(stan.model) == 4
        rk = RKBRMI(brmi)
        problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test LogDensityProblems.dimension(problem) == 4
        query = BRM._rk_pointwise_query(rk)
        for u in points(4)
            G = stan_matrix(stan_values(stan, u), "G", slices === :rows ? 2 : 3,
                slices === :rows ? 3 : 2)
            parts = slices === :rows ? eachrow(G) : eachcol(G)
            @test all(p -> sum(p) ≈ 1 && all(>=(0), p), parts)
            law = slices === :rows ? Dirichlet(3, 2.0) : Dirichlet([1.0, 2.0, 3.0])
            mu = mean(G)
            @test BridgeStan.log_density(stan.model, u; propto=false, jacobian=false) ≈
                sum(logpdf(law, collect(p)) for p in parts) +
                sum(logpdf(Normal(mu, 0.5), v) for y in DATA.y for v in y) atol=1e-10
            # RK reads the same matrix: its slices are simplexes, and each
            # observation's density reads the restored matrix.
            R = rk_matrix(rk, u, :G)
            @test size(R) == size(G)
            @test all(p -> sum(p) ≈ 1 && all(>=(0), p), slices === :rows ? eachrow(R) : eachcol(R))
            # Each cell's entries, each with its own pointwise density.
            @test reduce(vcat, BRM._rk_pointwise_values(query, u).y) ≈
                [logpdf(Normal(mean(R), 0.5), v) for y in DATA.y for v in y] atol=1e-12
            _, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            fd = map(eachindex(u)) do j
                plus, minus = copy(u), copy(u); plus[j] += 1e-5; minus[j] -= 1e-5
                (LogDensityProblems.logdensity(problem, plus) -
                 LogDensityProblems.logdensity(problem, minus)) / 2e-5
            end
            @test gradient ≈ fd atol=1e-6 rtol=1e-6
        end
        check_rk_source_roundtrip(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    end
end

@stestset "a data-sized Dirichlet(K, a)" begin
    model = @brm begin
        @plate for s in eachindex(y)
            y[s] ~ Normal(p[1] - p[2], 0.5)
        end
        p ~ Dirichlet(K, 1.5)
    end
    data = (; y=DATA.y, K=4)
    stan = consumer_stan(model(data), "dirichlet_data_K")
    @test BridgeStan.param_unc_num(stan.model) == 3
    rk = RKBRMI(model(data))
    @test ReactiveKernelsPPL.coordinate_names(rk.model.layout) == Symbol.(["p.1", "p.2", "p.3"])
end

@stestset "malformed matrix declarations are refused" begin
    @test_throws "a declared index range must be `1:n`" BRM.brm(:(begin
        @plate for s in eachindex(y)
            y[s] ~ Normal(W[1, 1], 1.0)
        end
        W[1:2, 2:3] .~ Normal.(0.0, 1.0)
    end); mod=@__MODULE__)
    sliced_scalar = @brm begin
        @plate for s in eachindex(y)
            y[s] ~ Normal(G[1, 1], 1.0)
        end
        eachrow(G[1:2, 1:3]) .~ Normal(0.0, 1.0)
    end
    @test_throws "draws each slice from `Dirichlet(...)`" SBBRMI(sliced_scalar(DATA); mod=@__MODULE__)
    mismatch = @brm begin
        @plate for s in eachindex(y)
            y[s] ~ Normal(G[1, 1], 1.0)
        end
        eachrow(G[1:2, 1:3]) .~ Dirichlet(2, 1)
    end
    @test_throws "each row has 3 entries" SBBRMI(mismatch(DATA); mod=@__MODULE__)
    @test_throws "each row has 3 entries" RKBRMI(mismatch(DATA))
end

@stestset "declared matrices in the plan and the description" begin
    # The matrix is a prior (its Stan rows `G_rows`), never an observation.
    plan = generative_plan(MATRIX_ROWS, DATA; mod=@__MODULE__)
    roles = Dict(d.target => d.role for d in plan.declarations)
    @test roles[:G_rows] === :prior
    @test roles[:y] === :observation
    @test !any(d -> d.role === :observation && d.target in (:G, :G_rows), plan.declarations)
    elementwise = generative_plan(MATRIX_NORMAL, DATA; mod=@__MODULE__)
    @test Dict(d.target => d.role for d in elementwise.declarations)[:W] === :prior
    for model in (MATRIX_NORMAL, MATRIX_ROWS, MATRIX_COLS)
        markdown = BRM.brm_description_markdown(BRM.brm_description(
            SBBRMI(model(DATA); mod=@__MODULE__)))
        @test !occursin("Incomplete", markdown)
    end
end
