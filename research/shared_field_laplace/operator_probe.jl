# Native derivative control for shared multigroup HSGP collapse.
# Run after test/setup_env.jl, under one compute token:
#   julia --project=test research/shared_field_laplace/operator_probe.jl
# This is a research control, not an SVI entry point or a streaming algorithm.
# The operator uses one compact HVP direction, never a field Hessian. Dense
# matrices below are confined to the eight-coefficient Gaussian oracle.
include(joinpath(@__DIR__, "../../test/nlme_fixtures.jl"))
using LinearAlgebra, Random, LogDensityProblems
const KrylovKit = BRM.KrylovKit

function shared_field_fixture(; groups=3, rows=3, temporal=2, spatial=2, poisson=false,
        prepare_model=true)
    rng = Xoshiro(20261101)
    data = (; probe=repeat(1:groups; inner=rows),
        time=repeat(collect(range(0.1, 2.0; length=rows)), groups),
        sx=cos.(1.3 .* (1:groups)), sy=sin.(0.7 .* (1:groups)),
        y=poisson ? rand(rng, Poisson(2.0), groups * rows) : randn(rng, groups * rows))
    response = poisson ? :(y[p] ~ Poisson(exp.(mu[p]))) : :(y[p] ~ Normal(mu[p], 0.7))
    builder = BRM.brm(quote
        mu ~ 1 + (1 | probe) + hsgp(time; k=$temporal, by=probe,
            cor=multigroup((sx, sy), 2.0; k=$spatial))
        @plate for p in groups(probe)
            $response
        end
    end; mod=@__MODULE__)
    rk = RKBRMI(builder(data))
    m = prepare_model ? brm_nlme_model(rk; ad_backend=NLME_AD) : nothing
    (; rk, m, data)
end

function shared_field_operator(m)
    records = [r for r in BRM._rk_coordinate_records(m.backend.plan)
        if r.kind === :hsgp]
    length(records) == 1 || error("probe requires exactly one HSGP")
    record = only(records)
    record.grouped && record.level_correlation || error(
        "probe requires a shared, level-correlated HSGP: $(record.declaration)")
    weights = Symbol(record.declaration, ".", record.locals.weights)
    field = findall(n -> BRM._brm_nlme_owned(n, weights), m.view.coordinates)
    isempty(field) && error("no standardized weights found for $weights")
    all(i -> m.view.role[i] === :shared_latent, field) || error(
        "HSGP $weights is not a shared field")
    t = BRM._brm_svi_target(m)
    ads = map(eachindex(m.cell_query)) do g
        i = findfirst(s -> !isempty(s[g]), t.plan.selections)
        i === nothing && error("empty observation group in control")
        selection = first(t.plan.selections[i][g])
        BRM._rk_spl_cell_ad(m.backend, m.cell_query[g].names, copy(m.unit),
            selection, NLME_AD, 1)
    end
    (; m, t, field, ads)
end

# All likelihood calculations stay in the model's RK compact-cell graph.
function shared_value_gradient(op, u)
    value, gradient = BRM._rk_nlme_population_prior_value_gradient(
        op.m.population_prior, u)
    for (g, cell) in enumerate(op.m.cell_query), i in eachindex(op.t.units),
            (k, p) in enumerate(cell.positions[i])
        selection = op.t.plan.selections[i][g][k]
        buffer = op.t.plan.buffers[g]
        buffer.active .= u[selection]
        v, _ = BRM._rk_nlme_compact_cell_value_and_gradient!(op.t.plan.samplers[g],
            buffer.gradient, u, buffer.active, selection, p)
        value += v
        gradient[selection] .+= buffer.gradient
    end
    value, gradient
end

# The likelihood HVP in a field-only direction, including its mixed outer
# derivatives. The field's independent N(0,I) prior is added by precision().
function shared_likelihood_hvp(op, u, v)
    direction = zeros(length(u)); direction[op.field] .= v
    result = zeros(length(u))
    for (g, cell) in enumerate(op.m.cell_query), i in eachindex(op.t.units),
            (k, p) in enumerate(cell.positions[i])
        selection = op.t.plan.selections[i][g][k]
        _, (hv,) = BRM._rk_spl_cell_block(op.ads[g], (direction[selection],),
            u, selection, p)
        result[selection] .+= hv
    end
    result
end

shared_precision(op, u, v) = v .- shared_likelihood_hvp(op, u, v)[op.field]

# D³log p[v,v,:]; the Gaussian weight prior has no third derivative.
function shared_third(op, u, v)
    direction = zeros(length(u)); direction[op.field] .= v
    result = zeros(length(u))
    for (g, cell) in enumerate(op.m.cell_query), i in eachindex(op.t.units),
            (k, p) in enumerate(cell.positions[i])
        selection = op.t.plan.selections[i][g][k]
        _, _, tv = BRM._rk_spl_cell_third(op.ads[g], direction[selection],
            u, selection, p)
        result[selection] .+= tv
    end
    result
end

# This single Newton step is exact only for this conditionally Gaussian
# fixture. A general likelihood still needs a deterministic converged mode
# search; the control deliberately makes no claim to implement that search.
function shared_gaussian_mode(op, u; rtol=1e-10, maxiter=128)
    start = copy(u); start[op.field] .= 0.0
    _, gradient = shared_value_gradient(op, start)
    H(v) = shared_precision(op, start, v)
    z, info = KrylovKit.linsolve(H, gradient[op.field]; issymmetric=true,
        isposdef=true, rtol, atol=1e-12, krylovdim=maxiter, maxiter=1)
    info.converged == 1 || error("field CG did not converge: $(info.normres)")
    mode = copy(start); mode[op.field] .= z
    (; mode, info)
end

@stestset "shared HSGP native operator against a small Gaussian control" begin
    println((; stage="prepare", julia=string(VERSION), enzyme=pkgdir(Enzyme),
        libenzyme=Enzyme.Compiler.Enzyme_jll.libEnzyme_path)); flush(stdout)
    (; rk, m, data) = shared_field_fixture()
    op = shared_field_operator(m)
    rng = Xoshiro(39)
    u = first(BRM._brm_nlme_prior_median_point(m))
    u .+= 0.07 .* randn(rng, length(u))
    field, d = op.field, length(op.field)
    @test d == 8
    @test field ⊆ op.t.globals
    @test isempty(intersect(field, reduce(vcat, op.t.units)))
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    f, g = shared_value_gradient(op, u)
    f_full, g_full = LogDensityProblems.logdensity_and_gradient(full, u)
    @test f ≈ f_full rtol=2e-12
    @test g ≈ g_full rtol=2e-10

    # The public observation-argument query supplies the authored means for an
    # independent affine-design oracle. Long-format cells return one vector
    # per group; flatten in their order, the same as this grouped fixture.
    plan = RKEXT._rk_translated(rk)
    observations = prepare_query(rk.model, plan, :observations)
    mean_query(point) = reduce(vcat, Base.invokelatest(observations, point).y.location)
    start = copy(u); start[field] .= 0.0
    mu0 = Base.invokelatest(mean_query, start)
    A = hcat(map(1:d) do j
        point = copy(start); point[field[j]] = 1.0
        Base.invokelatest(mean_query, point) .- mu0
    end...)
    H = Matrix{Float64}(I, d, d) + A' * A / 0.7^2
    z_exact = H \ (A' * (data.y .- mu0) / 0.7^2)
    v = randn(rng, d)
    println((; stage="native HVP", field_coordinates=d)); flush(stdout)
    @test shared_precision(op, u, v) ≈ H * v rtol=2e-10
    @test shared_precision(op, start, v) ≈ H * v rtol=2e-10
    prior, gp = BRM._rk_nlme_population_prior_value_gradient(m.population_prior, u)
    @test gp[field] ≈ -u[field] rtol=2e-12

    result = shared_gaussian_mode(op, u)
    @test result.mode[field] ≈ z_exact rtol=2e-9 atol=1e-10
    @test norm(last(shared_value_gradient(op, result.mode))[field]) < 1e-8
    # A changed preceding call cannot alter the zero-start mode.
    shared_gaussian_mode(op, u .+ 0.03)
    @test shared_gaussian_mode(op, u).mode == result.mode

    # Gaussian marginal density and exact implicit gradient. The dense factor
    # and its eight root columns exist ONLY in this independent tiny control.
    fm, gm = shared_value_gradient(op, result.mode)
    println((; stage="native third derivatives")); flush(stdout)
    root = inv(cholesky(Symmetric(H)).U)
    trace_third = sum(shared_third(op, result.mode, root[:, j]) for j in 1:d)
    correction = H \ trace_third[field]
    grad = gm + 0.5 .* (trace_third +
        shared_likelihood_hvp(op, result.mode, correction))
    outer = setdiff(eachindex(u), field)
    prior0 = BRM._rk_nlme_population_prior_value(m.population_prior, start) + d*log(2π)/2
    marginal_exact = prior0 + logpdf(MvNormal(mu0, Symmetric(
        0.7^2 * Matrix{Float64}(I, length(mu0), length(mu0)) + A*A')), data.y)
    marginal_laplace = fm + d*log(2π)/2 - logdet(Symmetric(H))/2
    @test marginal_laplace ≈ marginal_exact rtol=2e-11
    function oracle(point)
        base = copy(point); base[field] .= 0.0
        mean = Base.invokelatest(mean_query, base)
        design = hcat(map(1:d) do j
            e = copy(base); e[field[j]] = 1.0
            Base.invokelatest(mean_query, e) .- mean
        end...)
        p = BRM._rk_nlme_population_prior_value(m.population_prior, base) + d*log(2π)/2
        C = 0.7^2 * Matrix{Float64}(I, length(mean), length(mean)) + design*design'
        p + logpdf(MvNormal(mean, Symmetric(C)), data.y)
    end
    differences = map(outer) do j
        a, b = copy(u), copy(u)
        a[j] += 2e-5; b[j] -= 2e-5
        (oracle(a) - oracle(b)) / 4e-5
    end
    @test grad[outer] ≈ differences rtol=2e-6 atol=2e-7
    println((; field_coordinates=d, units=length(op.t.units), rows=length(data.y),
        cg_operations=result.info.numops, cg_residual=result.info.normres,
        gradient_max_error=maximum(abs.(grad[outer] .- differences)),
        scope="native Gaussian operator only; no SVI fit or streaming acceptance"))
end
