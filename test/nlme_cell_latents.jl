# test/nlme_cell_latents.jl — `@plate` cell latents in the NLME view and SVI.
#
# A `~` on a bare cell name declares a latent of the cell (brm-use kernel.md
# "Latents in a cell"), one draw per cell or per entry; a function-style
# component called in a cell declares its sampled locals per cell. Their laws
# are authored and stay in the model's prior, so `brm_nlme_view` reads their
# coordinates as `:shared_latent`, the reading the entries of a top-level
# vector `x[1:n] .~ Law` get: they enter θ, each unit's conditional likelihood
# reads its own cells' latents, and the SVI split into unit terms is exact.
# `brm_svi_fit` gives a cell component's locals and, from RKPPL 0.1.127, a
# dotted cell latent's entries to their unit. A latent under a transformed law
# (`LogNormal`) stays global until ReactiveKernels reads a constrained block per
# cell (`@test_broken` below). A latent whose law reads a value the model
# computes per cell is refused by name. Snag brm-nlme-view-re-ed98959b (before
# it: "coordinate record kind `cell_latent` has no NLME role").
#
# Run: julia --startup-file=no --project=test test/nlme_cell_latents.jl [filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using Random

# Four windows of unequal length; each cell's latent has its window's length.
const CELL_LATENT_DATA = (; win=["w1", "w2", "w3", "w4"],
    loc=[[0.1, -0.2], [0.3], [0.0, 0.4, -0.1], [0.2, 0.2]],
    w=[[0.5, -0.5], [1.0], [-0.3, 0.2, 0.7], [0.0, 0.9]],
    wc=[0.5, 1.0, -0.3, 0.0],
    y=[[0.4, -0.1], [0.8], [0.2, 0.6, -0.3], [0.5, 0.1]])

cell_latent_model(data) = @brm data begin
    sigma ~ Exponential(1)
    s ~ Exponential(1)
    sd(mu, win) ~ LogNormal(-1.0, 0.5)
    mu ~ 1 + (1 | win)
    @plate for i in eachindex(mu)
        x ~ Normal(loc[i], s)
        a ~ Normal(0.0, 1.0)
        y[i] ~ Normal(mu[i] .+ a .+ x, sigma)
    end
end

@brm function cell_calibration(x, scale)
    lo ~ Normal(0, scale)
    slope ~ Normal(1, 0.2)
    return lo + slope * x
end

cell_component_model(data) = @brm data begin
    sigma ~ Exponential(1)
    s ~ Exponential(1)
    mu ~ 1 + (1 | win)
    @plate for i in eachindex(mu)
        cell ~ cell_calibration(wc[i], s)
        y[i] ~ Normal(mu[i] .+ cell .* w[i], sigma)
    end
end

# A positive latent: its law transforms its coordinates.
positive_cell_latent_model(data) = @brm data begin
    sigma ~ Exponential(1)
    s ~ Exponential(1)
    mu ~ 1 + (1 | win)
    @plate for i in eachindex(mu)
        x ~ LogNormal(loc[i], s)
        y[i] ~ Normal(mu[i] .+ log.(x), sigma)
    end
end

# The latent's law reads a per-cell population predictor (no unit draw).
population_reading_latent_model(data) = @brm data begin
    sigma ~ Exponential(1)
    m ~ 1 + wc
    mu ~ 1 + (1 | win)
    @plate for i in eachindex(mu)
        x ~ Normal(m[i] .+ loc[i], 1.0)
        y[i] ~ Normal(mu[i] .+ x, sigma)
    end
end

# The latent's law reads the cell's predictor, which carries the unit's draw.
unit_reading_latent_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | win)
    @plate for i in eachindex(mu)
        x ~ Normal(mu[i] .+ loc[i], 1.0)
        y[i] ~ Normal(x, sigma)
    end
end

# The same model as `unit_reading_latent_model`, written as the refusal says.
deviation_latent_model(data) = @brm data begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | win)
    @plate for i in eachindex(mu)
        d ~ Normal(loc[i], 1.0)
        x = mu[i] .+ d
        y[i] ~ Normal(x, sigma)
    end
end

# A latent with its cell's data length that may have no entry: an optional step
# per window, read through its sum.
optional_step_model(data) = @brm data begin
    sigma ~ Exponential(1)
    step ~ Normal(0.0, 1.0)
    mu ~ 1 + (1 | win)
    @plate for i in eachindex(mu)
        x ~ Normal(loc[i], 1.0)
        y[i] ~ Normal(mu[i] .+ sum(step .+ x), sigma)
    end
end

# A cell latent's coordinates are `x.<cell>` (one draw) or `x.<cell>.<entry>`.
cell_coordinates(view, prefix, c) = [j for (j, n) in enumerate(view.coordinates)
    if String(n) == string(prefix, ".", c) || startswith(String(n), string(prefix, ".", c, "."))]
# A cell component's sampled locals are `<binding>.<local>.<cell>`.
component_coordinates(view, binding, c) = [j for (j, n) in enumerate(view.coordinates)
    if startswith(String(n), string(binding, ".")) && endswith(String(n), string(".", c))]

# The SVI split into one global and one term per unit is the whole density,
# in value and gradient, and averaging singleton minibatches is unbiased.
function check_cell_split(t)
    m = t.m
    sampler = prepare_sampler(m.backend.model, rk_bound_plan(m.backend), m.unit;
        backend=NLME_AD)
    for offset in (-0.23, 0.0, 0.31)
        u = [0.12sin(i) + offset for i in eachindex(m.unit)]
        g = u[t.globals]
        value, gg = BRM._brm_svi_global_term(t, g)
        global_value, global_gradient = value, copy(gg)
        grad = zeros(length(u))
        grad[t.globals] .= gg
        unit_values, unit_gradients = Float64[], Vector{Float64}[]
        for (i, block) in enumerate(t.units)
            gg, gz = zeros(length(g)), zeros(length(block))
            term = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[block])
            value += term
            grad[t.globals] .+= gg
            grad[block] .+= gz
            packed = zeros(length(u))
            packed[t.globals] .= gg
            packed[block] .= gz
            push!(unit_values, term)
            push!(unit_gradients, packed)
        end
        expected_grad = similar(u)
        expected, _ = sampler_value_and_gradient!(sampler, expected_grad, u)
        @test value ≈ expected atol=2e-10 rtol=2e-12
        @test grad ≈ expected_grad atol=2e-9 rtol=2e-10
        n = length(t.units)
        @test global_value + sum(n .* unit_values) / n ≈ expected atol=2e-10 rtol=2e-12
        average_gradient = zeros(length(u))
        average_gradient[t.globals] .= global_gradient
        average_gradient .+= sum(n .* unit_gradients) ./ n
        @test average_gradient ≈ expected_grad atol=2e-9 rtol=2e-10
    end
end

@stestset "cell latents are shared latents of the NLME view" begin
    data = CELL_LATENT_DATA
    backend = RKBRMI(cell_latent_model(data))
    view = brm_nlme_view(backend)
    @test view.levels == data.win
    for c in eachindex(data.win)
        x, a = cell_coordinates(view, :x, c), cell_coordinates(view, :a, c)
        @test length(x) == length(data.loc[c])
        @test length(a) == 1
        @test all(j -> view.role[j] === :shared_latent && view.subject[j] == 0, vcat(x, a))
    end
    @test count(==(:shared_latent), view.role) == sum(length, data.loc) + length(data.win)
    # The random intercepts stay the units' draws.
    @test all(length.(view.subject_coordinates) .== 1)
end

# RKPPL keeps a cell latent `x` as the layout entry `_rkppl_latent_x` and names
# its coordinates `x.<cell>…`. With no entry in any cell it is a zero-length
# entry and the declaration `x` owns no coordinate, which the view read as an
# inventory defect ("internal: declaration `x` owns no packed coordinate") until
# the layout's empty declarations carried authored names too (snag
# brm-start-rkbrmi-7ae680c8).
@stestset "a cell latent with no entry in any cell owns no coordinate" begin
    none = merge(CELL_LATENT_DATA, (; loc=[Float64[] for _ in CELL_LATENT_DATA.win]))
    some = merge(CELL_LATENT_DATA, (; loc=[Float64[], [0.1], Float64[], [0.2, -0.3]]))
    backend = RKBRMI(optional_step_model(none))
    @test :x in BRM._rk_layout_empty_declarations(backend)
    view = brm_nlme_view(backend)
    @test !any(n -> startswith(String(n), "x."), view.coordinates)
    @test count(==(:shared_latent), view.role) == 0
    # The same partition as data with some entries, less those entries.
    twin = brm_nlme_view(RKBRMI(optional_step_model(some)))
    @test :x ∉ BRM._rk_layout_empty_declarations(RKBRMI(optional_step_model(some)))
    @test count(==(:shared_latent), twin.role) == sum(length, some.loc)
    @test filter(n -> !startswith(String(n), "x."), twin.coordinates) == view.coordinates
    @test view.block_margins == twin.block_margins
    @test length.(view.subject_coordinates) == length.(twin.subject_coordinates)
    # Its units' conditionals and the SVI split are those of any other model.
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    rng = Xoshiro(3)
    θ, σ = 0.3 .* randn(rng, length(m.theta)), 0.2 .* randn(rng, length(m.sigma))
    H = 0.4 .* randn(rng, 1, length(m.view.levels))
    @test all(isfinite, brm_nlme_loglikelihoods(m, θ, σ, H))
    check_cell_split(BRM._brm_svi_target(m))
end

@stestset "each unit's conditional likelihood reads its own cell's latents" begin
    data = CELL_LATENT_DATA
    backend = RKBRMI(cell_latent_model(data))
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    shared = findall(==(:shared_latent), m.view.role)
    @test all(i -> i in m.theta, shared)
    rng = Xoshiro(7)
    θ = 0.3 .* randn(rng, length(m.theta))
    σ = 0.2 .* randn(rng, length(m.sigma))
    H = 0.4 .* randn(rng, 1, length(m.view.levels))
    u = BRM._brm_nlme_point(m, θ, σ, H)
    values, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    @test brm_nlme_loglikelihoods(m, θ, σ, H) == values
    @test all(isfinite, values)
    # The conditionals are the likelihood: the latents' laws stay in the prior.
    @test sum(values) ≈ Base.invokelatest(likelihood_query(backend), u) rtol=1e-12
    # Moving one cell's latent moves exactly its unit's conditional.
    for c in eachindex(data.win)
        moved = copy(θ)
        for j in cell_coordinates(m.view, :x, c)
            moved[findfirst(==(j), m.theta)] += 0.25
        end
        changed = brm_nlme_loglikelihoods(m, moved, σ, H)
        @test changed[c] != values[c]
        @test changed[setdiff(eachindex(values), c)] == values[setdiff(eachindex(values), c)]
    end
    # Fixed at a full point, the latents leave θ and keep their values.
    fixed_model = brm_nlme_model(backend; ad_backend=NLME_AD, shared_point=u)
    @test isempty(intersect(shared, fixed_model.theta))
    fixed_θ = [θ[findfirst(==(i), m.theta)] for i in fixed_model.theta]
    @test brm_nlme_loglikelihoods(fixed_model, fixed_θ, σ, H) ≈ values rtol=1e-12
end

# Whether SVI gave each unit exactly the coordinates `owned(c)` of its cell.
owns_cells(t, m, owned) = all(eachindex(m.view.levels)) do c
    sort(setdiff(t.units[c], m.view.subject_coordinates[c])) == sort(owned(c))
end

# RKPPL reads a dotted cell latent's block through a bound table of ranges
# (0.1.126, ReactiveKernels snag rkppl-dotted-pla-1629a747) and, when the cell
# also reads a computed location such as `mu[i]`, maps the typed view of that
# block to its parent's positions (0.1.127, snag dotted-cell-late-3789b8ea).
# Before, every unit's partition footprint read every cell's block, and SVI kept
# the latents global (exact, but not per unit).
@stestset "SVI gives each unit its cells' latents" begin
    data = CELL_LATENT_DATA
    m = brm_nlme_model(RKBRMI(cell_latent_model(data)); ad_backend=NLME_AD)
    t = BRM._brm_svi_target(m)
    @test owns_cells(t, m,
        c -> vcat(cell_coordinates(m.view, :x, c), cell_coordinates(m.view, :a, c)))
    @test !any(j -> startswith(String(m.view.coordinates[j]), r"[xa]\."), t.globals)
    check_cell_split(t)
    fit = brm_svi_fit(m; rng=Xoshiro(11), batch_size=2, max_iterations=6, ndraws=4,
        threads=1)
    @test size(fit.draws) == (length(m.unit), 4)  # coordinates × draws
    @test all(isfinite, fit.draws)
end

@stestset "a component called in a cell owns its unit's locals" begin
    data = CELL_LATENT_DATA
    m = brm_nlme_model(RKBRMI(cell_component_model(data)); ad_backend=NLME_AD)
    t = BRM._brm_svi_target(m)
    for c in eachindex(data.win)
        locals = component_coordinates(m.view, :cell, c)
        @test length(locals) == 2
        @test all(j -> m.view.role[j] === :shared_latent, locals)
    end
    @test owns_cells(t, m, c -> component_coordinates(m.view, :cell, c))
    @test !any(j -> startswith(String(m.view.coordinates[j]), "cell."), t.globals)
    check_cell_split(t)
end

# A transformed law's latent block is a constrained vector computed over every
# cell, which ReactiveKernels reads whole in each cell's footprint (RK todo
# 0sr6ilc), so SVI keeps it global. The split is exact either way.
@stestset "a positive cell latent stays global until RK reads it per cell" begin
    data = CELL_LATENT_DATA
    m = brm_nlme_model(RKBRMI(positive_cell_latent_model(data)); ad_backend=NLME_AD)
    @test count(==(:shared_latent), m.view.role) == sum(length, data.loc)
    t = BRM._brm_svi_target(m)
    @test_broken owns_cells(t, m, c -> cell_coordinates(m.view, :x, c))
    check_cell_split(t)
end

# Both readings are refused because RKPPL 0.1.126 still cannot evaluate such a
# law one cell at a time (RK todo 1olsblw: the per-cell law arguments are
# flattened without a row form); once it can, the unit-reading one is refused
# for reading the units' draws and the population-reading one has a reading.
# Either message names the latent and the rewrite.
@stestset "a cell latent whose law reads a per-cell model value is refused by name" begin
    data = CELL_LATENT_DATA
    for model in (unit_reading_latent_model, population_reading_latent_model)
        err = try
            brm_nlme_view(RKBRMI(model(data)))
            nothing
        catch e
            e
        end
        @test err isa BRM.BRMNLMEViewError
        @test occursin("cell latent `x`", sprint(showerror, err))
        @test occursin("`d ~ Normal(loc[i], s)`", sprint(showerror, err))
    end
    # The rewrite the message names is the same model and has a reading.
    m = brm_nlme_model(RKBRMI(deviation_latent_model(data)); ad_backend=NLME_AD)
    @test count(==(:shared_latent), m.view.role) == sum(length, data.loc)
    check_cell_split(BRM._brm_svi_target(m))
end
