# test/nlme_grouped_hsgp.jl — the NLME reading (`brm_nlme_view`) of a grouped
# HSGP, `hsgp(x; by=g)`. When the levels of `g` are the units (the subject
# grouping itself, a column labelled by subjects, or a nested grouping), each
# level's standardized basis weights are unit draws of the subject owning the
# level, in one family fixed at the identity; a hyper-predicted amplitude's
# `(1 | g)` effects are an estimated family over the same units; the shared
# length scale and amplitude stay shared. A term whose levels are not units, or
# whose `cor=` couples them, stays shared as a whole.
#
# Covered: roles and families on a flat model and on a subject plate that
# reads the term through `groups(ecg_subject => subject)`; a subject without a
# level (ragged blocks); lockstep, single-subject and unit-problem densities
# against the full RK density; hyper-predicted amplitudes; shared readings and
# the mixed-label refusal.
#
# Run: julia --project=test test/nlme_grouped_hsgp.jl

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LogDensityProblems, LinearAlgebra, Random

include(joinpath(@__DIR__, "nlme_grouped_hsgp_fixture.jl"))

# Random θ, σ and subject blocks; the packed point the NLME model evaluates.
function random_point(m, rng)
    θ = 0.2 .* randn(rng, length(m.theta))
    σ = 0.2 .* randn(rng, length(m.sigma))
    H = BRM._brm_nlme_ragged(m.view) ?
        [0.5 .* randn(rng, length(b)) for b in m.view.subject_coordinates] :
        0.5 .* randn(rng, length(m.view.block_margins), length(m.view.levels))
    θ, σ, H
end
column(H::AbstractMatrix, i) = H[:, i]
column(H::AbstractVector, i) = H[i]

# The full RK density's dependence on one unit's block equals that unit's
# problem's: density differences and the unit gradient (as in
# test/nlme_units_panel.jl).
function check_unit_problems(m, rng)
    v = m.view
    θ, σ, H = random_point(m, rng)
    Ω = BRM._brm_nlme_ragged(v) ? [Matrix{Float64}(I, K, K) for K in v.family_sizes] :
        Matrix{Float64}(I, length(v.block_margins), length(v.block_margins))
    full = rk_logdensity_problem(m.backend; ad_backend=NLME_AD)
    function point(i, z)
        u = copy(m.unit)
        u[m.theta] = θ
        u[m.sigma] = σ
        for (j, block) in enumerate(v.subject_coordinates)
            u[block] = j == i ? z : column(H, j)
        end
        u
    end
    for i in eachindex(v.levels)
        p = brm_nlme_unit_problem(m, i, θ, σ, Ω)
        d = LogDensityProblems.dimension(p)
        @test d == length(v.subject_coordinates[i])
        z1, z2 = 0.4 .* randn(rng, d), 0.4 .* randn(rng, d)
        @test LogDensityProblems.logdensity(p, z1) - LogDensityProblems.logdensity(p, z2) ≈
              LogDensityProblems.logdensity(full, point(i, z1)) -
              LogDensityProblems.logdensity(full, point(i, z2)) atol=1e-8
        _, gradient = LogDensityProblems.logdensity_and_gradient(p, z1)
        _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, point(i, z1))
        @test gradient ≈ full_gradient[v.subject_coordinates[i]] atol=1e-8
    end
end

# Lockstep likelihoods sum to the RK likelihood at the packed point, each
# subject's η gradient is the full RK density's gradient on its block without
# the block's standard-normal prior (every scale is 1 at the unit point), and
# each subject's term does not move with another subject's block.
function check_lockstep(m, rng)
    θ, σ, H = random_point(m, rng)
    values = brm_nlme_loglikelihoods(m, θ, σ, H)
    u = BRM._brm_nlme_point(m, θ, σ, H)
    @test sum(values) ≈ Base.invokelatest(likelihood_query(m.backend), u) rtol=1e-10
    _, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    full = rk_logdensity_problem(m.backend; ad_backend=NLME_AD)
    _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    for (i, block) in enumerate(m.view.subject_coordinates)
        @test column(G, i) ≈ full_gradient[block] .+ u[block] atol=1e-8
    end
    other = deepcopy(H)
    if H isa AbstractMatrix
        other[:, 2] .+= 0.7
    else
        other[2] .+= 0.7
    end
    moved = brm_nlme_loglikelihoods(m, θ, σ, other)
    @test moved[2] != values[2]
    @test moved[[1; 3:end]] == values[[1; 3:end]]
    θ, σ, H, values
end

@stestset "flat rows: grouped weights are each subject's unit draws" begin
    rk = RKBRMI(flat_hsgp(GH_FLAT))
    v = brm_nlme_view(rk)
    @test !any(==(:unclaimed), v.role)
    @test !BRM._brm_nlme_ragged(v)
    k = weight_family(v)
    family = v.families[k]
    @test family.size == 3 && family.fixed == Matrix{Float64}(I, 3, 3)
    @test isempty(family.scales) && isempty(family.factor)
    @test family.block.group === :subject
    weights = findall(n -> occursin(r"^hsgp_t[^.]*\.z\.\d+\.\d+$", String(n)), v.coordinates)
    @test length(weights) == 4 * 3
    @test all(i -> v.role[i] === :subject_effect, weights)
    # Each subject's block is its random intercept, then its three weights,
    # which are the weights of its own level.
    for (i, level) in enumerate(v.levels)
        @test v.subject_families[i] == [1, k]
        g = findfirst(==(level), sort(unique(GH_FLAT.subject)))
        @test v.coordinates[v.subject_coordinates[i][2:end]] ==
            [Symbol(split(String(v.coordinates[first(weights)]), ".z.")[1], ".z.", g, ".", j)
             for j in 1:3]
        @test all(b -> b.level == level, v.subject_margins[i])
    end
    @test [b.coefficient for b in v.block_margins[2:end]] ==
        [Symbol("hsgp(t)[", j, "]") for j in 1:3]
    # The length scale and amplitude stay shared; no weight does.
    shared = v.coordinates[v.role .=== :shared_latent]
    @test length(shared) == 2 && all(n -> !occursin(".z.", String(n)), shared)

    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    @test length(m.theta) == count(==(:population), v.role) + 2
    @test m.eta_blocks == [1, 3]
    # A flat-row model has no subject cell, so it has lockstep evaluation only
    # (no `brm_nlme_unit_problem`); the plate models below check unit problems.
    check_lockstep(m, Xoshiro(1))
end

@stestset "subject plate: weights labelled by subjects join their cells" begin
    rk = RKBRMI(plate_hsgp(GH_PLATE))
    v = brm_nlme_view(rk)
    @test v.group === :subject
    @test !BRM._brm_nlme_ragged(v)
    k = weight_family(v)
    @test v.families[k].block.group === :ecg_subject
    @test v.families[k].size == 4
    @test all(==([1, k]), v.subject_families)
    @test all(length.(v.subject_coordinates) .== 5)
    @test count(==(:subject_effect), v.role) == 4 * 5
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    θ, σ, H, values = check_lockstep(m, Xoshiro(3))
    # The single-subject cell reads its own weights through the binding.
    for i in eachindex(v.levels)
        @test brm_nlme_loglikelihood(m, i, θ, σ, H[:, i]) ≈ values[i] rtol=1e-10
        value, g = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[:, i])
        _, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
        @test value ≈ values[i] rtol=1e-10
        @test g ≈ G[:, i] rtol=1e-8
    end
    check_unit_problems(m, Xoshiro(4))
end

@stestset "a subject without a level of the term holds no weights" begin
    rk = RKBRMI(plate_hsgp(GH_PLATE_GAP))
    v = brm_nlme_view(rk)
    @test BRM._brm_nlme_ragged(v)
    k = weight_family(v)
    d = findfirst(==("d"), v.levels)
    @test v.subject_families[d] == [1]
    @test all(i -> i == d || v.subject_families[i] == [1, k], eachindex(v.levels))
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    check_lockstep(m, Xoshiro(5))
    check_unit_problems(m, Xoshiro(6))
end

@stestset "a hyper-predicted amplitude's group effects are a unit family" begin
    rk = RKBRMI(plate_hyper(GH_PLATE))
    v = brm_nlme_view(rk)
    @test !any(==(:unclaimed), v.role)
    hyper = only(f for f in v.families if f.record.kind === :hsgp_hyper)
    @test hyper.size == 1 && hyper.fixed === nothing && length(hyper.scales) == 1
    @test v.role[only(hyper.scales)] === :subject_scale
    @test only(hyper.record.margins) == (Symbol("log(sd(qt_time, hsgp(ecg_time)))"), :Intercept)
    h = findfirst(f -> f === hyper, v.families)
    k = weight_family(v)
    @test all(==([1, h, k]), v.subject_families)
    # The amplitude's intercept and the length scale stay shared.
    @test count(==(:shared_latent), v.role) == 2
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    check_lockstep(m, Xoshiro(7))
    check_unit_problems(m, Xoshiro(8))
end

@stestset "a term whose levels are not units, or are coupled, stays shared" begin
    by_region(data) = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + hsgp(t; by=region, k=3, domain=(-1.0, 1.0)) + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    v = brm_nlme_view(RKBRMI(by_region(GH_FLAT)))
    @test !any(==(:unclaimed), v.role)
    @test weight_family(v) === nothing
    @test count(==(:shared_latent), v.role) == 2 + 2 * 3
    @test all(length.(v.subject_coordinates) .== 1)

    known = [1.0 0.5 0.2 0.1; 0.5 1.0 0.3 0.2; 0.2 0.3 1.0 0.4; 0.1 0.2 0.4 1.0]
    coupled(data) = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + hsgp(t; by=subject, cor=C, k=3, domain=(-1.0, 1.0)) + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    v = brm_nlme_view(RKBRMI(coupled(merge(GH_FLAT, (; C=known)))))
    @test weight_family(v) === nothing
    @test count(==(:shared_latent), v.role) == 2 + 4 * 3

    # A declared level correlation is shared with the term's weights.
    declared(data) = @brm data begin
        sigma ~ Exponential(1)
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + x + hsgp(t; by=subject, cor=Omega, k=3, domain=(-1.0, 1.0)) + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    v = brm_nlme_view(RKBRMI(declared(GH_FLAT)))
    @test weight_family(v) === nothing
    omega = findall(n -> startswith(String(n), "Omega"), v.coordinates)
    @test length(omega) == 6 && all(i -> v.role[i] === :shared_latent, omega)
    @test count(==(:shared_latent), v.role) == 6 + 2 + 4 * 3
end

@stestset "levels that name some subjects but not others are refused" begin
    mixed(data) = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + hsgp(t; by=label, k=3, domain=(-1.0, 1.0)) + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    label = replace(GH_FLAT.subject, "d" => "z")
    # refused: BRM joins grouping columns by label, and a column whose labels
    # match only some subjects has no unambiguous owner for the others' weights
    # (an accidental label collision reads exactly like this).
    @test_throws "levels that are subjects of `subject` and levels that are not" brm_nlme_view(
        RKBRMI(mixed(merge(GH_FLAT, (; label)))))
end
