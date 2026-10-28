# test/nlme_crossed.jl — groupings crossed with the NLME units. A random-effect
# block over a grouping named in `shared_groups` (its levels occur in several
# subjects, so no subject owns them) is read by `brm_nlme_view` as shared
# latents: every coordinate of it (standardized draws, scales, correlation
# factor) sits in θ beside the population coefficients, and the subjects stay
# independent given θ. A `car(...)` field is always shared: its edges couple
# its areas.
#
# Covered: roles; the refusal that names a grouping named nowhere, and the
# refusals of a bad `shared_groups`; lockstep, single-subject and unit-problem
# densities against the full RK density; the per-unit Laplace (row-structured
# and whole-model) against the analytic Gaussian marginal; the minibatch SVI
# split against the full RK target; `brm_nlme_estimates` naming the block.
#
# Run: julia --project=test test/nlme_crossed.jl [testset-filter...]

include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random, Statistics

include(joinpath(@__DIR__, "nlme_crossed_fixture.jl"))

# Random θ, σ and subject blocks.
function random_point(m, rng)
    θ = 0.2 .* randn(rng, length(m.theta))
    σ = 0.2 .* randn(rng, length(m.sigma))
    H = 0.5 .* randn(rng, length(m.view.block_margins), length(m.view.levels))
    θ, σ, H
end

# Lockstep likelihoods sum to the RK likelihood at the packed point, each
# subject's η gradient is the full RK density's on its block without the
# block's standard-normal prior, and a subject's term does not move with
# another subject's block.
function check_lockstep(m, rng)
    θ, σ, H = random_point(m, rng)
    values = brm_nlme_loglikelihoods(m, θ, σ, H)
    u = BRM._brm_nlme_point(m, θ, σ, H)
    @test sum(values) ≈ Base.invokelatest(likelihood_query(m.backend), u) rtol=1e-10
    _, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    full = rk_logdensity_problem(m.backend; ad_backend=NLME_AD)
    _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    for (i, block) in enumerate(m.view.subject_coordinates)
        @test G[:, i] ≈ full_gradient[block] .+ u[block] atol=1e-8
    end
    other = copy(H)
    other[:, 2] .+= 0.7
    moved = brm_nlme_loglikelihoods(m, θ, σ, other)
    @test moved[2] != values[2]
    @test moved[[1; 3:end]] == values[[1; 3:end]]
    θ, σ, H, values
end

@stestset "a block over a crossed grouping is shared latents" begin
    rk = RKBRMI(crossed_flat(CR_FLAT))
    v = brm_nlme_view(rk; shared_groups=(:class,))
    @test v.group === :subject && v.shared_groups == (:class,)
    @test !any(==(:unclaimed), v.role)
    @test !BRM._brm_nlme_ragged(v)
    # Three levels × two margins of draws, two scales, one correlation.
    class = owned(v, BRM._brm_nlme_ranef_declaration(class_record(rk)))
    @test length(class) == 3 * 2 + 2 + 1
    @test all(i -> v.role[i] === :shared_latent, class)
    @test count(==(:shared_latent), v.role) == length(class)
    # The subjects hold only their own intercept.
    @test length(v.families) == 1 && v.families[1].block.group === :subject
    @test all(==([1]), v.subject_families)
    @test all(b -> b.block.group === :subject, v.block_margins)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, shared_groups=(:class,))
    @test sort(class) ⊆ m.theta
    check_lockstep(m, Xoshiro(1))
end

@stestset "an uncorrelated block is one shared block per margin" begin
    nocor(data) = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | subject) + (1 + x || class)
        y ~ Normal(mu, sigma)
    end
    rk = RKBRMI(nocor(CR_FLAT))
    v = brm_nlme_view(rk; shared_groups=(:class,))
    @test !any(==(:unclaimed), v.role)
    @test count(r -> r.kind === :ranef && r.group === :class,
        BRM._brm_nlme_records(rk.plan)) == 2
    class = class_coordinates(v, rk)
    @test length(class) == 3 * 2 + 2 && all(i -> v.role[i] === :shared_latent, class)
    @test count(==(:shared_latent), v.role) == length(class)
    check_lockstep(brm_nlme_model(rk; ad_backend=NLME_AD, shared_groups=(:class,)),
        Xoshiro(9))
end

@stestset "every grouping is named once; the refusal names the rest" begin
    rk = RKBRMI(crossed_flat(CR_FLAT))
    # refused: a grouping owned by no subject and named nowhere has no reading
    # the view may choose for the caller (decision 2026-10-28T13-05-01-304-0pfpp43).
    @test_throws "Named in neither: `class`" brm_nlme_view(rk; subject_group=:subject)
    @test_throws "random effects use 2 grouping columns" brm_nlme_view(rk)
    @test_throws "which no random-effect block" brm_nlme_view(rk;
        shared_groups=(:region,))
    @test_throws "is the subject grouping" brm_nlme_view(rk; subject_group=:subject,
        shared_groups=(:subject,))
    @test_throws "named in both nested_groups" brm_nlme_view(rk; subject_group=:subject,
        nested_groups=(class=:subject,), shared_groups=(:class,))
    @test_throws ArgumentError brm_nlme_view(rk; shared_groups=:class)
    @test_throws "nested_groups names `region`" brm_nlme_view(rk; subject_group=:subject,
        nested_groups=(region=:subject,), shared_groups=(:class,))
    # Both blocks shared leaves no subjects.
    @test_throws "over a grouping not in `shared_groups`" brm_nlme_view(rk;
        shared_groups=(:class, :subject))
end

@stestset "subject plate: unit problems and single-subject cells" begin
    rk = RKBRMI(crossed_plate(CR_PLATE))
    v = brm_nlme_view(rk; shared_groups=(:class,))
    @test !any(==(:unclaimed), v.role)
    class = class_coordinates(v, rk)
    @test length(class) == 3 + 1 && all(i -> v.role[i] === :shared_latent, class)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, shared_groups=(:class,))
    θ, σ, H, values = check_lockstep(m, Xoshiro(2))
    _, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    for i in eachindex(v.levels)
        value, g = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, H[:, i])
        @test value ≈ values[i] rtol=1e-10
        @test g ≈ G[:, i] rtol=1e-8
    end
    # A unit problem's density differences and gradient are the full RK
    # density's, with the class block at its θ values.
    rng = Xoshiro(3)
    Ω = Matrix{Float64}(I, 1, 1)
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    function point(i, z)
        u = BRM._brm_nlme_point(m, θ, σ, H)
        u[v.subject_coordinates[i]] = z
        u
    end
    for i in eachindex(v.levels)
        p = brm_nlme_unit_problem(m, i, θ, σ, Ω)
        z1, z2 = 0.4 .* randn(rng, 1), 0.4 .* randn(rng, 1)
        @test LogDensityProblems.logdensity(p, z1) - LogDensityProblems.logdensity(p, z2) ≈
              LogDensityProblems.logdensity(full, point(i, z1)) -
              LogDensityProblems.logdensity(full, point(i, z2)) atol=1e-8
        _, gradient = LogDensityProblems.logdensity_and_gradient(p, z1)
        _, full_gradient = LogDensityProblems.logdensity_and_gradient(full, point(i, z1))
        @test gradient ≈ full_gradient[v.subject_coordinates[i]] atol=1e-8
    end
end

@stestset "minibatch SVI: the crossed block is global; the split is the RK target" begin
    rk = RKBRMI(crossed_plate(CR_PLATE))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, shared_groups=(:class,))
    t = BRM._brm_svi_target(m)
    class = class_coordinates(m.view, rk)
    @test class ⊆ t.globals
    rng = Xoshiro(4)
    u = m.unit .+ 0.4 .* randn(rng, length(m.unit))
    g = u[t.globals]
    @test sort(vcat(t.globals, reduce(vcat, t.units))) == eachindex(u)
    P, gP = BRM._brm_svi_global_term(t, g)
    gg = zeros(length(g))
    terms = map(eachindex(t.units)) do i
        gz = zeros(length(t.units[i]))
        value = BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[t.units[i]])
        (; value, gg=copy(gg), gz)
    end
    full = rk_logdensity_problem(rk; ad_backend=NLME_AD)
    value, gradient = LogDensityProblems.logdensity_and_gradient(full, u)
    @test P + sum(x -> x.value, terms) ≈ value rtol=1e-12
    @test gP .+ sum(x -> x.gg, terms) ≈ gradient[t.globals] rtol=1e-10
    for (i, block) in enumerate(t.units)
        @test terms[i].gz ≈ gradient[block] rtol=1e-10
    end
end

# The model's priors and log-Jacobians of the population coordinates (the
# class block's included), less the subjects' standardized N(0, 1) terms.
function crossed_population_logprior(rk, view, u)
    plan = RKEXT._rk_translated_plan(rk.plan)
    prior = prepare_query(rk.model, plan, :prior)
    jacobian = prepare_query(rk.model, plan, :log_jacobian)
    draws = findall(==(:subject_effect), view.role)
    Base.invokelatest(prior, u) + Base.invokelatest(jacobian, u) -
        sum(logpdf(Normal(), u[j]) for j in draws)
end

# The exact marginal of the subjects' Gaussian intercepts: y_i ~ N(m_i, τ² 11ᵀ
# + σ² I), with m_i the rows' predictor at the class block's θ values.
function crossed_marginal(rk, view, u, data)
    values = BRM._rk_constrained_values(rk, u)
    record = only(r for r in BRM._brm_nlme_records(rk.plan)
        if r.kind === :ranef && r.group === :subject)
    τ = only(BRM._brm_nlme_joint_scale(values, record, 1))
    class = class_record(rk)
    A = BRM._brm_nlme_natural_factor(values, class, 2, nothing)
    z = BRM._brm_rk_declaration_value(values, class.z)
    levels = sort(unique(data.class))
    effects = Dict(level => A * z[k, :] for (k, level) in enumerate(levels))
    β = values.pop_mu.beta_pop
    mean_row(r) = β[1] + β[2] * data.x[r] + dot([1.0, data.x[r]], effects[data.class[r]])
    sum(view.levels) do level
        rows = findall(==(level), data.subject)
        Σ = Symmetric(fill(τ^2, length(rows), length(rows)) + values.sigma^2 * I)
        logpdf(MvNormal(mean_row.(rows), Σ), data.y[rows])
    end
end

@stestset "per-unit Laplace: the crossed block is a population coordinate" begin
    rk = RKBRMI(crossed_flat(CR_FLAT))
    rng = Xoshiro(5)
    for structured in (true, false)
        p = brm_laplace_problem(rk; ad_backend=NLME_AD, shared_groups=(:class,),
            structured)
        @test p isa (structured ? BRMStructuredLaplaceProblem : BRMLaplaceProblem)
        view = structured ? p.view : p.model.view
        class = owned(view, BRM._brm_nlme_ranef_declaration(class_record(rk)))
        @test class ⊆ p.population
        @test length(p.population) == length(view.coordinates) - length(view.levels)
        for _ in 1:3
            φ = 0.5 .* randn(rng, length(p.population))
            u = zeros(length(view.coordinates))
            u[p.population] .= φ
            @test LogDensityProblems.logdensity(p, φ) ≈
                crossed_population_logprior(rk, view, u) +
                crossed_marginal(rk, view, u, CR_FLAT) atol=1e-9
        end
    end
end

@stestset "brm_nlme_estimates names the crossed block" begin
    rk = RKBRMI(crossed_flat(CR_FLAT))
    m = brm_nlme_model(rk; ad_backend=NLME_AD, shared_groups=(:class,))
    rng = Xoshiro(6)
    θ, σ = 0.3 .* randn(rng, length(m.theta)), 0.3 .* randn(rng, length(m.sigma))
    Ω = fill(0.25, 1, 1)
    point = brm_nlme_estimates(m, θ, σ, Ω).point
    block = point.var"mu|class"
    u = copy(m.unit)
    u[m.theta] = θ
    u[m.sigma] = σ
    values = BRM._rk_constrained_values(rk, u)
    class = class_record(rk)
    A = BRM._brm_nlme_natural_factor(values, class, 2, nothing)
    z = BRM._brm_rk_declaration_value(values, class.z)
    @test block.group === :class && block.crossed
    @test [(b.predictor, b.coefficient) for b in block.margins] ==
        [(:mu, :Intercept), (:mu, :x)]
    @test block.covariance ≈ A * A' rtol=1e-12
    @test block.sd ≈ sqrt.(diag(A * A')) rtol=1e-12
    @test first.(block.effects) == ["k1", "k2", "k3"]
    @test reduce(hcat, last.(block.effects))' ≈ z * A' rtol=1e-12
    # The subject block is still read from Ω.
    @test point.var"mu|subject".sd ≈ [0.5]
end

@stestset "every block form over a crossed grouping is shared; estimates read it" begin
    shared = (:class, :region)
    rng = Xoshiro(8)
    for build in (crossed_stratified, crossed_declared, crossed_known)
        rk = RKBRMI(build(CR_FORMS))
        v = brm_nlme_view(rk; shared_groups=shared)
        @test !any(==(:unclaimed), v.role)
        @test all(==([1]), v.subject_families)
        records = [r for r in BRM._brm_nlme_records(rk.plan)
            if r.kind in (:ranef, :stratified_ranef) && r.group in shared]
        @test length(records) == 2
        for r in records
            @test all(i -> v.role[i] === :shared_latent,
                owned(v, BRM._brm_nlme_ranef_declaration(r)))
        end
        m = brm_nlme_model(rk; ad_backend=NLME_AD, shared_groups=shared)
        check_lockstep(m, rng)
        θ, σ = 0.3 .* randn(rng, length(m.theta)), 0.3 .* randn(rng, length(m.sigma))
        point = brm_nlme_estimates(m, θ, σ, fill(0.25, 1, 1)).point
        # Each level's effect, as the model adds it to the rows: the predictor
        # at a row less its population and subject parts.
        u = copy(m.unit)
        u[m.theta] = θ
        u[m.sigma] = σ
        values = BRM._rk_constrained_values(rk, u)
        mu = only(rk_draw_values(rk, reshape(u, :, 1), (:mu,)).mu)
        β = values.pop_mu.beta_pop
        subject = point.var"mu|subject"
        @test subject.sd ≈ [0.5]
        # A block addressed by a prior on its grouping lowers under a generated
        # `|ID|` (`mu__region` for `ranef(mu, region)`), so find each by group.
        block(g) = only(b for b in Base.values(point) if b isa NamedTuple &&
            get(b, :group, nothing) === g)
        blocks = (block(:class), block(:region))
        @test all(b -> b.crossed, blocks)
        level_effect(b, level) = only(last(b.effects[findfirst(p -> first(p) == level, b.effects)]))
        zsubject = BRM._brm_rk_declaration_value(values, only(r.z for r in
            BRM._brm_nlme_records(rk.plan) if r.kind === :ranef && r.group === :subject))
        τ = only(BRM._brm_nlme_joint_scale(values, only(r for r in
            BRM._brm_nlme_records(rk.plan) if r.kind === :ranef && r.group === :subject), 1))
        subjects = sort(unique(CR_FORMS.subject))
        for r in eachindex(CR_FORMS.y)
            s = findfirst(==(CR_FORMS.subject[r]), subjects)
            @test mu[r] ≈ β[1] + β[2] * CR_FORMS.x[r] + τ * zsubject[s] +
                level_effect(blocks[1], CR_FORMS.class[r]) +
                level_effect(blocks[2], CR_FORMS.region[r]) rtol=1e-10
        end
        if build === crossed_stratified
            @test blocks[1].by === :size && first.(blocks[1].strata) == ["large", "small"]
        elseif build === crossed_known
            @test blocks[1].sd ≈ [0.4]
        end
    end
end

@stestset "a car field is shared, whatever its area column" begin
    rk = RKBRMI(crossed_car(CR_CAR))
    v = brm_nlme_view(rk)
    @test !any(==(:unclaimed), v.role)
    field = only(r for r in BRM._brm_nlme_records(rk.plan) if r.kind === :car)
    car = owned(v, field.declaration)
    @test !isempty(car) && all(i -> v.role[i] === :shared_latent, car)
    @test count(==(:shared_latent), v.role) == length(car)
    @test all(==([1]), v.subject_families)
    m = brm_nlme_model(rk; ad_backend=NLME_AD)
    @test car ⊆ m.theta
    check_lockstep(m, Xoshiro(7))
end
