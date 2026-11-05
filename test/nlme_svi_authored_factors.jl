# Independent acceptance of the public authored-density API needed by SVI's
# block-local coordinate extension, and of SVI's authored prior term built on
# it (`_BRMSVIAuthoredPrior`, src/nlme_svi.jl). Unit membership is supplied
# here as an independent oracle beside production structural ownership.
# Run: julia --startup-file=no --project=test test/nlme_svi_authored_factors.jl
using Test, Random, LinearAlgebra, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using ADTypes: AutoEnzyme
import Enzyme

const FACTOR_AD = AutoEnzyme(; mode=Enzyme.Reverse)

function authored_factor_data(n=6)
    x = [[-0.7, 0.2, 0.9] for _ in 1:n]
    y = [[0.1i, -0.2, 0.4] for i in 1:n]
    (; subject=["s$i" for i in 1:n], x, y,
       c1=collect(1:2:2n), c2=collect(2:2:2n))
end

function standard_vector_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        tau ~ Exponential(1)
        z[1:12] .~ Normal(0., 1.)
        mu ~ 1 + (1 | unit | subject)
        @plate for i in eachindex(mu)
            y[i] ~ Normal(mu[i] .+ tau .* (z[c1[i]] .+ z[c2[i]] .* x[i]), sigma)
        end
    end
end

function dependent_constrained_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        tau ~ Exponential(1)
        z[1:12] .~ Normal(0.4, tau)
        positive[1:6] .~ Exponential(2.)
        bounded[1:6] .~ Beta(2., 3.)
        mu ~ 1 + (1 | unit | subject)
        @plate for i in eachindex(mu)
            y[i] ~ Normal(mu[i] .+ z[c1[i]] .+ z[c2[i]] .* x[i] .+
                positive[i] .* bounded[i], sigma)
        end
    end
end

const BRM = BayesianRegressionModels

# The same six subjects plus a simplex: a joint law with a coupled transform,
# whose two unconstrained coordinates one atomic event owns together. (An
# isotropic `MvNormal(zeros(n), s)` is not such a control: RK lowers it to n
# independent per-entry events.)
function joint_simplex_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        tau ~ Exponential(1)
        z[1:12] .~ Normal(0., 1.)
        s ~ Dirichlet([1.0, 2.0, 1.5])
        mu ~ 1 + (1 | unit | subject)
        @plate for i in eachindex(mu)
            y[i] ~ Normal(mu[i] .+ tau .* (z[c1[i]] .+ z[c2[i]] .* x[i]) .+ s[1], sigma)
        end
    end
end

# Subject i's own coordinates beyond its random-effect block: the coefficient
# pair it alone reads and, in the constrained model, its positive and bounded
# entries, selected by authored ownership.
function fixture_owned(catalogue, n; constrained=false)
    map(1:n) do i
        owned = Int[]
        sites = constrained ? ((:z, 2i - 1), (:z, 2i), (:positive, i), (:bounded, i)) :
            ((:z, 2i - 1), (:z, 2i))
        for (site, entry) in sites
            coordinates = factor_coordinates(select_factors(catalogue;
                kind=:prior, site, entries=entry:entry))
            @test length(coordinates) == 1
            append!(owned, coordinates)
        end
        owned
    end
end

# Use authored ownership, never support, to select a child's prior. The
# hyperparameter's prior must stay global even when every child reads it.
function check_authored_factor_split(rk; constrained=false)
    plan = BayesianRegressionModels.rk_bound_plan(rk)
    catalogue = density_factors(rk.model, plan)
    events = collect(catalogue)
    view = brm_nlme_view(rk)
    n = length(view.subject_coordinates)
    units = [vcat(block, owned) for (block, owned) in
        zip(view.subject_coordinates, fixture_owned(catalogue, n; constrained))]
    u0 = [0.17sin(k) for k in 1:rk.model.layout.total]
    owner = zeros(Int, length(u0))
    for (i, block) in enumerate(units)
        owner[block] .= i
    end
    @test length.(units) == fill(constrained ? 5 : 3, n)
    @test count(iszero, owner) == 4
    @test length(unique(vcat(units...))) == sum(length, units)

    # This fixture supplies its known unit membership. Production discovery
    # must obtain that membership from the connected structural cell plan.
    buckets = [eltype(events)[] for _ in 0:n]
    for event in events
        support = factor_support(catalogue, event; u0)
        owned = factor_coordinates(catalogue, event)
        local_owners = unique(filter(!iszero, owner[support]))
        @test length(local_owners) <= 1
        unit = isempty(local_owners) ? 0 : only(local_owners)
        @test all(k -> owner[k] == 0 || owner[k] == unit, owned)
        push!(buckets[unit + 1], event)
    end
    @test sum(length, buckets) == length(events)
    @test length(unique(vcat(buckets...))) == length(events)
    @test all(!isempty, buckets)

    tau = factor_coordinates(select_factors(catalogue; kind=:prior, site=:tau))
    z1 = select_factors(catalogue; kind=:prior, site=:z, entries=1:1)
    @test isempty(intersect(factor_coordinates(z1), tau))
    @test isempty(intersect(factor_support(z1; u0), tau)) == !constrained
    if constrained
        for site in (:positive, :bounded)
            jacobian = select_factors(catalogue,
                [event for event in events if event.site == site && event.kind != :prior])
            @test !isempty(factor_coordinates(jacobian))
            @test all(!iszero, owner[factor_coordinates(jacobian)])
        end
    end

    queries = [prepare_query_ad(prepare_ad, rk.model, plan,
        select_factors(catalogue, bucket), FACTOR_AD, u0; coordinates=:support)
        for bucket in buckets]
    likelihood = prepare_query_ad(prepare_ad, rk.model, plan,
        :likelihood, FACTOR_AD, u0)
    sampler = prepare_sampler(rk.model, plan, u0; backend=FACTOR_AD)
    for offset in (0.0, 0.31, -0.24)
        u = u0 .+ offset
        untouched = copy(u)
        value, gradient = ad_value_and_gradient(likelihood, u)
        total_gradient = copy(gradient)
        for (i, query) in enumerate(queries)
            v, g = ad_value_and_gradient(query, u, u[query.coordinates])
            value += v
            total_gradient[query.coordinates] .+= g
            @test all(k -> owner[k] == 0 || owner[k] == i - 1, query.coordinates)
        end
        full_gradient = similar(u)
        full_value, _ = sampler_value_and_gradient!(sampler, full_gradient, u)
        @test value ≈ full_value atol=2e-10 rtol=2e-12
        @test total_gradient ≈ full_gradient atol=2e-9 rtol=2e-10
        @test u == untouched
    end
    println("AUTHORED_FACTOR_SPLIT constrained=", constrained,
        " coordinates=", length(u0), " globals=", count(iszero, owner),
        " unit_sizes=", length.(units), " events=", length(events))
end

# The same split as SVI's prior term evaluates it (`_BRMSVIAuthoredPrior`,
# src/nlme_svi.jl). On the strict subject units it equals the standard-normal
# split term by term; with each subject also owning its coefficients, the
# global term plus every unit term equals RK's whole prior and Jacobian, with
# the units' reads of `tau` in the globals' gradient.
function check_svi_authored_prior(rk; constrained=false)
    m = brm_nlme_model(rk; ad_backend=FACTOR_AD)
    target = BRM._brm_svi_target(m)
    t = merge(target, (; globals=BRM._brm_svi_globals(m),
        units=m.view.subject_coordinates, prior=nothing))
    u0 = [0.17sin(k) for k in eachindex(m.unit)]
    strict = BRM._brm_svi_authored_prior(m, t.globals, t.units)
    @test count(!isnothing, strict.queries) == length(t.units) + 1
    catalogue = density_factors(rk.model, BRM.rk_bound_plan(rk))
    owned = fixture_owned(catalogue, length(t.units); constrained)
    globals = setdiff(t.globals, reduce(vcat, owned))
    units = [vcat(block, extra) for (block, extra) in zip(t.units, owned)]
    @test length(globals) == 4
    @test sort(vcat(globals, units...)) == eachindex(u0)
    local_units = (; globals, units)
    authored = BRM._brm_svi_authored_prior(m, globals, units)
    @test count(!isnothing, authored.queries) == length(units) + 1
    for offset in (0.0, 0.31, -0.24)
        u = u0 .+ offset
        untouched = copy(u)
        g = u[t.globals]
        P, gP = BRM._brm_svi_global_term(t, g)
        Pa, gPa = BRM._brm_svi_global_term(strict, t, g)
        @test Pa ≈ P atol=2e-10 rtol=2e-12
        @test gPa ≈ gP atol=2e-9 rtol=2e-10
        for (i, block) in enumerate(t.units)
            z = u[block]
            gg0, gz0, gg1, gz1 = zeros(length(g)), zeros(length(z)),
                zeros(length(g)), zeros(length(z))
            v0 = BRM._brm_svi_unit_prior!(gg0, gz0, nothing, t, i, g, z)
            v1 = BRM._brm_svi_unit_prior!(gg1, gz1, strict, t, i, g, z)
            @test v1 ≈ v0 atol=2e-12
            @test gz1 ≈ gz0 atol=2e-12
            @test iszero(gg1)
        end

        g = u[globals]
        value, gradient = BRM._brm_svi_global_term(authored, local_units, g)
        full_gradient = zeros(length(u))
        full_gradient[globals] .= gradient
        reads_globals = false
        for (i, block) in enumerate(units)
            gg, gz = zeros(length(g)), zeros(length(block))
            value += BRM._brm_svi_unit_prior!(gg, gz, authored, local_units, i, g, u[block])
            reads_globals |= !iszero(gg)
            full_gradient[globals] .+= gg
            full_gradient[block] .+= gz
        end
        @test reads_globals == constrained
        expected, expected_gradient =
            BRM._rk_nlme_population_prior_value_gradient(m.population_prior, u)
        @test value ≈ expected atol=2e-10 rtol=2e-12
        @test full_gradient ≈ expected_gradient atol=2e-9 rtol=2e-10
        @test u == untouched
    end
    println("SVI_AUTHORED_PRIOR constrained=", constrained, " globals=", length(globals),
        " unit_sizes=", length.(units), " strict_unit_sizes=", length.(t.units))
    @test target.globals == globals
    @test target.units == units
    check_svi_owned_target(target; constrained)
end

function check_svi_owned_target(t; constrained=false)
    m = t.m
    @test length(t.globals) == 4
    @test length.(t.units) == fill(constrained ? 5 : 3, 6)
    @test t.prior isa BRM._BRMSVIAuthoredPrior
    sampler = prepare_sampler(m.backend.model, BRM.rk_bound_plan(m.backend),
        m.unit; backend=FACTOR_AD)
    for offset in (0.0, 0.31, -0.24)
        u = [0.17sin(k) + offset for k in eachindex(m.unit)]
        g = u[t.globals]
        value, gg = BRM._brm_svi_global_term(t, g)
        gradient = zeros(length(u))
        gradient[t.globals] .= gg
        for (i, block) in enumerate(t.units)
            gg, gz = zeros(length(g)), zeros(length(block))
            value += BRM._brm_svi_unit_term!(gg, gz, t, i, g, u[block])
            gradient[t.globals] .+= gg
            gradient[block] .+= gz
        end
        expected_gradient = similar(u)
        expected, _ = sampler_value_and_gradient!(sampler, expected_gradient, u)
        @test value ≈ expected atol=2e-10 rtol=2e-12
        @test gradient ≈ expected_gradient atol=2e-9 rtol=2e-10
    end
    fit = brm_svi_fit(m; rng=Xoshiro(16), batch_size=2, max_iterations=2,
        ndraws=5, threads=1)
    @test fit.diagnostics.units == t.units
    @test fit.diagnostics.owner == t.ownership.owner
    @test size(fit.draws) == (length(m.unit), 5)
    @test all(isfinite, fit.draws)
    # With zero noise the packed draw must recover every factor's mean at
    # exactly its original coordinate, including the newly local vector.
    q = fit.diagnostics.q
    centered = merge(q, (; ρ=fill(-Inf, length(q.ρ)), Β=zero(q.Β),
        U=[Matrix{Float64}(I, length(b), length(b)) .* 1e100 for b in t.units]))
    draws = BRM._brm_svi_draws(Xoshiro(3), t, centered, 1)
    @test draws[t.globals, 1] == q.μ
    @test all(i -> draws[t.units[i], 1] ≈ q.M[i], eachindex(t.units))
end

@testset "BRM authored factors for block-local SVI" begin
    @testset "six-unit coefficient-vector reproduction" begin
        rk = RKBRMI(standard_vector_model(authored_factor_data()))
        check_authored_factor_split(rk)
        check_svi_authored_prior(rk)
    end
    @testset "non-unit dependent and constrained priors" begin
        rk = RKBRMI(dependent_constrained_model(authored_factor_data()))
        check_authored_factor_split(rk; constrained=true)
        check_svi_authored_prior(rk; constrained=true)
    end
    @testset "a joint law across units is refused" begin
        rk = RKBRMI(joint_simplex_model(authored_factor_data()))
        m = brm_nlme_model(rk; ad_backend=FACTOR_AD)
        t = BRM._brm_svi_target(m)
        catalogue = density_factors(rk.model, BRM.rk_bound_plan(rk))
        simplex = sort(factor_coordinates(select_factors(catalogue; kind=:prior, site=:s)))
        # Precondition: one atomic event owns both simplex coordinates.
        @test length(simplex) == 2
        @test any(e -> e.site == :s && length(factor_coordinates(catalogue, e)) == 2,
            collect(catalogue))
        owned = [i <= 2 ? [simplex[i]] : Int[] for i in eachindex(t.units)]
        globals = setdiff(t.globals, reduce(vcat, owned))
        units = [vcat(block, extra) for (block, extra) in zip(t.units, owned)]
        refusal(globals, units) = try
            BRM._brm_svi_authored_prior(m, globals, units)
            nothing
        catch e
            e isa BRM.BRMNLMEViewError || rethrow()
            sprint(showerror, e)
        end
        # The event owns coordinates of units 1 and 2, whatever it reads.
        @test occursin("coupled transform across units", something(refusal(globals, units), ""))
        # One coordinate in two places is refused before any factor is prepared.
        @test occursin("assigned to unit 1 and to the globals",
            something(refusal(t.globals, units), ""))
        @test occursin("appears twice in unit 1",
            something(refusal(globals, vcat([vcat(units[1], units[1][end])], units[2:end])), ""))
        # The strict subject units leave the simplex global, so they build, with
        # the simplex in the global term. RKPPL 0.1.72 reports a constant
        # Dirichlet's support as its own block (snag
        # ReactiveKernels/factor-support-r-2b8c2968); 0.1.70 reported the whole
        # packed vector, which made it read every subject and refused this.
        @test refusal(t.globals, t.units) === nothing
        @test sort(factor_support(select_factors(catalogue; site=:s);
            u0=m.unit)) == simplex
    end
end
