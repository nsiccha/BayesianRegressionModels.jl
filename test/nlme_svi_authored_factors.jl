# Independent acceptance of the public authored-density API needed by SVI's
# block-local coordinate extension. This does not claim that brm_svi_fit uses
# this layout yet: the connected-unit integration is a separate prerequisite.
# Run: julia --startup-file=no --project=test test/nlme_svi_authored_factors.jl
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
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

# Use authored ownership, never support, to select a child's prior. The
# hyperparameter's prior must stay global even when every child reads it.
function check_authored_factor_split(rk; constrained=false)
    plan = BayesianRegressionModels.rk_bound_plan(rk)
    catalogue = density_factors(rk.model, plan)
    events = collect(catalogue)
    view = brm_nlme_view(rk)
    units = deepcopy(view.subject_coordinates)
    n = length(units)
    u0 = [0.17sin(k) for k in 1:rk.model.layout.total]
    owner = zeros(Int, length(u0))
    for (i, block) in enumerate(units)
        owner[block] .= i
        for entry in (2i - 1, 2i)
            coordinates = factor_coordinates(select_factors(catalogue;
                kind=:prior, site=:z, entries=entry:entry))
            @test length(coordinates) == 1
            append!(block, coordinates)
            owner[coordinates] .= i
        end
        if constrained
            for site in (:positive, :bounded)
                coordinates = factor_coordinates(select_factors(catalogue;
                    kind=:prior, site, entries=i:i))
                @test length(coordinates) == 1
                append!(block, coordinates)
                owner[coordinates] .= i
            end
        end
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

@testset "BRM authored factors for block-local SVI" begin
    @testset "six-unit coefficient-vector reproduction" begin
        check_authored_factor_split(RKBRMI(standard_vector_model(authored_factor_data())))
    end
    @testset "non-unit dependent and constrained priors" begin
        check_authored_factor_split(RKBRMI(dependent_constrained_model(authored_factor_data()));
            constrained=true)
    end
end
