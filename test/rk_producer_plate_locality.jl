# Run: julia --project=test test/rk_producer_plate_locality.jl
# A reader selects rows of an array produced by another plate. Shared
# correlated effects exercise the view/reshape/product chain in BRM emission.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, LinearAlgebra, LogDensityProblems, Enzyme
using DifferentiationInterface: AutoEnzyme

const PRODUCER_PLATE_AD = AutoEnzyme(; mode=Enzyme.Reverse)
const PRODUCER_PLATE_DATA = (;
    group=["a", "b"], paths=[[0.2, 0.7, -0.4], [1.3, -0.1]],
    reader=[2, 1, 2], positions=[[2, 1], [3], [1]],
    y=[[0.0, 0.2], [0.1], [0.4]])

function producer_plate_model(indirect)
    source = quote
        a ~ 1 + (1 | shared | group)
        b ~ 1 + (1 | shared | group)
    end
    body = if indirect
        quote
            @plate for p in eachindex(paths)
                signal[p] = exp(a[p]) .* paths[p] .+ b[p]
            end
            @plate for i in eachindex(y)
                y[i] ~ Normal(signal[reader[i]][positions[i]], 1.0)
            end
        end
    else
        quote
            @plate for i in eachindex(y)
                y[i] ~ Normal(exp(a[reader[i]]) .* paths[reader[i]][positions[i]] .+ b[reader[i]], 1.0)
            end
        end
    end
    append!(source.args, body.args)
    RKBRMI(brm(source; mod=@__MODULE__)(PRODUCER_PLATE_DATA))
end

function producer_plate_fd(f, u; h=1e-5)
    map(eachindex(u)) do j
        plus = copy(u); minus = copy(u)
        plus[j] += h; minus[j] -= h
        (f(plus) - f(minus)) / (2h)
    end
end

@testset "producer plate selected rows retain strict values and supports" begin
    data_before = deepcopy(PRODUCER_PLATE_DATA)
    direct, producer = producer_plate_model(false), producer_plate_model(true)
    @test direct.model.layout.total == producer.model.layout.total
    @test coordinate_names(direct.model.layout) == coordinate_names(producer.model.layout)
    d = PRODUCER_PLATE_DATA
    records = []
    for rk in (direct, producer)
        bound = rk_bound_plan(rk)
        full = rk_logdensity_problem(rk; ad_backend=PRODUCER_PLATE_AD)
        cell = prepare_cell_query(rk.model, bound, :y; locality=:error)
        footprint = partition_footprint_kernel(cell.kernel)
        factors = prepare_cell_factors(rk.model, bound, :y; locality=:error)
        @test length(cell_reductions(factors)) == 1
        outer = prepare_cell_factor_query(factors, :outer)
        term = prepare_cell_factor_query(factors, 1)
        outer_footprint = partition_footprint_kernel(outer.kernel)
        term_footprint = partition_footprint_kernel(term.kernel)
        values = BayesianRegressionModels._rk_value_reader(rk, (:a, :b))
        u0 = zeros(rk.model.layout.total)
        sampler = prepare_cell_sampler(rk.model, bound, :y, u0;
            backend=PRODUCER_PLATE_AD, locality=:error)
        point_records = []
        for u in (u0, 0.07 .* collect(1:length(u0)))
            before = copy(u)
            v = values(u)
            cell_records = []
            for i in eachindex(d.y)
                g = d.reader[i]
                expected = sum(logpdf.(Normal.(exp(v.a[g]) .* d.paths[g][d.positions[i]] .+ v.b[g], 1.0), d.y[i]))
                support = Base.invokelatest(footprint, u, i).unconstrained
                @test support !== :whole
                @test length(support) < length(u)
                @test cell(u, i) ≈ expected atol=1e-12
                value, gradient = cell_value_and_gradient!(sampler, similar(u), u, i)
                @test value ≈ expected atol=1e-12
                @test gradient ≈ producer_plate_fd(x -> cell(x, i), u) rtol=2e-6 atol=2e-6
                @test all(iszero, gradient[setdiff(eachindex(u), support)])
                rows = reduction_rows(factors, 1, i)
                @test length(rows) == length(d.y[i])
                terms = [term(u, i, t) for t in eachindex(rows)]
                sums = [sum(terms; init=0.0)]
                @test outer(u, sums, i) ≈ expected atol=1e-12
                term_supports = [Base.invokelatest(term_footprint, u, i, t).unconstrained for t in eachindex(rows)]
                direct_support = Base.invokelatest(outer_footprint, u, sums, i).unconstrained
                @test all(s -> s !== :whole && issubset(s, support), term_supports)
                @test direct_support !== :whole
                @test sort!(unique(vcat(direct_support, term_supports...))) == support
                push!(cell_records, (; value, gradient=copy(gradient), support,
                    direct_support, term_supports, terms))
            end
            full_value, full_gradient = LogDensityProblems.logdensity_and_gradient(full, u)
            full_gradient = copy(full_gradient)
            @test isfinite(full_value)
            @test full_gradient ≈ producer_plate_fd(x -> LogDensityProblems.logdensity(full, x), u) rtol=2e-6 atol=2e-6
            @test u == before
            push!(point_records, (; full_value, full_gradient, cells=cell_records))
        end
        # Structural supports are unchanged at zero and nonzero inputs.
        @test getproperty.(point_records[1].cells, :support) == getproperty.(point_records[2].cells, :support)
        push!(records, point_records)
    end
    for (left, right) in zip(records...)
        @test left.full_value ≈ right.full_value atol=1e-12
        @test left.full_gradient ≈ right.full_gradient atol=1e-10
        for (a, b) in zip(left.cells, right.cells)
            @test a.value ≈ b.value atol=1e-12
            @test a.gradient ≈ b.gradient atol=1e-10
            @test a.support == b.support
            @test a.direct_support == b.direct_support
            @test a.term_supports == b.term_supports
            @test a.terms ≈ b.terms atol=1e-12
        end
    end
    @test isequal(PRODUCER_PLATE_DATA, data_before)
end

@testset "producer plate sparse Laplace agrees with direct reads" begin
    targets = map((false, true)) do indirect
        rk = producer_plate_model(indirect)
        plan = brm_laplace_plan(rk; ad_backend=PRODUCER_PLATE_AD,
            integrate=(:group,), draws=12)
        @test plan.integrated_dimension == 4
        automatic = brm_laplace_plan(rk; ad_backend=PRODUCER_PLATE_AD,
            integrate=:auto, draws=12)
        @test automatic.integrated_dimension == 0
        @test all(t -> !t.admitted && t.importance_ess === nothing, automatic.terms)
        target = brm_laplace_problem(rk; ad_backend=PRODUCER_PLATE_AD,
            integrate=plan, maxiter=100)
        @test length(target.components) == 2
        target
    end
    direct, producer = targets
    @test direct.population == producer.population
    for phi in (zeros(length(direct.population)), fill(0.07, length(direct.population)))
        value, gradient = LogDensityProblems.logdensity_and_gradient(direct, phi)
        expected_gradient = copy(gradient)
        actual, actual_gradient = LogDensityProblems.logdensity_and_gradient(producer, phi)
        @test isfinite(value)
        @test actual ≈ value atol=1e-9
        @test actual_gradient ≈ expected_gradient atol=2e-8 rtol=2e-8
        @test actual_gradient ≈ producer_plate_fd(x -> LogDensityProblems.logdensity(producer, x), phi) atol=2e-5 rtol=2e-5
    end
end
