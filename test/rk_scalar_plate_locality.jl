# A collected scalar plate must cost only the selected observation's rows.
# Run: julia --project=test test/rk_scalar_plate_locality.jl
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, LogDensityProblems, Enzyme
using DifferentiationInterface: AutoEnzyme

const SCALAR_PLATE_AD = AutoEnzyme(; mode=Enzyme.Reverse)
const collected_scalar_model = @brm begin
    beta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        signal[i] = beta * x[i]
    end
    location = signal + shift
    y ~ Normal(location, 1.0)
end
const direct_scalar_model = @brm begin
    beta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        y[i] ~ Normal(beta * x[i] + shift[i], 1.0)
    end
end

function scalar_cell_allocations(query, sampler, u, i)
    gradient = similar(u)
    # Warm both the primal and ordinary Reverse specialization before measuring.
    for _ in 1:3
        query(u, i)
        cell_value_and_gradient!(sampler, gradient, u, i)
    end
    value_bytes = minimum(@allocated(query(u, i)) for _ in 1:5)
    gradient_bytes = minimum(@allocated(
        cell_value_and_gradient!(sampler, gradient, u, i)) for _ in 1:5)
    (; value_bytes, gradient_bytes)
end

@testset "collected scalar plate values and gradients stay row-local" begin
    for (name, builder) in ((:collected, collected_scalar_model), (:direct, direct_scalar_model))
        costs = []
        for n in (16, 1024)
            data = (; x=collect(range(0.5, 2.0; length=n)),
                shift=collect(range(-0.2, 0.3; length=n)), y=fill(0.7, n))
            before = deepcopy(data)
            rk = RKBRMI(builder(data))
            problem = rk_logdensity_problem(rk; ad_backend=SCALAR_PLATE_AD)
            query = prepare_cell_query(rk.model, rk_bound_plan(rk), :y; locality=:error)
            sampler = prepare_cell_sampler(rk.model, rk_bound_plan(rk), :y, [0.3];
                backend=SCALAR_PLATE_AD, locality=:error)
            for beta in (-0.4, 0.3)
                u = [beta]
                residual = data.y .- beta .* data.x .- data.shift
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                @test value ≈ logpdf(Normal(), beta) + sum(logpdf.(Normal(), residual))
                @test gradient ≈ [-beta + sum(residual .* data.x)]
                for i in (1, n ÷ 2, n)
                    expected = logpdf(Normal(), residual[i])
                    @test query(u, i) ≈ expected
                    cell_value, cell_gradient = cell_value_and_gradient!(sampler, similar(u), u, i)
                    @test cell_value ≈ expected
                    @test cell_gradient ≈ [residual[i] * data.x[i]]
                    @test u == [beta]
                end
            end
            @test isequal(data, before)
            cost = scalar_cell_allocations(query, sampler, [0.3], n ÷ 2)
            push!(costs, cost)
            println("SCALAR_PLATE_LOCALITY form=$name n=$n cost=$cost")
        end
        # Allow small compiler/runtime bookkeeping changes, but no N-sized
        # predictor or derivative buffer for a single selected row.
        @test costs[2].value_bytes <= costs[1].value_bytes + 1024
        @test costs[2].gradient_bytes <= costs[1].gradient_bytes + 1024
    end
end
