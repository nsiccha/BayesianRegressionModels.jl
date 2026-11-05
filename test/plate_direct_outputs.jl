# A direct downstream read keeps the producer's cell axis, including when
# fixed-width vector cells are stored as matrix columns. Public toy models.
# Run: julia --startup-file=no --project=test test/plate_direct_outputs.jl
using Test, BayesianRegressionModels, StanBlocks
import BridgeStan
using Distributions: Normal, logpdf
const BRM = BayesianRegressionModels

function direct_output_builder(; side=false, shape=:vector, reader_first=false)
    value = shape === :scalar ? :(a + axis[i]) :
        shape === :ragged ? :(a .* paths[i]) :
        shape === :matrix ? :(reshape([a, axis[i], 2.0 * a, -0.5 * axis[i]], 2, 2)) :
        :([a + axis[i], 2.0 * a - 0.5 * axis[i]])
    outputs = Any[:(cells[i] = $value), :(spare[i] = axis[i])]
    side || reverse!(outputs)
    producer = quote
        @plate for i in eachindex(axis)
            $(outputs...)
        end
    end
    location = shape === :scalar ? :(cells[i]) : :(sum(cells[i]))
    reader = quote
        @plate for i in eachindex(axis)
            location[i] = $location
            y[i] ~ Normal(location[i], 0.7)
        end
    end
    plates = reader_first ? (reader, producer) : (producer, reader)
    brm(quote
        a ~ Normal(0, 1)
        $(plates...)
    end; mod=@__MODULE__)
end

function direct_output_problem(sb, tag)
    directory = mktempdir()
    BRM.stan_instantiate(sb; path=joinpath(directory, tag * ".stan"))
end

@testset "fixed-width cells retain their outer axis across plates" begin
    for side in (false, true), n in (2, 5)
        # n=2 would let a transposed read pass the extent check; unequal
        # values and the numerical oracle still detect that wrong axis.
        axis = collect(range(-0.8, 1.1; length=n))
        data = (; axis, y=collect(range(0.3, -0.4; length=n)))
        before = deepcopy(data)
        builder = direct_output_builder(; side, reader_first=side)
        sb = SBBRMI(builder(data); mod=@__MODULE__)
        @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
        problem = direct_output_problem(sb, "direct-$side-$n")
        @test BridgeStan.param_unc_num(problem.model) == 1
        for a in (-0.3, 0.0, 0.6)
            location = 3a .+ 0.5 .* axis
            expected = logpdf(Normal(0, 1), a) + sum(logpdf.(Normal.(location, 0.7), data.y))
            expected_gradient = -a + 3sum(data.y .- location) / 0.7^2
            value, gradient = BridgeStan.log_density_gradient(problem.model, [a]; propto=false)
            @test value ≈ expected atol=2e-11 rtol=2e-11
            @test only(gradient) ≈ expected_gradient atol=2e-11 rtol=2e-11
        end

        # The identical declaration without y must forward-generate the
        # scalar prior, every complete cell and each response.
        prior = SBBRMI(builder((; axis)); mod=@__MODULE__)
        @test StanBlocks.stanc_check(BRM.stan_code(prior); warn_pedantic=false).ok
        prior_problem = direct_output_problem(prior, "direct-prior-$side-$n")
        @test BridgeStan.param_unc_num(prior_problem.model) == 0
        names = BridgeStan.param_names(prior_problem.model; include_tp=true, include_gq=true)
        descriptor = brm_descriptor(prior)
        location_name = String(brm_output(descriptor, :location).name)
        indices = [only(findall(==(location_name * "." * string(i)), names)) for i in 1:n]
        a_index = only(findall(==("a"), names))
        rng = BridgeStan.StanRNG(prior_problem.model, 41)
        for _ in 1:3
            draw = BridgeStan.param_constrain(prior_problem.model, Float64[];
                include_tp=true, include_gq=true, rng)
            @test all(isfinite, draw)
            @test draw[indices] ≈ 3draw[a_index] .+ 0.5 .* axis atol=2e-12 rtol=2e-12
        end
        @test isequal(data, before)
    end
end

@testset "direct iteration preserves scalar, ragged and matrix cells" begin
    data = (; axis=[0.1, 0.7, -0.2], paths=[[1.0], [2.0, 3.0], [0.4, -0.1, 0.2]],
        y=[0.2, -0.3, 0.5])
    for shape in (:scalar, :ragged, :matrix), side in (false, true)
        builder = direct_output_builder(; shape, side)
        for values in (data, (; axis=data.axis, paths=data.paths))
            sb = SBBRMI(builder(values); mod=@__MODULE__)
            @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
        end
    end
end
