# A `gmrf` formula cell on SBBRMI that also reads another plate's complete
# cells. Preparing the formula removes the raw coordinates it consumed and adds
# its prepared inputs, so the plate's iterables follow that final list, and the
# other plate's output must still be read a whole cell at a time: passed as a
# plain positional, its matrix of cells does not transpile. A dense oracle
# (FEM Matérn ν=1 × OU time) checks that each cell reads its own field and
# offset. Public toy model.
# Run: julia --startup-file=no --project=test test/gmrf_stan_plate_outputs.jl
include("gmrf_rk_cell_models.jl")
import BridgeStan as BS

function gmrf_output_cell_model(input)
    @brm input begin
        a ~ Normal(0, 1)
        @plate for i in eachindex(units)
            offsets[i] = [a + shift[i], 2.0 * a - 0.5 * shift[i]]
        end
        rows = groups(unit => units)
        @plate for k in eachindex(units)
            field ~ 0 + gmrf((east[rows[k]], north[rows[k]]), time[rows[k]];
                mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east[rows[k]], north[rows[k]]), time[rows[k]])) = 1.2
            length_scale(field, gmrf((east[rows[k]], north[rows[k]]), time[rows[k]]), space) = 0.4
            length_scale(field, gmrf((east[rows[k]], north[rows[k]]), time[rows[k]]), time) = 1.7
            y[rows[k]] ~ Normal(field .+ sum(offsets[k]), 0.2)
        end
    end
end

function gmrf_output_cell_check()
    input = merge(data, (; y=[0.2, -0.1, 0.3, 0.05], unit=[10, 20, 10, 10], units=[10, 20],
        shift=[0.4, -0.7]))
    rows = ([1, 3, 4], [2])
    sb = SBBRMI(gmrf_output_cell_model(input); mod=@__MODULE__)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    problem = BRM.stan_instantiate(sb)
    names = BS.param_unc_names(problem.model)
    @test length(names) == 21
    a = only(findall(==("a"), names))
    output = only(brm_outputs(brm_descriptor(sb; highlights=());
        predictor=:field, component=:field, kind=:parameter))
    field = brm_output_coordinates(output, names)
    @test length(field) == 20
    cells = (field[1:15], field[16:20])
    function oracle(point)
        lp = logpdf(Normal(0, 1), point[a])
        for cell in 1:2
            # Normal(mu + offset, s) at y is Normal(mu, s) at y - offset.
            offset = 3point[a] + 0.5input.shift[cell]
            shifted = merge(input, (; y=input.y .- offset))
            z = point[cells[cell]]
            obs, times = gmrf_cell_rows_logpdf(z, 1.2, shifted, rows[cell], nothing)
            lp += obs + gmrf_cell_field_logpdf(z, 0.4, 1.7, ones(1, 1), times)
        end
        lp
    end
    for point in (zeros(length(names)), 0.05 .* sin.(eachindex(names)))
        value, gradient = BS.log_density_gradient(problem.model, point; propto=false, jacobian=true)
        @test value ≈ oracle(point) atol=2e-8 rtol=2e-10
        for j in eachindex(point)
            plus, minus = copy(point), copy(point)
            h = 1e-5
            plus[j] += h
            minus[j] -= h
            @test gradient[j] ≈ (oracle(plus) - oracle(minus)) / (2h) atol=3e-5 rtol=3e-6
        end
    end
end

@testset "SBBRMI gmrf formula cell reads another plate's complete cells" begin
    gmrf_output_cell_check()
end
