# test/rk_batched_field_cells.jl — a paired Fourier field written as array cells.
#
# RUN: julia --project=test test/rk_batched_field_cells.jl [testset-filter...]
#
# ODLPlaus's field components call a StanBlocks `@deffun @juliacompat` per row,
# `value[s] = paired_channel_state(frequency, lengths, amplitude, correlation,
# coefficients[block[s],:], coordinates[s])`. On RKBRMI that call is an opaque leaf,
# so the plate runs cell by cell. Written as array statements in the cell, over
# matrices every cell shares, the same field is one source for both backends:
# SBBRMI compiles it to Stan, and ReactiveKernels from 0.1.127 (RK `fa3deb83`,
# "Plates of products run as matrix products" in RK's `docs/src/compiler.md`)
# runs the plate as matrix products over tiles of its cells. BRM todo `07y7ghy`,
# from RK snag `plate-of-per-row-d1b4ed43`.
#
# Rules the cell spelling below follows, each found by a refusal or a mismatch:
# - The products are in the cell. Model-scope arithmetic is elementwise, so
#   `Z * odd` outside the plate broadcasts instead of multiplying.
# - `reshape` takes integer data sizes (`pairs`, not `2 * features`, which
#   StanBlocks types as real).
# - The cosine and sine coefficients are picked by constant selector matrices,
#   not slices: RKPPL refuses a range slice of a sampled array at model scope,
#   and Stan has no stepped range.
# - The frequencies come transposed as data (`frequency_t`), since Stan has no
#   `transpose` call.
# An `_rk_callable_source!` `@kernel` provider for the per-row helper does not
# batch: RK keeps the cell loop for a cell that calls a child graph.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Logging

# RK planning the field plate as matrix products. The landed number is read
# from ReactiveKernels' `Project.toml` at RK `fa3deb83`; a move of the test pin
# past it turns the `broken` line into a passing one.
const RK_BATCHES_PLATE_PRODUCTS = pkgversion(ReactiveKernels) >= v"0.1.127"

module BatchedFieldCells
using BayesianRegressionModels, StanBlocks, Distributions
using StanBlocks: @juliacompat, rep_vector

# The per-row reference: ODLPlaus `src/joint_field_components.jl` (kb-approved
# `0ef0de54`), coefficient k + C(2j-2) for the cosine of feature j and channel k,
# k + C(2j-1) for its sine.
StanBlocks.@deffun @juliacompat row_channel_state(frequency::matrix[D,F], lengths,
        amplitude, correlation::matrix[C,C], coefficients, coordinate)::vector[C] = begin
    raw = rep_vector(0.0, C)
    for j in 1:F
        phase = 0.0
        for d in 1:D
            phase += frequency[d,j] * coordinate[d] / lengths[d]
        end
        cos_phase = cos(phase)
        sin_phase = sin(phase)
        for k in 1:C
            raw[k] += coefficients[k+C*(2j-2)] * cos_phase +
                coefficients[k+C*(2j-1)] * sin_phase
        end
    end
    correlation * raw * amplitude / sqrt(F)
end

@brm function row_channel_field(coordinates, block, frequency, correlation, blocks, width,
        coefficient_count, dimensions)
    coefficients = reshape(z, blocks, width)
    @plate for s in eachindex(block)
        value[s] = row_channel_state(frequency, lengths, amplitude, correlation,
            coefficients[block[s],:], coordinates[s])
    end
    z[1:coefficient_count] .~ Normal.(0., 1.)
    amplitude ~ Exponential(1.0)
    lengths[1:dimensions] .~ LogNormal.(0.0, 1.0)
    return value
end

# The same field as array cells. `reshape(z, channels, pairs)` keeps the packing:
# column 2j-1 holds feature j's cosine coefficients, column 2j its sine ones, and
# `odd`/`even` (pairs × features) pick those columns.
@brm function cell_channel_field(coordinates, frequency_t, correlation, odd, even, pairs,
        channels, features, coefficient_count, dimensions)
    Z = reshape(z, channels, pairs)
    scale = amplitude / sqrt(features)
    @plate for s in eachindex(coordinates)
        Zc = Z * odd
        Zs = Z * even
        phase = frequency_t * (coordinates[s] ./ lengths)
        raw = Zc * cos.(phase) .+ Zs * sin.(phase)
        value[s] = (correlation * raw) * scale
    end
    z[1:coefficient_count] .~ Normal.(0., 1.)
    amplitude ~ Exponential(1.0)
    lengths[1:dimensions] .~ LogNormal.(0.0, 1.0)
    return value
end
end # module

function batched_field_data(; n=40, C=3, F=4, D=3)
    frequency = reshape(sin.(1:(D * F)), D, F)
    (; coordinates=[cos.(s .* (1:D) ./ 50) for s in 1:n], block=ones(Int, n),
        frequency, frequency_t=permutedims(frequency),
        correlation=[i >= j ? (i == j ? 1.0 : 0.2 / (i + j)) : 0.0 for i in 1:C, j in 1:C],
        odd=[i == 2j - 1 ? 1.0 : 0.0 for i in 1:2F, j in 1:F],
        even=[i == 2j ? 1.0 : 0.0 for i in 1:2F, j in 1:F],
        pairs=2F, blocks=1, width=2C * F, coefficient_count=2C * F, dimensions=D,
        features=F, channels=C, total=sin.(1:n) ./ 3)
end

function batched_field_model(spelling)
    field = spelling === :row ?
        :(row_channel_field(coordinates, block, frequency, correlation, blocks, width,
            coefficient_count, dimensions)) :
        :(cell_channel_field(coordinates, frequency_t, correlation, odd, even, pairs,
            channels, features, coefficient_count, dimensions))
    brm(quote
        field ~ $field
        @plate for i in eachindex(total)
            total[i] ~ Normal(sum(field[i]), sigma)
        end
        sigma ~ Exponential(1)
    end; mod=BatchedFieldCells)
end

const BATCHED_FIELD_AD = AutoEnzyme(; mode=Enzyme.Reverse)

function batched_field_sampler(model)
    artifact = BRM.emit_rk_artifact(model; case_id="batched-field-cells")
    bound = BRM.rk_translate_artifact(artifact)
    built = build_kernel(bound)
    u = collect(range(-0.3, 0.3; length=built.layout.total))
    logs, sampler = Test.collect_test_logs(min_level=Logging.Debug) do
        prepare_sampler(built, bound, u; backend=BATCHED_FIELD_AD)
    end
    (; sampler, u, names=coordinate_names(built.layout),
        batched=any(log -> log.group === :plate_batching, logs))
end

@stestset "an array-cell field equals the per-row @deffun field on RK" begin
    data = batched_field_data()
    row = batched_field_sampler(batched_field_model(:row)(data))
    cell = batched_field_sampler(batched_field_model(:cell)(data))
    @test cell.names == row.names
    for u in (row.u, zeros(length(row.u)))
        g_row, g_cell = zero(u), zero(u)
        v_row, _ = sampler_value_and_gradient!(row.sampler, g_row, u)
        v_cell, _ = sampler_value_and_gradient!(cell.sampler, g_cell, u)
        @test v_cell ≈ v_row rtol=1e-12
        @test g_cell ≈ g_row rtol=1e-10 atol=1e-12
    end
end

@stestset "RK plans an array-cell field plate as matrix products" begin
    data = batched_field_data()
    @test !batched_field_sampler(batched_field_model(:row)(data)).batched
    @test batched_field_sampler(batched_field_model(:cell)(data)).batched broken=!RK_BATCHES_PLATE_PRODUCTS
end

@stestset "SBBRMI compiles the array-cell field with RK's density and gradient" begin
    data = batched_field_data()
    model = batched_field_model(:cell)(data)
    rk, problem = consumer_problem(model)
    sb = SBBRMI(model; mod=BatchedFieldCells, total_groups=())
    stan = consumer_stan(model, "batched-field-cells"; mod=BatchedFieldCells)
    mapping = [pair.rk => pair.stan for pair in
        brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model)).pairs]
    names = coordinate_names(rk.model.layout)
    for u in (zeros(length(names)), collect(range(-0.2, 0.3; length=length(names))))
        check_consumer_stan(problem, stan, mapping, rk, u)
    end
end
