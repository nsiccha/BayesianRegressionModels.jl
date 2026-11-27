# The full Wren case-study declaration on small, simulated data. Run with
# julia --startup-file=no --project=test test/wren_hsgp.jl
using Test, BayesianRegressionModels, Random, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, WarmupHMC, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
include("wren_hsgp_fixture.jl")

const BRM = BayesianRegressionModels
const AD = AutoEnzyme(; mode=Enzyme.Reverse)
BLAS.set_num_threads(1)

@testset "Wren: six HSGPs, three row axes and every adaptive coordinate" begin
    data = WrenHSGPFixture.data()
    before = deepcopy(data)
    @test (length(data.cell), length(data.cy_decade), length(data.wren)) == (4, 8, 32)
    rk = RKBRMI(WrenHSGPFixture.model(data))
    @test Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt) !== nothing
    @test Base.get_extension(BRM, :BayesianRegressionModelsWarmupHMCExt) !== nothing
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    inventory = adaptive_centering_cells(rk)
    blocks = filter(b -> b.family === :hsgp, inventory.blocks)
    @test length(blocks) == 6
    @test sort([b.n_basis for b in blocks]) == [8, 8, 150, 150, 864, 900]
    @test count(b -> b.spectrum === :periodic, blocks) == 1
    @test length(inventory.cells) == 2084
    @test count(c -> c.family === :ranef, inventory.cells) == 4
    @test count(c -> c.family === :hsgp, inventory.cells) == 2080
    @test all(c -> c.family !== :hsgp || iszero(c.target), inventory.cells)
    @test allunique(c.effect for c in inventory.cells)

    # Read the actual program's six spectral vectors, independently of the
    # adaptive wrapper's log-scale reconstruction and tensor ordering.
    records = filter(r -> r.kind === :hsgp, BRM._rk_coordinate_records(rk))
    ports = Tuple(Symbol(r.declaration, ".", r.spectral) for r in records)
    spectrum = BRM._rk_value_reader(rk, ports)
    names = String.(BRM._rk_layout_coordinate_names(rk))
    positions = Dict(n => i for (i, n) in enumerate(names))
    ranef = only(adaptive_centering_blocks(rk))
    function log_scales(q)
        result = Dict(i => q[only(ranef.log_scales)] for i in ranef.effects)
        values = spectrum(q)
        for (r, port) in zip(records, ports), (b, s) in enumerate(values[port])
            # The centring map smooths spectral scales near exp(-150) to
            # keep source coordinates representable. This changes only the
            # coordinates: the compiled model still uses its exact spectrum.
            result[positions[string(r.declaration, ".beta_raw.", b)]] = log(s + exp(-150))
        end
        result
    end

    q = brm_start(rk, (;); base=:prior_median)
    q .+= 0.05 .* randn(Xoshiro(71), length(q))
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    gradient = copy(gradient)
    @test isfinite(value)
    @test all(isfinite, gradient)
    zero_frame = adaptive_centering_problem(rk, problem, AD)
    v0, g0 = LogDensityProblems.logdensity_and_gradient(zero_frame, q)
    @test v0 ≈ value rtol=1e-12
    @test g0 ≈ gradient rtol=1e-10 atol=1e-10

    # Both endpoints and genuinely continuous controls. Place the same
    # physical point in each source frame, so tiny high-frequency scales do
    # not turn the test point into an extreme model proposal.
    for controls in (ones(length(inventory.cells)),
                     rand(Xoshiro(72), length(inventory.cells)))
        adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
        report = adaptive_centering_controls(adaptive, rk)
        @test [r.index for r in report] == [c.effect for c in inventory.cells]
        scales = log_scales(q)
        source = copy(q)
        direction = 0.1 .* randn(Xoshiro(73), length(q))
        for (cell, c) in zip(inventory.cells, controls)
            multiplier = exp(c * scales[cell.effect])
            source[cell.effect] *= multiplier
            direction[cell.effect] *= multiplier
        end
        jac, recovered = WarmupHMC.reparametrizer(adaptive)(copy(source))
        @test recovered ≈ q rtol=1e-11 atol=1e-11
        @test jac ≈ -sum(c * scales[cell.effect]
            for (cell, c) in zip(inventory.cells, controls)) rtol=1e-12
        function oracle(u)
            ls = log_scales(u)
            mapped = copy(u)
            logjac = 0.0
            for (cell, c) in zip(inventory.cells, controls)
                mapped[cell.effect] *= exp(-c * ls[cell.effect])
                logjac -= c * ls[cell.effect]
            end
            logjac + LogDensityProblems.logdensity(problem, mapped)
        end
        va, ga = LogDensityProblems.logdensity_and_gradient(adaptive, source)
        ga = copy(ga)
        @test va ≈ value + jac rtol=1e-11 atol=1e-9
        @test all(isfinite, ga)
        h = 1e-5
        fd = (oracle(source + h * direction) - oracle(source - h * direction)) / (2h)
        @test dot(ga, direction) ≈ fd rtol=2e-5 atol=2e-5
    end
    @test isequal(data, before)
end
