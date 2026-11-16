# Adaptive partial centring of RKBRMI HSGP basis weights (todo 0nsdowt,
# increment 5): `adaptive_centering_problem(rk, ...)` adapts each weight's cell
# at the spectral scale RK's own program computes, `<decl>.spectral_scale`.
#
#     julia --project=test test/rk_adaptive_hsgp_centering.jl [filter...]
using Test, BayesianRegressionModels, Distributions, Random, LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems, WarmupHMC
using DifferentiationInterface: AutoEnzyme
using CategoricalArrays
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

const AD = AutoEnzyme(; mode=Enzyme.Reverse)

function hsgp_data(; seed=1, n=90)
    rng = Xoshiro(seed)
    x = sort(randn(rng, n))
    x2 = randn(rng, n)
    t = 48 .* rand(rng, n)
    g = categorical(rand(rng, ["a", "b", "c"], n))
    u = categorical(rand(rng, string.("u", 1:6), n))
    y = sin.(2x) .+ 0.3 .* x2 .+ 0.2 .* randn(rng, n)
    (; y, x, x2, t, g, u)
end

layout_names(rk) = String.(BRM._rk_layout_coordinate_names(rk))
hsgp_records(rk) = [r for r in BRM._rk_coordinate_records(rk.plan) if r.kind === :hsgp]

# Each HSGP weight's position and its log spectral scale as RK's program
# computes it (`<decl>.spectral_scale`, levels × modes for a per-level scale),
# in the wrapper's cell order: term by term, level by level, basis by basis.
function spectral_cells(rk)
    names = layout_names(rk)
    records = hsgp_records(rk)
    ports = Tuple(Symbol(r.declaration, ".", r.spectral) for r in records)
    read = BRM._rk_value_reader(rk, ports)
    cells = Tuple{Int,Function}[]
    for (r, port) in zip(records, ports)
        weights = r.grouped ? String(r.locals.weights) : "beta_raw"
        prefix = string(r.declaration, ".", weights, ".")
        levels = r.grouped ? length(r.levels) : 1
        B = count(n -> startswith(n, prefix), names) ÷ levels
        for g in 1:levels, b in 1:B
            name = r.grouped ? "$prefix$g.$b" : "$prefix$b"
            push!(cells, (findfirst(==(name), names),
                          x -> log(read(x)[port][b])))
        end
    end
    cells
end

# The independent map: each weight `y = x * s^(-c)` (RK's weights are compiled
# non-centred), the log-Jacobian `-c * log s`.
function scalar_oracle(x, cells, controls)
    y, ljac = copy(x), zero(eltype(x))
    for ((effect, log_s), c) in zip(cells, controls)
        y[effect] = x[effect] * exp(-c * log_s(x))
        ljac -= c * log_s(x)
    end
    ljac, y
end

# The wrapper against the oracle at random controls: map, log-Jacobian, the
# wrapped density and its Enzyme gradient against central differences. `skip`
# leading cells belong to other families, held at their compiled frame (`c = 0`).
function check_hsgp_cells(rk, cells; skip=0, seeds=(41, 42))
    problem = rk_logdensity_problem(rk; ad_backend=AD)
    n, d = length(cells), length(layout_names(rk))
    for seed in seeds
        controls = rand(Xoshiro(seed), n)
        adaptive = adaptive_centering_problem(rk, problem, AD;
            centeredness=vcat(zeros(skip), controls))
        rows = adaptive_centering_controls(adaptive, rk)
        @test length(rows) == skip + n
        @test all(r -> r.family === :hsgp, rows[skip+1:end])
        @test [r.index for r in rows[skip+1:end]] == first.(cells)
        x = 0.3 .* randn(Xoshiro(seed + 100), d)
        ljac, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
        oljac, oy = scalar_oracle(x, cells, controls)
        @test y ≈ oy rtol=1e-12 atol=1e-12
        @test ljac ≈ oljac rtol=1e-12 atol=1e-12
        wrapped(v) = (r = scalar_oracle(v, cells, controls);
                      r[1] + LogDensityProblems.logdensity(problem, r[2]))
        value, gradient = LogDensityProblems.logdensity_and_gradient(adaptive, x)
        @test value ≈ wrapped(x) rtol=1e-10
        fd = map(1:d) do j
            p, m = copy(x), copy(x)
            p[j] += 1e-6; m[j] -= 1e-6
            (wrapped(p) - wrapped(m)) / 2e-6
        end
        @test gradient ≈ fd rtol=1e-5 atol=1e-5
    end
end

@stestset "HSGP blocks read RK's own spectral scales, every covered shape" begin
    data = hsgp_data()
    shapes = [
        @brm(data, begin mu ~ 1 + hsgp(x; k=6); y ~ Normal(mu, 0.2) end),
        @brm(data, begin mu ~ 1 + hsgp(x, x2; k=(3, 4)); y ~ Normal(mu, 0.2) end),
        @brm(data, begin
            mu ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false)
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin mu ~ 1 + hsgp(x; k=5, domain=(-4.0, 4.0)); y ~ Normal(mu, 0.2) end),
        @brm(data, begin
            mu ~ 1 + hsgp(t; k=3, cov=:periodic, period=24.0)
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin mu ~ 1 + hsgp(x; k=4, by=g); y ~ Normal(mu, 0.2) end),
        @brm(data, begin
            mu ~ 1 + hsgp(x, x2; k=(3, 2), iso=false, by=g)
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x; k=4)
            length_scale(mu, hsgp(x)) = 0.8
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x; k=4, by=g)
            sd(mu, hsgp(x)) = 0.7
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x, x2; k=(3, 2), iso=false)
            length_scale(mu, hsgp(x, x2)) = [0.8, 1.2]
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x; k=4)
            length_scale(mu, hsgp(x)) ~ Uniform(0.5, 3.0)
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin mu ~ 1 + hsgp(x; k=4) + hsgp(x2; k=3); y ~ Normal(mu, 0.2) end),
    ]
    for brmi in shapes
        rk = RKBRMI(brmi)
        names = layout_names(rk)
        blocks = BRM._rk_adaptive_hsgp_blocks(rk)
        cells = spectral_cells(rk)
        @test sum(b -> length(b.effects), blocks) == length(cells)
        @test reduce(vcat, [b.effects for b in blocks]) == first.(cells)
        @test all(b -> all(iszero, b.target_c), blocks)
        for seed in (1, 2)
            x = 0.7 .* randn(Xoshiro(seed), length(names))
            mine = [BRM._adaptive_hsgp_log_scale(x, b, k) for b in blocks
                    for k in eachindex(b.effects)]
            @test mine ≈ [log_s(x) for (_, log_s) in cells] rtol=1e-12
        end
    end
end

@stestset "HSGP terms whose levels or hyperparameters are mapped elsewhere pass through" begin
    data = hsgp_data()
    passing = [
        # Correlated levels and per-level scales: SBBRMI maps those across
        # their levels; on RKBRMI they keep the emitted frame.
        @brm(data, begin
            Omega ~ LKJCholesky(3, 2.0)
            mu ~ 1 + hsgp(x; k=4, by=g, cor=Omega)
            y ~ Normal(mu, 0.2)
        end),
        @brm(data, begin
            mu ~ 1 + hsgp(x; k=4, by=g)
            log(sd(hsgp(x))) ~ 1 + (1 | g)
            y ~ Normal(mu, 0.2)
        end),
        # A scale that is a model value has no bounds the cell could read.
        @brm(data, begin
            a ~ Normal(0, 1)
            mu ~ 1 + hsgp(x; k=4)
            sd(mu, hsgp(x)) = exp(a)
            y ~ Normal(mu, 0.2)
        end),
    ]
    for brmi in passing
        rk = RKBRMI(brmi)
        @test isempty(BRM._rk_adaptive_hsgp_blocks(rk))
    end
    # With nothing else to adapt, the wrapper says so.
    rk = RKBRMI(first(passing))
    @test_throws r"no random-effect block or HSGP" adaptive_centering_problem(rk,
        rk_logdensity_problem(rk; ad_backend=AD), AD)
end

@stestset "HSGP cells alone: the map, Jacobian and gradient against RK's spectral scales" begin
    data = hsgp_data()
    for brmi in (
            @brm(data, begin
                mu ~ 1 + hsgp(x, x2; k=(3, 2), c=(1.5, 2.0), iso=false)
                y ~ Normal(mu, 0.2)
            end),
            @brm(data, begin mu ~ 1 + hsgp(x; k=3, by=g); y ~ Normal(mu, 0.2) end),
            @brm(data, begin
                mu ~ 1 + hsgp(t; k=2, cov=:periodic, period=24.0)
                y ~ Normal(mu, 0.2)
            end))
        rk = RKBRMI(brmi)
        check_hsgp_cells(rk, spectral_cells(rk))
    end
end

@stestset "HSGP cells beside a random-effect block follow its cells" begin
    data = hsgp_data()
    rk = RKBRMI(@brm data begin
        mu ~ 1 + hsgp(x; k=4) + (1 | u)
        sigma ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end)
    # The six levels' cells come first; the joint wrapper maps the HSGP cells
    # after them exactly as alone, with the block held at its compiled frame.
    rows = adaptive_centering_controls(adaptive_centering_problem(rk,
        rk_logdensity_problem(rk; ad_backend=AD), AD), rk)
    @test [r.family for r in rows] == vcat(fill(:ranef, 6), fill(:hsgp, 4))
    check_hsgp_cells(rk, spectral_cells(rk); skip=6)
end
