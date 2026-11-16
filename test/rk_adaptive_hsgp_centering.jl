# Adaptive partial centring of RKBRMI HSGP weights (todo 0nsdowt, increments 5
# and 6): `adaptive_centering_problem(rk, ...)` adapts each basis weight's cell
# at the spectral scale RK's own program computes, `<decl>.spectral_scale`, and
# maps a grouped term's correlated or per-level-scaled weights across its levels.
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

# A known level correlation for `cor = C`.
level_data(; kw...) = merge(hsgp_data(; kw...), (; C=[1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]))

# Grouped terms whose levels the wrapper maps across (`_rk_adaptive_hsgp_level_blocks`).
level_shapes(data) = [
    :cor_lkj => @brm(data, begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(x; k=4, by=g, cor=Omega)
        y ~ Normal(mu, 0.2)
    end),
    :cor_known => @brm(data, begin mu ~ 1 + hsgp(x; k=4, by=g, cor=C); y ~ Normal(mu, 0.2) end),
    :hyper_sd => @brm(data, begin
        mu ~ 1 + hsgp(x; k=4, by=g)
        log(sd(hsgp(x))) ~ 1 + (1 | g)
        y ~ Normal(mu, 0.2)
    end),
    :hyper_rho => @brm(data, begin
        mu ~ 1 + hsgp(x; k=4, by=g)
        log(length_scale(hsgp(x))) ~ 1 + (1 | g)
        y ~ Normal(mu, 0.2)
    end),
    :cor_hyper => @brm(data, begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(x; k=4, by=g, cor=Omega)
        log(sd(hsgp(x))) ~ 1 + (1 | g)
        y ~ Normal(mu, 0.2)
    end),
    :cor_aniso => @brm(data, begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(x, x2; k=(3, 2), iso=false, by=g, cor=Omega)
        y ~ Normal(mu, 0.2)
    end),
]

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

@stestset "HSGP terms whose levels are mapped, or whose scale is a model value, are not basis cells" begin
    data = level_data()
    # Correlated levels and per-level scales are level cells (below); a scale
    # that is a model value has no bounds a cell could read and passes through.
    for (label, brmi) in level_shapes(data)
        rk = RKBRMI(brmi)
        @test isempty(BRM._rk_adaptive_hsgp_blocks(rk))
        @test length(BRM._rk_adaptive_hsgp_level_blocks(rk)) == 1
    end
    rk = RKBRMI(@brm(data, begin
        a ~ Normal(0, 1)
        mu ~ 1 + hsgp(x; k=4)
        sd(mu, hsgp(x)) = exp(a)
        y ~ Normal(mu, 0.2)
    end))
    @test isempty(BRM._rk_adaptive_hsgp_blocks(rk))
    @test isempty(BRM._rk_adaptive_hsgp_level_blocks(rk))
    # With nothing else to adapt, the wrapper says so.
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

# Stan's `cholesky_factor_corr[K]` from its free values (row-major CPCs), the
# test's own reading of a sampled level correlation.
function cpc_cholesky(w, K)
    L = zeros(eltype(w), K, K)
    m = 0
    for i in 1:K
        rest = one(eltype(w))
        for j in 1:i-1
            m += 1
            r = tanh(w[m])
            L[i, j] = r * sqrt(rest)
            rest -= L[i, j]^2
        end
        L[i, i] = sqrt(rest)
    end
    L
end

# A level block's correlation factor from the test's own reading: the CPCs of
# the sampled `Omega`, the model's known factor, or the identity.
function level_factor(x, block, known)
    G = size(block.effects, 1)
    block.correlation === :parameter && return cpc_cholesky(x[block.cholesky_free], G)
    block.correlation === :data && return Matrix{Float64}(cholesky(Symmetric(known)).L)
    Matrix{Float64}(I, G, G)
end

# RK's per-level spectral scales `S[g, b]` of a level-mapped term at `x` (one
# vector over the bases when every level shares its hyperparameters).
function level_spectral(read, record, x, G, B)
    S = read(x)[Symbol(record.declaration, ".", record.spectral)]
    S isa AbstractVector && length(S) == B && return repeat(permutedims(S), G)
    size(S) == (G, B) && return S
    size(S) == (B, G) && return permutedims(S)
    error("spectral scale of size $(size(S)) for $G levels × $B bases")
end

# The independent level map: per term, per basis `b`, per level `g` in level
# order, `z_g = (u_g - c m_g) / s^c` with `C = diag(S[:, b]) * L`,
# `m_g = sum_{j<g} C[g, j] z_j` and `s = C[g, g]`; `ljac = -sum c log s`. `S` is
# RK's own spectral scale and `L` the test's factor.
function level_oracle(x, blocks, records, read, controls, known; skip=0)
    y, ljac = copy(x), zero(eltype(x))
    p = skip
    for (block, record) in zip(blocks, records)
        G, B = size(block.effects)
        S = level_spectral(read, record, x, G, B)
        L = level_factor(x, block, known)
        for b in 1:B
            C = Diagonal(S[:, b]) * L
            z = zeros(eltype(x), G)
            for g in 1:G
                p += 1
                c = controls[p]
                m = sum((C[g, j] * z[j] for j in 1:g-1); init=zero(eltype(x)))
                s = C[g, g]
                z[g] = (x[block.effects[g, b]] - c * m) / s^c
                y[block.effects[g, b]] = z[g]
                ljac -= c * log(s)
            end
        end
    end
    ljac, y
end

@stestset "HSGP level blocks read RK's own per-level spectral scales and level weights" begin
    data = level_data()
    for (label, brmi) in level_shapes(data)
        rk = RKBRMI(brmi)
        names = layout_names(rk)
        record = only(r for r in hsgp_records(rk) if r.grouped)
        block = only(BRM._rk_adaptive_hsgp_level_blocks(rk))
        G, B = size(block.effects)
        @test G == 3
        @test block.effects == [findfirst(==("$(record.declaration).z.$g.$b"), names)
                                for g in 1:G, b in 1:B]
        @test block.correlation === (label in (:cor_lkj, :cor_hyper, :cor_aniso) ? :parameter :
                                     label === :cor_known ? :data : :none)
        @test block.sigma.predicted == (label in (:hyper_sd, :cor_hyper))
        @test block.rho.predicted == (label === :hyper_rho)
        ports = (Symbol(record.declaration, ".", record.spectral),
                 Symbol(record.declaration, ".z_level"))
        read = BRM._rk_value_reader(rk, block.correlation === :none ? ports[1:1] : ports)
        for seed in (1, 2, 3)
            x = 0.7 .* randn(Xoshiro(seed), length(names))
            # Seed 3 drives a predicted length scale below its floor.
            seed == 3 && block.rho.predicted && (x[block.rho.intercept] = -6.0)
            S = level_spectral(read, record, x, G, B)
            mine = [BRM._adaptive_hsgp_level_log_scale(x, block, g, b) for g in 1:G, b in 1:B]
            @test mine ≈ log.(S) rtol=1e-12
            if block.correlation !== :none
                Z = [x[block.effects[g, b]] for g in 1:G, b in 1:B]
                L = BRM._adaptive_hsgp_level_correlation_factor(x, block)
                zl = read(x)[ports[2]]
                @test L * Z ≈ (size(zl) == (G, B) ? zl : permutedims(zl)) rtol=1e-12
                @test L ≈ level_factor(x, block, data.C) rtol=1e-12
            end
        end
    end
end

@stestset "HSGP level cells: the map, Jacobian and gradient against RK's own scales" begin
    data = level_data()
    for (label, brmi) in level_shapes(data)
        rk = RKBRMI(brmi)
        names = layout_names(rk)
        records = [r for r in hsgp_records(rk) if r.grouped]
        blocks = BRM._rk_adaptive_hsgp_level_blocks(rk)
        n = sum(b -> length(b.effects), blocks)
        read = BRM._rk_value_reader(rk,
            Tuple(Symbol(r.declaration, ".", r.spectral) for r in records))
        problem = rk_logdensity_problem(rk; ad_backend=AD)
        for seed in (51, 52)
            controls = rand(Xoshiro(seed), n)
            adaptive = adaptive_centering_problem(rk, problem, AD; centeredness=controls)
            rows = adaptive_centering_controls(adaptive, rk)
            @test length(rows) == n
            @test all(r -> r.family === :hsgp && r.level !== nothing, rows)
            # Pair order: term, basis, level.
            @test [r.index for r in rows] == vec(only(blocks).effects)
            x = 0.3 .* randn(Xoshiro(seed + 100), length(names))
            ljac, y = WarmupHMC.reparametrizer(adaptive)(copy(x))
            oljac, oy = level_oracle(x, blocks, records, read, controls, data.C)
            @test y ≈ oy rtol=1e-12 atol=1e-12
            @test ljac ≈ oljac rtol=1e-12 atol=1e-12
            wrapped(v) = (r = level_oracle(v, blocks, records, read, controls, data.C);
                          r[1] + LogDensityProblems.logdensity(problem, r[2]))
            value, gradient = LogDensityProblems.logdensity_and_gradient(adaptive, x)
            @test value ≈ wrapped(x) rtol=1e-10
            fd = map(eachindex(x)) do j
                p, m = copy(x), copy(x)
                p[j] += 1e-6; m[j] -= 1e-6
                (wrapped(p) - wrapped(m)) / 2e-6
            end
            @test gradient ≈ fd rtol=1e-5 atol=1e-5
        end
        # `hsgp = false` keeps the level weights in their emitted frame.
        @test_throws r"no random-effect block" adaptive_centering_problem(rk, problem, AD;
            hsgp=false)
    end
end

@stestset "HSGP level cells follow a random-effect block's and basis cells" begin
    data = level_data()
    rk = RKBRMI(@brm data begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(x; k=3, by=g, cor=Omega) + hsgp(x2; k=2) + (1 | u)
        sigma ~ Exponential(1.0)
        y ~ Normal(mu, sigma)
    end)
    rows = adaptive_centering_controls(adaptive_centering_problem(rk,
        rk_logdensity_problem(rk; ad_backend=AD), AD), rk)
    # Six random-effect levels, the ungrouped term's two weights, then the
    # grouped term's 3 levels × 3 bases.
    @test [r.family for r in rows] == vcat(fill(:ranef, 6), fill(:hsgp, 2 + 9))
    @test all(r -> r.level === nothing, rows[7:8])
    @test all(r -> r.level !== nothing, rows[9:end])
end

@stestset "a multigroup coherence raises under hsgp = true" begin
    data = hsgp_data()
    rk = RKBRMI(@brm(data, begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(x; k=4, by=g, cor=multigroup(Omega, 0.7))
        y ~ Normal(mu, 0.2)
    end))
    @test_throws r"cor = multigroup" BRM._rk_adaptive_hsgp_level_blocks(rk)
    @test isempty(BRM._rk_adaptive_hsgp_blocks(rk))
end
