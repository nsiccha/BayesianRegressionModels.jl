# test/grouped_hsgp_cost.jl — a grouped HSGP `hsgp(x...; by=g)` on SBBRMI
# costs one rows x basis pass per gradient, like an ungrouped field: the
# spectral weights scale the levels x basis weights (`<term>_beta_scaled`), and
# `brm_hsgp_by_level_rows` multiplies each level's basis rows, sorted by level
# once in transformed data, by its level's weights. No rows x basis value is a
# transformed parameter (snag grouped-correlat-68e4a63c: the hyper-predicted
# form stored a rows x basis `S`, 7x the gradient cost of ungrouped fields).
#
#     julia --project=test test/grouped_hsgp_cost.jl
#
# Set `BRM_GROUPED_HSGP_RUNTIME=0` to skip the compiled-model sets.

using Test
using BayesianRegressionModels
using StanBlocks
using LinearAlgebra
using Random

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const RUNTIME = get(ENV, "BRM_GROUPED_HSGP_RUNTIME", "1") != "0"
const CACHE = joinpath(tempdir(), "brm-grouped-hsgp-cost")

# Levels interleaved out of order, so the level sort and the scatter back to
# the rows are exercised.
function level_df(; N=30, seed=5)
    rng = Random.Xoshiro(seed)
    (; t=collect(range(0.0, 10.0; length=N)), u=collect(range(-1.0, 1.0; length=N)),
       g=[(7i + i ÷ 5) % 3 + 1 for i in 1:N], y=randn(rng, N))
end

const MODELS = (
    shared = (@brm begin
        mu ~ 1 + hsgp(t; k=5, by=g)
        y ~ Normal(mu, 1.0)
    end),
    aniso = (@brm begin
        mu ~ 0 + hsgp(t, u; k=3, by=g, iso=false)
        y ~ Normal(mu, 1.0)
    end),
    icm = (@brm begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 0 + hsgp(t, u; k=3, by=g, cor=Omega, orthogonal_to=:intercept)
        y ~ Normal(mu, 1.0)
    end),
    multigroup = (@brm begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(Omega, 0.7))
        y ~ Normal(mu, 1.0)
    end),
    # The reporter's shape: per-level amplitudes of correlated levels.
    icm_amplitude = (@brm begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 0 + hsgp(t, u; k=3, by=g, cor=Omega, orthogonal_to=:intercept)
        log(sd(hsgp(t, u))) ~ 1 + (1 | g)
        y ~ Normal(mu, 1.0)
    end),
    multigroup_amplitude = (@brm begin
        Omega ~ LKJCholesky(3, 2.0)
        mu ~ 1 + hsgp(t; k=5, by=g, cor=multigroup(Omega, 0.7))
        log(sd(hsgp(t))) ~ 1 + (1 | g)
        y ~ Normal(mu, 1.0)
    end),
    amplitude_length_scale = (@brm begin
        mu ~ 1 + hsgp(t; k=5, by=g)
        log(sd(hsgp(t))) ~ 1 + (1 | g)
        log(length_scale(hsgp(t))) ~ 1 + (1 | g)
        y ~ Normal(mu, 1.0)
    end),
)

term_of(name) = name in (:aniso, :icm, :icm_amplitude) ? "hsgp_t_u_by_g" : "hsgp_t_by_g"
hyper(name) = name in (:icm_amplitude, :multigroup_amplitude, :amplitude_length_scale)

@testset "the grouped field is level blocks, not a rows x basis value" begin
    df = level_df()
    for (name, model) in pairs(MODELS)
        sb = SBBRMI(model(df); mod=@__MODULE__)
        code = BRM.stan_code(sb)
        term = term_of(name)
        PHI = sb.data[Symbol(:PHI_, term)]
        @testset "$name" begin
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
            @test occursin("array[g_idx_n] int $(term)_level_order = sort_indices_asc(g_idx);", code)
            @test occursin("$(term) = brm_hsgp_by_level_rows(", code)
            @test !occursin("PHI_scaled", code)
            @test !occursin("brm_hsgp_by_hyper_S", code)
            # Every output a draw carries is far smaller than rows x basis.
            d = StanBlocks.stan_descriptor(sb.model)
            @test all(o -> prod(o.dims; init=1) < length(PHI), d.outputs)
            # The scaled weights are levels x basis.
            scaled = only(o for o in d.outputs if o.name === Symbol(term, :_beta_scaled))
            @test scaled.dims == (3, size(PHI, 2))
            @test hyper(name) == occursin("brm_hsgp_by_hyper_sqrt_spd(", code)
        end
    end
end

compiled(sb, tag) = (mkpath(CACHE);
    BRM.stan_instantiate(sb; path=joinpath(CACHE,
        tag * "_" * string(hash(BRM.stan_code(sb))) * ".stan")))

# The constrained value of output `name` (first index fastest, as Stan names it).
function output(model, constrained, name, dims)
    names = BS.param_names(model; include_tp=true, include_gq=true)
    at = findall(n -> n == name || startswith(n, name * "."), names)
    @assert length(at) == prod(dims; init=1) "$name: $(length(at)) entries, dims $dims"
    reshape(constrained[at], dims)
end

# The field each row should carry: its basis row times its level's scaled
# weights, and those weights the level weights times the level's spectral
# density (one shared one, or the level's own (rho, sigma) for a hyper term).
function check_field(sb, problem, name, seed)
    term = term_of(name)
    PHI = sb.data[Symbol(:PHI_, term)]
    omega2 = sb.data[Symbol(:omega2_, term)]
    g = sb.data[:g_idx]
    G, m = 3, size(PHI, 2)
    model = problem.model
    q = 0.3 .* randn(Random.Xoshiro(seed), BS.param_unc_num(model))
    lp, grad = BS.log_density_gradient(model, q)
    @test isfinite(lp) && all(isfinite, grad)
    constrained = BS.param_constrain(model, q; include_tp=true, include_gq=true,
        rng=BS.StanRNG(model, seed))
    W = output(model, constrained, term * "_beta_scaled", (G, m))
    field = output(model, constrained, term, (length(g),))
    @test field ≈ [dot(PHI[i, :], W[g[i], :]) for i in eachindex(g)] atol=1e-12
    level_weights = name in (:shared, :aniso, :amplitude_length_scale) ?
        output(model, constrained, "b_hsgpw_" * replace(term, "hsgp_" => "", "_by_g" => "_g"), (G, m)) :
        output(model, constrained, term * "_beta_level", (G, m))
    if hyper(name)
        rho_lower = sb.data[Symbol(:rho_lower_, term)]
        rho = output(model, constrained, term * "_rho_vec", (G,))
        sigma = output(model, constrained, term * "_sigma_vec", (G,))
        spd = permutedims(hcat([brm_hsgp_sqrt_spd(omega2, sigma[l],
            fill(max(rho[l], rho_lower), size(omega2, 2))) for l in 1:G]...))
        @test W ≈ level_weights .* spd atol=1e-12
    else
        spd = output(model, constrained, term * "_sqrt_spd", (m,))
        @test W ≈ level_weights .* spd' atol=1e-12
    end
    lp
end

if RUNTIME
    @testset "each row carries its basis row times its level's scaled weights" begin
        df = level_df()
        for (name, model) in pairs(MODELS)
            @testset "$name" begin
                sb = SBBRMI(model(df); mod=@__MODULE__)
                check_field(sb, compiled(sb, string(name)), name, 11)
            end
        end
    end

    @testset "a replay whose rows miss a level, or hold one row" begin
        df = level_df()
        for (name, model) in pairs(MODELS), rows in ([5, 2, 12, 8, 15], [1])
            # level 2 is absent from the first set; the second is one row
            @testset "$name $rows" begin
                sb = SBBRMI(model(df); mod=@__MODULE__)
                replay = reprocess(sb, map(column -> column[rows], df))
                @test sort(unique(replay.data[:g_idx])) != [1, 2, 3]
                check_field(replay, compiled(replay, string(name, "_replay")), name, 12)
            end
        end
    end
else
    @info "Skipping compiled grouped-HSGP sets (BRM_GROUPED_HSGP_RUNTIME=0)"
end
