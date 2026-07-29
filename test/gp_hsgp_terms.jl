using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LogDensityProblems

const GP_TERM_N = 8
const GP_RUN_BRIDGESTAN = get(ENV, "BRM_GP_RUNTIME", "1") != "0"
const GP_TERM_CACHE = joinpath(tempdir(), "brm-gp-hsgp")

function gp_term_df(; shift_x=0.0, shift_x2=0.0)
    x = collect(range(-1.0, 1.0; length=GP_TERM_N)) .+ shift_x
    x2 = (x .- shift_x) .^ 2 .+ 0.1 .* (x .- shift_x) .+ shift_x2
    y = sin.(x)
    g = repeat(["a", "b"], inner=GP_TERM_N ÷ 2)
    (; x, x2, y, g)
end

exact_gp_model(df) = @brm df begin
    loc ~ 1 + gp(x, x2; iso=false)
    y ~ Normal(loc, 1)
end

hsgp_model(df) = @brm df begin
    loc ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false)
    y ~ Normal(loc, 1)
end

grouped_hsgp_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    y ~ Normal(loc, 1)
end

bad_exact_k(df) = @brm df begin
    loc ~ gp(x; k=10)
    y ~ Normal(loc, 1)
end

bad_exact_by(df) = @brm df begin
    loc ~ gp(x; by=g)
    y ~ Normal(loc, 1)
end

bad_hsgp_k_arity(df) = @brm df begin
    loc ~ hsgp(x, x2; k=(3,))
    y ~ Normal(loc, 1)
end

@testset "exact gp and Hilbert-space hsgp are separate terms" begin
    df = gp_term_df()
    exact = exact_gp_model(df)
    approx = hsgp_model(df)
    grouped = grouped_hsgp_model(df)

    @testset "public contract and validation" begin
        @test popcoefnames(exact, :loc) == [:Intercept]
        @test popcoefnames(approx, :loc) == [:Intercept]
        @test grouping_factors(grouped, :loc) == [:g]
        @test_throws "Use `hsgp" bad_exact_k(df)
        @test_throws "Use `hsgp" bad_exact_by(df)
        @test_throws "one value per axis" SBBRMI(bad_hsgp_k_arity(df); mod=@__MODULE__)
    end

    sb_exact = SBBRMI(exact; mod=@__MODULE__)
    sb_hsgp = SBBRMI(approx; mod=@__MODULE__)
    sb_grouped = SBBRMI(grouped; mod=@__MODULE__)

    @testset "data shapes and replay" begin
        @test size(sb_exact.data[:X_gp_x_x2]) == (GP_TERM_N, 2)
        @test size(sb_hsgp.data[:PHI_hsgp_x_x2]) == (GP_TERM_N, 12)
        @test size(sb_hsgp.data[:omega2_hsgp_x_x2]) == (12, 2)
        @test !haskey(sb_grouped.data, :g)
        @test sb_grouped.data[:g_idx] == [1, 1, 1, 1, 2, 2, 2, 2]

        exact_input = only(i for i in brm_descriptor(sb_exact).inputs
                           if i.transform === :gp)
        hsgp_input = only(i for i in brm_descriptor(sb_hsgp).inputs
                          if i.transform === :hsgp)
        @test exact_input.name === :X_gp_x_x2
        @test hsgp_input.name === :PHI_hsgp_x_x2
        @test exact_input.column === nothing
        @test hsgp_input.column === nothing

        replay_df = gp_term_df(; shift_x=0.25, shift_x2=0.4)
        replay_exact = reprocess(sb_exact, replay_df)
        replay_hsgp = reprocess(sb_hsgp, replay_df)
        @test replay_exact.data[:X_gp_x_x2] != sb_exact.data[:X_gp_x_x2]
        @test replay_hsgp.data[:PHI_hsgp_x_x2] != sb_hsgp.data[:PHI_hsgp_x_x2]
        @test StanBlocks.stan_code(replay_exact.model) == StanBlocks.stan_code(sb_exact.model)
        @test StanBlocks.stan_code(replay_hsgp.model) == StanBlocks.stan_code(sb_hsgp.model)
    end

    @testset "transpile and stanc" begin
        for (label, sb) in ((:exact, sb_exact), (:hsgp, sb_hsgp), (:grouped_hsgp, sb_grouped))
            @test StanBlocks.stan.transpiles(sb.model)
            code = StanBlocks.stan_code(sb.model)
            @test occursin(label === :exact ? "brm_exp_quad_cov" : "brm_hsgp_sqrt_spd", code)
            if label === :exact
                @test occursin("gp_exp_quad_cov", code)
                @test occursin("add_diag", code)
                @test occursin("brm_gp_locations", code)
                @test !occursin("sqdist", code)
                @test !occursin("K_axis", code)
            end
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        end
    end

    @testset "BridgeStan finite density and gradient" begin
        if GP_RUN_BRIDGESTAN
            isdir(GP_TERM_CACHE) || mkpath(GP_TERM_CACHE)
            for (label, sb) in ((:exact, sb_exact), (:hsgp, sb_hsgp),
                                (:grouped_hsgp, sb_grouped))
                code = StanBlocks.stan_code(sb.model)
                problem = StanBlocks.stan_instantiate(
                    sb.model; path=joinpath(GP_TERM_CACHE, string(hash(code)) * ".stan"))
                dimension = LogDensityProblems.dimension(problem)
                q = [0.05 * ((i % 5) - 2) for i in 1:dimension]
                lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
                @test isfinite(lp)
                @test length(gradient) == dimension
                @test all(isfinite, gradient)
            end
        else
            @info "Skipping BridgeStan GP/HSGP runtime gate (BRM_GP_RUNTIME=0)"
        end
    end
end
