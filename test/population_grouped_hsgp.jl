# test/population_grouped_hsgp.jl — the population and transport routes on a
# grouped HSGP, `hsgp(t; by = g)` (snag `brm-use-silent-o-0039d181`, decision
# `0wlazld`).
#
# A grouped HSGP draws its standardized basis weights as a plain
# `zflat_<suffix> ~ std_normal()` vector and a per-level hyper-predictor its
# level effects inside the term's submodel, so `ranef_blocks` does not list
# them. They are still the grouping's effects:
#
# - `population_draws(…; groups = :g)` zeroes them, so the term is exactly zero
#   on every row (before, it kept every group's fitted curve, and refused a
#   grouping with no other block);
# - `population_model(sb; groups = :g)` emits them multiplied by zero, with
#   the fitted coordinates, and agrees with `population_draws`;
# - `transport_draws` copies them by level label, and refuses a level the
#   source lacks with the `resample_groups` route (before, it copied them by
#   position, so a dropped level shifted later levels onto earlier weights).
#
# Run: julia --project=test test/population_grouped_hsgp.jl
# (`BRM_PRED_RUNTIME=0` skips the compiled-model sets.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels
const PRED_RUNTIME = get(ENV, "BRM_PRED_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

# Four subjects, each over its own stretch of the time axis.
function frame(subjects; seed = 1)
    rng = MersenneTwister(seed)
    ranges = Dict(1 => 40.0, 2 => 60.0, 3 => 80.0, 4 => 100.0, 5 => 50.0)
    t = Float64[]; g = Int[]
    for s in subjects
        ts = collect(range(0.0, ranges[s]; length = 12))
        append!(t, ts); append!(g, fill(s, length(ts)))
    end
    y = 0.01 .* t .+ 0.3 .* sin.(t ./ 15) .+ 0.1 .* randn(rng, length(t))
    (; t, subject = g, ecg_subject = g, y)
end

# A shared trend, per-subject deviations with a per-subject amplitude, and an
# ordinary subject intercept on the SAME grouping column.
shared = @brm begin
    base ~ 1 + (1 | subject)
    trend ~ 0 + hsgp(t; k = 8, c = 1.5, extrapolate = :clamp, anchor = 0.0)
    dev ~ 0 + hsgp(t; by = subject, k = 8, c = 1.5, extrapolate = :clamp, anchor = 0.0)
    log(sd(dev, hsgp(t))) ~ 1 + (1 | subject)
    m = base + trend + dev
    sigma ~ Exponential(1)
    y ~ Normal(m, sigma)
end

# The deviations grouped by a column no other block reads.
own_column = @brm begin
    base ~ 1
    dev ~ 0 + hsgp(t; by = ecg_subject, k = 8, c = 1.5)
    m = base + dev
    sigma ~ Exponential(1)
    y ~ Normal(m, sigma)
end

df = frame(1:4)
sb = SBBRMI(shared(df); mod = @__MODULE__)
sb_own = SBBRMI(own_column(df); mod = @__MODULE__)

@testset "the reader lists the grouped HSGP that ranef_blocks does not" begin
    @test [b.binding for b in ranef_blocks(sb)] == [:r_base_subject]
    blocks = BRM._grouped_hsgp_blocks(sb)
    @test length(blocks) == 1
    b = only(blocks)
    @test (b.block, b.weights, b.term, b.group) ==
        (:b_hsgpw_t_subject, :zflat_hsgpw_t_subject, :hsgp_t_by_subject, :subject)
    @test b.levels == [1, 2, 3, 4]
    @test (b.n_basis, b.per_level, b.generated) == (8, true, false)
    @test [(h.hyper, h.z, h.generated) for h in b.hypers] ==
        [(:sd, :hsgp_t_by_subject_z_sigma, false)]
    @test isempty(ranef_blocks(sb_own))
    @test only(BRM._grouped_hsgp_blocks(sb_own)).group === :ecg_subject
    @test isempty(only(BRM._grouped_hsgp_blocks(sb_own)).hypers)
end

@testset "an unqualified hyper-predictor names the qualified fix" begin
    two = @brm begin
        trend ~ 0 + hsgp(t; k = 8, c = 1.5)
        dev ~ 0 + hsgp(t; by = subject, k = 8, c = 1.5)
        log(sd(hsgp(t))) ~ 1 + (1 | subject)
        m = trend + dev
        sigma ~ Exponential(1)
        y ~ Normal(m, sigma)
    end
    err = try
        SBBRMI(two(df); mod = @__MODULE__); nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("`log(sd(:, hsgp(t))) ~ …` is ambiguous", err.msg)
    @test occursin("`log(sd(trend, hsgp(t))) ~ …` or `log(sd(dev, hsgp(t))) ~ …`", err.msg)
end

if PRED_RUNTIME
    sm = stan_instantiate(sb).model
    unc = BS.param_unc_names(sm)
    draws = randn(MersenneTwister(2), 3, length(unc))
    names_tp = BS.param_names(sm; include_tp = true)
    tp(model, draw) = BS.param_constrain(model, draw; include_tp = true)
    at(prefix) = findall(nm -> startswith(nm, prefix), names_tp)
    zeroed(out) = Set(unc[j] for j in eachindex(unc) if out[:, j] != draws[:, j])

    @testset "population_draws zeroes the grouped HSGP with the subject blocks" begin
        pop = population_draws(sb, draws, unc; groups = :subject)
        expected = Set(nm for nm in unc if startswith(nm, "r_base_subject_xi.") ||
            startswith(nm, "zflat_hsgpw_t_subject.") ||
            startswith(nm, "hsgp_t_by_subject_z_sigma."))
        @test length(expected) == 4 + 32 + 4
        @test zeroed(pop) == expected
        @test all(iszero, pop[:, [findfirst(==(nm), unc) for nm in expected]])
        for i in axes(draws, 1)
            fitted, population = tp(sm, draws[i, :]), tp(sm, pop[i, :])
            @test maximum(abs, fitted[at("dev.")]) > 0.0
            @test all(iszero, population[at("dev.")])           # zero on every row
            @test allequal(population[at("base.")])             # every subject at the mean
            @test population[at("trend.")] == fitted[at("trend.")]
        end
    end

    @testset "population_model: the same program at the fitted coordinates" begin
        pm = population_model(sb; groups = :subject)
        pmm = stan_instantiate(pm).model
        @test BS.param_unc_names(pmm) == unc
        @test BS.param_names(pmm; include_tp = true) == names_tp
        pop = population_draws(sb, draws, unc; groups = :subject)
        # The parameters themselves differ by construction (the population
        # program keeps the fitted coordinates, `population_draws` zeroes
        # them); everything the program computes from them agrees.
        computed = (length(BS.param_names(sm)) + 1):length(names_tp)
        @test !isempty(computed)
        for i in axes(draws, 1)
            on_model = tp(pmm, draws[i, :])
            @test all(iszero, on_model[at("dev.")])
            @test on_model[computed] ≈ tp(sm, pop[i, :])[computed]
        end
        # Applying it again, and replaying it, keep the program.
        code = BRM.stan_code(pm)
        @test code != BRM.stan_code(sb)
        @test BRM.stan_code(population_model(pm; groups = :subject)) == code
        regrid = merge(df, (; t = df.t .* 0.5))
        @test BRM.stan_code(reprocess(pm, regrid)) == code
        @test BRM.stan_code(reprocess(pm, regrid)) ==
            BRM.stan_code(population_model(reprocess(sb, regrid); groups = :subject))
        # refused: a typo'd factor must not silently return the fitted program.
        @test_throws "no random-effect block on grouping factor `subjct`" population_model(
            sb; groups = :subjct)
    end

    @testset "a grouping whose only effects are a grouped HSGP" begin
        sm_own = stan_instantiate(sb_own).model
        unc_own = BS.param_unc_names(sm_own)
        d_own = randn(MersenneTwister(3), 2, length(unc_own))
        pop = population_draws(sb_own, d_own, unc_own; groups = :ecg_subject)
        @test Set(unc_own[j] for j in eachindex(unc_own) if pop[:, j] != d_own[:, j]) ==
            Set(nm for nm in unc_own if startswith(nm, "zflat_hsgpw_t_ecg_subject."))
        names_own = BS.param_names(sm_own; include_tp = true)
        dev = findall(nm -> startswith(nm, "dev."), names_own)
        @test all(iszero, BS.param_constrain(sm_own, pop[1, :]; include_tp = true)[dev])
        pm = stan_instantiate(population_model(sb_own; groups = :ecg_subject)).model
        @test BS.param_unc_names(pm) == unc_own
        @test all(iszero, BS.param_constrain(pm, d_own[1, :]; include_tp = true)[dev])
        # The other grouping column names no block of this model.
        @test_throws "no random-effect block on grouping factor `subject`" population_draws(
            sb_own, d_own, unc_own; groups = :subject)
    end

    @testset "transport_draws copies the weights by level label" begin
        to = SBBRMI(shared(frame([1, 3, 4])); mod = @__MODULE__)
        unc_to = BS.param_unc_names(stan_instantiate(to).model)
        moved = transport_draws(sb, to, draws, unc, unc_to)
        pf = Dict(nm => j for (j, nm) in enumerate(unc))
        pt = Dict(nm => j for (j, nm) in enumerate(unc_to))
        for (g, label) in enumerate([1, 3, 4]), j in 1:8
            @test moved[:, pt["zflat_hsgpw_t_subject.$((g - 1) * 8 + j)"]] ==
                draws[:, pf["zflat_hsgpw_t_subject.$((label - 1) * 8 + j)"]]
        end
        for (g, label) in enumerate([1, 3, 4])
            @test moved[:, pt["hsgp_t_by_subject_z_sigma.$g"]] ==
                draws[:, pf["hsgp_t_by_subject_z_sigma.$label"]]
            @test moved[:, pt["r_base_subject_xi.$g"]] == draws[:, pf["r_base_subject_xi.$label"]]
        end
        @test moved[:, pt["hsgp_t_sigma"]] == draws[:, pf["hsgp_t_sigma"]]
        # refused: a level the fit never saw needs fresh weights, which come
        # from StanBlocks' generated-quantity re-draw, not from Julia-side
        # randomness (the non-centered route rule of `transport_draws`). The
        # model without an ordinary block reaches the grouped-HSGP rule itself.
        unc_own = BS.param_unc_names(stan_instantiate(sb_own).model)
        d_own = randn(MersenneTwister(4), 2, length(unc_own))
        fresh = SBBRMI(own_column(frame([1, 5])); mod = @__MODULE__)
        unc_fresh = BS.param_unc_names(stan_instantiate(fresh).model)
        @test_throws "grouped HSGP `b_hsgpw_t_ecg_subject` (group `ecg_subject`) needs fresh per-level draws for level(s) 5" transport_draws(
            sb_own, fresh, d_own, unc_own, unc_fresh)
        @test_throws "every level (`resample` names `ecg_subject`)" transport_draws(
            sb_own, sb_own, d_own, unc_own, unc_own; resample = :ecg_subject)
        # The same levels in the same order copy straight across.
        @test transport_draws(sb_own, sb_own, d_own, unc_own, unc_own) == d_own
    end
end
