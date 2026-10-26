# test/stratified_reprocess.jl — replaying a fit with a stratified
# `gr(g, by=b)` random-effect block onto new rows of its fitted groups (todo
# `08nab3q`), and drawing another grouping's new levels beside it (snag
# `forecast-replay-a44fea87`: atlas-16's stratified Gemeinde block beside the
# Kreis-years a forecast resamples).
#
# The stratified index records its levels, its strata and each level's stratum
# as a `:group_index` record, so `reprocess` codes the replay rows into the
# fitted levels and keeps every level in its fitted stratum. Drawing new levels
# of a stratified block (`resample_groups=[:g]`) stays refused.
#
# Run: julia --project=test test/stratified_reprocess.jl [filter...]
using Test, BayesianRegressionModels, StanBlocks
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

const GROUPS = repeat(1:8; inner=4)
const DF = (; g=GROUPS, b=[g <= 4 ? 1 : 2 for g in GROUPS], h=repeat(1:4; outer=8),
            x=round.(sin.(1:32); digits=3), y=round.(cos.(3 .* (1:32)); digits=3))
# Rows of fitted groups 3, 6 and 8, each in its fitted stratum.
const ROWS = (; g=[3, 3, 6, 8], b=[1, 1, 2, 2], h=[1, 2, 3, 4],
              x=[0.1, -0.2, 0.3, 0.0], y=[0.0, 0.1, -0.1, 0.2])

build(b) = SBBRMI(b; mod=@__MODULE__, total_groups=())
code(sb) = BRM.stan_code(sb)
stanc_ok(sb) = StanBlocks.stanc_check(code(sb); warn_pedantic=false).ok
stan_block(c, name) = (k = findfirst("\n" * name * " {", c);
    c[k[1]+1:findnext("\n}", c, k[1] + 1)[end]])

# The plain block: one scale vector and correlation factor per stratum.
plain = @brm begin
    mu ~ 1 + x + (1 + x | gr(g, by=b))
    y ~ Normal(mu, 1.0)
end
# One stratified block shared by two predictors.
shared = @brm begin
    mu ~ 1 + x + (1 | p | gr(g, by=b))
    log(sigma) ~ 1 + (1 | p | gr(g, by=b))
    y ~ Normal(mu, sigma)
end
# Beside an ordinary grouping `h`.
beside = @brm begin
    mu ~ 1 + x + (1 | gr(g, by=b)) + (1 | h)
    y ~ Normal(mu, 1.0)
end

@stestset "frozen replay onto the fitted groups" begin
    for builder in (plain, shared, beside)
        sb = build(builder(DF))
        replay = reprocess(sb, ROWS)
        @test code(replay) == code(sb)
        @test replay.data[:g__by__b_idx] == [3, 3, 6, 8]
        @test replay.data[:n_g__by__b] == 8
        @test replay.data[:g__by__b_stratum_idx] == sb.data[:g__by__b_stratum_idx] ==
              [1, 1, 1, 1, 2, 2, 2, 2]
        @test replay.data[:n_strata_g__by__b] == 2
        # onto the fit's own rows, the replay is the fit
        @test BRM.stan_data(reprocess(sb, DF)) == BRM.stan_data(sb)
        @test :reprocess in [o.name for o in brm_descriptor(sb).operations]
        # replayed again from the replay
        @test reprocess(replay, DF).data[:g__by__b_idx] == sb.data[:g__by__b_idx]
    end
end

@stestset "fresh replay refits the levels and strata" begin
    sb = build(plain(DF))
    fresh = reprocess(sb, ROWS; freeze_constants=false)
    @test code(fresh) == code(sb)
    @test fresh.data[:g__by__b_idx] == [1, 1, 2, 3]
    @test fresh.data[:n_g__by__b] == 3
    @test fresh.data[:g__by__b_stratum_idx] == [1, 2, 2]
end

@stestset "refusals" begin
    sb = build(plain(DF))
    @test_throws "unseen level `9`" reprocess(sb, merge(ROWS, (; g=[3, 3, 6, 9])))
    @test_throws "unseen level `3`" reprocess(sb, merge(ROWS, (; b=[1, 1, 3, 2])))
    # refused: a level's stratum is part of the fitted model (its scales are
    # the stratum's), so a row cannot move it to another stratum
    @test_throws "fit has it in stratum `1`" reprocess(sb, merge(ROWS, (; b=[2, 1, 2, 2])))
    # not built: new levels of a stratified block (the cv emission draws them
    # from one population, a stratified block has one per stratum)
    @test_throws "stratified `gr(g, by=b)`" reprocess(sb, ROWS; resample_groups=[:g])
    @test_throws "stratified `gr(g, by=b)`" reprocess(build(shared(DF)), ROWS;
                                                      resample_groups=[:g])
end

@stestset "new levels of another grouping beside a stratified block" begin
    sb = build(beside(DF))
    rows = (; g=[3, 6], b=[1, 2], h=[5, 6], x=[0.1, 0.2], y=[0.0, 0.1])
    replay = reprocess(sb, rows; resample_groups=[:h])
    c = code(replay)
    @test occursin("r_mu_h_xi", stan_block(c, "generated quantities"))
    @test replay.data[:g__by__b_idx] == [3, 6]
    @test BRM.stan_data(replay)[:h_idx] == [1, 2]
    @test stanc_ok(replay)
    # and a forecast: the same rows without the response
    forecast = reprocess(sb, (; g=[3, 6], b=[1, 2], h=[5, 6], x=[0.1, 0.2]);
                         resample_groups=[:h])
    @test forecast.held_out == Set([:y])
    @test !occursin("y_likelihood", code(forecast))
    @test stan_block(code(forecast), "parameters") == stan_block(c, "parameters")
    @test stanc_ok(forecast)
end
