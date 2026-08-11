# test/resample_groups.jl — first-class new-population replay.
#
# `resample_groups` is the RESAMPLE twin of frozen same-group `reprocess`:
# it re-emits the ordinary fitted BRMI with cv-contagious group sizing, derives
# the requested groups' levels from the new dataframe, marks those indices so
# their standardized effects move to generated quantities, and keeps every
# predictor preprocessing constant frozen by default.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Normal

resample_builder = @brm begin
    mu ~ 1 + zscale(zage) + (1 + zage | p | subject)
    y ~ Normal(mu, 1.0)
end

plain_resample_builder = @brm begin
    mu ~ 1 + zage + (1 | subject)
    y ~ Normal(mu, 1.0)
end

resample_train = (;
    subject = repeat([1, 2, 3], inner=2),
    zage = collect(range(-1.0, 1.0; length=6)),
    y = zeros(6),
)

resample_future = (;
    subject = repeat([101, 203, 307, 409], inner=2),
    zage = collect(range(10.0, 20.0; length=8)),
    y = zeros(8),
)

@testset "reprocess resample_groups — first-class new-population artifact" begin
    fitted = SBBRMI(resample_builder(resample_train); mod=@__MODULE__)

    replay = reprocess(fitted, resample_future; resample_groups=:subject)
    replay_data = StanBlocks.stan_data(replay.model)
    @test replay.data[:n_subject] == 4
    @test replay_data[:subject_idx] == repeat(1:4, inner=2)
    @test replay.preproc[:subject_idx].const_.levels == [101, 203, 307, 409]
    @test replay.preproc[:zscale_zage].const_ ==
          fitted.preproc[:zscale_zage].const_
    @test replay.data[:zscale_zage] != fitted.data[:zscale_zage]

    code = StanBlocks.stan_code(replay.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    params = code[first(findfirst("parameters {", code)):first(findfirst("model {", code))]
    gq = code[first(findfirst("generated quantities {", code)):end]
    @test !occursin("b_p_subject_z_flat", params)
    @test occursin("b_p_subject_z_flat = std_normal_vector_rng", gq)
    @test occursin("b_p_subject_L", params)
    @test occursin("b_p_subject_tau", params)

    # Exact source parity with the former hand-wired recipe: re-emit with
    # cv_groups, then mark the new group index by hand. The public path also
    # replaces the fresh transform fit with the training one (asserted above).
    manual = SBBRMI(resample_builder(resample_future);
                    mod=@__MODULE__, cv_groups=[:subject])
    manual_data = Dict{Symbol,Any}(manual.data)
    manual_data[:subject_idx] =
        StanBlocks.stan.maybecv(:subject_idx, manual_data[:subject_idx])
    manual_model = StanBlocks.SlicModel(
        manual.model.model, manual_data, manual.model.mod)
    @test StanBlocks.stan_code(manual_model) == code

    @test restan_data(fitted, resample_future;
                      resample_groups=[:subject]) == replay_data
    plan = reprocess(generative_plan(fitted), resample_future;
                     resample_groups=[:subject])
    @test plan.cv_groups == Set([:subject])
    @test StanBlocks.stan_data(plan.model) == replay_data
    descriptor = brm_descriptor(
        resample_builder, resample_train; mod=@__MODULE__)
    replay_descriptor = brm_execute(
        descriptor, :reprocess, resample_future;
        resample_groups=[:subject])
    @test StanBlocks.stan_code(replay_descriptor.plan.model) == code

    fresh = reprocess(fitted, resample_future; freeze_constants=false,
                      resample_groups=[:subject])
    @test fresh.preproc[:zscale_zage].const_ !=
          fitted.preproc[:zscale_zage].const_

    @test_throws "no ordinary random-effect block" reprocess(
        fitted, resample_future; resample_groups=[:nosuchgroup])
    centered = SBBRMI(resample_builder(resample_train); mod=@__MODULE__,
                      centered_groups=[:subject])
    @test_throws "default non-centered" reprocess(
        centered, resample_future; resample_groups=[:subject])

    # The smallest public acceptance shape: a plain `(1 | subject)` fit built
    # without cv_groups gains a different-size new population automatically.
    plain = SBBRMI(plain_resample_builder(resample_train); mod=@__MODULE__)
    plain_replay = reprocess(
        plain, resample_future; resample_groups=[:subject])
    plain_code = StanBlocks.stan_code(plain_replay.model)
    @test plain_replay.data[:n_subject] == 4
    @test occursin("r_mu_subject_xi = std_normal_vector_rng", plain_code)
    plain_params = plain_code[
        first(findfirst("parameters {", plain_code)):first(findfirst(
            "model {", plain_code))]
    @test !occursin("r_mu_subject_xi", plain_params)
end
