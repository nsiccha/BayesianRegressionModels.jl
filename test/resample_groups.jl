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
using Distributions: Exponential, Normal

resample_builder = @brm begin
    mu ~ 1 + zscale(zage) + (1 + zage | p | subject)
    y ~ Normal(mu, 1.0)
end

# NOTE (2026-09-18): these fixtures exercise the CONVENTIONAL GQ resample
# path, so every SBBRMI below pins `total_groups=()`. The default `:auto`
# integrates eligible blocks (e.g. the plain `(1 | subject)` shape) into
# group totals, and `resample_groups` refuses totals models — their route is
# `generative_plan` + `transport_draws` recovery. Without the pin the plain
# and HSGP fixtures throw ArgumentError instead of resampling.
plain_resample_builder = @brm begin
    mu ~ 1 + zage + (1 | subject)
    y ~ Normal(mu, 1.0)
end

hsgp_resample_builder = @brm begin
    mu ~ 1 + hsgp(dose; k = 5) + (1 | subject)
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

@brm function resample_instrument(config)
    if config.use_site
        level ~ 1 + (1 | parentblock(:p) | config.site)
    else
        level ~ 1 + (1 | parentblock(:p) | config.reader)
    end
    shift ~ 1 + (1 | parentblock(:p) | config.site)
    return sum(level + shift)
end

@testset "shared component margins resample by their authored grouping field" begin
    builder = @brm begin
        baseline ~ 0 + (1 | p | site)
        other ~ 0 + (1 | q | site)
        for config in instruments
            readings[config.id] ~ resample_instrument(config)
        end
        y ~ Normal(baseline + other, 1)
        z ~ Normal(readings, 1)
    end
    records(site; second_uses_reader=false) = [
        (; id=:a, use_site=true, site=site[1:2], reader=site[1:2]),
        (; id=:b, use_site=!second_uses_reader, site=reverse(site), reader=reverse(site))]
    fitted_sites = [1, 2, 3]
    train = (; site=fitted_sites, instruments=records(fitted_sites), y=zeros(3), z=zeros(2))
    fit = SBBRMI(builder(train); mod=@__MODULE__)
    # New labels, reversed input record order, and both responses absent.
    new_sites = [4, 5, 6, 7]
    new = (; site=new_sites, instruments=reverse(records(new_sites)))
    member_indices = [k for (k, e) in fit.preproc if e.kind === :group_index && k !== :site_idx]
    @test length(member_indices) == 2
    for design in (train, new)
        replay = reprocess(fit, design; resample_groups=[:site])
        kinds = Dict(o.name => o.kind for o in brm_descriptor(replay).outputs)
        @test kinds[:b_p_site_z_flat] === :generated_quantity
        @test kinds[:b_p_site_L] === kinds[:b_p_site_tau] === :parameter
        @test !haskey(kinds, :b_p_site_new)
        @test StanBlocks.stanc_check(stan_code(replay); warn_pedantic=false).ok
        data = stan_data(replay)
        @test data[:site_idx] == eachindex(design.site)
        @test all(k -> data[k] == [1; 2; reverse(eachindex(design.site))], member_indices)
        @test data[:col_idx_baseline_p_site] == fit.data[:col_idx_baseline_p_site]
        @test replay.data[:n_site] == length(design.site)
    end
    fresh = reprocess(fit, merge(new, (; y=zeros(4), z=zeros(2)));
                      resample_groups=:site, freeze_constants=false)
    @test all(k -> fresh.preproc[k].const_.levels == new_sites, member_indices)

    # Older fits recorded only the standalone block for the shared index.
    entry = fit.preproc[:site_idx]
    fit.preproc[:site_idx] = BayesianRegressionModels.PreprocEntry(:group_index,
        (; entry.const_.levels, entry.const_.n_groups_key), entry.raw_ref, true)
    more_member_sites = (; site=[5, 7], instruments=records(new_sites))
    restored = reprocess(fit, more_member_sites; resample_groups=:site)
    @test stan_data(restored)[:site_idx] == [2, 4]
    @test restored.preproc[:site_idx].const_.levels == new_sites

    # A differently named field participates only when requested. The
    # conditional redraw still uses the single-column address of baseline.
    mixed = merge(train, (; instruments=records(train.site; second_uses_reader=true)))
    fit = SBBRMI(builder(mixed); mod=@__MODULE__)
    a, b = mixed.instruments
    by_site = (; site=[2, 4], instruments=[merge(a, (; site=[4, 2])), merge(b, (; site=[4, 2, 4]))])
    by_reader = (; mixed.site, instruments=[a, merge(b, (; site=[3, 2], reader=[4, 3]))])
    for (selected, design, n_kept, fitted_position) in
            ((:site, by_site, 1, 2), (:reader, by_reader, 4, 3))
        replay = reprocess(fit, design; resample_groups=selected)
        data = stan_data(replay)
        @test data[:b_p_site_new_n_kept] == n_kept
        @test data[:b_p_site_new_pos] == [fitted_position, 0]
        @test data[:col_idx_baseline_p_site] == fit.data[:col_idx_baseline_p_site]
        @test StanBlocks.stanc_check(stan_code(replay); warn_pedantic=false).ok
    end
    all_new = (; new.site, instruments=reverse(records(new.site; second_uses_reader=true)))
    together = reprocess(fit, all_new; resample_groups=[:site, :reader])
    @test !haskey(stan_data(together), :b_p_site_new_n_kept)
    @test together.data[:n_site] == length(new.site)
end

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

    fitted_outputs = brm_descriptor(fitted).outputs
    replay_outputs = brm_descriptor(replay).outputs
    fitted_parameters = Set(o.name for o in fitted_outputs if o.kind === :parameter)
    replay_parameters = Set(o.name for o in replay_outputs if o.kind === :parameter)
    replay_kinds = Dict(o.name => o.kind for o in replay_outputs)
    @test issubset(replay_parameters, fitted_parameters)
    @test replay_parameters != fitted_parameters
    for name in (:b_p_subject_z_flat, :b_p_subject_z, :b_p_subject,
                 :r_mu_p_subject, :mu, :y_gen, :y_likelihood)
        @test replay_kinds[name] === :generated_quantity
    end
    @test replay_kinds[:pop_mu] === :transformed_parameter
    for name in (:b_p_subject_L, :b_p_subject_tau, :pop_mu_beta_pop)
        @test replay_kinds[name] === :parameter
    end

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
    plain = SBBRMI(plain_resample_builder(resample_train); mod=@__MODULE__,
                     total_groups=())
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

@testset "resampled kernel cell moves the whole prediction chain to GQ" begin
    builder = @brm begin
        sigma_pk ~ Exponential(1)
        sigma_qt ~ Exponential(1)
        log_scale ~ 1 + (1 | p | subject)
        pred ~ kernel(dose, pk_y, qt_y, log_scale) do dd, pk_obs, qt_obs, ls
            pred_location = (dd / 10.0) * exp(ls)
            pk_obs ~ normal(pred_location, sigma_pk)
            qt_obs ~ normal(pred_location, sigma_qt)
            pred_location
        end
    end
    train = (; subject=[:a, :b, :c], dose=fill(100.0, 3),
              pk_y=[1.0, 1.5, 2.0], qt_y=[0.8, 1.2, 1.7])
    future = (; subject=[:w, :x, :y, :z], dose=fill(120.0, 4),
               pk_y=fill(0.0, 4), qt_y=fill(0.0, 4))
    fitted = SBBRMI(builder(train); mod=@__MODULE__, total_groups=())
    replay = reprocess(fitted, future; resample_groups=[:subject])
    fitted_outputs = brm_descriptor(fitted).outputs
    replay_outputs = brm_descriptor(replay).outputs
    fitted_parameters = Set(o.name for o in fitted_outputs if o.kind === :parameter)
    replay_parameters = Set(o.name for o in replay_outputs if o.kind === :parameter)
    replay_kinds = Dict(o.name => o.kind for o in replay_outputs)
    @test issubset(replay_parameters, fitted_parameters)
    @test replay_parameters != fitted_parameters
    for name in (:b_p_subject_z_flat, :b_p_subject_z, :b_p_subject,
                 :r_log_scale_p_subject, :log_scale, :pred,
                 :pk_y_gen, :qt_y_gen)
        @test replay_kinds[name] === :generated_quantity
    end
    @test only(o.kind for o in replay_outputs if o.logical === :pred_location) ===
          :generated_quantity
    # The `|p|` block has one margin, so it declares no correlation factor
    # (c98c058b): only its scale stays a sampled parameter.
    for name in (:sigma_pk, :sigma_qt, :b_p_subject_tau, :pop_log_scale_beta_pop)
        @test replay_kinds[name] === :parameter
    end
    @test !haskey(replay_kinds, :b_p_subject_L)
end

@testset "resampled plain block stays introspectable" begin
    # Regression: the grouping-origin record requirement briefly left the
    # record handle unbound for ordinary Stan-expression indices, so
    # `ranef_blocks` on a resample replay threw `UndefVarError` instead of
    # describing the re-drawn block.
    fitted = SBBRMI(plain_resample_builder(resample_train); mod=@__MODULE__,
                     total_groups=())
    replayed = reprocess(fitted, resample_future; resample_groups=[:subject])
    blocks = ranef_blocks(replayed)
    @test length(blocks) == 1
    @test only(blocks).group === :subject
    @test only(blocks).generated
    @test only(blocks).levels == [101, 203, 307, 409]
end

# Regression: `resample_groups` re-emits the fitted BRMI against the new frame,
# so a prediction slice whose `hsgp(...)` axis is CONSTANT (a fixed future dose)
# reaches HSGP emission with an all-equal axis. Frozen replay must apply the
# fitted basis there — it must NOT re-fit and trip the degeneracy guard. A
# lazy-fit regression in `_sb_hsgp_fit_for_emission` (eager `fresh = _brm_fit_hsgp`
# ahead of the frozen-entry early return) made this crash `hsgp: degenerate
# input`; plain frozen `reprocess` was always lazy and unaffected. Snag
# `reprocess-freeze-c4ac8241` (a downstream PKPD consumer); the entangled kernel/ragged
# twin of this lock lives in test/kernel_ragged_reprocess_snag.jl.
@testset "frozen HSGP basis survives resample onto a constant axis" begin
    hsgp_train = (; subject = repeat([1, 2, 3], inner=2),
                    dose = collect(range(-1.0, 1.0; length=6)), y = zeros(6))
    # A single repeated dose value → the HSGP axis is degenerate on the slice.
    # 1.25 lies past the training doses but inside the basis interval
    # (-1.5, 1.5).
    hsgp_future = (; subject = repeat([101, 203, 307, 409], inner=2),
                     dose = fill(1.25, 8), y = zeros(8))

    fitted = SBBRMI(hsgp_resample_builder(hsgp_train); mod=@__MODULE__,
                     total_groups=())
    fit_fits = fitted.preproc[:PHI_hsgp_dose].const_.fits

    replay = reprocess(fitted, hsgp_future; freeze_constants=true,
                       resample_groups=[:subject])
    # The fitted (mu, L) basis is reused verbatim, applied at the constant axis.
    @test replay.preproc[:PHI_hsgp_dose].const_.fits == fit_fits
    @test size(replay.data[:PHI_hsgp_dose]) == (8, 5)
    @test all(isfinite, replay.data[:PHI_hsgp_dose])
    @test all(r -> r == replay.data[:PHI_hsgp_dose][1, :],
              eachrow(replay.data[:PHI_hsgp_dose]))
    # The spectral weights and the basis-validity rho floor derive from the
    # fitted basis, so both must match the training data verbatim.
    @test replay.data[:omega2_hsgp_dose] == fitted.data[:omega2_hsgp_dose]
    @test replay.data[:rho_lower_hsgp_dose] == fitted.data[:rho_lower_hsgp_dose]
    @test replay.data[:n_subject] == 4

    # `freeze_constants=false` genuinely has no domain to fit on this slice.
    @test_throws "hsgp: degenerate input" reprocess(
        fitted, hsgp_future; freeze_constants=false, resample_groups=[:subject])
    # refused: a dose past the basis interval would wrap the sines around and
    # evaluate a different point (decision 1fk9b29)
    @test_throws "outside the fitted basis interval" reprocess(
        fitted, merge(hsgp_future, (; dose=fill(2.0, 8))); freeze_constants=true,
        resample_groups=[:subject])
end
