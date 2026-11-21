# test/resample_grouped_hsgp.jl — new levels of a grouped HSGP on replay.
#
# `reprocess(sb, rows; resample_groups=[:g])` draws a new level of
# `hsgp(x...; by=g)` as it draws a new level of a `(… | g)` block: its
# standardized basis weights are drawn `Normal(0, 1)` in generated quantities
# and scaled by the spectral density at the draw's fitted length scale and
# amplitude, the law of a fitted level. A per-level hyper-predictor
# (`log(sd(hsgp(x))) ~ 1 + (1 | g)`) draws the new level's hyper effect from
# the fitted SD. `cor=` and `orthogonal_to=` have no value for a new level and
# are refused (snag `resample-groups-6a6b2d58`). On RKBRMI the term's per-level
# rows are sized by its in-graph level set, so a resampled replay keeps the
# fit's program, and a design binder takes the new levels' standardized
# weights and hyper effects in HAVE (`binder.level_inputs`; snag
# `rk-design-binder-45d0b445`).
#
# Set `BRM_RESAMPLE_HSGP_RUNTIME=0` to skip the compiled-model sets.

include(joinpath(@__DIR__, "fitting_packages.jl"))
using Test
using Random
using Statistics
using LinearAlgebra
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal, LKJCholesky
const BRM = BayesianRegressionModels
const RUNTIME = get(ENV, "BRM_RESAMPLE_HSGP_RUNTIME", "1") != "0"

shared_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5)
    y ~ Normal(mu, sigma)
end

# A per-level amplitude beside an ordinary random intercept on the same
# grouping: both draw their new levels.
hyper_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5) + (1 | subject)
    log(sd(hsgp(t))) ~ 1 + (1 | subject)
    y ~ Normal(mu, sigma)
end

# The option set of a per-subject time deviation as a downstream consumer
# writes it: a larger basis, clamped past the fitted interval, anchored at 0.
anchored_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by = subject, k = 16, c = 1.5, extrapolate = :clamp, anchor = 0.0) +
        (1 | subject)
    log(sd(hsgp(t))) ~ 1 + (1 | subject)
    y ~ Normal(mu, sigma)
end

cor_builder = @brm begin
    sigma ~ Exponential(1)
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5, cor = Omega)
    y ~ Normal(mu, sigma)
end

projected_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(t; by = subject, k = 6, c = 1.5, orthogonal_to = :intercept)
    y ~ Normal(mu, sigma)
end

const T_GRID = collect(range(0.0, 10.0; length = 6))
train = (; subject = repeat(["s1", "s2", "s3"], inner = 6), t = repeat(T_GRID, 3),
         y = sin.(repeat(T_GRID, 3)) .+ 0.1 .* (1:18))
future = (; subject = repeat(["n1", "n2", "n3", "n4"], inner = 2),
          t = repeat([2.0, 7.0], 4))

section(code, from, to) =
    code[first(findfirst(from, code)):first(findfirst(to, code))]
params_block(code) = section(code, "parameters {", "model {")
gq_block(code) = code[first(findfirst("generated quantities {", code)):end]

@testset "SBBRMI: new levels of a grouped HSGP are drawn in generated quantities" begin
    fitted = SBBRMI(shared_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    @test replay.data[:n_subject] == 4
    @test StanBlocks.stan_data(replay.model)[:subject_idx] == repeat(1:4, inner = 2)
    @test replay.preproc[:subject_idx].const_.levels == ["n1", "n2", "n3", "n4"]
    # The fitted basis interval is kept and evaluated at the new rows.
    @test replay.preproc[:PHI_hsgp_t_by_subject].const_.fits ==
          fitted.preproc[:PHI_hsgp_t_by_subject].const_.fits
    @test size(replay.data[:PHI_hsgp_t_by_subject]) == (8, 6)

    code = StanBlocks.stan_code(replay.model)
    @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    @test occursin("b_hsgpw_t_subject_n_g = max(subject_idx)", code)
    @test !occursin("zflat_hsgpw_t_subject", params_block(code))
    @test occursin("zflat_hsgpw_t_subject = std_normal_vector_rng", gq_block(code))

    outputs = brm_descriptor(replay).outputs
    kinds = Dict(o.name => o.kind for o in outputs)
    fitted_parameters = Set(o.name for o in brm_descriptor(fitted).outputs
                            if o.kind === :parameter)
    replay_parameters = Set(o.name for o in outputs if o.kind === :parameter)
    @test issubset(replay_parameters, fitted_parameters)
    @test setdiff(fitted_parameters, replay_parameters) == Set([:zflat_hsgpw_t_subject])
    for name in (:zflat_hsgpw_t_subject, :b_hsgpw_t_subject, :mu, :y_gen)
        @test kinds[name] === :generated_quantity
    end

    # The ordinary build is unchanged: the data-sized block stays sampled.
    @test !occursin("_n_g", StanBlocks.stan_code(fitted.model))

    # `cv_groups` alone sizes the block by `max(subject_idx)` but does not
    # mark the index, so the weights stay parameters.
    cv = SBBRMI(shared_builder(train); mod = @__MODULE__, cv_groups = [:subject])
    cv_code = StanBlocks.stan_code(cv.model)
    @test occursin("b_hsgpw_t_subject_n_g = max(subject_idx)", cv_code)
    @test occursin("zflat_hsgpw_t_subject", params_block(cv_code))

    # The other public spellings of the same replay.
    @test restan_data(fitted, future; resample_groups = [:subject]) ==
          StanBlocks.stan_data(replay.model)
    descriptor = brm_descriptor(shared_builder, train; mod = @__MODULE__)
    executed = brm_execute(descriptor, :reprocess, future; resample_groups = [:subject])
    @test StanBlocks.stan_code(executed.plan.model) == code
    # A fresh rebuild is fitted on the rows, so it needs their response.
    fresh = reprocess(fitted, merge(future, (; t = repeat([1.0, 9.0], 4), y = zeros(8)));
                      freeze_constants = false, resample_groups = [:subject])
    @test fresh.preproc[:PHI_hsgp_t_by_subject].const_.fits !=
          fitted.preproc[:PHI_hsgp_t_by_subject].const_.fits
    @test !occursin("zflat_hsgpw_t_subject", params_block(StanBlocks.stan_code(fresh.model)))
end

@testset "SBBRMI: a per-level hyper-predictor draws the new level's hyper effect" begin
    fitted = SBBRMI(hyper_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    code = StanBlocks.stan_code(replay.model)
    @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    params, gq = params_block(code), gq_block(code)
    # New levels: the basis weights, the hyper effect's standardized draw and
    # the ordinary intercept's.
    for name in ("zflat_hsgpw_t_subject", "hsgp_t_by_subject_z_sigma", "r_mu_subject_xi")
        @test !occursin(name, params)
        @test occursin(name, gq)
    end
    # Fitted: the hyper intercept and SD, the shared length scale, the
    # intercept's SD.
    for name in ("hsgp_t_by_subject_beta0_sigma", "hsgp_t_by_subject_sd_sigma",
                 "hsgp_t_by_subject_rho_iso")
        @test occursin(name, params)
    end

    # Anchored and clamped: the frozen basis is zero at the anchor and holds
    # its edge value past the fitted interval, for new levels too.
    anchored = SBBRMI(anchored_builder(train); mod = @__MODULE__)
    edge = (; subject = repeat(["n1", "n2"], inner = 3), t = repeat([0.0, 10.0, 14.0], 2))
    anchored_replay = reprocess(anchored, edge; resample_groups = [:subject])
    anchored_code = StanBlocks.stan_code(anchored_replay.model)
    @test StanBlocks.stanc_check(anchored_code; warn_pedantic = false).ok
    @test occursin("zflat_hsgpw_t_subject = std_normal_vector_rng", gq_block(anchored_code))
    @test occursin("hsgp_t_by_subject_z_sigma", gq_block(anchored_code))
    PHI = anchored_replay.data[:PHI_hsgp_t_by_subject]
    @test size(PHI) == (6, 16)
    @test all(abs.(PHI[[1, 4], :]) .< 1e-12)
    @test PHI[3, :] ≈ PHI[2, :]
end

# A per-unit deviation on its own row set, grouped by an integer unit id and
# read through a `groups(...)` binding in a `@plate` cell beside the units'
# `|p|` block: the replay names both groupings.
plate_builder = @brm begin
    sigma ~ Exponential(1)
    log_level ~ 1 + weight + (1 | p | subject)
    log_rate ~ 1 + (1 | p | subject)
    deviation ~ 0 + hsgp(obs_time; k = 16, c = 1.5, by = obs_subject,
                         extrapolate = :clamp, anchor = 0.0)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_level)
        loc[i] = exp(log_level[i]) .+ exp(log_rate[i]) .* obs_time[obs[i]] .+ deviation[obs[i]]
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

@testset "SBBRMI: a plate cell reads the new levels of a grouped HSGP" begin
    plate_train = (; subject = [1, 2, 3], weight = [60.0, 75.0, 90.0],
        obs_subject = repeat([1, 2, 3], inner = 5),
        obs_time = repeat([0.0, 1.0, 2.0, 4.0, 8.0], 3), y = 1.0 .+ 0.1 .* (1:15))
    fitted = SBBRMI(plate_builder(plate_train); mod = @__MODULE__)
    fitted_parameters = Set(o.name for o in brm_descriptor(fitted).outputs
                            if o.kind === :parameter)
    # New ids, and new units that reuse the fitted ids: resampling draws both.
    for ids in ([11, 12], [1, 2])
        design = (; subject = ids, weight = [70.0, 80.0],
            obs_subject = repeat(ids, inner = 3), obs_time = repeat([0.0, 3.0, 12.0], 2))
        replay = reprocess(fitted, design; resample_groups = [:subject, :obs_subject])
        code = StanBlocks.stan_code(replay.model)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
        params, gq = params_block(code), gq_block(code)
        @test !occursin("zflat_hsgpw_obs_time_obs_subject", params)
        @test occursin("zflat_hsgpw_obs_time_obs_subject = std_normal_vector_rng", gq)
        @test !occursin("b_p_subject_z", params)
        @test occursin("deviation = hsgp_obs_time_by_obs_subject", gq)
        @test occursin("kernel_loc_y_ragged_gen", gq)
        replay_parameters = Set(o.name for o in brm_descriptor(replay).outputs
                                if o.kind === :parameter)
        @test replay_parameters == setdiff(fitted_parameters,
            Set([:zflat_hsgpw_obs_time_obs_subject, :b_p_subject_z_flat]))
        @test all(abs.(replay.data[:PHI_hsgp_obs_time_by_obs_subject][[1, 4], :]) .< 1e-12)
    end
    # Naming only the subjects leaves the deviation's grouping frozen, which
    # refuses an unseen level instead of reusing a fitted level's weights.
    unseen = (; subject = [11, 12], weight = [70.0, 80.0],
        obs_subject = repeat([11, 12], inner = 3), obs_time = repeat([0.0, 3.0, 12.0], 2))
    @test_throws "unseen level" reprocess(fitted, unseen; resample_groups = [:subject])
end

@testset "new levels of a construction fitted per level" begin
    # Not built: new levels of a level-correlated grouping need the
    # correlation over a larger level set (todo 0lopk07); refused by name.
    with_cor = SBBRMI(cor_builder(train); mod = @__MODULE__)
    @test_throws "whose levels" reprocess(with_cor, future; resample_groups = [:subject])
    @test_broken (reprocess(with_cor, future; resample_groups = [:subject]); true)
    # refused: `orthogonal_to=` is fitted on each level's training rows and
    # frozen, so a row's value never depends on the other replay rows
    # (decision 15hn91t); a new level has no training rows to fit it on
    projected = SBBRMI(projected_builder(train); mod = @__MODULE__)
    @test_throws "projection is undefined" reprocess(
        projected, future; resample_groups = [:subject])
    # Their frozen replays of the fitted levels are unchanged.
    @test StanBlocks.stan_code(reprocess(projected, train).model) ==
          StanBlocks.stan_code(projected.model)
end

using ReactiveKernels, ReactiveKernelsPPL

rk_value(query, args...) = (v = Base.invokelatest(query, args...); v isa Tuple ? only(v) : v)
# `want` of `rk` at the draw `u` with `cut` (name => value pairs) in HAVE.
function rk_cut(rk, inputs, want, u, cut...)
    query = ReactiveKernels.prepare(rk.model.spec;
        have = (:unconstrained, first.(cut)..., keys(inputs)...), want = (want,), bound = inputs)
    rk_value(query, u, last.(cut)...)
end
# A draw of the fit carried onto a replay's own coordinates by name: a
# coordinate the fit lacks (a new level's draw, which the cut values replace)
# is zero.
function carried(fit_names, u, replay)
    map(BRM._rk_bound_coordinate_names(replay)) do name
        i = findfirst(==(name), fit_names)
        i === nothing ? 0.0 : u[i]
    end
end
point(rk) = collect(range(-0.6, 0.8; length = rk.model.layout.total))
# The prior variance of a level's field at each row, given the draw's length
# scale and amplitude: sum_j PHI[i, j]^2 S(omega_j).
field_variance(PHI, omega2, sigma, rho) =
    vec(sum(PHI .^ 2 .* (brm_hsgp_sqrt_spd(omega2, sigma, [rho]) .^ 2)'; dims = 2))

@testset "RKBRMI: new levels of a grouped HSGP" begin
    # The term's per-level rows (its standardized weights `<term>.z`, and a
    # hyper-predictor's per-level effects) are sized by the in-graph level set
    # `subject_fit_levels = brm_fit_levels(subject_fit)`, so a resampled
    # replay keeps the fit's program for any design, and a design binder keeps
    # the fit's level set and takes the new levels' values in HAVE (todo
    # 10h1myt; snag rk-design-binder-45d0b445).
    reused = (; subject = repeat(["s1", "s2", "s3"], inner = 2), t = repeat([2.0, 7.0], 3))
    for (builder, inputs) in ((shared_builder, (Symbol("hsgp_t.z"),)),
            (hyper_builder, (:b_subject, Symbol("hsgp_t.z"), Symbol("hsgp_t.sigma_z"))))
        rk = RKBRMI(builder(train))
        @test any(s -> s isa Expr && s == :(subject_fit_levels = brm_fit_levels(subject_fit)),
            BRM._rk_retained_program(rk).main.args)
        u = point(rk)
        fit_names = BRM._rk_layout_coordinate_names(rk)
        binder = rk_design_binder(rk; resample_groups = [:subject])
        @test Set(keys(binder.level_inputs)) == Set(inputs)
        @test binder.level_inputs[Symbol("hsgp_t.z")] == (; columns = (:subject,), kind = :hsgp_weights)
        builder === hyper_builder &&
            @test binder.level_inputs[Symbol("hsgp_t.sigma_z")].kind === :hsgp_sd_effects
        state = brm_fitted_state(rk)
        for design in (reused, future)
            n = length(unique(design.subject))
            replay = reprocess(rk, design; resample_groups = [:subject])
            # The replay's coordinates hold its own levels' draws.
            @test count(name -> startswith(string(name), "hsgp_t.z."),
                BRM._rk_bound_coordinate_names(replay)) == 6n
            # The data-free replay of the fitted state is the same model.
            free = RKBRMI(builder(design), state; resample_groups = [:subject])
            @test BRM._rk_bound_coordinate_names(free) == BRM._rk_bound_coordinate_names(replay)
            @test Set(keys(rk_bound_data(free))) == Set(keys(rk_bound_data(replay)))
            # The binder keeps the fit's level set; at a draw of the fit with
            # the new levels' values in HAVE it is the replay carrying them.
            bound = binder(design)
            @test bound.subject_fit_levels == ["s1", "s2", "s3"]
            cut = Pair{Symbol,Any}[]
            for name in inputs
                width = name === Symbol("hsgp_t.z") ? 6 : 1
                values = reshape(collect(range(-1.0, 1.0; length = n * width)), n, width)
                push!(cut, name => (name === Symbol("hsgp_t.sigma_z") ? vec(values) : values))
            end
            @test rk_cut(rk, bound, :mu, u, cut...) ≈
                rk_cut(replay, rk_bound_data(replay), :mu, carried(fit_names, u, replay), cut...) atol = 1e-12
            free_binder = rk_design_binder(free; resample_groups = [:subject])
            @test rk_cut(free, free_binder(design), :mu, u, cut...) ≈
                rk_cut(rk, bound, :mu, u, cut...) atol = 1e-12
        end
        # The fitted levels' own values, read from the draw by name, give the
        # frozen replay of those levels.
        fitted_rows = (; subject = ["s2", "s3"], t = [2.0, 7.0])
        frozen = reprocess(rk, fitted_rows)
        @test BRM._rk_bound_coordinate_names(frozen) == fit_names
        c = constrain(rk.model.layout, u)
        cut = Pair{Symbol,Any}[Symbol("hsgp_t.z") => c.hsgp_t.z[2:3, :]]
        if builder === hyper_builder
            push!(cut, Symbol("hsgp_t.sigma_z") => c.hsgp_t.sigma_z[2:3])
            b = rk_value(ReactiveKernels.prepare(rk.model.spec;
                have = (:unconstrained, keys(rk_bound_data(rk))...), want = (:b_subject,),
                bound = rk_bound_data(rk)), u)
            push!(cut, :b_subject => b[2:3, :])
        end
        @test rk_cut(rk, binder(fitted_rows), :mu, u, cut...) ≈
            rk_cut(rk, rk_bound_data(frozen), :mu, u) atol = 1e-12
    end
end

@testset "RKBRMI: a new level's field has the fitted law" begin
    # With standardized weights e_j at every new level, the field is the j-th
    # basis function weighted by the square root of the spectral density at
    # the draw's fitted length scale and amplitude, so summing the squares
    # over j gives the fitted level's prior variance. The basis is SBBRMI's
    # replay data: an oracle independent of the RK graph.
    sb_replay = reprocess(SBBRMI(shared_builder(train); mod = @__MODULE__), future;
        resample_groups = [:subject])
    PHI = sb_replay.data[:PHI_hsgp_t_by_subject]
    omega2 = sb_replay.data[:omega2_hsgp_t_by_subject]
    for builder in (shared_builder, hyper_builder)
        rk = RKBRMI(builder(train))
        u = point(rk)
        c = constrain(rk.model.layout, u)
        binder = rk_design_binder(rk; resample_groups = [:subject])
        bound = binder(future)
        # One amplitude per new level: the fitted one, or exp(intercept +
        # sd * z) of the hyper-predictor at the given standardized effects.
        hyper_z = [0.5, -1.0, 0.0, 2.0]
        amplitude = builder === shared_builder ? fill(c.hsgp_t.sigma, 4) :
            exp.(c.hsgp_t.sigma_Intercept .+ c.hsgp_t.sigma_sd .* hyper_z)
        extra = builder === shared_builder ? () :
            (Symbol("hsgp_t.sigma_z") => hyper_z, :b_subject => zeros(4, 1))
        squares = zeros(8)
        for j in 1:6
            unit = zeros(4, 6)
            unit[:, j] .= 1
            field = rk_cut(rk, bound, :mu, u, Symbol("hsgp_t.z") => unit, extra...) .-
                c.mu_Intercept
            squares .+= (field ./ amplitude[repeat(1:4, inner = 2)]) .^ 2
        end
        @test squares ≈ field_variance(PHI, omega2, 1.0, c.hsgp_t.rho) rtol = 1e-10
        # Zero weights (and zero effects): the typical level, the intercept alone.
        @test rk_cut(rk, bound, :mu, u, Symbol("hsgp_t.z") => zeros(4, 6), extra...) ≈
            fill(c.mu_Intercept, 8) atol = 1e-12
    end
end

@testset "RKBRMI: a grouped HSGP on its own rows beside a |p| block" begin
    # A downstream PKPD consumer's shape: the deviation is grouped by the
    # subject column of its own rows, read through `groups(...)` in a plate
    # cell; the binder resamples both groupings.
    plate_train = (; subject = [1, 2, 3], weight = [60.0, 75.0, 90.0],
        obs_subject = repeat([1, 2, 3], inner = 5),
        obs_time = repeat([0.0, 1.0, 2.0, 4.0, 8.0], 3), y = 1.0 .+ 0.1 .* (1:15))
    rk = RKBRMI(plate_builder(plate_train))
    u = point(rk)
    fit_names = BRM._rk_layout_coordinate_names(rk)
    groups = [:subject, :obs_subject]
    binder = rk_design_binder(rk; resample_groups = groups)
    weights = only(k for (k, v) in pairs(binder.level_inputs) if v.kind === :hsgp_weights)
    @test binder.level_inputs[weights].columns == (:obs_subject,)
    @test binder.level_inputs[:b_p_subject] == (; columns = (:subject,), kind = :group_effects)
    # More new units than the fit has, and new units reusing fitted ids.
    for ids in ([11, 12, 13, 14, 15], [1, 2])
        n = length(ids)
        design = (; subject = ids, weight = 60.0 .+ 5.0 .* collect(1:n),
            obs_subject = repeat(ids, inner = 3), obs_time = repeat([0.0, 3.0, 12.0], n))
        replay = reprocess(rk, design; resample_groups = groups)
        cut = (:b_p_subject => 0.1 .* reshape(collect(1.0:2n), n, 2),
            weights => reshape(collect(range(-1.0, 1.0; length = 16n)), n, 16))
        @test rk_cut(rk, binder(design), :loc, u, cut...) ≈
            rk_cut(replay, rk_bound_data(replay), :loc, carried(fit_names, u, replay), cut...)
    end
end

@testset "RKBRMI: new levels of a construction fitted per level are refused" begin
    @test_throws "whose levels are correlated" reprocess(
        RKBRMI(cor_builder(train)), future; resample_groups = [:subject])
    projected = RKBRMI(projected_builder(train))
    @test_throws "projection is undefined" reprocess(
        projected, future; resample_groups = [:subject])
    @test_throws "projection is undefined" rk_design_binder(
        projected; resample_groups = [:subject])(future)
    @test_throws "projection is undefined" RKBRMI(projected_builder(future),
        brm_fitted_state(projected); resample_groups = [:subject])
    # Their frozen replays of the fitted levels are unchanged.
    @test BRM._rk_bound_coordinate_names(reprocess(projected, train)) ==
        BRM._rk_layout_coordinate_names(projected)
end

if !RUNTIME
    @info "BRM_RESAMPLE_HSGP_RUNTIME=0 — skipping the compiled-model sets"
else

import BridgeStan as BS

function constrained_by_name(sm, theta; seed = nothing)
    include_gq = !isnothing(seed)
    names = BS.param_names(sm; include_tp = true, include_gq)
    values = BS.param_constrain(sm, theta; include_tp = true, include_gq,
        rng = include_gq ? BS.StanRNG(sm, seed) : nothing)
    Dict(n => v for (n, v) in zip(names, values))
end
entries(d, prefix) = [d[k] for k in sort!([k for k in keys(d) if startswith(k, prefix * ".")];
                                          by = k -> parse(Int, split(k, '.')[end]))]

@testset "SBBRMI: a new level's field has the fitted law" begin
    fitted = SBBRMI(shared_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    fit_model = StanBlocks.stan_instantiate(fitted.model).model
    rp_model = StanBlocks.stan_instantiate(replay.model).model
    unc_fit, unc_rp = BS.param_unc_names(fit_model), BS.param_unc_names(rp_model)
    draws = 0.3 .* randn(Xoshiro(3), 2, length(unc_fit))
    moved = transport_draws(fitted, replay, draws, unc_fit, unc_rp)
    @test size(moved) == (2, length(unc_rp))
    position = Dict(n => i for (i, n) in enumerate(unc_fit))
    @test moved == draws[:, [position[n] for n in unc_rp]]

    PHI = replay.data[:PHI_hsgp_t_by_subject]
    omega2 = replay.data[:omega2_hsgp_t_by_subject]
    theta = moved[1, :]
    fixed = constrained_by_name(rp_model, theta)
    rho = fixed["hsgp_t_by_subject_rho_iso"]
    sigma = fixed["hsgp_t_by_subject_sigma"]
    expected = field_variance(PHI, omega2, sigma, rho)

    n = 4000
    field = Matrix{Float64}(undef, n, 8)
    z = Matrix{Float64}(undef, n, 4 * 6)
    for s in 1:n
        values = constrained_by_name(rp_model, theta; seed = s)
        field[s, :] = entries(values, "hsgp_t_by_subject")
        z[s, :] = entries(values, "zflat_hsgpw_t_subject")
        # The fitted hypers are the draw's, whatever the generated draw.
        @test values["hsgp_t_by_subject_rho_iso"] == rho
    end
    # Fresh standardized weights for every new level.
    @test all(abs.(mean(z; dims = 1)) .< 0.1)
    @test all(abs.(std(z; dims = 1) .- 1) .< 0.06)
    # Each new level's field: mean zero, the fitted variance at each row,
    # independent across levels.
    @test all(abs.(vec(mean(field; dims = 1))) .< 4 .* sqrt.(repeat(expected ./ n)))
    @test all(abs.(vec(var(field; dims = 1)) ./ expected .- 1) .< 0.1)
    @test abs(cor(field[:, 1], field[:, 3])) < 0.06
end

@testset "SBBRMI: a new level's amplitude is drawn from the fitted hyper SD" begin
    fitted = SBBRMI(hyper_builder(train); mod = @__MODULE__)
    replay = reprocess(fitted, future; resample_groups = [:subject])
    rp_model = StanBlocks.stan_instantiate(replay.model).model
    unc = BS.param_unc_names(rp_model)
    theta = 0.2 .* randn(Xoshiro(5), length(unc))
    fixed = constrained_by_name(rp_model, theta)
    beta0 = fixed["hsgp_t_by_subject_beta0_sigma"]
    sd = fixed["hsgp_t_by_subject_sd_sigma"]
    rho = fixed["hsgp_t_by_subject_rho_iso"]
    PHI = replay.data[:PHI_hsgp_t_by_subject]
    omega2 = replay.data[:omega2_hsgp_t_by_subject]
    level = StanBlocks.stan_data(replay.model)[:subject_idx]
    # Each new level's amplitude is sigma_g = exp(beta0 + sd * z_g) with a
    # fresh z_g ~ Normal(0, 1) at the draw's fitted beta0 and sd, and its
    # field divided by sigma_g is a unit-amplitude field at the fitted rho.
    expected = field_variance(PHI, omega2, 1.0, rho)
    n = 4000
    unit = Matrix{Float64}(undef, n, 8)
    hyper_z = Matrix{Float64}(undef, n, 4)
    amplitudes_follow = true
    for s in 1:n
        values = constrained_by_name(rp_model, theta; seed = s)
        hyper_z[s, :] = entries(values, "hsgp_t_by_subject_z_sigma")
        sigma_g = entries(values, "hsgp_t_by_subject_sigma_vec")
        amplitudes_follow &= sigma_g ≈ exp.(beta0 .+ sd .* hyper_z[s, :])
        unit[s, :] = entries(values, "hsgp_t_by_subject") ./ sigma_g[level]
    end
    @test amplitudes_follow
    @test all(abs.(vec(mean(hyper_z; dims = 1))) .< 0.1)
    @test all(abs.(vec(std(hyper_z; dims = 1)) .- 1) .< 0.06)
    @test all(abs.(vec(var(unit; dims = 1)) ./ expected .- 1) .< 0.1)
end

end
