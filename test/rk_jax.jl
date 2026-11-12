# test/rk_jax.jl — RKBRMI backends as Python/JAX modules,
# `jax_source(rk.model, rk_bound_plan(rk))` (brm-use
# references/reactivekernels.md, "A Python/JAX module"; snag
# `rkbrmi-to-rk-jax-a465a405`).
#
# Every run checks that representative backends, fits and replays, emit a
# module whose `prepare_data` holds BRM's preprocessing (level codes, fitted
# `zscale`/`center` anchors, cell flattens) as translated host code. With
# `RK_JAX_PYTHON` naming a Python that has `jax`, each module's `log_density`
# and `jax.grad` are also compared with the native density and its central
# differences, through the pinned RK checkout's own driver
# (`test/jax_python.jl`); without it those checks are skipped.
#
# Run: RK_JAX_PYTHON=<python with jax> julia --project=test test/rk_jax.jl
using Test
using BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL
import CategoricalArrays

include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

module JaxModels
using BayesianRegressionModels, Distributions
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x) + center(w) + (1 | g)
    y ~ Normal(mu, sigma)
end
counts(d) = @brm d begin
    log(lambda) ~ 1 + x + (1 | g)
    c ~ Poisson(lambda)
end
toy(d) = @brm d begin
    mu ~ 1 + (1 | g)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
member(d) = @brm d begin
    mu ~ 1 + (1 | mm(g, g2))
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end
# Grouped reads through `groups(group => cells)`: dose rows and long-format
# observation rows joined to their subject by label (`test/rk_design_binder.jl`'s
# `flat_pk`), and dose rows read in `order=dose_rank`.
flat_pk(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V ~ 1 + (1 | p | subject)
    log_F ~ 0 + log_dose
    effect(log_CL, Intercept) ~ Normal(0, 1)
    effect(log_V, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    sd(:, p) ~ Exponential(1)
    doses = groups(dose_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        dose = sum(exp.(log_F[doses[i]]))
        loc[i] = dose / exp(log_V[i]) .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end
# Two columns read through each join: the second column of a join reads the
# first one's partition (`brm_ragged_rows_shared`).
flat_pk_two(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V ~ 1 + (1 | p | subject)
    log_F ~ 0 + log_dose
    effect(log_CL, Intercept) ~ Normal(0, 1)
    effect(log_V, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    sd(:, p) ~ Exponential(1)
    doses = groups(dose_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        dose = sum(dose_amount[doses[i]] .* exp.(log_F[doses[i]]))
        loc[i] = dose / exp(log_V[i]) .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]]) .*
            obs_scale[obs[i]]
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end
flat_pk_ordered(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | subject)
    log_F ~ 0 + log_dose
    effect(log_CL, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    doses = groups(dose_subject => subject; order=dose_rank)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        first_dose = exp(first(log_F[doses[i]]))
        loc[i] = first_dose .* exp.(-exp(log_CL[i]) .* obs_time[obs[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end
# A per-subject `zscale` read in a `@plate`: its anchor is computed in the
# program, not bound, so it reaches the module's `prepare_data`.
pk(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dose[i] .* exp.(-exp(log_CL[i]) .* t[i])
        y[i] ~ normal(conc[i], sigma)
    end
end
end
const JM = JaxModels

const TRAIN = (; x=collect(range(0.3, 4.1; length=9)), w=cos.(1:9),
    g=["a", "b", "c"][mod1.(1:9, 3)], y=sin.(1:9), c=[0, 2, 1, 3, 0, 1, 4, 2, 1])
const PK = (; subject=["s1", "s2", "s3", "s4", "s5"],
    weight=[60.0, 72.0, 81.0, 55.0, 90.0], dose=[100.0, 120.0, 80.0, 100.0, 150.0],
    t=[[0.5, 1.0, 2.0], [0.25, 1.5], [0.5, 1.0, 2.0, 4.0], [1.0], [0.5, 2.0]],
    y=[[1.0, 0.8, 0.5], [1.2, 0.7], [0.9, 0.7, 0.4, 0.2], [0.6], [1.3, 0.5]])

# Subjects with uneven dose and observation rows, listed out of subject order;
# `s4` has no dose row.
const FLAT_PK = (; subject=["s1", "s2", "s3", "s4"], weight=[61.0, 75.0, 68.0, 82.0],
    dose_subject=["s2", "s1", "s2", "s3", "s1", "s2"],
    log_dose=log.([1.4, 1.1, 1.7, 1.2, 1.9, 1.3]), dose_rank=[3, 2, 1, 1, 1, 2],
    obs_subject=["s1", "s3", "s2", "s1", "s4", "s2", "s3", "s4", "s1"],
    obs_time=[0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0],
    y=[0.9, 0.6, 1.4, 0.7, 0.2, 0.8, 0.3, 0.1, 0.4])
const FLAT_PK_TWO = merge(FLAT_PK, (; dose_amount=[100.0, 80.0, 120.0, 90.0, 60.0, 110.0],
    obs_scale=[1.0, 0.8, 1.2, 0.9, 1.1, 1.0, 0.7, 1.3, 0.95]))
# The ordered join reads each subject's first dose in `dose_rank` order, so
# every subject has one here.
const FLAT_PK_ORDERED = merge(FLAT_PK, (; dose_subject=[FLAT_PK.dose_subject; "s4"],
    log_dose=[FLAT_PK.log_dose; log(1.6)], dose_rank=[FLAT_PK.dose_rank; 1]))

point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
jax_function(src, name) = only(f for f in src.functions if f.name == name)

# The module's `prepare_data` takes the bound columns it lists, by name.
function raw_inputs(rk, src)
    bound = rk_bound_data(rk, src)
    Tuple(bound[Symbol(first(input))] for input in jax_function(src, "prepare_data").inputs)
end

function check_module(rk; anchors)
    src = jax_source(rk.model, rk_bound_plan(rk))
    @test src isa JAXSource
    @test issubset(("prepare_data", "log_density", "log_prior", "log_likelihood",
        "log_jacobian", "constrain"), [f.name for f in src.functions])
    # Module inputs have exactly the names prepare_data lists. The native
    # binding can also hold intermediate values and omitted fit-role columns.
    inputs = Set(Symbol(first(input)) for input in jax_function(src, "prepare_data").inputs)
    @test inputs == Set(keys(rk_bound_data(rk, src)))
    # A fitted anchor computed in the program is translated host code, its
    # guards included.
    @test occursin("requires at least one training value", src.text) == anchors
    if rk_jax_available()
        u = point(rk)
        density = prepare_query(rk.model, rk_bound_plan(rk), :sampler)
        r = rk_jax_evaluate(src, "log_density", (u, raw_inputs(rk, src)...);
            grad=true, prepare=true)
        @test r.value ≈ Base.invokelatest(density, u) rtol=1e-12
        @test r.grad ≈ rk_fd_gradient(v -> Base.invokelatest(density, v), u) rtol=1e-6 atol=1e-6
    else
        @test_skip "RK_JAX_PYTHON names no Python with jax"
    end
    src
end

@testset "bound anchors and level codes: the fit and a replay" begin
    rk = RKBRMI(JM.grouped(TRAIN))
    check_module(rk; anchors=false)
    check_module(reprocess(rk, (; x=[1.0, 2.0, 3.0, 4.0], w=[0.1, -0.2, 0.3, 0.0],
        g=["a", "c", "c", "b"], y=[0.2, -0.1, 0.4, 0.0])); anchors=false)
    check_module(RKBRMI(JM.counts(TRAIN)); anchors=false)
end

# A grouping's level coding, its fitted levels (`g_fit_levels =
# brm_fit_levels(g_fit)`) and each row's position among them (`g_level`), is
# `prepare_data` host code translated from the source Julia runs, so a module
# fed the raw labels alone runs without Julia (snag `rkbrmi-grouping-589e4a34`).
# So are a multi-membership block's positions among its members' joint labels.
# A categorical column's declared level order is not in the labels a module
# reads: its levels and positions stay inputs.
prepare_inputs(src) = Set(Symbol(first(i)) for i in jax_function(src, "prepare_data").inputs)
untranslated(src) = something(match(r"^UNTRANSLATED = (.*)$"m, src.text), (; captures=[""])).captures[1]
@testset "a grouping's level coding is prepare_data host code" begin
    y = [0.1, -0.2, 0.3, 0.5, -0.1, 0.2]
    for g in ([1, 1, 2, 3, 3, 3], ["a", "a", "b", "c", "c", "c"], [30, 10, 20, 10, 30, 5])
        src = check_module(RKBRMI(JM.toy((; g, y))); anchors=false)
        @test prepare_inputs(src) == Set([:g, :g_fit, :y])
        @test untranslated(src) == "{}"
    end
    src = check_module(RKBRMI(JM.member((; g=["b", "a", "c", "a", "b", "b"],
        g2=["d", "a", "a", "c", "b", "d"], y))); anchors=false)
    @test prepare_inputs(src) == Set([:g, :g2, :y])
    @test untranslated(src) == "{}"
    g = CategoricalArrays.categorical(["b", "a", "c", "a", "b", "b"];
        levels=["c", "b", "a", "z"])
    src = check_module(RKBRMI(JM.toy((; g, y))); anchors=false)
    @test issubset((:g_fit_levels, :g_level), prepare_inputs(src))
    # A replay's labels outside the fitted levels: Julia and the module refuse
    # them with one message.
    rk = RKBRMI(JM.toy((; g=["a", "a", "b"], y=y[1:3])))
    @test_throws "is not one of the 2 fitted levels" BayesianRegressionModels.brm_level_indices(
        ["a", "q"], ["a", "b"])
    if rk_jax_available()
        src = jax_source(rk.model, rk_bound_plan(rk))
        bad = merge(rk_bound_data(rk, src), (; g=["a", "q", "b"]))
        @test_throws "is not one of the 2 fitted levels" rk_jax_evaluate(src, "log_density",
            (point(rk), Tuple(bad[Symbol(first(i))]
                for i in jax_function(src, "prepare_data").inputs)...); prepare=true)
    end
end

const PK_DESIGN = (; subject=["s2", "s4"], weight=[72.0, 55.0], dose=[120.0, 100.0],
    t=[[0.25, 1.5, 3.0], [1.0, 2.0]], y=[[1.1, 0.6, 0.3], [0.5, 0.4]])

@testset "an anchor computed in the program is prepare_data host code" begin
    rk = RKBRMI(JM.pk(PK))
    src = check_module(rk; anchors=true)
    @test occursin("requires nonzero sample variance", src.text)
    check_module(reprocess(rk, PK_DESIGN); anchors=true)
end

# A `@plate` cell's collected value (`conc`, one vector per subject, read
# from the sampled `log_CL`) in `values`, which `constrain` returns flat.
# Not built yet (capability gap): the emitted plate iterates
# `eachindex(dose)` and its cells read `t[i]`, so RK's JAX flatten finds no
# prepared ragged input carrying the cells' lengths ("flattening a ragged
# value whose lengths are not those of a prepared ragged input"; RK
# `356142a62`). Todo `2026-10-24T11-58-45-172-1e0pxzp`.
@testset "a plate cell's collected value in `values`" begin
    rk = RKBRMI(JM.pk(PK))
    src = try
        jax_source(rk.model, rk_bound_plan(rk); values=(:conc,))
    catch e
        e isa JAXEmissionError || rethrow()
        nothing
    end
    @test_broken src isa JAXSource
    if src === nothing
        # nothing to evaluate until the module emits
    elseif rk_jax_available()
        u = point(rk)
        bound = rk_bound_data(rk)
        conc = ReactiveKernels.prepare(rk.model.spec;
            have=(:unconstrained, keys(bound)...), want=(:conc,), bound)
        native = first(Base.invokelatest(conc, u))
        r = rk_jax_evaluate(src, "constrain", (u, raw_inputs(rk, src)...); prepare=true)
        @test r.value["conc"] ≈ reduce(vcat, native) rtol=1e-12
    else
        @test_skip "RK_JAX_PYTHON names no Python with jax"
    end
end

# `RKBRMI(build(design), brm_fitted_state(rk))` binds the fitted values, not
# the training rows: they are `prepare_data` inputs, which
# `rk_bound_data(rk, src)` holds under the names `src.functions` lists.
@testset "a data-free replay passes its fitted values to prepare_data" begin
    rk = RKBRMI(JM.grouped(TRAIN))
    design = (; x=[1.0, 2.0, 3.0, 4.0], w=[0.1, -0.2, 0.3, 0.0],
        g=["a", "c", "c", "b"], y=[0.2, -0.1, 0.4, 0.0])
    check_module(RKBRMI(JM.grouped(design), brm_fitted_state(rk)); anchors=false)
    pk = RKBRMI(JM.pk(PK))
    check_module(RKBRMI(JM.pk(PK_DESIGN), brm_fitted_state(pk)); anchors=false)
end

# A grouped read's row partition (`brm_ragged_rows`, `brm_ragged_rows_ordered`;
# `src/ragged_rows.jl`) is `prepare_data` host code, translated from the same
# source Julia runs (snag `jax-source-brm-r-f6503407`). It needs ReactiveKernels
# 0.1.138: host writes into a list of arrays the source built and only the
# taken arm of a decided type test (RK 0.1.136), and reductions over a gather
# through a ragged index masked (RK 0.1.138), which the cells'
# `sum(exp.(log_F[doses[i]]))` and observations make. The ordered join's merge
# sort runs in `prepare_data` too.
@testset "a grouped read's row partition is prepare_data host code" begin
    for (model, data) in ((JM.flat_pk, FLAT_PK), (JM.flat_pk_ordered, FLAT_PK_ORDERED),
            (JM.flat_pk_two, FLAT_PK_TWO))
        rk = RKBRMI(model(data))
        src = check_module(rk; anchors=false)
        @test occursin("def prepare_data", src.text)
        @test occursin("_brm_partition_rows", src.text)
        model === JM.flat_pk_two && @test occursin("brm_ragged_rows_shared",
            sprint(Base.show_unquoted, BayesianRegressionModels._rk_emit_ast(rk.plan).main))
    end
end
