# test/rk_categorical_levels.jl — a categorical response's level table and the
# concise form's coefficient prior address (snag `rkbrmi-categoric-afd009c8`).
#
# Oracles:
#   * one declaration, many label sets: models that differ only in the labels
#     of a `CategoricalLogit` or `Ordinal` response emit the identical RK
#     program and lowered program (user principle 0ikeidi), on the formula
#     route and the authored-location (value) route; the level table is the
#     fitted value `y_fit_levels = brm_fit_levels(y_fit)`, not a literal;
#   * the density is the categorical-logit density of the rows coded against
#     the sorted labels (or a `CategoricalVector`'s declared order), checked
#     against an independent formula and central differences;
#   * a frozen `reprocess` and a data-free replay (`brm_fitted_state`) code a
#     design's rows against the fit's levels and refuse a label the fit never
#     saw; a design holding only some classes keeps the fitted class count,
#     and a row's one-class held-out probabilities sum to one;
#   * `effect(y, term) ~ law` on a concise `CategoricalLogit(@brm(...))`
#     response sets `term`'s coefficient in every non-reference class on both
#     backends, reading a data-valued law argument as data.
#
# Run: julia --startup-file=no --project=test test/rk_categorical_levels.jl

using Test, BayesianRegressionModels, Distributions, StanBlocks
using Enzyme, LogDensityProblems
# Qualified: RKPPL also exports `LogitLink`, which the `Ordinal` model names.
import ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface: AutoEnzyme
import CategoricalArrays as CA
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

emitted_program(rk) = BRM._rk_program_statements(BRM._rk_emit_ast(rk.plan))
lowered_program(rk) = string(rk.model.program)
kernel_text(rk) = replace(string(ReactiveKernelsPPL.kernel_expr(rk_bound_plan(rk),
    rk.model.layout)), r"##\w+#\d+" => "##gen")

x1 = [-1.0, 0.2, 0.5, -0.4, 1.1, 0.0, 0.7, -0.9]
x2 = [0.3, -0.6, 1.2, 0.1, -0.2, 0.8, -1.1, 0.4]
codes = [2, 1, 3, 2, 1, 3, 3, 1]
labelled(levels) = (; x1, x2, y=[levels[c] for c in codes], coef_scale=1.5)

explicit = @brm begin
    eta2 ~ 1 + x1 + x2
    eta3 ~ 1 + x1 + x2
    y ~ CategoricalLogit(eta2, eta3)
end
# The authored-location route: the class logits are assignments.
authored = @brm begin
    b2 ~ Normal(0, 1)
    b3 ~ Normal(0, 1)
    eta2 = b2 .* x1
    eta3 = b3 .* x2
    y ~ CategoricalLogit(eta2, eta3)
end
ordinal = @brm begin
    eta ~ 0 + x1 + x2
    y ~ Ordinal(Cumulative(), LogitLink(), eta)
end
concise = @brm begin
    effect(y, x1) ~ Normal(0, coef_scale)
    y ~ CategoricalLogit(@brm(1 + x1 + x2))
end

# The categorical-logit density of `codes` (1 = reference) under the explicit
# model's coordinates (Intercept, x1, x2 per class, standard-normal priors).
function explicit_oracle(u; c=codes, a=x1, b=x2, prior=u -> sum(logpdf.(Normal(), u)))
    eta2 = u[1] .+ u[2] .* a .+ u[3] .* b
    eta3 = u[4] .+ u[5] .* a .+ u[6] .* b
    sum(eachindex(c)) do i
        logits = [0.0, eta2[i], eta3[i]]
        logits[c[i]] - log(sum(exp.(logits)))
    end + prior(u)
end

function check_density(rk, oracle)
    problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    N = rk.model.layout.total
    for u in (zeros(N), collect(range(-0.3, 0.4; length=N)))
        lp, g = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test lp ≈ oracle(u) atol=1e-11
        h = 1e-5
        fd = map(eachindex(u)) do j
            plus, minus = copy(u), copy(u)
            plus[j] += h; minus[j] -= h
            (oracle(plus) - oracle(minus)) / (2h)
        end
        @test g ≈ fd atol=1e-6 rtol=1e-5
    end
end

@stestset "a categorical response's level table is a fitted value" begin
    label_sets = ([1, 2, 3], [2, 5, 9], ["a", "b", "c"], ["p", "q", "r"])
    models = [RKBRMI(explicit(labelled(levels))) for levels in label_sets]
    @test allequal(emitted_program(rk) for rk in models)
    # The lowered program types its bound columns, so labels of one element
    # type share it; Int and String labels differ in those types only.
    for group in (models[1:2], models[3:4])
        @test allequal(lowered_program(rk) for rk in group)
        @test allequal(kernel_text(rk) for rk in group)
    end
    typed(text) = replace(text, "Vector{String}" => "Vector{Int64}")
    @test typed(lowered_program(models[3])) == lowered_program(models[1])
    main = join(string.(emitted_program(first(models))), "\n")
    @test occursin("y_fit_levels = brm_fit_levels(y_fit)", main)
    @test occursin(r"y = \S*brm_prepared_indices\(y_raw, y_fit_levels\)", main)
    @test !occursin("Base.vect(1, 2, 3)", kernel_text(first(models)))
    for rk in models
        check_density(rk, explicit_oracle)
    end
    # A categorical column keeps its declared level order, reference first.
    declared = CA.categorical(["b", "a", "c", "b", "a", "c", "c", "a"];
        levels=["b", "a", "c"])
    rk = RKBRMI(explicit((; x1, x2, y=declared, coef_scale=1.5)))
    @test emitted_program(rk) == emitted_program(first(models))
    check_density(rk, u -> explicit_oracle(u; c=[1, 2, 3, 1, 2, 3, 3, 2]))
end

@stestset "the authored-location route and Ordinal read their level table too" begin
    for builder in (authored, ordinal)
        models = [RKBRMI(builder(labelled(levels)))
                  for levels in ([1, 2, 3], [4, 7, 8], ["lo", "mid", "top"])]
        @test allequal(emitted_program(rk) for rk in models)
        @test allequal(lowered_program(rk) for rk in models[1:2])
        main = join(string.(emitted_program(first(models))), "\n")
        @test occursin("brm_fit_levels(y_fit)", main)
    end
    rk = RKBRMI(authored(labelled(["lo", "mid", "top"])))
    check_density(rk, u -> begin
        eta2, eta3 = u[1] .* x1, u[2] .* x2
        sum(eachindex(codes)) do i
            logits = [0.0, eta2[i], eta3[i]]
            logits[codes[i]] - log(sum(exp.(logits)))
        end + sum(logpdf.(Normal(), u))
    end)
end

@stestset "a replay codes its rows against the fit's levels" begin
    fit = RKBRMI(explicit(labelled([2, 5, 9])))
    u = collect(range(-0.3, 0.4; length=fit.model.layout.total))
    # The design's rows in another order, every label present.
    order = [3, 1, 2, 8, 7, 6, 5, 4]
    design = (; x1=x1[order], x2=x2[order], y=[2, 5, 9][codes[order]], coef_scale=1.5)
    oracle(v) = explicit_oracle(v; c=codes[order], a=design.x1, b=design.x2)
    replay = reprocess(fit, design)
    @test emitted_program(replay) == emitted_program(fit)
    check_density(replay, oracle)
    state = brm_fitted_state(fit)
    dataless = RKBRMI(explicit(design), state)
    check_density(dataless, oracle)
    # A label the fit never saw is refused, not coded as another class.
    unseen = merge(design, (; y=[2, 5, 7, 9, 2, 5, 9, 2]))
    @test_throws "not a training level" reprocess(fit, unseen)
    @test_throws "not a training level" RKBRMI(explicit(unseen), state)
    # A fresh replay takes the new data's own levels.
    fresh = reprocess(fit, merge(design, (; y=[1, 4, 6][codes[order]]));
        freeze_constants=false)
    check_density(fresh, oracle)
end

@stestset "a replay design keeps the fitted class count" begin
    # A replay keeps the fit's three classes: its rows are coded against them,
    # not against the design's own levels (before 0.2.285 a design holding two
    # classes was planned as a two-class model and refused by arity).
    plain = @brm begin
        y ~ CategoricalLogit(@brm(1 + x1 + x2))
    end
    fit = RKBRMI(plain(labelled([1, 2, 3])))
    @test fit.model.layout.total == 6
    # A design holding only some of the fitted classes binds (RKPPL 0.1.168,
    # ReactiveKernels snag `rkppl-leveled-re-328038d0`) and is scored as a
    # three-class model, on a frozen and on a data-free replay.
    state = brm_fitted_state(fit)
    for c in ([1, 2, 1, 2, 1, 2, 1, 2], fill(2, 8), [1, 3, 3, 1, 1, 3, 1, 3])
        design = (; x1, x2, y=c, coef_scale=1.5)
        replay = reprocess(fit, design)
        @test emitted_program(replay) == emitted_program(fit)
        check_density(replay, u -> explicit_oracle(u; c))
        check_density(RKBRMI(plain(design), state), u -> explicit_oracle(u; c))
    end
    # Held-out class probabilities: set every row to class k and read the
    # pointwise likelihood. A row's K one-class probabilities sum to one and
    # match the softmax of its class logits.
    u = collect(range(-0.3, 0.4; length=fit.model.layout.total))
    rows = map(1:3) do k
        replay = reprocess(fit, (; x1, x2, y=fill(k, 8), coef_scale=1.5))
        pointwise = ReactiveKernelsPPL.prepare_query(replay.model,
            rk_bound_plan(replay), :pointwise)
        collect(Base.invokelatest(pointwise, u).y)
    end
    probabilities = reduce(hcat, map(r -> exp.(r), rows))
    @test size(probabilities) == (8, 3)
    @test all(sum(probabilities; dims=2) .≈ 1)
    logits = hcat(zeros(8), u[1] .+ u[2] .* x1 .+ u[3] .* x2,
        u[4] .+ u[5] .* x1 .+ u[6] .* x2)
    @test probabilities ≈ exp.(logits) ./ sum(exp.(logits); dims=2) atol=1e-12
    # A code outside the fitted classes is still refused, by the fitted count.
    @test_throws "must hold codes in 1..3" BRM._rk_gate_response_values!(
        :categorical_logit, [1, 4, 2], :y; n_levels=3)
    # Without a stated class count every level 1..K must appear.
    @test_throws "must hold every one of its 3 levels" BRM._rk_gate_response_values!(
        :categorical_logit, [1, 3, 3], :y)
    # A declared level with no rows is a class of the model: three classes,
    # rows coded against the declared order.
    declared = CA.categorical(["a", "c", "a", "c", "a", "c", "c", "a"];
        levels=["a", "b", "c"])
    rk = RKBRMI(explicit((; x1, x2, y=declared, coef_scale=1.5)))
    check_density(rk, u -> explicit_oracle(u; c=[1, 3, 1, 3, 1, 3, 3, 1]))
end

@stestset "a JAX module codes the response in prepare_data" begin
    plain = @brm begin
        y ~ CategoricalLogit(@brm(1 + x1 + x2))
    end
    for levels in ([2, 5, 9], ["a", "b", "c"])
        rk = RKBRMI(plain(labelled(levels)))
        text = ReactiveKernelsPPL.jax_source(rk.model, rk_bound_plan(rk)).text
        @test occursin("UNTRANSLATED = {}", text)
        @test occursin("_brm_fit_levels_np(y_fit)", text)
        @test occursin(r"_indices_np\(y_raw, y_fit_levels\)", text)
    end
end

@stestset "effect(y, term) sets a concise categorical coefficient in every class" begin
    data = labelled([1, 2, 3])
    resolved = string(concise(data))
    for class in (:y_nested_arg1_class2, :y_nested_arg1_class3)
        @test occursin("effect($class, x1) ~ Normal(0, coef_scale)", resolved)
    end
    @test !occursin("effect(y, x1)", resolved)
    named = @brm begin
        eta2 ~ 1 + x1 + x2
        eta3 ~ 1 + x1 + x2
        effect(eta2, x1) ~ Normal(0, coef_scale)
        effect(eta3, x1) ~ Normal(0, coef_scale)
        y ~ CategoricalLogit(eta2, eta3)
    end
    a, b = RKBRMI(concise(data)), RKBRMI(named(data))
    # Same density as the explicit twin: x1's coefficient has scale 1.5 in both
    # classes, every other coefficient the default standard normal.
    prior(v) = sum(logpdf.(Normal(), v[[1, 3, 4, 6]])) +
        sum(logpdf.(Normal(0, 1.5), v[[2, 5]]))
    check_density(a, v -> explicit_oracle(v; prior))
    check_density(b, v -> explicit_oracle(v; prior))
    # The scale is read as data: another value, one program.
    other = RKBRMI(concise(merge(data, (; coef_scale=0.4))))
    @test emitted_program(other) == emitted_program(a)
    @test lowered_program(other) == lowered_program(a)
    # The class count is still a program shape (one predictor per class).
    four = RKBRMI(concise(merge(data, (; y=[1, 2, 3, 4, 1, 2, 3, 4]))))
    @test emitted_program(four) != emitted_program(a)
    # SBBRMI reads the same addresses.
    code = stan_code(SBBRMI(concise(data); mod=@__MODULE__))
    @test count("coef_scale", code) >= 2
    # A per-class address beside the response address is refused by name.
    duplicate = @brm begin
        effect(y, x1) ~ Normal(0, coef_scale)
        effect(y_nested_arg1_class2, x1) ~ Normal(0, 1)
        y ~ CategoricalLogit(@brm(1 + x1 + x2))
    end
    @test_throws "Keep only `effect(y, x1)`" duplicate(data)
    # `effect(y, …)` on a response without a concise categorical formula keeps
    # its meaning: it names no linear predictor.
    plain = @brm begin
        effect(y, x1) ~ Normal(0, 1)
        eta2 ~ 1 + x1
        eta3 ~ 1 + x1
        y ~ CategoricalLogit(eta2, eta3)
    end
    @test_throws Exception RKBRMI(plain(data))
end
