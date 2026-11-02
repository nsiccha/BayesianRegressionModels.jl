# Runtime fit inputs must execute constant expressions without interpreter dispatch.
# Run: julia --project=test test/runtime_constants.jl
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Statistics
const BRM = BayesianRegressionModels
include("rk_source_roundtrip.jl")

module RuntimeConstantScope
using Statistics, BayesianRegressionModels
shifted_mean(x; shift=0.0) = mean(x) + shift
subtract_total(x, k) = x .- sum(k)
scale_rows(x, a) = x .* a
const v = 100.0 # Inner lambda and comprehension bindings shadow this global.
end

function runtime_constant_cut(rk, want; bound=(;))
    inputs = rk_bound_data(rk)
    have = Tuple(unique((keys(inputs)..., keys(bound)...)))
    cut = ReactiveKernels.prepare(rk.model.spec; have, want, bound,
        type_stability=:error, type_stability_level=:all)
    call(values=inputs) = Base.invokelatest(cut,
        (values[key] for key in keys(inputs) if !haskey(bound, key))...)
    call
end

@testset "runtime constants from source builders pass strict inference" begin
    data = (; x=[1.0, 3.0, 5.0], y=[0.2, 0.4, 0.6], anchor=0.5)
    local_helper = let shift = 0.25
        x -> sum(x) + shift
    end
    interpolated = :($local_helper(x))
    cases = (
        (:(anchor), (x, a) -> a),
        (:(mean(x)), (x, a) -> mean(x)),
        (:(shifted_mean(x; shift=anchor)), (x, a) -> mean(x) + a),
        (:(sum(v -> v^2, x)), (x, a) -> sum(abs2, x)),
        (:(sum([v^2 for v in x if v > 2])), (x, a) -> sum(abs2, filter(>(2), x))),
        (:([mean(x), anchor]), (x, a) -> [mean(x), a]),
        (:(reshape(x, 1, length(x))), (x, a) -> reshape(x, 1, length(x))),
        (:(sum(map(x) do v; v^2 + 1; end)), (x, a) -> sum(abs2, x) + length(x)),
        (:(sum(x[begin:end-1])), (x, a) -> sum(x[1:end-1])),
        (:(sum(x .+ anchor)), (x, a) -> sum(x .+ a)),
        (:($sum(x)), (x, a) -> sum(x)),
        (interpolated, (x, a) -> sum(x) + 0.25),
    )
    for (expression, expected_constant) in cases
        source = quote
            k = constant($expression)
            z = subtract_total(x, k)
            a ~ Normal(1, 0.5)
            mu = scale_rows(z, a)
            y ~ Normal(mu, 1)
        end
        builder = brm(source; mod=RuntimeConstantScope)
        rk = RKBRMI(builder(data))
        inputs = rk_bound_data(rk)
        # The whole-array helper reads a parameter, so the constant remains a
        # graph recipe. A bind-time folded constant would not test this bug.
        @test !haskey(inputs, :k)
        saved = deepcopy(inputs)
        cut = runtime_constant_cut(rk, :z)
        expected = data.x .- sum(expected_constant(data.x, data.anchor))
        @test cut() == expected
        @test isequal(inputs, saved)

        # Binding fitted values is equivalent on the training inputs. With
        # runtime fit inputs, changing those inputs must recompute the constant.
        frozen = runtime_constant_cut(rk, :z; bound=rk_design_binder(rk).fitted)
        @test frozen() == expected
        if haskey(inputs, :x_fit)
            changed = merge(inputs, (; x_fit=[2.0, 4.0, 8.0]))
            @test cut(changed) == data.x .-
                sum(expected_constant(changed.x_fit, data.anchor))
            @test frozen(changed) == expected
        end
        if expression in (:(mean(x)), :([mean(x), anchor]), :($sum(x)), interpolated)
            check_rk_source_roundtrip(rk)
        end
        if expression in (:(mean(x)), :([mean(x), anchor]))
            design = (; x=[10.0, 20.0], y=[0.1, 0.2], anchor=data.anchor)
            replayed = design.x .- sum(expected_constant(data.x, data.anchor))
            @test runtime_constant_cut(reprocess(rk, design), :z)() == replayed
            state = brm_fitted_state(rk)
            @test runtime_constant_cut(RKBRMI(builder(design), state), :z)() == replayed
        end
    end
end

@testset "runtime array constants own their result" begin
    data = (; x=[1.0, 3.0, 5.0], y=[0.2, 0.4, 0.6])
    rk = RKBRMI(brm(quote
        k = constant(x)
        z = subtract_total(x, k)
        a ~ Normal(1, 0.5)
        mu = scale_rows(z, a)
        y ~ Normal(mu, 1)
    end; mod=RuntimeConstantScope)(data))
    inputs = rk_bound_data(rk)
    cut = runtime_constant_cut(rk, :k)
    result = cut()
    @test result == inputs.x_fit
    @test result !== inputs.x_fit
    result[1] = -100.0
    @test cut() == data.x
    @test_throws "k = constant(...)" cut(merge(inputs, (; x_fit=[Inf, 3.0, 5.0])))
end
