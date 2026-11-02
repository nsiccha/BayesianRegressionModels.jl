# Run in a fresh process so Revise tracks BRM from its initial load:
# julia --startup-file=no --project=test test/revise_ar1.jl
using Revise
using BayesianRegressionModels, StanBlocks, Test

const BRM = BayesianRegressionModels

@testset "AR1 submodel values survive package rescans" begin
    models = copy(BRM._sb_ar1_normal_models)
    @test length(models) == 16
    for (key, model) in models
        # The previous registration evaluated this anonymous macro. Direct
        # construction must preserve its entire value, including its scope.
        body = BRM._sb_ar1_normal_body(key...)
        reference = Core.eval(BRM, :(StanBlocks.@slic $body))
        @test all(name -> isequal(getfield(model, name), getfield(reference, name)),
            fieldnames(StanBlocks.SlicModel))
    end
    for _ in 1:2
        Revise.revise(BRM; force=false)
        @test isempty(Revise.queue_errors)
        @test keys(BRM._sb_ar1_normal_models) == keys(models)
        @test all(key -> BRM._sb_ar1_normal_models[key] === models[key], keys(models))
    end
end

@testset "Cell source edits remain visible after AR1 registration" begin
    # Edit an owned temporary copy of the real source, never the checkout.
    # The whole file is tracked, including the AR1 registration, while an
    # unrelated method changes. Explicit rescans need no watcher event.
    fixture = Module(gensym(:AR1ReviseFixture))
    Core.eval(fixture, :(using Distributions, LogExpFunctions, StanBlocks, StatsFuns))
    source = read(joinpath(dirname(pathof(BRM)), "sb_cell_julia.jl"), String)
    original = "_sb_cell_binding(scope, head) = nothing"
    edited = "_sb_cell_binding(scope, head) = :revised"
    @test occursin(original, source)
    mktempdir() do dir
        path = joinpath(dir, "sb_cell_julia.jl")
        write(path, source)
        Revise.includet(fixture, path)
        @test Base.invokelatest(fixture._sb_cell_binding, nothing, nothing) === nothing
        models = copy(fixture._sb_ar1_normal_models)
        for (text, expected) in ((replace(source, original => edited), :revised),
                (source, nothing))
            replacement = path * ".new"
            write(replacement, text)
            mv(replacement, path; force=true)
            Revise.revise(fixture; force=false)
            @test isempty(Revise.queue_errors)
            @test Base.invokelatest(fixture._sb_cell_binding, nothing, nothing) === expected
            @test all(key -> fixture._sb_ar1_normal_models[key] === models[key], keys(models))
        end
    end
end
