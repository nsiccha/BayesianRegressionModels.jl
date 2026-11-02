using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels
include("rk_source_roundtrip.jl")

@testset "Data-owned vector extents survive fitted preprocessing replay" begin
    for extent in (:n, :(length(axis) / 2))
        builder = brm(quote
            s ~ Exponential(1.0)
            z[1:$extent] .~ Normal.(0.0, 1.0)
            mu = z[block]
            y ~ Normal(mu, s)
        end; mod=@__MODULE__)
        training = (; n=2, axis=zeros(4), block=[1, 1, 2], y=[0.1, 0.2, -0.3])
        window = (; n=3, axis=zeros(6), block=[3, 3], y=[0.4, 0.5])
        fitted = check_rk_source_roundtrip(RKBRMI(builder(training)))
        state = brm_fitted_state(fitted)
        fresh = check_rk_source_roundtrip(RKBRMI(builder(window)))
        @test BRM._rk_program_statements(BRM._rk_retained_program(fitted)) ==
            BRM._rk_program_statements(BRM._rk_retained_program(fresh))
        @test length(coordinate_names(fitted.model.layout)) == 3
        @test length(coordinate_names(fresh.model.layout)) == 4

        # Ordinary prediction still needs draws on exactly the fitted frame.
        @test_throws r"changes the fitted coordinates" RKBRMI(builder(window), state)
        same_frame = merge(window, (; n=2, axis=zeros(4), block=[2, 2]))
        replay = RKBRMI(builder(same_frame), state)
        @test coordinate_names(replay.model.layout) == state.coordinates

        # The window-only replay permits growth after the same source,
        # preprocessing, terms and group-level checks as ordinary replay.
        grown = BRM._rk_fitted_replay(builder(window), window, state;
            coordinates=:growing)
        @test coordinate_names(grown.model.layout) == coordinate_names(fresh.model.layout)
        smaller = merge(window, (; n=1, axis=zeros(2), block=[1, 1]))
        @test_throws r"cannot remove fitted coordinates" BRM._rk_fitted_replay(
            builder(smaller), smaller, state; coordinates=:growing)
    end
end

@testset "A computed extent cannot alias an equal scalar data input" begin
    builder = @brm begin
        z[1:length(axis) / 2] .~ Normal.(0.0, 1.0)
        mu = z[block] .+ offset
        y ~ Normal(mu, 1)
    end
    training = (; axis=zeros(4), block=[1, 2], offset=2, y=[0.0, 0.0],
        z_length=99)
    rk = RKBRMI(builder(training))
    binder = rk_design_binder(rk)
    first_window = merge(training, (; y=[0.1, 0.2]))
    @test binder(first_window) == rk_bound_data(reprocess(rk, first_window))
    second_window = merge(first_window, (; offset=7))
    @test binder(second_window) == rk_bound_data(reprocess(rk, second_window))
    @test length(coordinate_names(reprocess(rk, second_window).model.layout)) == 2
end
