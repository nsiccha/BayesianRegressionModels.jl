using Test, BayesianRegressionModels, StanBlocks, BridgeStan
using TreeArrays, Statistics, Tables, LogDensityProblems

const RAGGED_TOPLEVEL = @brm begin
    theta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        mu[i] = theta .+ x[i]
    end
    y ~ Normal(mu, 0.5)
end

const RAGGED_IN_CELL = @brm begin
    theta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        y[i] ~ Normal(theta .+ x[i], 0.5)
    end
end

const RAGGED_DATA = (; x=[[-0.4, 0.2], [0.1, 0.8, 1.2]],
                       y=[[-0.1, 0.3], [0.4, 0.7, 1.4]])
const RAGGED_BANDS = (lo=0.025, mid=0.5, hi=0.975)

function check_ragged_bands(tree, matrices, positions; dims)
    columns = Tables.columntable(TreeTable(quantile(tree, RAGGED_BANDS; dims)))
    @test columns.observation == positions
    for (key, p) in pairs(RAGGED_BANDS)
        expected = [quantile(vcat((m[:, j] for m in matrices)...), p)
                    for j in axes(first(matrices), 2)]
        @test getproperty(columns, key) ≈ expected atol=1e-12 rtol=1e-12
    end
end

function check_ragged_outputs(sb, data)
    d = brm_descriptor(sb)
    predictive = brm_output(d, :y; role=:posterior_predictive)
    likelihood = brm_output(d, :y; role=:pointwise_loglik)
    ends = cumsum(length.(data.x))
    @test predictive.segments == likelihood.segments == ends
    @test predictive.dims == (sum(length, data.x),)
    @test likelihood.dims == (length(data.x),)

    problem = stan_instantiate(sb)
    @test LogDensityProblems.dimension(problem) == 1
    U = reshape([-0.3, 0.1, 0.4, 0.7], :, 1)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    rng = BridgeStan.StanRNG(problem.model, 42)
    C = permutedims(reduce(hcat, [BridgeStan.param_constrain(problem.model,
        collect(q); include_tp=true, include_gq=true, rng) for q in eachrow(U)]))
    original = copy(C)

    for role in (:posterior_predictive, :pointwise_loglik)
        n = role === :posterior_predictive ? sum(length, data.x) : length(data.x)
        raw = brm_output_draws(d, C, names; logical=:y, role)
        tree = brm_output_tree(d, C, names; logical=:y, role)
        @test size(raw) == (size(U, 1), n)
        @test parent(tree) == raw
        @test parent(parent(tree)) === C
        @test coords(tree, :observation) == 1:n
        check_ragged_bands(tree, [raw], 1:n; dims=:draw)

        # Numeric element order must survive arbitrary carrier interleaving.
        order = vcat(collect(2:2:length(names)), collect(1:2:length(names)))
        reversed = reverse(order)
        for permutation in (order, reversed)
            permuted = view(C, :, permutation)
            permuted_names = names[permutation]
            moved = brm_output_tree(d, permuted, permuted_names; logical=:y, role)
            original_columns = brm_output_coordinates(d, :y, names; role)
            lookup = Dict(column => j for (j, column) in enumerate(original_columns))
            positions = [lookup[column] for column in permutation if haskey(lookup, column)]
            @test coords(moved, :observation) == positions
            @test parent(moved) == raw[:, positions]
            @test Base.mightalias(parent(moved), C)
            check_ragged_bands(moved, [raw[:, positions]], positions; dims=:draw)
        end

        inputs = [C, view(C, 1:2, :)]
        chains = brm_output_tree(d, inputs, names; logical=:y, role)
        @test coords(chains, :chain) == 1:2
        @test size.(parent.(parent(chains)), 1) == [4, 2]
        @test all(Base.mightalias(parent(leaf), C) for leaf in parent(chains))
        check_ragged_bands(chains, [raw, raw[1:2, :]], 1:n; dims=(:chain, :draw))

        # The descriptor's role-specific width is checked even when names
        # still describe a complete but undersized vector.
        last_column = last(brm_output_coordinates(d, :y, names; role))
        keep = setdiff(eachindex(names), [last_column])
        @test_throws DimensionMismatch brm_output_tree(d, view(C, :, keep), names[keep];
                                                       logical=:y, role)
    end

    predicted = brm_predictive_draws(d, U; problem, seed=17)
    tree = brm_predictive_tree(d, predicted)
    @test parent(tree.y) === predicted.y
    @test coords(tree.y, :observation) == 1:last(ends)
    @test parent(brm_predictive_tree(d, U; problem, seed=17).y) == predicted.y
    check_ragged_bands(tree.y, [predicted.y], 1:last(ends); dims=:draw)
    @test_throws DimensionMismatch brm_predictive_tree(d, (; y=predicted.y[:, 1:end-1]))
    @test_throws DimensionMismatch brm_predictive_tree(d, (; y=zeros(4, length(ends))))

    inputs = [U, view(U, 1:2, :)]
    chains = brm_predictive_tree(d, inputs; problem, seed=17)
    expected_second = brm_predictive_draws(d, inputs[2]; problem, seed=21)
    @test coords(chains, :chain) == 1:2
    @test parent(parent(chains)[1].y) == predicted.y
    @test parent(parent(chains)[2].y) == expected_second.y
    check_ragged_bands(chains.y, [predicted.y, expected_second.y], 1:last(ends);
                      dims=(:chain, :draw))
    @test C == original

    # Wrapping a large draw buffer allocates only coordinates and views.
    backing = zeros(100_000, length(names))
    brm_output_tree(d, backing, names; logical=:y, role=:posterior_predictive)
    bytes = @allocated brm_output_tree(d, backing, names;
                                       logical=:y, role=:posterior_predictive)
    @test bytes < 100_000
    @test bytes < sizeof(backing) ÷ 10
end

@testset "ragged draw trees: top-level and in-cell responses" begin
    for builder in (RAGGED_TOPLEVEL, RAGGED_IN_CELL)
        sb = SBBRMI(builder(RAGGED_DATA); mod=@__MODULE__)
        check_ragged_outputs(sb, RAGGED_DATA)
    end
end

@testset "ragged draw trees: changed design and omitted response" begin
    design = (; x=[[0.1, 0.3, 0.6], [-0.2], [0.8, 1.0]])
    for builder in (RAGGED_TOPLEVEL, RAGGED_IN_CELL)
        fitted = SBBRMI(builder(RAGGED_DATA); mod=@__MODULE__)
        replayed = reprocess(fitted, design)
        d = brm_descriptor(replayed)
        output = brm_output(d, :y; role=:posterior_predictive)
        @test output.segments == [3, 4, 6]
        @test output.dims == (6,)
        values = reshape(collect(1.0:18.0), 3, 6)
        tree = brm_predictive_tree(d, (; y=values))
        @test parent(tree.y) === values
        @test coords(tree.y, :observation) == 1:6
        check_ragged_bands(tree.y, [values], 1:6; dims=:draw)
    end
end
