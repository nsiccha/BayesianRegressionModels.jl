using Test, BayesianRegressionModels, StanBlocks, BridgeStan
using TreeArrays, NaNStatistics, Tables, AlgebraOfVega
using WarmupHMC, Random, Statistics, LinearAlgebra, LogDensityProblems, Treebars
using ReactiveKernels, ReactiveKernelsPPL

const BRM = BayesianRegressionModels
const BS = BridgeStan
const DATA = (; x=[-1.0, -0.7, -0.3, 0.1, 0.4, 0.8, 1.1],
    g=["west", "east", "west", "north", "east", "west", "north"],
    h=["B", "A", "A", "B", "A", "B", "B"],
    y=[-0.7, -0.5, -0.2, 0.2, 0.1, 0.6, 0.9])
const MODEL = @brm begin
    sd(:, g) = 0.3
    sd(:, h) = 0.2
    mu ~ 1 + x + (1 + x | g) + (1 | h)
    y ~ Normal(mu, 0.4)
end

function fitted_fixture()
    sb = SBBRMI(MODEL(DATA); mod=@__MODULE__, total_groups=())
    d = brm_descriptor(sb)
    problem = stan_instantiate(sb)
    n = LogDensityProblems.dimension(problem)
    fits = with_progress(:state; description="TreeArrays fit fixture") do p
        [adaptive_warmup_mcmc(Xoshiro(seed), problem;
            init=(; position=zeros(n), squared_scale=ones(n)),
            n_draws=24, n_evaluations=120, stepsize_adaptation_limit=20,
            nonlinear_adapt=false, max_tree_depth=6, progress=p, monitor_ess=false)
         for seed in (17, 23)]
    end
    U = [permutedims(f.posterior_position) for f in fits]
    names = BS.param_names(problem.model; include_tp=true, include_gq=true)
    C = map(U) do draws
        rng = BS.StanRNG(problem.model, 4)
        permutedims(reduce(hcat, [BS.param_constrain(problem.model, collect(q);
            include_tp=true, include_gq=true, rng) for q in eachrow(draws)]))
    end
    (; d, problem, U, C, names)
end

@testset "scalar and explicitly indexed output storage" begin
    sb = SBBRMI(@brm (; y=[0.1, 0.2]) begin
        theta ~ Normal(0, 1)
        q[1:2] .~ Normal.(0, 1)
        y ~ Normal(theta + q[1], 1)
    end; mod=@__MODULE__)
    d = brm_descriptor(sb)
    C = reshape(collect(-0.4:0.1:0.4), 3, 3)
    names = ["theta", "q.1", "q.2"]
    tree = brm_output_tree(d, C, names; logical=:theta)
    @test parent(tree) == C[:, 1]
    @test parent(parent(tree)) === C
    @test coords(tree, :draw) == 1:3
    vector = brm_output_tree(d, C, names; logical=:q)
    @test parent(vector) == C[:, 2:3]
    @test parent(parent(vector)) === C
    @test coords(vector, :element) == 1:2
end

@testset "RK values retain individual draw storage" begin
    rk = RKBRMI(@brm (; x=[0.2, -0.3, 0.4], y=[0.1, 0.2, -0.1]) begin
        theta ~ Normal(0, 1)
        mu = theta .+ x
        y ~ Normal(mu, 1)
    end)
    U = reshape([-0.1, 0.2, 0.5], 1, :)
    raw = rk_draw_values(rk, U, (:theta, :mu))
    saved = rk_draw_tree(raw)
    @test parent(saved.theta) === raw.theta
    @test all(parent(leaf) === value for (leaf, value) in zip(parent(saved.mu), raw.mu))
    @test coords(saved.theta, :draw) == 1:3
    tree = rk_draw_tree(rk, U, (:theta, :mu))
    @test parent(tree.theta) == raw.theta
    @test [parent(leaf) for leaf in parent(tree.mu)] == raw.mu
    @test coords(first(parent(tree.mu)), :element) == 1:3
    chains = rk_draw_tree(rk, [U, view(U, :, 1:2)], (:theta,))
    @test coords(chains, :chain) == 1:2
    @test parent(parent(chains)[2].theta) == raw.theta[1:2]
end

@testset "draw trees over a fitted hierarchical model" begin
    (; d, problem, U, C, names) = fitted_fixture()
    @test all(all(isfinite, c) for c in C)
    original = copy(C[1])
    tree = brm_output_tree(d, C[1], names; logical=:mu)
    @test parent(parent(tree)) === C[1]
    @test parent(tree) == brm_output_draws(d, C[1], names; logical=:mu)
    @test coords(tree, :observation) == 1:length(DATA.y)
    @test C[1] == original

    # Permuting a caller's coordinates must permute its labels with them.
    order = reverse(eachindex(names))
    reversed = brm_output_tree(d, view(C[1], :, order), names[order]; logical=:mu)
    @test coords(reversed, :observation) == reverse(1:length(DATA.y))
    @test parent(reversed) == reverse(parent(tree); dims=2)
    beta = only(o for o in d.outputs if o.component === :coefficients && !isnothing(o.labels))
    coefficients = brm_output_tree(d, C[1], names; logical=beta)
    @test coords(coefficients, :coefficient) == beta.labels
    @test Base.mightalias(parent(coefficients), C[1])
    @test_throws ArgumentError brm_output_tree(d, C[1], names; logical=beta, role=:membership)
    # Every sampled parameter of this model is reachable for diagnostics.
    for output in brm_outputs(d; kind=:parameter)
        post = brm_output_tree(d, C, names; logical=output)
        numeric = TreeActualArray(post; dims=:draw, align=:truncate_min)
        selected = brm_output_coordinates(output, names)
        @test sort(vec(reshape(Array(numeric), size(numeric, 1), 2, :)[:, 1, :])) ==
              sort(vec(C[1][1:size(numeric, 1), selected]))
    end

    bands = nanquantile(tree, (lo=0.025, mid=0.5, hi=0.975); dims=:draw)
    columns = Tables.columntable(TreeTable(bands))
    @test columns.observation == 1:length(DATA.y)
    for (key, p) in pairs((lo=0.025, mid=0.5, hi=0.975))
        @test getproperty(columns, key) ≈ [quantile(parent(tree)[:, j], p) for j in axes(parent(tree), 2)] atol=1e-12 rtol=1e-12
    end
    @test AlgebraOfVega.data(TreeTable(bands)) !== nothing

    chain_inputs = [C[1], view(C[2], 1:2, :)]
    chains = brm_output_tree(d, chain_inputs, names; logical=:mu)
    @test coords(chains, :chain) == 1:2
    @test Base.mightalias(parent(parent(chains)[2]), C[2])
    pooled = nanquantile(chains, (mid=0.5,); dims=(:chain, :draw))
    pooled_columns = Tables.columntable(TreeTable(pooled))
    @test pooled_columns.mid == [median(vcat(parent(tree)[:, j], parent(parent(chains)[2])[:, j]))
                                for j in axes(parent(tree), 2)]
    numeric = TreeActualArray(chains; dims=:draw, align=:truncate_min)
    @test size(numeric) == (2, 2, length(DATA.y))
    @test numeric[1, 2, 3] == parent(parent(chains)[2])[1, 3]

    # Different grouping cardinalities remain ragged; no padding or stacking.
    raw = brm_ranef_draws(d, C[1], names)
    ranef = brm_ranef_tree(d, C[1], names)
    saved_ranef = brm_ranef_tree(raw)
    @test parent(first(parent(saved_ranef)).effects) === first(raw).effects
    @test parent(first(parent(saved_ranef)).sd) === first(raw).sd
    @test parent(first(parent(saved_ranef)).cor) === first(raw).cor
    @test length(parent(ranef)) == 2
    @test sort(length.(getproperty.(raw, :levels))) == [2, 3]
    @test sort(length.(getproperty.(raw, :margins))) == [1, 2]
    for (entry, leaf) in zip(raw, parent(ranef))
        @test coords(leaf.effects, :level) == entry.levels
        @test coords(leaf.effects, :margin) == entry.margins
        @test coords(leaf.cor, :margin2) == entry.margins
        @test parent(leaf.effects) == entry.effects
        @test parent(leaf.sd) == entry.sd
        @test parent(leaf.cor) == entry.cor
        if length(entry.margins) == 2
            # This two-margin block emits its effects as an existing carrier.
            @test Base.mightalias(parent(leaf.effects), C[1])
        end
        @test !Base.mightalias(entry.effects, C[1])
        # Compiled Stan's effects are an independent oracle for the sampled
        # standardized carrier's level/margin placement.
        output = only(o for o in brm_outputs(d; component=:standardized, kind=:parameter)
                      if o.declaration.target === entry.binding)
        z = brm_output_tree(d, C[1], names; logical=output)
        @test coords(z, :level) == entry.levels
        if length(entry.margins) == 1
            @test parent(z) .* entry.sd ≈ entry.effects[:, :, 1]
        else
            for i in axes(C[1], 1)
                covariance_factor = Diagonal(entry.sd[i, :]) * cholesky(Symmetric(entry.cor[i, :, :])).L
                @test parent(z)[i, :, :] * covariance_factor' ≈ entry.effects[i, :, :]
            end
        end
    end
    @test coords(brm_ranef_tree(d, C, names), :chain) == 1:2

    predicted = brm_predictive_draws(d, U[1]; problem, seed=10)
    predictive = brm_predictive_tree(d, predicted)
    @test parent(predictive.y) === predicted.y
    @test coords(predictive.y, :observation) == 1:length(DATA.y)
    @test parent(brm_predictive_tree(d, U[1]; problem, seed=10).y) == predicted.y
    predictive_chains = brm_predictive_tree(d, U; problem, seed=10)
    @test coords(predictive_chains, :chain) == 1:2
    @test parent(parent(predictive_chains)[1].y) == predicted.y

    grid = brm_prediction_grid(d; focal=[:x => [-0.5, 0.5]], fixed=(; g="west", h="B"))
    conditional = brm_conditional_draws(d, U[1], grid; focal=:x)
    labelled = brm_conditional_tree(conditional)
    @test parent(labelled.draws) === conditional.draws
    @test coords(labelled.draws, :grid) == [(; x=-0.5), (; x=0.5)]
    @test parent(brm_conditional_tree(d, U[1], grid; focal=:x,
                                    problem=conditional.problem).draws) == conditional.draws
    conditional_chains = brm_conditional_tree(d, U, grid; focal=:x,
                                             problem=conditional.problem)
    @test coords(conditional_chains.draws, :chain) == 1:2
    @test parent(first(parent(conditional_chains.draws))) == conditional.draws

    # Construction costs depend on metadata, never on the number of draws.
    backing = zeros(100_000, length(names))
    brm_output_tree(d, backing, names; logical=:mu)
    bytes = @allocated brm_output_tree(d, backing, names; logical=:mu)
    @info "zero-copy output construction" bytes backing_bytes=sizeof(backing)
    @test bytes < 100_000
    @test bytes < sizeof(backing) ÷ 10
end
