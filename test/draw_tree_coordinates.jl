using Test, BayesianRegressionModels, StanBlocks, TreeArrays, Random
using MCMCDiagnosticTools

@test !isnothing(Base.get_extension(BayesianRegressionModels, :BayesianRegressionModelsTreeArraysExt))

const DATA = (; x=[-0.8, 0.2, 0.7, -0.3, 1.1, 0.4],
    g=["west", "east", "north", "west", "east", "north"],
    by=["B", "A", "B", "B", "A", "B"], y=[0.1, 0.4, 0.8, 0.2, 0.3, 0.7])

function carrier_fixture(d)
    # Constrained names use column-major indices of each descriptor shape.
    # Distinct values expose permutations; no density or sampling is needed
    # for an adapter that only labels existing constrained storage.
    names = String[]
    for o in d.outputs
        isnothing(o.dims) && continue
        if isempty(o.dims)
            push!(names, string(o.name))
        else
            append!(names, [string(o.name, '.', join(Tuple(i), '.'))
                            for i in CartesianIndices(o.dims)])
        end
    end
    draws = randn(Xoshiro(8), 64, length(names))
    (; names, draws)
end

@testset "row assignments inherit observation coordinates" begin
    model = @brm begin
        t_loc ~ Normal(0, 1)
        q[1:6] .~ Normal.(0, 1)
        dev ~ 0 + (1 | g)
        sd(dev, g) = 3.0
        log_b ~ 1 + x
        log_c ~ 1
        log_T = t_loc + dev
        mu_dbs = log(exp(log_c) + exp(log_T + log_b))
        sig_dbs = exp(mu_dbs / 2)
        same_length = exp(q)
        reordered = mu_dbs[permutation]
        y ~ Normal(mu_dbs + q[1], sig_dbs)
    end
    data = (; DATA..., permutation=collect(6:-1:1))
    d = brm_descriptor(SBBRMI(model(data); mod=@__MODULE__, total_groups=()))
    (; names, draws) = carrier_fixture(d)
    outputs = [brm_output(d, name) for name in (:log_T, :mu_dbs, :sig_dbs)]
    append!(outputs, brm_outputs(d; component=:contribution))
    @test length(outputs) > 3
    for output in outputs
        tree = brm_output_tree(d, draws, names; logical=output)
        @test coords(tree, :observation) == 1:6
        @test parent(tree) == draws[:, brm_output_coordinates(output, names)]
        @test Base.mightalias(parent(tree), draws)
        order = reverse(eachindex(names))
        reversed = brm_output_tree(d, view(draws, :, order), names[order]; logical=output)
        @test coords(reversed, :observation) == 6:-1:1
        @test parent(reversed) == reverse(parent(tree); dims=2)
    end
    # These vectors have the same length but no observation-preserving path.
    for name in (:same_length, :reordered)
        @test brm_output(d, name).dims == (6,)
        @test_throws ArgumentError brm_output_tree(d, draws, names; logical=name)
    end
end

function check_parameters(d)
    (; names, draws) = carrier_fixture(d)
    order = randperm(Xoshiro(9), length(names))
    input = view(draws, :, order)
    snapshot = copy(draws)
    for output in brm_outputs(d; kind=:parameter)
        tree = brm_output_tree(d, draws, names; logical=output)
        @test Base.mightalias(parent(tree), draws)
        shuffled = brm_output_tree(d, input, names[order]; logical=output)
        if !isnothing(output.component)
            @test parent(shuffled) == parent(tree)
        end
        chains = brm_output_tree(d, [draws, view(draws, 1:48, :)], names; logical=output)
        actual = TreeActualArray(chains; dims=:draw, align=:truncate_min)
        @test size(actual, 1) == 48
        @test size(actual, 2) == 2
        columns = brm_output_coordinates(output, names)
        flat = reshape(Array(actual), 48, 2, :)
        @test sort(vec(flat[:, 1, :])) == sort(vec(draws[1:48, columns]))
        # The same diagnostics as the app's column-selection workaround.
        reference = cat(draws[1:48, columns], draws[1:48, columns]; dims=3)
        reference = PermutedDimsArray(reference, (1, 3, 2))
        diagnostic = reshape(actual, 48, 2, :)
        @test sort(vec(MCMCDiagnosticTools.rhat(diagnostic))) ≈
              sort(vec(MCMCDiagnosticTools.rhat(reference)))
        @test sort(vec(MCMCDiagnosticTools.ess(diagnostic; kind=:bulk))) ≈
              sort(vec(MCMCDiagnosticTools.ess(reference; kind=:bulk)))
        @test sort(vec(MCMCDiagnosticTools.ess(diagnostic; kind=:tail))) ≈
              sort(vec(MCMCDiagnosticTools.ess(reference; kind=:tail)))
    end
    @test draws == snapshot
    (; names, draws)
end

@testset "known-SD random intercept has fitted level labels" begin
    model = @brm begin
        dev ~ 0 + (1 | g)
        sd(dev, g) = 3.0
        y ~ Normal(dev, 1.0)
    end
    data = (; g=["sample_$i" for i in 69:-1:1], y=zeros(69))
    d = brm_descriptor(SBBRMI(model(data); mod=@__MODULE__, total_groups=()))
    (; names, draws) = check_parameters(d)
    output = only(brm_outputs(d; component=:standardized, kind=:parameter))
    tree = brm_output_tree(d, draws, names; logical=output)
    @test coords(tree, :level) == sort(data.g)
    @test size(parent(tree)) == (64, 69)
    @test parent(tree) == draws[:, brm_output_coordinates(output, names)]
    @test_throws DimensionMismatch brm_output_tree(d, draws[:, 1:end-1], names; logical=output)
end

@testset "correlated block coordinates follow carrier storage" begin
    model = @brm begin
        mu ~ 1 + (1 + x | g)
        y ~ Normal(mu, 1.0)
    end
    for centered in ((), (:g,))
        d = brm_descriptor(SBBRMI(model(DATA); mod=@__MODULE__, total_groups=(), centered_groups=centered))
        (; names, draws) = check_parameters(d)
        block = only(ranef_blocks(d))
        output = only(o for o in brm_outputs(d; kind=:parameter) if o.name === block.z)
        tree = brm_output_tree(d, draws, names; logical=output)
        @test coords(tree, :level) == ["east", "north", "west"]
        @test coords(tree, :margin) == [(predictor=:mu, coefficient=:Intercept),
                                       (predictor=:mu, coefficient=:x)]
        @test size(parent(tree)) == (64, 3, 2)
        for g in 1:3, k in 1:2
            suffix = isempty(centered) ? string(k + 2(g - 1)) : string(g, '.', k)
            column = findfirst(==(string(output.name, '.', suffix)), names)
            @test parent(tree)[:, g, k] == draws[:, column]
        end
    end
end

@testset "stratified, known, Student-t and shared block parameters" begin
    models = [
        @brm(begin
            mu ~ 1 + (1 + x | p | gr(g, by=by))
            y ~ Normal(mu, 1.0)
        end),
        @brm(begin
            mu ~ 1 + (1 + x | p | g)
            sd(mu, p, Intercept) = 0.4
            y ~ Normal(mu, 1.0)
        end),
        @brm(begin
            mu ~ 1 + (1 + x | p | g)
            ranef(:, p) ~ TDist(5)
            y ~ Normal(mu, 1.0)
        end),
        @brm(begin
            mu ~ 1 + (1 | p | g)
            log(sigma) ~ 1 + (1 | p | g)
            y ~ Normal(mu, sigma)
        end),
        @brm(begin
            mu ~ 1 + (1 + x | p | g)
            nu ~ 1 + (1 | p | g)
            y ~ Normal(mu, 1.0)
            z ~ Normal(nu, 1.0)
        end),
    ]
    for (i, model) in enumerate(models)
        d = brm_descriptor(SBBRMI(model(DATA); mod=@__MODULE__, total_groups=()))
        (; names, draws) = check_parameters(d)
        if i == 1
            margins = [(predictor=:mu, coefficient=:Intercept), (predictor=:mu, coefficient=:x)]
            for output in brm_outputs(d; kind=:parameter)
                output.component in (:scale, :correlation_cholesky) || continue
                tree = brm_output_tree(d, draws, names; logical=output)
                @test coords(tree, :margin) == margins
                @test coords(tree, :stratum) == ["A", "B"]
                if output.component === :scale
                    for s in 1:2, k in 1:2
                        column = findfirst(==(string(output.name, '.', k, '.', s)), names)
                        @test parent(tree)[:, k, s] == draws[:, column]
                    end
                else
                    @test coords(tree, :margin2) == margins
                    for s in 1:2, k in 1:2, j in 1:2
                        column = findfirst(==(string(output.name, '.', s, '.', k, '.', j)), names)
                        @test parent(tree)[:, s, k, j] == draws[:, column]
                    end
                end
            end
        elseif i == 2
            scale = only(brm_outputs(d; kind=:parameter, component=:scale))
            @test coords(brm_output_tree(d, draws, names; logical=scale), :margin) ==
                  [(predictor=:mu, coefficient=:x)]
        elseif i == 4
            z = only(brm_outputs(d; kind=:parameter, component=:standardized))
            @test coords(brm_output_tree(d, draws, names; logical=z), :margin) ==
                  [(predictor=:mu, coefficient=:Intercept), (predictor=:sigma, coefficient=:Intercept)]
        end
    end
end
