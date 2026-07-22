# test/descriptor.jl — BRMDescriptor acceptance.
#
# Covers the five things the descriptor must be able to promise a consumer:
#   1. reflection            — inputs/outputs/operations derived from ONE declaration
#   2. stable identity       — same declaration, same id; different model, different id
#   3. schema propagation    — Stan inputs carry their dataframe column + transform
#   4. generative semantics  — BRM roles on the outputs, coefficient labels
#   5. execution             — BRM -> StanBlocks -> BridgeStan actually runs
# plus the fail-closed cases and the documented extension points.
#
# Run: julia --project=. test/descriptor.jl
# (BRM has no runnable test harness — see the primer; these are standalone.)

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

df = (; x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        g=[1, 1, 2, 2, 3, 3],
        y=[1.0, 1.5, 2.0, 2.4, 3.1, 3.4],
        z=[2.0, 2.5, 3.0, 3.4, 4.1, 4.4])

hier_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | g)
    y ~ Normal(mu, sigma)
    z ~ Normal(mu, 2 * sigma)
end

@testset "reflection — one declaration, derived surface" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)

    @test d isa BRMDescriptor
    @test d.name === :hier
    @test !isempty(d.id)

    # Operations are DERIVED, not listed. This model has parameters, a
    # likelihood, predictive draws and pointwise log-liks, and was built from a
    # reusable builder -> everything but :reprocess (it has a random effect).
    ops = Symbol[op.name for op in d.operations]
    @test :transpile in ops
    @test :instantiate in ops
    @test :fit in ops
    @test :predict in ops
    @test :pointwise_loglik in ops
    @test :replay in ops
    @test :reprocess ∉ ops          # ranef-bearing: reprocess is not supported
    @test isempty(d.unpredictable)  # both observations get a real *_gen

    # An operation's origin says which namespace its `inputs` live in.
    @test brm_operation(d, :fit).origin === :stan
    @test brm_operation(d, :replay).origin === :brm
    @test brm_operation(d, :replay).inputs == d.columns

    # :predict reports only draws the program actually emits.
    @test Set(brm_operation(d, :predict).outputs) == Set([:y_gen, :z_gen])

    # required inputs exclude the sizes StanBlocks derives from other inputs.
    req = required_brm_inputs(d)
    @test :y in req && :x in req && :g_idx in req
    @test :y_n ∉ req && :x_n ∉ req
end

@testset "stable identity" begin
    a = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)
    b = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:something_else)
    # Identity is content, not label.
    @test a.id == b.id
    @test a.name !== b.name
    # ...and it is the key the compiled artifact caches under.
    @test a.id == StanBlocks.stan_descriptor(a.plan.model).id

    flat_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    @test brm_descriptor(flat_builder, df; mod=@__MODULE__).id != a.id
end

@testset "schema propagation — Stan inputs carry their dataframe column" begin
    zs_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + zscale(x)
        y ~ Normal(mu, sigma)
    end
    d = brm_descriptor(zs_builder, df; mod=@__MODULE__)

    @test :x in d.columns          # a predictor, reached through zscale(...)
    @test :y in d.columns          # the response — a `~` op, not a DataColumn op
    @test brm_columns(d) === d.columns

    # The formula is DERIVED from the parsed declaration, so it cannot drift
    # from the model that runs — unlike a hand-kept `formula_src` string.
    @test occursin("zscale", d.formula)
    @test d.formula == sprint(show, d.plan.parent)

    byname = Dict(i.name => i for i in d.inputs)
    @test byname[:y].column === :y
    @test byname[:y].transform === nothing
    @test byname[:y].observed

    # The z-scaled predictor lands under a transformed key that still knows the
    # raw column it came from and how it was transformed.
    zs = [i for i in d.inputs if i.transform === :zscale]
    @test length(zs) == 1
    @test only(zs).column === :x

    # Sizes are `derived` -> a form must not ask for them.
    @test byname[:y_n].derived
    @test byname[:y_n].column === nothing
end

@testset "generative semantics — BRM roles and labels" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__)
    byname = Dict(o.name => o for o in d.outputs)

    @test byname[:sigma].role === :parameter
    @test byname[:sigma].declaration.family === :exponential

    # The population block and its internals both resolve to the popefs
    # declaration; the linear predictor comes from the formula, not a `~`.
    @test byname[:pop_mu].role === :population_effect
    @test byname[:pop_mu_beta_pop].role === :population_effect
    @test byname[:pop_mu_beta_pop].declaration.target === :pop_mu
    @test byname[:mu].role === :linear_predictor
    @test byname[:mu].declaration === nothing

    @test byname[:r_mu_g].role === :random_effect
    @test byname[:r_mu_g_xi].role === :random_effect

    # Predictive draws and pointwise log-liks are tagged from StanBlocks'
    # `source` link, never by parsing the `_gen` suffix.
    @test byname[:y_gen].role === :posterior_predictive
    @test byname[:y_gen].source === :y
    @test byname[:y_likelihood].role === :pointwise_loglik

    # Coefficient labels replace `pop_mu_beta_pop.N` guessing.
    @test byname[:pop_mu_beta_pop].labels == [:Intercept, :x]
    @test byname[:mu].labels === nothing
end

@testset "execution — BRM -> StanBlocks -> BridgeStan" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__)

    src = brm_execute(d, :transpile)
    @test src isa AbstractString
    @test occursin("y_gen", src)
    @test src == BayesianRegressionModels.stan_code(d.plan)

    prob = brm_execute(d, :fit)
    n = StanBlocks.LogDensityProblems.dimension(prob)
    @test n > 0
    theta = 0.1 .* randn(n)
    lp = StanBlocks.LogDensityProblems.logdensity(prob, theta)
    @test isfinite(lp)

    out = brm_execute(d, :predict; problem=prob, draws=theta, seed=1234)
    @test haskey(out, :y_gen)
    @test length(out.y_gen) == length(df.y)

    ll = brm_execute(d, :pointwise_loglik; problem=prob, draws=theta, seed=1234)
    @test length(ll.y_likelihood) == length(df.y)
end

@testset "replay — the same declaration on genuinely new groups" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)
    new_df = (; x=[0.0, 1.0, 2.0, 3.0],
                g=[7, 7, 8, 8],
                y=[0.2, 0.4, 0.6, 0.8],
                z=[1.2, 1.4, 1.6, 1.8])
    d2 = brm_execute(d, :replay, new_df)
    @test d2 isa BRMDescriptor
    @test d2.plan.data[:x] == new_df.x
    @test d2.plan.data[:n_g] == 2          # the new cohort has two groups
    @test Symbol[o.name for o in d2.outputs] == Symbol[o.name for o in d.outputs]
end

@testset "reprocess is offered exactly when it is supported" begin
    flat_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
        y ~ Normal(mu, sigma)
    end
    d = brm_descriptor(flat_builder, df; mod=@__MODULE__)
    @test :reprocess in Symbol[op.name for op in d.operations]

    d2 = brm_execute(d, :reprocess, (; x=[9.0, 10.0], y=[0.0, 0.0]))
    @test d2.plan.data[:x] == [9.0, 10.0]
end

StanBlocks.@deffun begin
    descriptor_kernel_cell(ts::vector[nt], dd::real, eta::vector[ne])::vector[nt] = begin
        CL = 1.0 * exp(eta[1])
        V = 10.0 * exp(eta[2])
        dd / V * exp(-(CL / V) * ts)
    end
end

kernel_builder = @brm begin
    sigma_a ~ Exponential(1)
    sigma_p ~ Exponential(1)
    pred ~ kernel(t, dose; by=subject, model=descriptor_kernel_cell, n_eta=2,
                  obs=CombinedError(dv, sigma_a, sigma_p))
end

kernel_schedule(n; subject=collect(1:n)) = (;
    t=[collect(1.0:3.0) for _ in 1:n],
    dose=fill(100.0, n),
    dv=[collect(1.0:3.0) ./ 10 for _ in 1:n],
    subject,
)

@testset "kernel(...) — a plate-nested observation" begin
    d = brm_descriptor(kernel_builder, kernel_schedule(3); mod=@__MODULE__, name=:pk)
    byname = Dict(o.name => o for o in d.outputs)

    # A declaration inside the plate cell resolves to the CONTEXT-JOINED Stan
    # name: `kernel_z` declared in the `pred` cell is the parameter
    # `pred_kernel_z`. Same join BRM already uses to build `draw`.
    @test haskey(byname, :pred_kernel_z)
    @test byname[:pred_kernel_z].declaration.target === :kernel_z
    @test byname[:pred_kernel_z].role === :group_block

    # The BSV block's SHARED captures are emitted at top level — one
    # correlation matrix and one scale vector for the whole model — so they
    # read as ordinary parameters, which is what they are. Nobody wrote them.
    @test byname[:kernel_L_pred].role === :parameter
    @test byname[:kernel_L_pred].declaration.family === :lkj_corr_cholesky
    @test byname[:kernel_om_pred].declaration.family === :std_normal
    @test byname[:sigma_a].role === :parameter

    # The plate's transformed-parameter carriers. StanBlocks dropped EVERY bare
    # declaration from `outputs` before 42fd83a (merged 825776e), so this set
    # was quietly short for every plate model — a gap in a place nobody was
    # looking, since the parameters it did report looked complete. Pin it.
    @test any(o -> o.name === :pred_kernel_eta, d.outputs)
    @test byname[:pred_kernel_eta].kind === :transformed_parameter
    @test byname[:pred_kernel_eta].role === :group_block
    @test !isempty([o for o in d.outputs
                    if o.kind === :transformed_parameter && o.role === :group_block])

    @test :dv in d.columns && :subject in d.columns && :dose in d.columns

    # The ragged plate-sliced observation is recognised as conditioned-on, and
    # the model is fittable. Both were briefly wrong upstream — the traced-`~`
    # walk stopped at the `getfield` in `dv.mem[start:end]` — and BRM carried
    # workarounds for them until `44f58fa` fixed the walk. These now assert
    # PLAIN DELEGATION: if a future regression reintroduces that blind spot,
    # these fail rather than being silently absorbed.
    @test Dict(i.name => i for i in d.inputs)[:dv].observed

    ops = Symbol[op.name for op in d.operations]
    @test :fit in ops
    prob = brm_execute(d, :fit)
    n = StanBlocks.LogDensityProblems.dimension(prob)
    @test n > 0
    @test isfinite(StanBlocks.LogDensityProblems.logdensity(prob, 0.1 .* randn(n)))

    # :reprocess and :replay are OFFERED on a kernel model, so they must RUN.
    # The reprocess gate keys on the absence of a `ranef_*` block, and a
    # kernel's eta block is a plate rather than a ranef — so a kernel model
    # sails through that gate. That is only correct if reprocess genuinely
    # supports a plate model; offering one that errors would be this
    # descriptor committing the exact failure it exists to prevent.
    @test brm_execute(d, :reprocess, kernel_schedule(3)) isa BRMDescriptor
    @test brm_execute(d, :replay, kernel_schedule(5)) isa BRMDescriptor

    # Whether a plate-nested observation gets a predictive draw is StanBlocks'
    # call, not BRM's — a ragged observation base gets no `_gen`
    # (stanblocks-use §9). The descriptor reports whichever is true
    # CONSISTENTLY: either the draw exists and :predict is offered, or it does
    # not, the observation is named in `unpredictable`, and no button appears.
    if isempty(d.unpredictable)
        @test :predict in ops
        @test !isempty(brm_operation(d, :predict).outputs)
    else
        @test :predict ∉ ops
        @test :kernel_y in d.unpredictable
    end
end

StanBlocks.@deffun begin
    descriptor_scalar_cell(dd::real, eta::vector[ne])::real =
        (dd / 10.0) * exp(eta[1])
end

scalar_kernel_builder = @brm begin
    sigma_a ~ Exponential(1)
    sigma_p ~ Exponential(1)
    pred ~ kernel(dose; by=subject, model=descriptor_scalar_cell, n_eta=2,
                  obs=CombinedError(dv, sigma_a, sigma_p))
end

scalar_schedule(n) = (; dose=fill(100.0, n), dv=collect(1.0:n) ./ 10,
                        subject=collect(1:n))

# A plate observation on a NON-RAGGED (scalar-per-subject) base. Stan really
# does emit `<data_source>_gen` here, but `stan_descriptor` collects neither a
# bare gq declaration nor a for-loop fill, so the twin is absent from its
# `outputs` (StanBlocks snag `descriptor-misse-1149b397`).
#
# This closes the question brm-use §3 leaves explicitly open ("I did NOT verify
# `:predict` on a plate model against it"). What must hold is the INVARIANT,
# not either branch of it: a `:predict` that is offered must be executable, and
# an observation with no reachable draw must be named in `unpredictable`. That
# way this test keeps holding when the snag lands and the branch flips.
@testset "non-ragged plate observation — :predict stays consistent" begin
    d = brm_descriptor(scalar_kernel_builder, scalar_schedule(4);
                       mod=@__MODULE__, name=:pk_scalar)
    ops = Symbol[op.name for op in d.operations]

    @test :fit in ops
    if :predict in ops
        # offered => there is a real draw behind it, and it runs
        @test !isempty(brm_operation(d, :predict).outputs)
        @test isempty(d.unpredictable)
        prob = brm_execute(d, :fit)
        n = StanBlocks.LogDensityProblems.dimension(prob)
        out = brm_execute(d, :predict; problem=prob, draws=0.1 .* randn(n), seed=7)
        @test !isempty(keys(out))
    else
        # withheld => the observation is named, so nobody silently loses it
        @test !isempty(d.unpredictable)
        @test :kernel_y in d.unpredictable
        @test_throws ErrorException brm_execute(d, :predict)
    end

    @test brm_execute(d, :reprocess, scalar_schedule(4)) isa BRMDescriptor
end

@testset "fail closed" begin
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__, name=:hier)

    # 1. An operation the model does not offer errors and NAMES the ones it does.
    err = try; brm_operation(d, :simulate); catch e; e; end
    @test err isa ErrorException
    @test occursin("does not offer operation `simulate`", err.msg)
    @test occursin("predict", err.msg)
    @test_throws ErrorException brm_execute(d, :simulate)

    # 2. Suppressing an operation that was never derived is loud — a caller
    #    doing that is holding exactly the stale list this type removes.
    @test_throws ErrorException brm_descriptor(
        hier_builder, df; mod=@__MODULE__, operations=Dict(:simulate => nothing))

    # 3. Retitling an operation that is not offered is loud for the same reason.
    @test_throws ErrorException brm_descriptor(
        hier_builder, df; mod=@__MODULE__, titles=Dict(:simulate => "nope"))

    # 4. A model with no predictive draw is not offered :predict at all,
    #    instead of offering one that would return nothing usable.
    prior_only = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x
    end
    dp = brm_descriptor(prior_only, df; mod=@__MODULE__)
    @test :predict ∉ Symbol[op.name for op in dp.operations]
    @test :transpile in Symbol[op.name for op in dp.operations]
end

@testset "extension points" begin
    # (a) add an operation the derivation cannot know about
    d = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                       operations=Dict(:summarise => (dd; kwargs...) -> length(dd.outputs)))
    @test :summarise in Symbol[op.name for op in d.operations]
    @test brm_operation(d, :summarise).origin === :override
    @test brm_execute(d, :summarise) == length(d.outputs)

    # (b) replace a derived operation's behaviour, keeping its identity
    d2 = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                        operations=Dict(:transpile => (dd; kwargs...) -> "STUB"))
    @test brm_execute(d2, :transpile) == "STUB"
    @test brm_operation(d2, :transpile).origin === :override

    # (c) suppress a derived operation — the button simply is not there
    d3 = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                        operations=Dict(:pointwise_loglik => nothing))
    @test :pointwise_loglik ∉ Symbol[op.name for op in d3.operations]
    @test :fit in Symbol[op.name for op in d3.operations]

    # (d) relabel
    d4 = brm_descriptor(hier_builder, df; mod=@__MODULE__,
                        titles=Dict(:fit => "Run the sampler"))
    @test brm_operation(d4, :fit).title == "Run the sampler"
    @test brm_operation(d4, :fit).origin === :stan   # still the derived runner
end
