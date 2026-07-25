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

kernel_builder = @brm begin
    sigma_a ~ Exponential(1)
    sigma_p ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, dv, log_CL, log_V) do ts, dd, yy, lCL, lV
        CL = exp(lCL)
        V = exp(lV)
        mu = dd / V * exp(-(CL / V) * ts)
        yy ~ normal(mu, addprop(mu, sigma_a, sigma_p))
        mu
    end
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

    # Kernel v2 owns BSV through ordinary formula declarations. The population
    # blocks and the shared `|p|` random-effect block therefore retain their
    # ordinary BRM roles; no anonymous kernel-owned L/om/z block exists.
    @test byname[:pop_log_CL_beta_pop].role === :population_effect
    @test byname[:pop_log_V_beta_pop].role === :population_effect
    @test byname[:b_p_subject_L].role === :random_effect
    @test byname[:b_p_subject_L].declaration.family === :ranef_correlated_draws
    @test byname[:b_p_subject_tau].role === :random_effect
    @test !any(o -> startswith(string(o.name), "kernel_L_") ||
                   startswith(string(o.name), "kernel_om_") ||
                   startswith(string(o.name), "kernel_z"), d.outputs)
    @test byname[:sigma_a].role === :parameter

    # The plate's transformed-parameter carriers and cell locals resolve to the
    # plate declaration, independent of the generated carrier suffix.
    plate_outputs = [o for o in d.outputs
                     if o.kind === :transformed_parameter && o.role === :group_block]
    @test !isempty(plate_outputs)
    @test all(o -> o.declaration.target === :pred, plate_outputs)
    @test any(o -> startswith(string(o.name), "pred__pl_mem_"), plate_outputs)
    @test byname[:pred_CL].role === :group_block
    @test byname[:pred_V].role === :group_block

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

    # A builder-backed kernel can rebuild for genuinely new groups. It cannot
    # reprocess in place because its formula-declared BSV is a random-effect
    # block, exactly the documented reprocess boundary.
    @test :replay in ops
    @test :reprocess ∉ ops
    @test brm_execute(d, :replay, kernel_schedule(5)) isa BRMDescriptor

    # A ragged plate observation has no declarable generated-quantity twin.
    # BRM therefore names it in `unpredictable` and withholds `:predict`.
    @test :predict ∉ ops
    @test d.unpredictable == (:yy,)
end

scalar_kernel_builder = @brm begin
    sigma ~ Exponential(1)
    log_scale ~ 1 + (1 | p | subject)
    pred ~ kernel(dose, dv, log_scale) do dd, yy, ls
        mu = (dd / 10.0) * exp(ls)
        yy ~ normal(mu, sigma)
        mu
    end
end

scalar_schedule(n) = (; dose=fill(100.0, n), dv=collect(1.0:n) ./ 10,
                        subject=collect(1:n))

# A plate observation on a NON-RAGGED (scalar-per-subject) base. StanBlocks
# emits and describes the `<data_source>_gen` twin, so BRM must offer a
# prediction operation that resolves through the upstream `source` link.
@testset "non-ragged plate observation — :predict stays consistent" begin
    d = brm_descriptor(scalar_kernel_builder, scalar_schedule(4);
                       mod=@__MODULE__, name=:pk_scalar)
    ops = Symbol[op.name for op in d.operations]

    @test :fit in ops
    @test :predict in ops
    @test brm_operation(d, :predict).outputs == (:dv_gen,)
    @test isempty(d.unpredictable)
    prob = brm_execute(d, :fit)
    n = StanBlocks.LogDensityProblems.dimension(prob)
    out = brm_execute(d, :predict; problem=prob, draws=0.1 .* randn(n), seed=7)
    @test haskey(out, :dv_gen)

    @test :reprocess ∉ ops
    @test brm_execute(d, :replay, scalar_schedule(5)) isa BRMDescriptor
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
