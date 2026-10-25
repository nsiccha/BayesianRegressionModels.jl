# test/effect_prior_vector_arguments.jl — one prior per coefficient from data:
# `effect(lp, block) .~ Family.(args...)` (decision `1xaf4fx`, snag
# `brm-data-field-i-62ec2369`).
#
# An addressed prior reaching several coefficients took one scalar law, and an
# array-valued argument was refused on both backends. The broadcast spelling a
# declared vector uses (`x[1:n] .~ Family.(...)`, decision `19ssmud`) now reads
# it per coefficient: the `k`-th coefficient of the block, in coefficient-label
# order, gets `Family(args...)` with each array-valued argument at `k`. Each
# case is checked against the model written with one statement per
# coefficient, against an independent density, and, on RKBRMI, against the
# compiled Stan program.
#
# Run: julia --startup-file=no --project=test test/effect_prior_vector_arguments.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const LEVELS = (; y=[0.4, 1.3, -0.2, 2.1, 0.9, 1.7],
                  g=["a", "b", "c", "d", "a", "c"])
const SCALES = [0.5, 1.0, 2.0, 4.0]

sb(brmi) = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
code(brmi) = BRM.stan_code(sb(brmi))
reparsed(brmi, data) = Base.invokelatest(
    Base.invokelatest(Core.eval, @__MODULE__, BRM._brm(string(brmi))), data)
rk_source(backend) = sprint(Base.show_unquoted, BRM._rk_emit_ast(backend.plan).main)

# The compiled program's log density at fixed unconstrained points, keyed by
# coordinate name.
function stan_points(brmi, label)
    stan = BRM.stan_instantiate(sb(brmi);
        path=joinpath(mktempdir(), "effect_vector_$label.stan"))
    names = BridgeStan.param_unc_names(stan.model)
    n = length(names)
    points = (zeros(n), [0.2 * sin(i) for i in 1:n], [-0.3 * cos(i) for i in 1:n])
    names, points, [BridgeStan.log_density(stan.model, u; propto=false) for u in points]
end

# RKBRMI against the compiled Stan program through the coordinate transport.
function check_rk_against_stan(brmi, label)
    backend, problem = consumer_problem(brmi)
    stan = BRM.stan_instantiate(sb(brmi);
        path=joinpath(mktempdir(), "effect_vector_rk_$label.stan"))
    transport = brm_coordinate_transport(backend, sb(brmi),
        BridgeStan.param_unc_names(stan.model))
    n = backend.model.layout.total
    for u in (zeros(n), [0.15 * sin(i) for i in 1:n], [-0.1 * cos(i) for i in 1:n])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_u = brm_rk_point_to_stan(transport, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
            stan_gradient; propto=false, jacobian=true)
        @test value ≈ stan_value + transport.logdensity_offset atol=2e-10 rtol=2e-10
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-8 rtol=2e-8
    end
    backend
end

const cellmeans_bounded = @brm begin
    w ~ 0 + factor(g)
    effect(w, g) .~ Normal.(0.0, wscale; lower = 0.0)
    y ~ Normal(w, 1.0)
end
const cellmeans_bounded_levels = @brm begin
    w ~ 0 + factor(g)
    effect(w, g_lvl_1) ~ Normal(0.0, wscale[1]; lower = 0.0)
    effect(w, g_lvl_2) ~ Normal(0.0, wscale[2]; lower = 0.0)
    effect(w, g_lvl_3) ~ Normal(0.0, wscale[3]; lower = 0.0)
    effect(w, g_lvl_4) ~ Normal(0.0, wscale[4]; lower = 0.0)
    y ~ Normal(w, 1.0)
end
const cellmeans_bounded_literal = @brm begin
    w ~ 0 + factor(g)
    effect(w, g_lvl_1) ~ Normal(0.0, 0.5; lower = 0.0)
    effect(w, g_lvl_2) ~ Normal(0.0, 1.0; lower = 0.0)
    effect(w, g_lvl_3) ~ Normal(0.0, 2.0; lower = 0.0)
    effect(w, g_lvl_4) ~ Normal(0.0, 4.0; lower = 0.0)
    y ~ Normal(w, 1.0)
end

@stestset "cell means: one bounded prior per level from a data vector (SBBRMI)" begin
    data = merge(LEVELS, (; wscale=SCALES))
    before = deepcopy(data)
    brmi = cellmeans_bounded(data)
    program = code(brmi)
    @test StanBlocks.stanc_check(program; warn_pedantic=false).ok
    # The same model as one statement per level, byte for byte.
    @test program == code(cellmeans_bounded_levels(data))
    @test BRM.stan_data(sb(brmi))[:wscale] == SCALES
    @test stan_points(brmi, "cellmeans") ==
          stan_points(cellmeans_bounded_literal(LEVELS), "cellmeans_literal")
    # It prints as written and re-parses to the same program.
    @test occursin("effect(w, g) .~ Normal.(0.0, wscale; lower=0.0)", string(brmi))
    @test code(reparsed(brmi, data)) == program
    spec = only(effect_priors(brmi))
    @test spec.elementwise && spec.family === Normal && spec.coefficient === :g
    @test isequal(data, before)
end

# A per-contrast location vector on a treatment-coded block: K−1 entries, the
# non-reference levels in fitted order. The density is written out by name.
const treatment = @brm begin
    w ~ 1 + factor(g)
    effect(w, g) .~ Normal.(wloc, 2.0)
    y ~ Normal(w, 1.0)
end
const WLOC = [0.1, -0.2, 0.3]

@stestset "treatment contrasts: one location per contrast" begin
    data = merge(LEVELS, (; wloc=WLOC))
    brmi = treatment(data)
    @test StanBlocks.stanc_check(code(brmi); warn_pedantic=false).ok
    names, points, densities = stan_points(brmi, "treatment")
    level = Dict("a" => 0, "b" => 1, "c" => 2, "d" => 3)
    for (u, density) in zip(points, densities)
        at(name) = u[findfirst(==(name), names)]
        intercept = at("pop_w_beta_pop.1")
        beta = [at("cat_w_g_beta.$k") for k in 1:3]
        mu = [intercept + (level[l] == 0 ? 0.0 : beta[level[l]]) for l in LEVELS.g]
        oracle = logpdf(Normal(), intercept) + sum(logpdf.(Normal.(WLOC, 2.0), beta)) +
                 sum(logpdf.(Normal.(mu, 1.0), LEVELS.y))
        @test density ≈ oracle atol=1e-10 rtol=1e-10
    end
    check_rk_against_stan(brmi, "treatment")
end

# RKBRMI takes the same statements within its population-prior vocabulary
# (unbounded families): the block's per-level values are one vector argument
# of its broadcast prior, and per-level statements give the same program.
const cellmeans_cauchy = @brm begin
    w ~ 0 + factor(g)
    effect(w, g) .~ Cauchy.(0.0, wscale)
    y ~ Normal(w, 1.0)
end
const cellmeans_cauchy_levels = @brm begin
    w ~ 0 + factor(g)
    effect(w, g_lvl_1) ~ Cauchy(0.0, wscale[1])
    effect(w, g_lvl_2) ~ Cauchy(0.0, wscale[2])
    effect(w, g_lvl_3) ~ Cauchy(0.0, wscale[3])
    effect(w, g_lvl_4) ~ Cauchy(0.0, wscale[4])
    y ~ Normal(w, 1.0)
end

@stestset "cell means on RKBRMI: per-level scales are one vector argument" begin
    data = merge(LEVELS, (; wscale=SCALES))
    backend = check_rk_against_stan(cellmeans_cauchy(data), "cellmeans_cauchy")
    @test occursin("Cauchy.(0.0, [0.5, 1.0, 2.0, 4.0])", rk_source(backend))
    twin, _ = consumer_problem(cellmeans_cauchy_levels(data))
    @test rk_source(twin) == rk_source(backend)
    @test code(cellmeans_cauchy_levels(data)) == code(cellmeans_cauchy(data))
end

# The model with one `effect(lp, <label>) ~ law_k` statement per label, built
# from source like any `@brm` block.
function per_label_twin(formula, lp, labels, law)
    lines = ["$lp ~ $formula";
             ["effect($lp, $label) ~ $(law(k))" for (k, label) in enumerate(labels)];
             "y ~ Normal($lp, 1.0)"]
    Base.invokelatest(Core.eval, @__MODULE__,
        BRM._brm(Meta.parse("begin\n" * join(lines, "\n") * "\nend")))
end

# A bare string column's levels and a whole interaction term are `beta_pop`
# columns reached by one address; the vector follows their labels.
@stestset "design-coded levels and interaction terms read one entry per column" begin
    data = merge(LEVELS, (; x=[-1.0, -0.4, 0.1, 0.6, 1.2, -0.7], h=[1, 2, 3, 1, 2, 3],
                            wscale=SCALES, iscale=SCALES[1:2]))
    levels_model = @brm begin
        w ~ 0 + g
        effect(w, g) .~ Normal.(0.0, wscale)
        y ~ Normal(w, 1.0)
    end
    labels = popcoefnames(levels_model(data), :w)
    @test length(labels) == 4
    twin = per_label_twin("0 + g", :w, labels, k -> "Normal(0.0, wscale[$k])")
    @test code(levels_model(data)) == code(twin(data))
    # An integer column is categorical: `x & h` emits one column per
    # non-reference level of `h`.
    interaction = @brm begin
        w ~ 1 + x + x & h
        effect(w, x & h) .~ Normal.(0.0, iscale)
        y ~ Normal(w, 1.0)
    end
    terms = filter(l -> startswith(string(l), "int_"), popcoefnames(interaction(data), :w))
    @test length(terms) == 2
    twin = per_label_twin("1 + x + x & h", :w, terms, k -> "Normal(0.0, iscale[$k])")
    @test code(interaction(data)) == code(twin(data))
end

# A wildcard predictor slot gives every predictor owning the block the vector.
@stestset "a wildcard predictor reads the vector for each owner" begin
    data = merge(LEVELS, (; wscale=SCALES))
    wildcard = @brm begin
        w ~ 0 + factor(g)
        v ~ 0 + factor(g)
        effect(:, g) .~ Normal.(0.0, wscale)
        y ~ Normal(w + v, 1.0)
    end
    each = @brm begin
        w ~ 0 + factor(g)
        v ~ 0 + factor(g)
        effect(w, g) .~ Normal.(0.0, wscale)
        effect(v, g) .~ Normal.(0.0, wscale)
        y ~ Normal(w + v, 1.0)
    end
    @test code(wildcard(data)) == code(each(data))
end

# A more specific statement still overrides one coefficient, in either order.
@stestset "a level address overrides one element" begin
    data = merge(LEVELS, (; wscale=SCALES))
    first_block = @brm begin
        w ~ 0 + factor(g)
        effect(w, g) .~ Normal.(0.0, wscale)
        effect(w, g_lvl_2) ~ Normal(5.0, 1.0)
        y ~ Normal(w, 1.0)
    end
    first_level = @brm begin
        w ~ 0 + factor(g)
        effect(w, g_lvl_2) ~ Normal(5.0, 1.0)
        effect(w, g) .~ Normal.(0.0, wscale)
        y ~ Normal(w, 1.0)
    end
    literal = @brm begin
        w ~ 0 + factor(g)
        effect(w, g_lvl_1) ~ Normal(0.0, 0.5)
        effect(w, g_lvl_2) ~ Normal(5.0, 1.0)
        effect(w, g_lvl_3) ~ Normal(0.0, 2.0)
        effect(w, g_lvl_4) ~ Normal(0.0, 4.0)
        y ~ Normal(w, 1.0)
    end
    @test code(first_block(data)) == code(first_level(data))
    @test stan_points(first_block(data), "override") == stan_points(literal(LEVELS), "override_literal")
    # The block-wide elementwise statement outranks a predictor default.
    defaulted = @brm begin
        w ~ 1 + factor(g)
        effect(w, :) ~ Normal(0.0, 3.0)
        effect(w, g) .~ Normal.(wloc, 2.0)
        y ~ Normal(w, 1.0)
    end
    wdata = merge(LEVELS, (; wloc=WLOC))
    names, points, densities = stan_points(defaulted(wdata), "defaulted")
    _, _, plain = stan_points(treatment(wdata), "defaulted_plain")
    # Only the intercept's prior differs: Normal(0, 3) instead of Normal(0, 1).
    for (u, a, b) in zip(points, densities, plain)
        b0 = u[findfirst(==("pop_w_beta_pop.1"), names)]
        @test a - b ≈ logpdf(Normal(0.0, 3.0), b0) - logpdf(Normal(), b0) atol=1e-10
    end
end

@stestset "elementwise priors: refusals name the fix" begin
    data = merge(LEVELS, (; wscale=SCALES))
    # refused: the vector must hold one entry per coefficient the address
    # reaches (decision `1xaf4fx`).
    @test_throws r"reaches 4 \(a, b, c, d\), but `wscale` has 3 entries" sb(
        cellmeans_bounded(merge(LEVELS, (; wscale=SCALES[1:3]))))
    @test_throws r"reaches 3 \(b, c, d\), but `wloc` has 4 entries" sb(
        treatment(merge(LEVELS, (; wloc=[WLOC; 0.0]))))
    @test_throws r"reaches .*but `wscale` has 3 entries" RKBRMI(
        cellmeans_cauchy(merge(LEVELS, (; wscale=SCALES[1:3]))))
    # refused: `:` reaches several blocks with no single coefficient order
    # (decision `1xaf4fx`: one entry per coefficient of the addressed block).
    wildcard = @brm begin
        w ~ 0 + factor(g)
        effect(w, :) .~ Normal.(0.0, wscale)
        y ~ Normal(w, 1.0)
    end
    @test_throws r"name the block" sb(wildcard(data))
    # The undotted spelling with a vector stays refused and names this one:
    # `test/effect_prior_data_fields.jl`.
end
