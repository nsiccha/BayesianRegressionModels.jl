# Address-keyed initial values -> complete RK and Stan starting points.
# RUN: julia --project=test test/start_values.jl
#
# `brm_start` resolves the same prior addresses on both backends: the RK point
# and the Stan point must be one physical point (the semantic transport is the
# independent oracle), the addressed constrained values must be exact, and
# every unaddressed coordinate keeps its base value.
using BridgeStan, CategoricalArrays, LinearAlgebra, Random
import StanBlocks
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

const START_CACHE = mktempdir()
compiled(sb, name) = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(START_CACHE, "$name.stan")).model

# Two correlated margins of `|pk|` in `ranefcoefnames` order.
const PK_SIGMA = [0.09 0.018; 0.018 0.16]
const PK_INITS = (;
    log_CL=(; Intercept=log(2.7), weight=0.75),
    log_V=(; Intercept=log(31.5)),
    pk=PK_SIGMA,
    sigma=0.1,
)

# RK and Stan points of the same BRMI agree physically and by coordinate.
function same_point(brmi, inits, name; rk_base=nothing, kwargs...)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), kwargs...)
    model = compiled(sb, name)
    t = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(model))
    u = isnothing(rk_base) ? brm_start(rk, inits) : brm_start(rk, inits; base=rk_base)
    s = isnothing(rk_base) ? brm_start(sb, model, inits) :
        brm_start(sb, model, inits; base=brm_rk_point_to_stan(t, rk_base))
    (; rk, sb, model, t, u, s)
end

constrained(model, s) = (BridgeStan.param_constrain(model, s),
    String.(BridgeStan.param_names(model)))

@stestset "address-keyed values start the population-PK model on both backends" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    @test [(m.predictor, m.coefficient) for m in ranefcoefnames(brmi, :pk)] ==
        [(:log_CL, :Intercept), (:log_V, :Intercept)]
    saved = deepcopy(PK_INITS)
    (; rk, sb, model, t, u, s) = same_point(brmi, PK_INITS, "plate_pk")
    @test PK_INITS == saved
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    @test brm_check_coordinate_transport(t, rk, model, u).factors == 1

    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    population(lp, c) = θ[only(brm_population_effect_coordinates(d, lp, names;
        coefficient=c).coordinates)]
    @test population(:log_CL, :Intercept) ≈ log(2.7) rtol=1e-12
    @test population(:log_CL, :weight) ≈ 0.75 rtol=1e-12
    @test population(:log_V, :Intercept) ≈ log(31.5) rtol=1e-12
    tau = [θ[only(brm_ranef_sd_coordinates(d, lp, names; id=:pk).coordinates)]
        for lp in (:log_CL, :log_V)]
    @test tau ≈ sqrt.(diag(PK_SIGMA)) rtol=1e-12
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.1 rtol=1e-12
    # Cov = diag(tau) L L' diag(tau) is the given covariance.
    view = brm_nlme_view(rk)
    natural = BRM._brm_nlme_natural(rk, view, BRM._brm_nlme_families(rk, view), u)
    order = [findfirst(b -> b.predictor === lp, view.block_margins)
        for lp in (:log_CL, :log_V)]
    @test natural.omega[order, order] ≈ PK_SIGMA rtol=1e-10
    # The standardized subject draws are untouched: the natural effects are zero.
    @test all(iszero, natural.eta)
    stan_pos = Dict(n => i for (i, n) in enumerate(t.stan_names))
    @test all(iszero, s[[stan_pos[p.stan] for p in t.pairs if p.address.kind === :ranef_z]])

    # `(; sd, cor)` addresses the same block parts.
    C = PK_SIGMA ./ (sqrt.(diag(PK_SIGMA)) * sqrt.(diag(PK_SIGMA))')
    C[diagind(C)] .= 1
    parts = merge(PK_INITS, (; pk=(; sd=sqrt.(diag(PK_SIGMA)), cor=C)))
    @test brm_start(rk, parts) ≈ u rtol=1e-10 atol=1e-12
    @test brm_start(sb, model, parts) ≈ s rtol=1e-10 atol=1e-12
    # One part alone leaves the other at the base: sd only keeps L = I.
    sd_only = brm_start(rk, (; pk=(; sd=0.5)))
    only_natural = BRM._brm_nlme_natural(rk, view,
        BRM._brm_nlme_families(rk, view), sd_only)
    @test only_natural.omega ≈ 0.25I rtol=1e-12

    # A random base: addressed values are set, every other coordinate keeps it.
    base = 0.3 .* randn(Xoshiro(7), length(u))
    (; u, s) = same_point(brmi, PK_INITS, "plate_pk"; rk_base=base)
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    addressed = [i for (i, p) in enumerate(t.pairs) if p.address.kind in
        (:population, :scalar, :ranef_scale, :ranef_correlation)]
    rk_pos = Dict(n => i for (i, n) in enumerate(t.rk_names))
    untouched = [rk_pos[p.rk] for (i, p) in enumerate(t.pairs) if !(i in addressed)]
    @test !isempty(untouched)
    @test u[untouched] == base[untouched]
end

@stestset "a linked predictor is addressed by its public name" begin
    brmi = nlme_pk(NLME_DATA)
    inits = (; CL=(; Intercept=log(3.0), wt=0.5), V=(; Intercept=log(20.0)),
        pk=(; sd=[0.3, 0.2]), sigma=1.5)
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "linked_pk")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    @test θ[only(brm_population_effect_coordinates(d, :CL, names;
        coefficient=:wt).coordinates)] ≈ 0.5 rtol=1e-12
    # refused: the emitted binding is not the public address.
    @test_throws ErrorException brm_start(rk, (; log_CL=(; Intercept=1.0)))
    @test_throws ErrorException brm_start(sb, model, (; log_CL=(; Intercept=1.0)))
end

@stestset "exact total blocks start at the population value" begin
    data = (; g=repeat(["a", "b", "c", "d"], 3), x=collect(range(-1, 1; length=12)),
        y=sin.(1:12))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | g)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=:auto)
    blocks = total_effect_blocks(sb)
    @test length(blocks) == 1
    model = compiled(sb, "totals")
    inits = (; mu=(; Intercept=1.5, x=0.2), sigma=0.4)
    s = brm_start(sb, model, inits)
    θ, names = constrained(model, s)
    block = only(blocks)
    totals = BRM._total_coordinates(sb, block, names).totals
    β = [Dict(:Intercept => 1.5, :x => 0.2)[c] for c in block.population_columns]
    @test θ[totals] ≈ repeat(transpose(block.A * β), size(totals, 1)) rtol=1e-12
    @test :Intercept in block.population_columns
    d = brm_descriptor(sb)
    if !(:x in block.population_columns)  # a retained, sampled slope
        @test θ[only(brm_population_effect_coordinates(d, :mu, names;
            coefficient=:x).coordinates)] ≈ 0.2 rtol=1e-12
    end
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.4 rtol=1e-12

    # The conventional emission of the same model samples the coefficients.
    (; u, s) = same_point(brmi, inits, "totals_conventional")
    @test length(s) == length(u)
end

# One lone `|ID|` margin per predictor: independent effects, which the default
# `total_groups=:auto` turns into exact total blocks.
function lone_eta_pk(data)
    @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | eta_cl | subject)
        log_V ~ 1 + (1 | eta_v | subject)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            Vc = exp(log_V[i])
            pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
            dv[i] ~ normal(pred[i], sigma)
        end
    end
end

@stestset "an |ID| absorbed by an exact total block keeps its scale address" begin
    brmi = lone_eta_pk(NLME_PLATE_DATA)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=:auto)
    blocks = Dict(b.predictor => b for b in total_effect_blocks(sb))
    @test sort!(collect(keys(blocks))) == [:log_CL, :log_V]
    model = compiled(sb, "lone_eta_totals")
    inits = (; log_CL=(; Intercept=log(2.7), weight=0.75),
        log_V=(; Intercept=log(31.5)), eta_cl=[0.09;;], eta_v=(; sd=0.2), sigma=0.7)
    saved = deepcopy(inits)
    s = brm_start(sb, model, inits)
    @test inits == saved
    @test isfinite(BridgeStan.log_density(model, s))
    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    # The `|ID|` scale is the block's total scale: the SD of each group's
    # total around `A β`, read back through the same address.
    scale(lp, id) = θ[only(brm_ranef_sd_coordinates(d, lp, names; id).coordinates)]
    @test scale(:log_CL, :eta_cl) ≈ 0.3 rtol=1e-12
    @test scale(:log_V, :eta_v) ≈ 0.2 rtol=1e-12
    for (lp, sd) in (:log_CL => 0.3, :log_V => 0.2)
        @test θ[BRM._total_coordinates(sb, blocks[lp], names).scales] ≈ [sd] rtol=1e-12
    end
    # Absorbed coefficients place every total at its subject's population
    # value; `weight` is one value per subject, so it is absorbed too.
    @test blocks[:log_CL].population_columns == (:Intercept, :weight)
    @test blocks[:log_CL].group_columns == (:weight,)
    @test blocks[:log_V].population_columns == (:Intercept,)
    for (lp, β) in (:log_CL => log(2.7) .+ 0.75 .* NLME_PLATE_DATA.weight,
                    :log_V => fill(log(31.5), 3))
        totals = BRM._total_coordinates(sb, blocks[lp], names).totals
        @test vec(θ[totals]) ≈ β rtol=1e-12
    end
    gq = String.(BridgeStan.param_names(model; include_tp=true, include_gq=true))
    @test brm_population_effect_coordinates(d, :log_CL, gq; coefficient=:weight).recovered
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.7 rtol=1e-12
    # Unaddressed coordinates keep a random base: here everything but one scale.
    unc = String.(BridgeStan.param_unc_names(model))
    k = only(BRM._total_coordinates(sb, blocks[:log_V], unc).scales)
    base = 0.3 .* randn(Xoshiro(11), length(s))
    u = brm_start(sb, model, (; eta_v=(; sd=0.2)); base)
    @test exp(u[k]) ≈ 0.2 rtol=1e-12
    rest = setdiff(eachindex(u), k)
    @test u[rest] ≈ base[rest] rtol=1e-10 atol=1e-12
    # refused: a one-margin block takes a 1×1 covariance and a positive scale.
    @test_throws DimensionMismatch brm_start(sb, model, (; eta_cl=fill(0.1, 2, 2)))
    @test_throws ErrorException brm_start(sb, model, (; eta_cl=(; sd=-0.1)))

    # Omitting the response leaves no likelihood reaching the totals: the block
    # and its scale are drawn in generated quantities, under the same address.
    prior_sb = SBBRMI(lone_eta_pk(Base.structdiff(NLME_PLATE_DATA, (; dv=nothing)));
        mod=@__MODULE__, total_groups=:auto)
    prior = compiled(prior_sb, "lone_eta_prior")
    pd = brm_descriptor(prior_sb)
    gq_names = String.(BridgeStan.param_names(prior; include_tp=true, include_gq=true))
    q = brm_ranef_sd_coordinates(pd, :log_V, gq_names; id=:eta_v)
    @test q.output.kind === :generated_quantity
    @test gq_names[only(q.coordinates)] ==
        "$(only(b for b in total_effect_blocks(prior_sb) if b.predictor === :log_V).scales).1"
    # refused: the generated scale is not among the sampled names, and a start
    # point has no coordinate for it.
    @test_throws ErrorException brm_ranef_sd_coordinates(pd, :log_V,
        String.(BridgeStan.param_names(prior)); id=:eta_v)
    @test_throws ErrorException brm_start(prior_sb, prior, (; eta_v=(; sd=0.2)))
end

@stestset "a categorical predictor takes level-keyed values" begin
    data = (; subject=repeat(["a", "b", "c", "d"], 3),
        arm=categorical(repeat(["p", "q", "r"], 4)), y=sin.(1:12))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + arm + (1 | e | subject)
        y ~ Normal(mu, sigma)
    end
    inits = (; mu=(; Intercept=-0.5, arm=["q" => 0.25, "r" => -0.4]),
        e=fill(0.04, 1, 1), sigma=0.2)
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "categorical")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    θ, names = constrained(model, s)
    d = brm_descriptor(sb)
    q = brm_population_effect_coordinates(d, :mu, names; coefficient=:arm)
    levels = Dict(BRM._brm_transport_level(c.nonreference_level) => θ[c.coordinate]
        for c in q.contrasts)
    @test levels["q"] ≈ 0.25 rtol=1e-12
    @test levels["r"] ≈ -0.4 rtol=1e-12
    @test θ[only(brm_population_effect_coordinates(d, :mu, names;
        coefficient=:Intercept).coordinates)] ≈ -0.5 rtol=1e-12
    # refused: the reference level has no coefficient.
    @test_throws ErrorException brm_start(rk, (; mu=(; arm=["p" => 0.1])))
    @test_throws ErrorException brm_start(sb, model, (; mu=(; arm=["p" => 0.1])))
    # refused: a categorical column takes level pairs, not one number.
    @test_throws ErrorException brm_start(sb, model, (; mu=(; arm=0.1)))
end

# The first categorical term of an intercept-free predictor is coded as cell
# means (decision `0woa6hh`): no reference level, one coefficient per level.
@stestset "an intercept-free categorical term takes a value for every level" begin
    cell_values(sb, model, s, lp, column) = begin
        θ, names = constrained(model, s)
        q = brm_population_effect_coordinates(brm_descriptor(sb), lp, names;
            coefficient=column)
        @test q.coding === :cellmeans
        Dict(BRM._brm_transport_level(c.level) => θ[c.coordinate] for c in q.cells)
    end

    data = (; subject=repeat(["a", "b", "c", "d"], 3),
        arm=categorical(repeat(["p", "q", "r"], 4)), y=sin.(1:12))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 0 + arm + (1 | e | subject)
        y ~ Normal(mu, sigma)
    end
    inits = (; mu=(; arm=["p" => 0.3, "q" => 0.25, "r" => -0.4]),
        e=fill(0.04, 1, 1), sigma=0.2)
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "cellmeans")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    cells = cell_values(sb, model, s, :mu, :arm)
    @test [cells[l] for l in ("p", "q", "r")] ≈ [0.3, 0.25, -0.4] rtol=1e-12
    # An unaddressed level keeps its base value.
    partial = brm_start(sb, model, (; mu=(; arm=["q" => 0.25])))
    @test cell_values(sb, model, partial, :mu, :arm)["p"] == 0
    @test brm_rk_point_to_stan(t, brm_start(rk, (; mu=(; arm=["q" => 0.25])))) ≈
        partial rtol=1e-10 atol=1e-12
    # refused: a level the column never takes has no coefficient.
    @test_throws ErrorException brm_start(rk, (; mu=(; arm=["s" => 0.1])))
    @test_throws ErrorException brm_start(sb, model, (; mu=(; arm=["s" => 0.1])))

    # A linked predictor over a plain String column through `factor`, with an
    # elementwise prior: values are on the link scale, keyed by the public name.
    linked = (; subject=repeat(["a", "b", "c", "d"], 3),
        kind=repeat(["x", "y", "z"], 4), n=[1, 2, 4, 2, 1, 3, 0, 2, 5, 1, 3, 2],
        y=2 .+ sin.(1:12), w_loc=[0.1, 0.6, 1.2])
    brmi = @brm linked begin
        log(w) ~ 0 + factor(kind) + (1 | e | subject)
        effect(w, kind) .~ Normal.(w_loc, 0.7)
        n ~ Poisson(w)
    end
    inits = (; w=(; kind=["x" => log(1.5), "y" => log(2.0), "z" => log(3.0)]),
        e=fill(0.09, 1, 1))
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "cellmeans_linked")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    cells = cell_values(sb, model, s, :w, :kind)
    @test [cells[l] for l in ("x", "y", "z")] ≈ log.([1.5, 2.0, 3.0]) rtol=1e-12

    # A plain String column without `factor` codes its levels in the predictor's
    # own coefficient vector, and takes the same pairs.
    brmi = @brm linked begin
        sigma ~ Exponential(1)
        mu ~ 0 + kind
        y ~ Normal(mu, sigma)
    end
    inits = (; mu=(; kind=["x" => 1.0, "y" => 2.0, "z" => 3.0]))
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "cellmeans_string")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    θ, names = constrained(model, s)
    q = brm_population_effect_coordinates(brm_descriptor(sb), :mu, names;
        coefficient=:kind)
    @test q.coding === :cellmeans
    @test θ[q.coordinates] ≈ [1.0, 2.0, 3.0] rtol=1e-12
end

@stestset "a vector parameter is set on its constrained scale" begin
    data = (; x=collect(range(-1, 1; length=8)), y=cos.(1:8))
    brmi = @brm data begin
        s ~ Dirichlet([1.0, 2.0, 1.5])
        mu ~ 1 + x
        loc = mu + 0.8 * s[1] - s[3]
        y ~ Normal(loc, 1)
    end
    inits = (; s=[0.5, 0.3, 0.2], mu=(; x=-0.3))
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "simplex")
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    θ, names = constrained(model, s)
    @test θ[brm_output_coordinates(brm_descriptor(sb), :s, names)] ≈
        [0.5, 0.3, 0.2] rtol=1e-10
    # refused: a simplex of another length, or off the simplex.
    @test_throws DimensionMismatch brm_start(rk, (; s=[0.5, 0.5]))
    @test_throws DimensionMismatch brm_start(sb, model, (; s=[0.5, 0.5]))
    @test_throws Exception brm_start(rk, (; s=[0.5, 0.6, 0.2]))
    @test_throws Exception brm_start(sb, model, (; s=[0.5, 0.6, 0.2]))
end

# A latent declared in a `@plate` cell, one value per entry of the cell's
# location: 3 values in cell 1, 2 in cell 2 (brm-use kernel.md).
const CELL_LATENT_DATA = (; mu=[[10.0, 20.0, 5.0], [7.0, 3.0]], obs=[[1, 2], [1]],
    y=[[11.0, 19.0], [6.5]], total=[36.0, 10.5])

@stestset "a @plate cell latent takes one flat vector or one vector per cell" begin
    brmi = @brm CELL_LATENT_DATA begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(total)
            loc = log.(mu[i])
            log_T ~ Normal(loc, 0.5)
            T = exp.(log_T)
            y[i] ~ Normal(T[obs[i]], sigma)
            total[i] ~ Normal(sum(T), 0.01 * total[i])
        end
    end
    sb = SBBRMI(brmi; mod=@__MODULE__)
    model = compiled(sb, "cell_latent")
    d = brm_descriptor(sb)
    @test brm_output(d, :log_T).segments == [3, 5]
    per_cell = [log.([9.0, 21.0, 6.0]), log.([7.5, 3.0])]
    inits = (; log_T=per_cell, sigma=0.5)
    saved = deepcopy(inits)
    s = brm_start(sb, model, inits)
    @test inits == saved
    @test isfinite(BridgeStan.log_density(model, s))
    θ, names = constrained(model, s)
    @test θ[brm_output_coordinates(d, :log_T, names)] ≈ reduce(vcat, per_cell) rtol=1e-12
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.5 rtol=1e-12
    # One flat vector, cell after cell, is the same point.
    @test brm_start(sb, model, (; log_T=reduce(vcat, per_cell), sigma=0.5)) == s
    # Unaddressed coordinates keep a random base.
    base = 0.3 .* randn(Xoshiro(5), length(s))
    u = brm_start(sb, model, (; log_T=per_cell); base)
    k = brm_output_coordinates(d, :log_T, String.(BridgeStan.param_unc_names(model)))
    rest = setdiff(eachindex(u), k)
    @test u[rest] ≈ base[rest] rtol=1e-10 atol=1e-12
    # refused: the cells' counts and lengths are the model's.
    @test_throws DimensionMismatch brm_start(sb, model, (; log_T=per_cell[1:1]))
    @test_throws DimensionMismatch brm_start(sb, model,
        (; log_T=[per_cell[1], [per_cell[2]; 0.0]]))
    @test_throws DimensionMismatch brm_start(sb, model, (; log_T=zeros(4)))
    @test_throws ErrorException brm_start(sb, model, (; log_T=[per_cell[1], [NaN, 1.0]]))
    # refused: per-cell vectors address a cell latent; `sigma` is one value.
    @test_throws ErrorException brm_start(sb, model, (; sigma=[[0.5]]))
end

@stestset "a bounded cell latent is set on its constrained scale" begin
    brmi = @brm CELL_LATENT_DATA begin
        sigma ~ Exponential(1.0)
        @plate for i in eachindex(total)
            T ~ LogNormal(log.(mu[i]), 0.5)
            y[i] ~ Normal(T[obs[i]], sigma)
            total[i] ~ Normal(sum(T), 0.01 * total[i])
        end
    end
    sb = SBBRMI(brmi; mod=@__MODULE__)
    model = compiled(sb, "cell_latent_bounded")
    d = brm_descriptor(sb)
    @test brm_output(d, :T).constraints == (; lower=0.0)
    per_cell = [[9.0, 21.0, 6.0], [7.5, 3.0]]
    s = brm_start(sb, model, (; T=per_cell))
    θ, names = constrained(model, s)
    @test θ[brm_output_coordinates(d, :T, names)] ≈ reduce(vcat, per_cell) rtol=1e-12
    # The unconstrained coordinates are the lower-bound transform's, not the values.
    k = brm_output_coordinates(d, :T, String.(BridgeStan.param_unc_names(model)))
    @test s[k] ≈ log.(reduce(vcat, per_cell)) rtol=1e-12
    # refused: a value outside the latent's support.
    @test_throws Exception brm_start(sb, model, (; T=[per_cell[1], [-1.0, 3.0]]))
end

@stestset "per-cell values have no target on RKBRMI" begin
    data = (; x=collect(range(-1, 1; length=8)), y=cos.(1:8))
    brmi = @brm data begin
        s ~ Dirichlet([1.0, 2.0, 1.5])
        mu ~ 1 + x
        loc = mu + 0.8 * s[1] - s[3]
        y ~ Normal(loc, 1)
    end
    # refused: `s` is one vector, not a `@plate` cell latent.
    @test_throws ErrorException brm_start(RKBRMI(brmi), (; s=[[0.5], [0.3], [0.2]]))
end

# Per-level effects (`ID = (; sd, cor, effects)`, decision 15s7csf): each
# listed group starts at its own effect. The read-back oracle on SBBRMI is
# `brm_ranef_draws`, which reads Stan's own effect carriers; RK's is the
# NLME view's natural effects, and the RK and Stan points must agree through
# the semantic transport.
const PK_EFFECTS = ["a" => [0.2, -0.1], "c" => [-0.3, 0.25]]

# The effects `brm_ranef_draws` reads at the Stan point `s`, levels × margins.
function ranef_effects(sb, model, s; id)
    names = String.(BridgeStan.param_names(model; include_tp=true))
    draws = permutedims(BridgeStan.param_constrain(model, s; include_tp=true))
    p = only(brm_ranef_draws(brm_descriptor(sb), draws, names; id))
    Dict(BRM._brm_transport_level(l) => p.effects[1, i, :] for (i, l) in enumerate(p.levels))
end

@stestset "per-level effects start each group at its own value" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    C = PK_SIGMA ./ (sqrt.(diag(PK_SIGMA)) * sqrt.(diag(PK_SIGMA))')
    C[diagind(C)] .= 1
    inits = merge(PK_INITS, (; pk=(; sd=sqrt.(diag(PK_SIGMA)), cor=C, effects=PK_EFFECTS)))
    saved = deepcopy(inits)
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "plate_pk")
    @test inits == saved
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    @test brm_check_coordinate_transport(t, rk, model, u).factors == 1
    effects = ranef_effects(sb, model, s; id=:pk)
    @test effects["a"] ≈ [0.2, -0.1] rtol=1e-10
    @test effects["c"] ≈ [-0.3, 0.25] rtol=1e-10
    # An unlisted level keeps its standardized draws at the base: effect 0.
    @test effects["b"] == [0.0, 0.0]
    # The scales and correlation are still the given ones.
    view = brm_nlme_view(rk)
    natural = BRM._brm_nlme_natural(rk, view, BRM._brm_nlme_families(rk, view), u)
    order = [findfirst(b -> b.predictor === lp, view.block_margins)
        for lp in (:log_CL, :log_V)]
    @test natural.omega[order, order] ≈ PK_SIGMA rtol=1e-10
    # NLME's η (`brm_nlme_start`) are the same natural effects.
    for (level, e) in PK_EFFECTS
        @test natural.eta[order, findfirst(==(level), view.levels)] ≈ e rtol=1e-10
    end
    # Effects alone, and a random base: the effects hold under the base's
    # scales and correlation, and every other coordinate keeps the base.
    base = 0.3 .* randn(Xoshiro(3), length(u))
    alone = (; pk=(; effects=PK_EFFECTS))
    (; u, s) = same_point(brmi, alone, "plate_pk"; rk_base=base)
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    effects = ranef_effects(sb, model, s; id=:pk)
    @test effects["a"] ≈ [0.2, -0.1] rtol=1e-10
    @test effects["c"] ≈ [-0.3, 0.25] rtol=1e-10
    rk_pos = Dict(n => i for (i, n) in enumerate(t.rk_names))
    moved = [rk_pos[p.rk] for p in t.pairs
        if p.address.kind === :ranef_z && p.address.level in ("a", "c")]
    @test length(moved) == 4
    rest = setdiff(eachindex(u), moved)
    @test u[rest] ≈ base[rest] rtol=1e-10 atol=1e-12
    # A dictionary takes the same pairs.
    @test brm_start(rk, (; pk=(; effects=Dict(PK_EFFECTS)))) ≈
        brm_start(rk, (; pk=(; effects=PK_EFFECTS))) rtol=1e-12
end

@stestset "per-level effects of a centred block" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    inits = merge(PK_INITS, (; pk=(; sd=[0.3, 0.4], effects=PK_EFFECTS)))
    (; rk, sb, model, t, u, s) = same_point(brmi, inits, "plate_pk_centered";
        centered_groups=[:subject])
    @test !only(ranef_blocks(brm_descriptor(sb))).noncentered
    # The transport maps RK's standardized draws to Stan's centred effects.
    @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
    effects = ranef_effects(sb, model, s; id=:pk)
    @test effects["a"] ≈ [0.2, -0.1] rtol=1e-10
    @test effects["c"] ≈ [-0.3, 0.25] rtol=1e-10
    @test effects["b"] == [0.0, 0.0]
end

const STUDENT_T_DATA = (; subject=[1, 1, 2, 2, 3, 3, 4, 4],
    site=["s1", "s2", "s1", "s2", "s3", "s3", "s1", "s2"],
    x=[0.1, 0.4, -0.2, 0.3, 0.7, 0.5, 0.2, -0.1],
    y=[0.4, 1.1, -0.3, 0.6, 1.7, 1.2, 0.2, -0.5],
    y2=[0.3, 0.2, 0.1, -0.4, 0.6, 0.8, 0.1, 0.0])

function student_t_blocks(data)
    @brm data begin
        nu ~ Gamma(2, 10; lower=1.0)
        mu ~ 1 + x + (1 + x | effect | subject)
        sd(:, effect) ~ Exponential(0.8)
        ranef(:, effect) ~ TDist(nu)
        eta ~ 1 + (1 | site_block | site)
        ranef(:, site_block) ~ TDist(4)
        sigma ~ Exponential(0.7)
        y ~ Normal(mu, sigma)
        y2 ~ Normal(eta, sigma)
    end
end

@stestset "per-level effects of Student-t blocks hold at the base mixing weights" begin
    brmi = student_t_blocks(STUDENT_T_DATA)
    inits = (; effect=(; sd=[0.5, 0.2], effects=[2 => [0.4, -0.05], 4 => [-0.6, 0.1]]),
        site_block=(; effects=["s3" => 0.7]))
    base = 0.3 .* randn(Xoshiro(13),
        length(BRM._rk_layout_coordinate_names(RKBRMI(brmi))))
    for centered_groups in ((), [:subject, :site])
        name = "student_t_$(length(centered_groups))"
        (; t, u, s) = same_point(brmi, inits, name; centered_groups)
        @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
        # An unlisted level keeps its coordinates at the base: its standardized
        # draws on RK and in a non-centred block, its effects in a centred one.
        # Under the changed scale the backends then place it differently.
        (; rk, sb, model, t, u, s) = same_point(brmi, inits, name; centered_groups,
            rk_base=base)
        @test all(b -> occursin("student_t", String(b.family)),
            ranef_blocks(brm_descriptor(sb)))
        isempty(centered_groups) &&
            @test brm_rk_point_to_stan(t, u) ≈ s rtol=1e-10 atol=1e-12
        effects = ranef_effects(sb, model, s; id=:effect)
        @test effects[2] ≈ [0.4, -0.05] rtol=1e-10
        @test effects[4] ≈ [-0.6, 0.1] rtol=1e-10
        @test only(ranef_effects(sb, model, s; id=:site_block)["s3"]) ≈ 0.7 rtol=1e-10
        # The mixing weights keep the base.
        rk_pos = Dict(n => i for (i, n) in enumerate(t.rk_names))
        weights = [rk_pos[p.rk] for p in t.pairs if p.address.kind === :ranef_mixing]
        @test length(weights) == 7
        @test u[weights] ≈ base[weights] rtol=1e-10 atol=1e-12
    end
end

const LEVEL_COR_DATA = (; t=collect(range(0.0, 10.0; length=24)), g=repeat(1:4, 6),
    y=sin.(1:24), C=[1.0 0.5 0.2 0.1; 0.5 1.0 0.3 0.2; 0.2 0.3 1.0 0.4; 0.1 0.2 0.4 1.0])

@stestset "per-level effects of a level-correlated block" begin
    sampled = @brm LEVEL_COR_DATA begin
        Omega ~ LKJCholesky(4, 2.0)
        mu ~ 1 + t + (1 + t | r | gr(g; cor=Omega))
        y ~ Normal(mu, 1.0)
    end
    known = @brm LEVEL_COR_DATA begin
        mu ~ 1 + t + (1 + t | r | gr(g; cor=C))
        y ~ Normal(mu, 1.0)
    end
    # Two of the four levels: the levels' factor couples them to the others,
    # which keep their standardized draws.
    given = [3 => [0.5, -0.2], 1 => [-0.4, 0.1]]
    for (name, brmi) in (("level_cor_sampled", sampled), ("level_cor_known", known))
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        @test BRM._ranef_level_correlated(only(ranef_blocks(brm_descriptor(sb))).family)
        model = compiled(sb, name)
        base = 0.3 .* randn(Xoshiro(17), BridgeStan.param_unc_num(model))
        s = brm_start(sb, model, (; r=(; sd=[0.6, 0.3], effects=given)); base)
        effects = ranef_effects(sb, model, s; id=:r)
        @test effects[3] ≈ [0.5, -0.2] rtol=1e-10
        @test effects[1] ≈ [-0.4, 0.1] rtol=1e-10
        block = only(ranef_blocks(brm_descriptor(sb)))
        z = ranef_coordinates(block, String.(BridgeStan.param_unc_names(model)))
        @test s[vec(z[:, [2, 4]])] ≈ base[vec(z[:, [2, 4]])] rtol=1e-12
        # RK solves the same effects through its own level factor.
        rk = RKBRMI(brmi)
        record = only(r for r in BRM._rk_coordinate_records(rk.plan) if r.kind === :ranef)
        @test haskey(record, :level_correlation)
        u = brm_start(rk, (; r=(; sd=[0.6, 0.3], effects=given)))
        values = BRM._rk_constrained_values(rk, u)
        E = BRM._brm_rk_block_effects(values, record, 2)
        levels = BRM._rk_grouping_levels(rk.plan.columns[record.group])
        @test E[findfirst(==(3), levels), :] ≈ [0.5, -0.2] rtol=1e-10
        @test E[findfirst(==(1), levels), :] ≈ [-0.4, 0.1] rtol=1e-10
    end
end

@stestset "per-level effects of a stratified block" begin
    data = (; subject=repeat(["a", "b", "c", "d"], 3), arm=repeat(["p", "p", "q", "q"], 3),
        x=collect(range(-1, 1; length=12)), y=sin.(1:12))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | gr(subject, by = arm))
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test !isnothing(only(ranef_blocks(brm_descriptor(sb))).by)
    model = compiled(sb, "stratified")
    base = 0.3 .* randn(Xoshiro(19), BridgeStan.param_unc_num(model))
    # Each level's effects hold under its own stratum's scales and correlation.
    s = brm_start(sb, model, (; p=(; effects=["b" => [0.3, -0.2], "c" => [-0.5, 0.4]])); base)
    names = String.(BridgeStan.param_names(model; include_tp=true))
    draws = permutedims(BridgeStan.param_constrain(model, s; include_tp=true))
    strata = brm_ranef_draws(brm_descriptor(sb), draws, names; id=:p)
    @test length(strata) == 2
    level(l) = only(e.effects[1, findfirst(==(l), e.levels), :] for e in strata if l in e.levels)
    @test level("b") ≈ [0.3, -0.2] rtol=1e-10
    @test level("c") ≈ [-0.5, 0.4] rtol=1e-10
end

# Each stratum of a stratified block has its own scales and correlation. Stan's
# are read through `brm_ranef_draws`, RK's from its constrained values; strata
# are compared by label. Snag `stratified-block-73231199`.
const STRATA_DATA = (; subject=repeat(["a", "b", "c", "d", "e", "f"], 3),
    arm=repeat(["p", "p", "q", "q", "r", "r"], 3),
    x=collect(range(-1, 1; length=18)), y=sin.(1:18))

function stan_strata(sb, model, s; id)
    names = String.(BridgeStan.param_names(model; include_tp=true))
    draws = permutedims(BridgeStan.param_constrain(model, s; include_tp=true))
    Dict(BRM._brm_transport_level(e.stratum) => (; sd=e.sd[1, :], cor=e.cor[1, :, :],
            effects=Dict(BRM._brm_transport_level(l) => e.effects[1, i, :]
                for (i, l) in enumerate(e.levels)))
        for e in brm_ranef_draws(brm_descriptor(sb), draws, names; id))
end

function rk_strata(rk, u; id)
    record = only(BRM._brm_rk_start_record(rk, r) for r in BRM._rk_coordinate_records(rk.plan)
        if r.kind === :stratified_ranef && r.id === id)
    values = BRM._rk_constrained_values(rk, u)
    K = length(record.margins)
    tau = isnothing(record.scales) ? BRM._brm_rk_declaration_value(values, record.scale) :
        reduce(hcat, [BRM._brm_rk_declaration_value(values, n) for n in record.scales])
    L = isnothing(record.L) ? nothing : BRM._brm_rk_declaration_value(values, record.L)
    E = BRM._brm_rk_block_effects(values, record, K)
    levels = BRM._rk_grouping_levels(rk.plan.columns[record.group])
    Dict(BRM._brm_transport_level(stratum) => (; sd=collect(tau[s, :]),
            cor=isnothing(L) ? ones(1, 1) : L[s] * L[s]',
            effects=Dict(BRM._brm_transport_level(levels[g]) => E[g, :]
                for g in eachindex(levels) if record.level_strata[g] == s))
        for (s, stratum) in enumerate(record.strata))
end

@stestset "scales and correlation of a stratified block, per stratum" begin
    brmi = @brm STRATA_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | gr(subject, by = arm))
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    model = compiled(sb, "stratified_scales")
    rk = RKBRMI(brmi)
    base = 0.3 .* randn(Xoshiro(23), BridgeStan.param_unc_num(model))
    rk_base = 0.3 .* randn(Xoshiro(23), length(BRM._rk_layout_coordinate_names(rk)))
    C = [1.0 0.4; 0.4 1.0]
    C_r = [1.0 -0.3; -0.3 1.0]
    Σ_q = [0.25 0.03; 0.03 0.09]
    at_base = stan_strata(sb, model, base; id=:p)
    rk_at_base = rk_strata(rk, rk_base; id=:p)

    # `sd` and `cor` set every stratum, on both backends.
    inits = (; p=(; sd=[0.6, 0.3], cor=C))
    stan, rk_values = stan_strata(sb, model, brm_start(sb, model, inits; base); id=:p),
        rk_strata(rk, brm_start(rk, inits; base=rk_base); id=:p)
    @test sort!(collect(keys(stan))) == sort!(collect(keys(rk_values))) == ["p", "q", "r"]
    for x in (stan, rk_values), st in ("p", "q", "r")
        @test x[st].sd ≈ [0.6, 0.3] rtol=1e-10
        @test x[st].cor ≈ C rtol=1e-10
    end
    # One value for all margins of all strata (the snag's `sd = 0.4`).
    stan = stan_strata(sb, model, brm_start(sb, model, (; p=(; sd=0.4)); base); id=:p)
    @test all(stan[st].sd ≈ [0.4, 0.4] for st in ("p", "q", "r"))

    # `stratum => value` pairs set the listed strata; the others keep the base.
    inits = (; p=(; sd=["q" => [0.5, 0.2], "p" => 0.7], cor=["r" => C_r],
        effects=["c" => [0.3, -0.2], "a" => [-0.1, 0.4]]))
    saved = deepcopy(inits)
    s, u = brm_start(sb, model, inits; base), brm_start(rk, inits; base=rk_base)
    @test inits == saved
    for (x, x0) in ((stan_strata(sb, model, s; id=:p), at_base),
                    (rk_strata(rk, u; id=:p), rk_at_base))
        @test x["p"].sd ≈ [0.7, 0.7] rtol=1e-10
        @test x["q"].sd ≈ [0.5, 0.2] rtol=1e-10
        @test x["r"].sd ≈ x0["r"].sd rtol=1e-10
        @test x["r"].cor ≈ C_r rtol=1e-10
        @test x["p"].cor ≈ x0["p"].cor rtol=1e-10
        @test x["q"].cor ≈ x0["q"].cor rtol=1e-10
        # Each listed level's effects hold under its own stratum's values.
        @test x["q"].effects["c"] ≈ [0.3, -0.2] rtol=1e-10
        @test x["p"].effects["a"] ≈ [-0.1, 0.4] rtol=1e-10
    end
    # A covariance per stratum sets that stratum's scales and correlation.
    for (x, x0) in (
            (stan_strata(sb, model, brm_start(sb, model, (; p=["q" => Σ_q]); base); id=:p),
                at_base),
            (rk_strata(rk, brm_start(rk, (; p=["q" => Σ_q]); base=rk_base); id=:p),
                rk_at_base))
        @test x["q"].sd ≈ sqrt.(diag(Σ_q)) rtol=1e-10
        @test x["q"].cor .* (x["q"].sd * x["q"].sd') ≈ Σ_q rtol=1e-10
        @test x["p"].sd ≈ x0["p"].sd rtol=1e-10
    end

    # refused: a stratum the block does not have, and a repeated one.
    for f in (i -> brm_start(sb, model, i), i -> brm_start(rk, i))
        @test_throws "not a stratum of this block" f((; p=(; sd=["z" => 0.5])))
        @test_throws "repeats a stratum" f((; p=(; sd=["q" => 0.5, "q" => 0.6])))
    end
    # refused: strata of a block that has none.
    plain = @brm STRATA_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subject)
        y ~ Normal(mu, sigma)
    end
    plain_sb = SBBRMI(plain; mod=@__MODULE__, total_groups=())
    plain_model = compiled(plain_sb, "unstratified_scales")
    @test_throws "is not a stratified" brm_start(plain_sb, plain_model, (; p=(; sd=["q" => 0.5])))
    @test_throws "is not a stratified" brm_start(RKBRMI(plain), (; p=(; cor=["q" => C])))
end

@stestset "a one-margin stratified block with a configured scale prior" begin
    # The snag's shape: `sd(lp, ID) ~ …` selects the generic stratified family.
    brmi = @brm STRATA_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | p | gr(subject, by = arm))
        sd(mu, p) ~ LogNormal(-1, 0.75)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test only(ranef_blocks(brm_descriptor(sb))).family === :ranef_correlated_by_draws_generic
    model = compiled(sb, "stratified_one_margin")
    rk = RKBRMI(brmi)
    inits = (; p=(; sd=0.4, effects=["b" => 0.3, "e" => -0.2]))
    for x in (stan_strata(sb, model, brm_start(sb, model, inits); id=:p),
              rk_strata(rk, brm_start(rk, inits); id=:p))
        @test all(x[st].sd ≈ [0.4] for st in ("p", "q", "r"))
        @test x["p"].effects["b"] ≈ [0.3] rtol=1e-10
        @test x["r"].effects["e"] ≈ [-0.2] rtol=1e-10
    end
    inits = (; p=(; sd=["r" => 0.25]))
    stan = stan_strata(sb, model, brm_start(sb, model, inits); id=:p)
    @test stan["r"].sd ≈ [0.25] rtol=1e-12
    @test stan["p"].sd ≈ [1.0] rtol=1e-12
end

@stestset "per-level effects place exact totals" begin
    brmi = lone_eta_pk(NLME_PLATE_DATA)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=:auto)
    blocks = Dict(b.predictor => b for b in total_effect_blocks(sb))
    model = compiled(sb, "lone_eta_totals")
    inits = (; log_CL=(; Intercept=log(2.7), weight=0.75), log_V=(; Intercept=log(31.5)),
        eta_cl=(; sd=0.3, effects=["a" => 0.2, "c" => -0.1]),
        eta_v=(; effects=["b" => 0.15]), sigma=0.7)
    s = brm_start(sb, model, inits)
    θ, names = constrained(model, s)
    # Each total is its subject's population value plus its effect: the
    # deviation `total - A (W[g,:] .* β)` that `brm_ranef_draws` reports.
    population = Dict(:log_CL => log(2.7) .+ 0.75 .* NLME_PLATE_DATA.weight,
        :log_V => fill(log(31.5), 3))
    deviation = Dict(:log_CL => [0.2, 0.0, -0.1], :log_V => [0.0, 0.15, 0.0])
    for lp in (:log_CL, :log_V)
        totals = BRM._total_coordinates(sb, blocks[lp], names).totals
        @test vec(θ[totals]) ≈ population[lp] .+ deviation[lp] rtol=1e-12
    end
    @test θ[BRM._total_coordinates(sb, blocks[:log_CL], names).scales] ≈ [0.3] rtol=1e-12
    # Effects alone count the unaddressed absorbed coefficients as 0.
    s = brm_start(sb, model, (; eta_v=(; effects=["b" => 0.15])))
    θ, names = constrained(model, s)
    @test vec(θ[BRM._total_coordinates(sb, blocks[:log_V], names).totals]) ≈
        [0.0, 0.15, 0.0] rtol=1e-12 atol=1e-14
    # refused: the totals of an omitted response are drawn in generated quantities.
    prior_sb = SBBRMI(lone_eta_pk(Base.structdiff(NLME_PLATE_DATA, (; dv=nothing)));
        mod=@__MODULE__, total_groups=:auto)
    @test_throws "generated quantities" brm_start(prior_sb, compiled(prior_sb, "lone_eta_prior"),
        (; eta_v=(; effects=["b" => 0.15])))
end

@stestset "per-level effects of a sum-to-zero block" begin
    data = (; site=repeat(["s1", "s2", "s3", "s4"], 4), x=collect(range(-1, 1; length=16)),
        y=cos.(1:16))
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | a | site)
        y ~ Normal(mu, sigma)
    end
    effects = ["s1" => 0.5, "s2" => -0.3, "s3" => 0.9, "s4" => 0.1]
    for s2z_rho in (0.0, 0.5, 1.0)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), s2z_groups=[:site], s2z_rho)
        block = only(BRM.s2z_effect_blocks(brm_descriptor(sb).plan))
        model = compiled(sb, "s2z_$(s2z_rho)")
        s = brm_start(sb, model, (; mu=(; Intercept=1.2), a=(; sd=0.7, effects)))
        θ, names = constrained(model, s)
        c = BRM._s2z_coordinates(brm_descriptor(sb).plan, block, names)
        levels = brm_descriptor(sb).plan.preproc[block.group_index].const_.levels
        e = [Dict(effects)[l] for l in levels]
        # The deviations from the mean, and `theta = β + B μ` with `μ` the mean.
        @test θ[c.scales] ≈ [0.7] rtol=1e-12
        @test BRM._s2z_block_deviations(block, θ[c.contrasts[:, 1]], 0.7, 1) ≈
            e .- sum(e) / 4 rtol=1e-10
        @test θ[c.theta] ≈ [1.2 + only(block.B) * sum(e) / 4] rtol=1e-12
        # Stan's own deviation carrier agrees.
        tp = String.(BridgeStan.param_names(model; include_tp=true))
        r = BridgeStan.param_constrain(model, s; include_tp=true)[
            findall(n -> startswith(n, "$(block.deviations)."), tp)]
        @test r ≈ e .- sum(e) / 4 rtol=1e-10
        # refused: the deviations need every level's effect.
        @test_throws "all 4 levels" brm_start(sb, model, (; a=(; effects=effects[1:3])))
    end
end

@stestset "invalid per-level effects are refused" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    model = compiled(sb, "plate_pk")
    for backend in ((rk,), (sb, model))
        # refused: a level the grouping column never takes.
        @test_throws "no level" brm_start(backend..., (; pk=(; effects=["z" => [0.1, 0.2]])))
        # refused: one effect per margin of `|pk|`.
        @test_throws DimensionMismatch brm_start(backend...,
            (; pk=(; effects=["a" => [0.1]])))
        @test_throws DimensionMismatch brm_start(backend...,
            (; pk=(; effects=["a" => 0.1])))
        # refused: a level given twice, a non-finite effect, effects without levels.
        @test_throws "repeats a level" brm_start(backend...,
            (; pk=(; effects=["a" => [0.1, 0.2], "a" => [0.0, 0.0]])))
        @test_throws "finite" brm_start(backend..., (; pk=(; effects=["a" => [NaN, 0.2]])))
        @test_throws "level => value pairs" brm_start(backend..., (; pk=(; effects=[0.1, 0.2])))
        # refused: `effects` is no coefficient of a linear predictor.
        @test_throws ErrorException brm_start(backend..., (; log_V=(; effects=["a" => 0.1])))
    end
end

@stestset "invalid addresses and values are refused" begin
    brmi = nlme_plate_pk(NLME_PLATE_DATA)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    model = compiled(sb, "plate_pk")
    for backend in ((rk,), (sb, model))
        # refused: no such parameter, coefficient or block part.
        @test_throws ErrorException brm_start(backend..., (; tau=0.1))
        @test_throws ErrorException brm_start(backend..., (; log_V=(; weight=0.1)))
        @test_throws ErrorException brm_start(backend..., (; pk=(; scale=0.1)))
        # refused: a scale or covariance outside its support.
        @test_throws ErrorException brm_start(backend..., (; pk=(; sd=-0.1)))
        @test_throws ErrorException brm_start(backend..., (; pk=[1.0 2.0; 2.0 1.0]))
        @test_throws ErrorException brm_start(backend...,
            (; pk=(; cor=[1.0 0.2; 0.3 1.0])))
        # refused: the block has two margins.
        @test_throws DimensionMismatch brm_start(backend..., (; pk=fill(0.1, 3, 3)))
        # refused: a linear predictor's coefficients are a NamedTuple.
        @test_throws ErrorException brm_start(backend..., (; log_V=1.0))
        @test_throws ErrorException brm_start(backend..., (; sigma=NaN))
    end
    @test_throws DimensionMismatch brm_start(rk, PK_INITS; base=zeros(2))
end
