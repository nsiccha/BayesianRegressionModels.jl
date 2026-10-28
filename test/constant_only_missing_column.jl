# test/constant_only_missing_column.jl — a data column that only a
# `name = constant(expr)` reads may hold `missing` (snag
# `constant-only-co-b89230c6`).
#
# The model reads a constant's value, never its inputs: `expr` takes them whole
# when the constant resolves, so `c = constant(mean(skipmissing(y_data)))`
# gives one finite number however many cells of `y_data` are missing. SBBRMI
# and RKBRMI resolve constants before they collect data, so they built such a
# model; the passes over the unresolved model (`brm_descriptor` and with it
# `brm_start`, `brm_description`, `prune_responses`, `modeled_covariates`)
# collected `y_data` as model data and refused its missing cells. Each is
# checked against the twin whose `y_data` holds only the published cells: the
# constant, and so everything the model reads, is the same.
#
# Run: julia --startup-file=no --project=test test/constant_only_missing_column.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics: mean
const stan_code = BRM.stan_code
const stan_data = BRM.stan_data

const COMPLETED = @brm begin
    c = constant(mean(skipmissing(y_data)))
    mu ~ 1 + x
    sigma ~ Exponential(1)
    m = mu + c
    mi(y) ~ Normal(m, sigma)
end

const Y = Union{Missing,Float64}[0.3, missing, 1.6, 0.4, missing, 2.0, 0.1, 0.8]
const X = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, -0.9, -0.3]
const DATA = (; x=X, y=Y, y_data=Y)
const TWIN = (; x=X, y=Y, y_data=collect(skipmissing(Y)))
const INITS = (; mu=(; Intercept=1.0, x=0.5), sigma=0.3)

# The data the preparation over the unresolved model collects, and the data
# the backend collects once its constants resolve, less the constants' values.
function collected_columns(brmi)
    resolved, constants = BRM._brm_resolve_constants(brmi)
    (Set(keys(BRM._brm_backend_context(brmi).data)),
     setdiff(Set(keys(BRM._brm_backend_context(resolved).data)), (c.name for c in constants)))
end

@stestset "a column only a constant reads is not model data" begin
    brmi = COMPLETED(DATA)
    unresolved, backend = collected_columns(brmi)
    @test !(:y_data in unresolved)
    @test unresolved == backend
    @test collected_columns(COMPLETED(TWIN)) == (unresolved, backend)
    program = BRM._brm_prepare_program(brmi)
    @test program.order == BRM._brm_prepare_program(COMPLETED(TWIN)).order
end

@stestset "descriptor, description and Stan start equal the complete twin" begin
    sb = SBBRMI(COMPLETED(DATA); mod=@__MODULE__)
    twin = SBBRMI(COMPLETED(TWIN); mod=@__MODULE__)
    @test stan_data(sb)[:c] == mean(skipmissing(Y))
    @test stan_code(sb) == stan_code(twin)
    @test isequal(stan_data(sb), stan_data(twin))

    d = brm_descriptor(sb)
    dt = brm_descriptor(twin)
    @test brm_columns(d) == brm_columns(dt)
    @test !(:y_data in brm_columns(d))
    @test [(i.name, i.column, i.transform) for i in d.inputs] ==
        [(i.name, i.column, i.transform) for i in dt.inputs]
    input = only(i for i in d.inputs if i.name === :c)
    @test input.transform === :constant && isnothing(input.column)
    @test brm_description(d) isa BRMDescription

    model = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(mktempdir(), "constant_only_missing.stan")).model
    s = brm_start(sb, model, INITS)
    @test s == brm_start(twin, model, INITS)
    θ = BridgeStan.param_constrain(model, s)
    names = String.(BridgeStan.param_names(model))
    population(c) = θ[only(brm_population_effect_coordinates(d, :mu, names;
        coefficient=c).coordinates)]
    @test population(:Intercept) ≈ 1.0 rtol=1e-12
    @test population(:x) ≈ 0.5 rtol=1e-12
    @test θ[only(brm_output_coordinates(d, :sigma, names))] ≈ 0.3 rtol=1e-12
end

@stestset "RKBRMI, response pruning and covariate inventory equal the twin" begin
    brmi = COMPLETED(DATA)
    rk = RKBRMI(brmi)
    rk_twin = RKBRMI(COMPLETED(TWIN))
    @test brm_start(rk, INITS) == brm_start(rk_twin, INITS)
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    p = rk_logdensity_problem(rk; ad_backend)
    p_twin = rk_logdensity_problem(rk_twin; ad_backend)
    for u in (zeros(LogDensityProblems.dimension(p)), fill(0.3, LogDensityProblems.dimension(p)))
        @test LogDensityProblems.logdensity(p, u) == LogDensityProblems.logdensity(p_twin, u)
    end
    @test sprint(show, BRM.prune_responses(brmi; keep=(:y,))) ==
        sprint(show, BRM.prune_responses(COMPLETED(TWIN); keep=(:y,)))
    @test modeled_covariates(brmi) == modeled_covariates(COMPLETED(TWIN))
end

@stestset "a masked HSGP axis that a constant also reads keeps its mask" begin
    t = collect(1.0:12.0)
    carried = t .<= 6.0
    gaps = ifelse.(carried, t, missing)
    y = [0.1, 0.4, 0.2, 0.7, 0.5, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    builder = @brm begin
        t0 = constant(mean(skipmissing(t)))
        mu ~ 1 + hsgp(t; k=5, rows=carried)
        sigma ~ Exponential(1)
        m = mu + t0
        y ~ Normal(m, sigma)
    end
    brmi = builder((; t=gaps, y, carried))
    unresolved, backend = collected_columns(brmi)
    @test unresolved == backend
    sb = SBBRMI(brmi; mod=@__MODULE__)
    twin = SBBRMI(builder((; t=ifelse.(carried, t, 123.0), y, carried)); mod=@__MODULE__)
    @test stan_code(sb) == stan_code(twin)
    @test stan_data(sb)[:t0] == mean(t[carried])
    @test brm_columns(brm_descriptor(sb)) == brm_columns(brm_descriptor(twin))
end

@stestset "a multi-membership source a constant also reads stays Julia-side" begin
    builder = @brm begin
        n1 = constant(length(unique(g1)))
        sigma ~ Exponential(1)
        loc ~ 1 + (1 | mm(g1, g2))
        m = loc + n1
        y ~ Normal(m, sigma)
    end
    brmi = builder((; g1=["a", "a", "b"], g2=["b", "c", "c"], y=[0.1, 0.2, 0.3]))
    unresolved, backend = collected_columns(brmi)
    @test !(:g1 in unresolved)
    @test unresolved == backend
end

@stestset "a column the model itself reads still refuses its missing cells" begin
    # refused: BRM never silently drops a row the model reads (`_brm_data_vec`);
    # the constant reading the same column does not change that.
    read = @brm begin
        c = constant(mean(skipmissing(y_data)))
        mu ~ 1 + x + y_data
        sigma ~ Exponential(1)
        m = mu + c
        mi(y) ~ Normal(m, sigma)
    end
    message = "data column `y_data` contains `missing` values"
    @test_throws message SBBRMI(read(DATA); mod=@__MODULE__)
    @test_throws message BRM._brm_prepare_program(read(DATA))
end
