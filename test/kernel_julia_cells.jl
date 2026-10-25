# Julia-first kernel cells: one authored Julia body runs on RKBRMI as written,
# and SBBRMI gives StanBlocks the Stan spellings of its Julia callables and
# Distributions observation constructors.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LogExpFunctions

# A caller's own function that happens to share a Julia name keeps its
# meaning: callables are matched by identity, never by spelling.
module OwnLogistic
logistic(x) = 2x
end

# One helper name, two implementations: StanBlocks traces the `@deffun`, the
# RK backend calls the Julia method.
module DualHelper
using BayesianRegressionModels, Distributions, LogExpFunctions, StanBlocks
StanBlocks.@deffun saturating(x::vector[n], s::real) = 2.0 * inv_logit(s * x) - 1.0
saturating(x::AbstractVector, s::Real) = 2.0 .* logistic.(s .* x) .- 1.0
function build(data)
    @brm data begin
        sigma ~ Exponential(1)
        base ~ Normal(0, 1)
        log_s ~ Normal(0, 0.5)
        amp ~ Normal(0, 1)
        @plate for i in eachindex(c)
            loc[i] = base .+ amp .* saturating(c[i], exp(log_s))
            y[i] ~ Normal(loc[i], sigma)
        end
    end
end
end

const STAN = BRM._sb_cell_statements

@stestset "cell callables take their Stan spelling by identity" begin
    scope = @__MODULE__
    spell(ex) = only(STAN(Any[ex], scope))
    @test spell(:(e = 2.0 .* logistic.(s .* x) .- 1.0)) ==
        :(e = 2.0 .* inv_logit.(s .* x) .- 1.0)
    @test spell(:(z = cumsum(x))) == :(z = cumulative_sum(x))
    @test spell(:(z = logsumexp(x) .+ log1pexp.(x))) ==
        :(z = log_sum_exp(x) .+ log1p_exp.(x))
    @test spell(:(z = LogExpFunctions.logistic(a))) == :(z = inv_logit(a))
    @test spell(:(z = fill(a, n) .+ zeros(n))) ==
        :(z = rep_vector(a, n) .+ rep_vector(0.0, n))
    @test spell(:(z = ones(m, n))) == :(z = rep_matrix(1.0, m, n))
    @test spell(:(z = vcat(a, b, c))) == :(z = append_row(append_row(a, b), c))
    # Stan spellings, Julia names Stan shares, and unbound names pass through.
    for ex in (:(e = 2.0 * inv_logit(s * x) - 1.0), :(z = rep_vector(a, 3)),
            :(z = exp.(x) .+ sum(x)), :(z = my_helper(x; k = 2)))
        @test spell(ex) == ex
    end
    @test only(STAN(Any[:(z = logistic.(x))], OwnLogistic)) == :(z = logistic.(x))
end

@stestset "cell observations take Stan families" begin
    scope = @__MODULE__
    spell(ex) = only(STAN(Any[ex], scope))
    @test spell(:(y ~ Normal(mu, sigma))) == :(y ~ normal(mu, sigma))
    @test spell(:(y ~ LogNormal(mu, sigma))) == :(y ~ lognormal(mu, sigma))
    # Distributions parameterizations follow the top-level translation table.
    @test spell(:(y ~ Exponential(theta))) == :(y ~ exponential(1.0 ./ theta))
    @test spell(:(y ~ censored(Normal(mu, s); lower = lo))) ==
        :(y ~ censored(normal(mu, s); lower = lo))
    @test spell(:(y ~ weighted(Normal(logistic.(mu), s), w))) ==
        :(y ~ weighted(normal(inv_logit.(mu), s), w))
    for ex in (:(y ~ normal(mu, sigma)), :(y ~ weighted(normal, w, mu, sigma)),
            :(y ~ logistic(mu, s)))
        @test spell(ex) == ex
    end
end

const DATA = (; c=[[0.1, 0.5, 1.0], [0.2, 0.4], [0.0, 0.3, 0.6, 0.9]],
    y=[[0.2, 0.1, -0.3], [0.4, 0.0], [0.1, 0.2, 0.3, -0.1]])

julia_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    log_s ~ Normal(0, 0.5)
    amp ~ Normal(0, 1)
    @plate for i in eachindex(c)
        effect = 2.0 .* logistic.(exp(log_s) .* c[i]) .- 1.0
        loc[i] = base .+ amp .* effect
        y[i] ~ Normal(loc[i], sigma)
    end
end

stan_spelled_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    log_s ~ Normal(0, 0.5)
    amp ~ Normal(0, 1)
    @plate for i in eachindex(c)
        effect = 2.0 .* inv_logit.(exp(log_s) .* c[i]) .- 1.0
        loc[i] = base .+ amp .* effect
        y[i] ~ normal(loc[i], sigma)
    end
end

function saturating_oracle(data, names)
    index = Dict(n => findfirst(==(n), names) for n in (:sigma, :base, :log_s, :amp))
    u -> begin
        sigma, base, log_s, amp = exp(u[index[:sigma]]), u[index[:base]],
            u[index[:log_s]], u[index[:amp]]
        logpdf(Exponential(1), sigma) + u[index[:sigma]] + logpdf(Normal(0, 1), base) +
            logpdf(Normal(0, 0.5), log_s) + logpdf(Normal(0, 1), amp) +
            sum(sum(logpdf.(Normal.(base .+ amp .* (2.0 .* logistic.(exp(log_s) .* c) .- 1.0),
                sigma), y)) for (c, y) in zip(data.c, data.y))
    end
end

const MAPPING = [:sigma => "sigma", :base => "base", :log_s => "log_s", :amp => "amp"]

@stestset "one Julia cell runs on RK and compiles to the Stan cell" begin
    saved = deepcopy(DATA)
    julia, stan_spelled = julia_cell(DATA), stan_spelled_cell(DATA)
    julia_code = BRM.stan_code(SBBRMI(julia; mod=@__MODULE__, total_groups=()))
    @test julia_code == BRM.stan_code(SBBRMI(stan_spelled; mod=@__MODULE__, total_groups=()))
    @test occursin("inv_logit", julia_code) && !occursin("logistic", julia_code)
    backend, problem = consumer_problem(julia)
    names = coordinate_names(backend.model.layout)
    @test sort(names) == [:amp, :base, :log_s, :sigma]
    oracle = saturating_oracle(DATA, names)
    stan = consumer_stan(julia, "julia-first-cell")
    # The StanBlocks spelling stays accepted on RK as a compatibility layer.
    stan_backend, stan_problem = consumer_problem(stan_spelled)
    @test coordinate_names(stan_backend.model.layout) == names
    for u in ([0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 0.4], [-0.4, 0.25, -0.3, 0.2])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, MAPPING, backend, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_value, stan_gradient = LogDensityProblems.logdensity_and_gradient(stan_problem, u)
        @test stan_value ≈ value atol=1e-12 rtol=1e-12
        @test stan_gradient ≈ gradient atol=1e-12 rtol=1e-12
    end
    @test isequal(DATA, saved)
end

prims_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    log_s ~ Normal(0, 0.5)
    amp ~ Normal(0, 1)
    @plate for i in eachindex(c)
        z = vcat(0.0, cumsum(abs2.(c[i])))
        w = fill(base, length(c[i])) .+ zeros(length(c[i]))
        loc[i] = w .+ amp .* z[2:(length(c[i]) + 1)] .+
            exp(log_s) .* logsumexp(c[i]) .+ 0.1 .* log1pexp.(c[i])
        y[i] ~ Normal(loc[i], sigma)
    end
end

@stestset "Julia constructors and reductions agree on RK and Stan" begin
    brmi = prims_cell(DATA)
    backend, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "julia-first-prims")
    code = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
    for name in ("append_row", "cumulative_sum", "rep_vector", "log_sum_exp", "log1p_exp")
        @test occursin(name, code)
    end
    for u in ([0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 0.4], [-0.4, 0.25, -0.3, 0.2])
        check_consumer_stan(problem, stan, MAPPING, backend, u)
    end
end

# A model-scope value is Julia too: `@brm` holds the callable it evaluated,
# and SBBRMI spells it through the cells' table. Three areal fields joined
# once at model scope and read per cell, as a zone model with one field per
# branch group reads them.
const FIELDS = (; zone=collect(1:6), from=[1, 2, 3, 4, 5, 1], to=[2, 3, 4, 5, 6, 3],
    uidx=[[1, 2, 3], [8, 10], [13, 17, 18, 3]],
    y=[[0.2, -0.1, 0.3], [0.5, 0.0], [-0.4, 0.1, 0.2, 0.3]])

joined_fields(data) = @brm data begin
    sigma ~ Exponential(1)
    u_a ~ 0 + car(zone; edges=(from, to), type=:bym2)
    u_b ~ 0 + car(zone; edges=(from, to), type=:bym2)
    u_c ~ 0 + car(zone; edges=(from, to), type=:bym2)
    u = vcat(u_a, u_b, u_c)
    @plate for s in eachindex(y)
        y[s] ~ Normal(u[uidx[s]], sigma)
    end
end

joined_in_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    u_a ~ 0 + car(zone; edges=(from, to), type=:bym2)
    u_b ~ 0 + car(zone; edges=(from, to), type=:bym2)
    u_c ~ 0 + car(zone; edges=(from, to), type=:bym2)
    @plate for s in eachindex(y)
        y[s] ~ Normal(vcat(u_a, u_b, u_c)[uidx[s]], sigma)
    end
end

# Two ordinary LPs joined at model scope, which RKPPL lowers today.
const LPS = (; x=[0.1, 0.5, 1.0, 1.5], idx=[[1, 5], [2, 6, 7], [8]],
    y=[[0.2, -0.1], [0.5, 0.0, 0.3], [-0.4]])

joined_lps(data) = @brm data begin
    sigma ~ Exponential(1)
    m_a ~ 1 + x
    m_b ~ 1 + x
    m = vcat(m_a, m_b)
    @plate for s in eachindex(y)
        y[s] ~ Normal(m[idx[s]], sigma)
    end
end

scope_calls(data) = @brm data begin
    sigma ~ Exponential(1)
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    steps = cumsum([a, b, a])
    level = fill(a, 3) + zeros(3)
    p = logistic([a, b, a])
    y ~ Normal(steps + level + p + logsumexp([a, b]), sigma)
end

function check_transported(brmi, name)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    rk, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, name)
    transport = brm_coordinate_transport(rk, sb, BridgeStan.param_unc_names(stan.model))
    for u in (zeros(length(transport)), collect(range(-0.43, 0.51; length=length(transport))))
        stan_u = brm_rk_point_to_stan(transport, u)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_gradient = similar(stan_u)
        stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u, stan_gradient;
            propto=false, jacobian=true)
        @test value ≈ stan_value + transport.logdensity_offset atol=1e-9 rtol=1e-12
        @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=1e-8 rtol=1e-10
    end
end

@stestset "model-scope values take the cells' Stan spellings" begin
    brmi = joined_fields(FIELDS)
    code = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
    @test occursin("u = append_row(append_row(u_a, u_b), u_c);", code)
    @test !occursin("vcat", code)
    # The same density as the cell spelling, at the same coordinates.
    names, values = stan_density_points(brmi, "model-scope-vcat")
    cell_names, cell_values = stan_density_points(joined_in_cell(FIELDS), "in-cell-vcat")
    @test names == cell_names
    @test values ≈ cell_values atol=1e-10 rtol=1e-12
    # Not built on RKBRMI yet: RKPPL refuses a model-scope `vcat` of the
    # values `car` submodels return (ReactiveKernels snag lower-a-brm-emit-2136dcc0);
    # the cell spelling above runs there.
    @test_broken (RKBRMI(brmi); true)
    # Joined ordinary LPs run the authored Julia on RKBRMI, equal to SBBRMI.
    lps = joined_lps(LPS)
    @test occursin("m = append_row(m_a, m_b);",
        BRM.stan_code(SBBRMI(lps; mod=@__MODULE__, total_groups=())))
    check_transported(lps, "model-scope-vcat-lps")
    # The rest of the table at model scope, compiled by stanc.
    calls = BRM.stan_code(SBBRMI(scope_calls((; y=[0.3, -0.2, 0.5])); mod=@__MODULE__,
        total_groups=()))
    for spelled in ("cumulative_sum([a, b, a]')", "rep_vector(a, 3)", "rep_vector(0.0, 3)",
            "inv_logit([a, b, a]')", "log_sum_exp([a, b]')")
        @test occursin(spelled, calls)
    end
    @test consumer_stan(scope_calls((; y=[0.3, -0.2, 0.5])), "model-scope-calls") !== nothing
end

@stestset "one helper name dispatches to each backend's implementation" begin
    brmi = DualHelper.build(DATA)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    oracle = saturating_oracle(DATA, names)
    stan = consumer_stan(brmi, "julia-first-helper"; mod=DualHelper)
    @test occursin("saturating", BRM.stan_code(SBBRMI(brmi; mod=DualHelper, total_groups=())))
    for u in ([0.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, 0.4])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, MAPPING, backend, u)
    end
end
