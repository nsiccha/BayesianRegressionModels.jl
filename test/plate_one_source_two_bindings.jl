# test/plate_one_source_two_bindings.jl — one vector read through two
# `groups` bindings in one `@plate` cell.
#
# A cell's sum over its suppressed rows, one sum per class with a coefficient
# each: `eta[rows1[k]]` and `eta[rows2[k]]` read the same linear predictor (and
# `elig[...]` the same data column) at different rows. Each read is its own
# gather. The twin reads every suppressed row through one binding and weights
# it by its class indicator, which is the same density. Snag
# reading-one-form-4a7413e7 (reporter ArbeitsplatzAtlas).
#
# Run: julia --startup-file=no --project=test test/plate_one_source_two_bindings.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

# 24 rows in 4 groups of 6, classes alternating; the suppressed rows of groups
# 1, 3 and 4 hold both classes.
function two_binding_data(; seed=1)
    rng = Xoshiro(seed)
    n = 24
    g = repeat(1:4, inner=6)
    class = repeat([1, 2], 12)
    x = randn(rng, n)
    elig = 0.5 .+ 0.5 .* rand(rng, n)
    sup = falses(n); sup[[1, 2, 4, 13, 14, 19, 20, 21]] .= true
    cells = unique(g[sup])
    label(c) = [s && ci == c ? gi : 0 for (s, ci, gi) in zip(sup, class, g)]
    lam = exp.(1 .+ 0.4 .* x)
    R = [rand(rng, Poisson(sum(lam[sup .& (g .== k)]))) for k in cells]
    (; x, g, elig, g_sup1=label(1), g_sup2=label(2), g_sup=[s ? gi : 0 for (s, gi) in zip(sup, g)],
       d1=Float64.(class .== 1), d2=Float64.(class .== 2), cells, R, sup, class)
end
columns(d) = (; d.x, d.g, d.elig, d.g_sup1, d.g_sup2, d.g_sup, d.d1, d.d2, d.cells, d.R)

two_bindings(d) = @brm columns(d) begin
    lw1 ~ Normal(0.0, 1.0)
    lw2 ~ Normal(0.0, 1.0)
    eta ~ 1 + x + (1 | g)
    rows1 = groups(g_sup1 => cells)
    rows2 = groups(g_sup2 => cells)
    @plate for k in eachindex(R)
        R[k] ~ Poisson(exp(lw1) * sum(exp.(eta[rows1[k]]) .* elig[rows1[k]]) +
                       exp(lw2) * sum(exp.(eta[rows2[k]]) .* elig[rows2[k]]))
    end
end
# A cell local spelled like the second read's generated cell name.
two_bindings_local(d) = @brm columns(d) begin
    lw1 ~ Normal(0.0, 1.0)
    lw2 ~ Normal(0.0, 1.0)
    eta ~ 1 + x + (1 | g)
    rows1 = groups(g_sup1 => cells)
    rows2 = groups(g_sup2 => cells)
    @plate for k in eachindex(R)
        eta_rows2 = sum(exp.(eta[rows2[k]]) .* elig[rows2[k]])
        R[k] ~ Poisson(exp(lw1) * sum(exp.(eta[rows1[k]]) .* elig[rows1[k]]) + exp(lw2) * eta_rows2)
    end
end
indicator_twin(d) = @brm columns(d) begin
    lw1 ~ Normal(0.0, 1.0)
    lw2 ~ Normal(0.0, 1.0)
    eta ~ 1 + x + (1 | g)
    rows = groups(g_sup => cells)
    @plate for k in eachindex(R)
        R[k] ~ Poisson(sum(exp.(eta[rows[k]]) .* elig[rows[k]] .*
                           (exp(lw1) .* d1[rows[k]] .+ exp(lw2) .* d2[rows[k]])))
    end
end

const FOLDER = joinpath(tempdir(), "brm-plate-one-source-two-bindings")
function compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    mkpath(FOLDER)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(FOLDER, label * "_" * string(hash(code); base=16) * ".stan"))
    (; sb, code, stan, names=BridgeStan.param_unc_names(stan.model))
end

# An independent reading at the unconstrained point (Stan's coordinate order).
function oracle(d, names, q)
    at(name) = q[coordinate(names, name)]
    lw1, lw2 = at("lw1"), at("lw2")
    beta = [at("pop_eta_beta_pop.1"), at("pop_eta_beta_pop.2")]
    log_scale = at("r_eta_g_log_scale")
    xi = [at("r_eta_g_xi.$j") for j in 1:4]
    eta = beta[1] .+ beta[2] .* d.x .+ exp(log_scale) .* xi[d.g]
    lp = logpdf(Normal(), lw1) + logpdf(Normal(), lw2) + sum(logpdf.(Normal(), beta)) +
        logpdf(Normal(), log_scale) + sum(logpdf.(Normal(), xi))
    for (k, r) in zip(d.cells, d.R)
        rows(c) = d.sup .& (d.g .== k) .& (d.class .== c)
        mean = exp(lw1) * sum(exp.(eta[rows(1)]) .* d.elig[rows(1)]) +
               exp(lw2) * sum(exp.(eta[rows(2)]) .* d.elig[rows(2)])
        lp += logpdf(Poisson(mean), r)
    end
    lp
end

@stestset "SBBRMI: two bindings of one predictor equal the indicator twin" begin
    d = two_binding_data()
    c = compiled(two_bindings(d), "two")
    twin = compiled(indicator_twin(d), "twin")
    local_ = compiled(two_bindings_local(d), "local")
    @test c.names == twin.names == local_.names
    # Each read is its own gather of the predictor's rows, and of the column's.
    for gather in ("eta", "eta_rows2", "elig", "elig_rows2")
        @test occursin("kernel_R_cell_$(gather)_ragged;", c.code)
    end
    rng = Xoshiro(3)
    for _ in 1:4
        q = 0.3 .* randn(rng, length(c.names))
        value = BridgeStan.log_density(c.stan.model, q; propto=false)
        @test value ≈ oracle(d, c.names, q) atol=1e-8 rtol=1e-12
        @test value ≈ BridgeStan.log_density(twin.stan.model, q; propto=false) atol=1e-10 rtol=1e-12
        @test value ≈ BridgeStan.log_density(local_.stan.model, q; propto=false) atol=1e-10 rtol=1e-12
    end
    # A frozen replay onto the same rows keeps the density.
    again = reprocess(c.sb, columns(d))
    stan = BRM.stan_instantiate(again; path=joinpath(FOLDER, "replay_" *
        string(hash(BRM.stan_code(again)); base=16) * ".stan"))
    q = 0.2 .* randn(Xoshiro(4), length(c.names))
    @test BridgeStan.log_density(stan.model, q; propto=false) ≈
        BridgeStan.log_density(c.stan.model, q; propto=false) atol=1e-10
end

@stestset "RKBRMI agrees with SBBRMI: value and gradient" begin
    d = two_binding_data()
    for (model, label) in ((two_bindings, "rk_two"), (two_bindings_local, "rk_local"))
        brmi = model(d)
        rk = RKBRMI(brmi)
        problem = rk_logdensity_problem(check_rk_source_roundtrip(rk);
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        c = compiled(brmi, label)
        t = brm_coordinate_transport(rk, c.sb, String.(c.names))
        rng = Xoshiro(17)
        for _ in 1:3
            u = 0.3 .* randn(rng, LogDensityProblems.dimension(problem))
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_value, stan_gradient = BridgeStan.log_density_gradient(c.stan.model,
                brm_rk_to_stan(t, u); propto=false, jacobian=true)
            @test value ≈ stan_value atol=1e-10 rtol=1e-11
            @test gradient ≈ brm_stan_to_rk(t, stan_gradient) atol=1e-9 rtol=1e-10
        end
    end
end

@stestset "a response observed through two bindings is refused" begin
    # refused: a binding that observes a response places every entry in a cell
    # (decision `0q8hcwq`), so a second observing binding counts every entry
    # twice. Reading a vector through several bindings stays allowed (above).
    twice = :(@brm cols begin
        eta ~ 1 + x
        rows1 = groups(g => cells)
        rows2 = groups(h => cells)
        @plate for k in eachindex(cnt)
            y[rows1[k]] ~ Poisson(exp.(eta[rows1[k]]))
            y[rows2[k]] ~ Poisson(exp.(eta[rows2[k]]))
        end
    end)
    @test_throws "observes `y` through both `rows1` and `rows2`" macroexpand(@__MODULE__, twice)
end
