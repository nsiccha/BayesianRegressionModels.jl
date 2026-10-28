# test/plate_subset_response_sums.jl — a response on some of a predictor's
# rows, the other rows observed only through group sums.
#
# Suppressed cells of a count table whose margins are published: the published
# rows are observed through a model-scope gather (`lam_pub = lam[pos_pub]`), and
# each cell's sum of the suppressed rows in a plate that reads the same linear
# predictor through a full-length grouping column. That column gives each
# suppressed row its cell's label and each published row a label no cell lists,
# so no cell reads a published row (decision `0z97g0b`). No latent stands for a
# suppressed count. Snag mi-completion-wi-54222432.
#
# Run: julia --startup-file=no --project=test test/plate_subset_response_sums.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

# 20 rows in 4 groups of 5; rows 2 and 3 of group 1 and 1, 4 and 5 of group 3
# are suppressed.
function subset_data(; seed=1)
    rng = Xoshiro(seed)
    n = 20
    g = repeat(1:4, inner=5)
    x = randn(rng, n)
    lam = exp.(2 .+ 0.5 .* x)
    y = [rand(rng, NegativeBinomial(5.0, 5.0 / (5.0 + l))) for l in lam]
    sup = falses(n); sup[[2, 3, 11, 14, 15]] .= true
    cells = unique(g[sup])
    g_sup = [s ? gi : 0 for (s, gi) in zip(sup, g)]
    R = [sum(y[sup .& (g .== k)]) for k in cells]
    (; x, g, pos_pub=findall(!, sup), y_pub=y[.!sup], g_sup, cells, R, sup)
end
columns(d) = (; d.x, d.g, d.pos_pub, d.y_pub, d.g_sup, d.cells, d.R)

poisson_model(d) = @brm columns(d) begin
    log(lam) ~ 1 + x + (1 | g)
    lam_pub = lam[pos_pub]
    y_pub ~ Poisson(lam_pub)
    rows = groups(g_sup => cells)
    @plate for k in eachindex(R)
        R[k] ~ Poisson(sum(lam[rows[k]]))
    end
end
nb_model(d) = @brm columns(d) begin
    phi ~ Exponential(0.2)
    log(lam) ~ 1 + x + (1 | g)
    lam_pub = lam[pos_pub]
    y_pub ~ NegativeBinomial2(lam_pub, phi)
    rows = groups(g_sup => cells)
    @plate for k in eachindex(R)
        R[k] ~ NegativeBinomial2(sum(lam[rows[k]]), phi)
    end
end

const FOLDER = joinpath(tempdir(), "brm-plate-subset-response-sums")
function compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    mkpath(FOLDER)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(FOLDER, label * "_" * string(hash(code); base=16) * ".stan"))
    (; sb, stan, names=BridgeStan.param_unc_names(stan.model))
end
nb2(mu, phi) = NegativeBinomial(phi, phi / (mu + phi))

# An independent reading of each model at the unconstrained point: the
# non-centred random intercept, the published rows and each cell's sum.
function oracle(d, q, family)
    k = family === :nb ? 1 : 0
    phi = family === :nb ? exp(q[1]) : nothing
    beta = q[k+1:k+2]
    log_scale = q[k+3]
    xi = q[k+4:k+7]
    lam = exp.(beta[1] .+ beta[2] .* d.x .+ exp(log_scale) .* xi[d.g])
    law(m) = family === :nb ? nb2(m, phi) : Poisson(m)
    lp = sum(logpdf.(Normal(), beta)) + logpdf(Normal(), log_scale) + sum(logpdf.(Normal(), xi))
    family === :nb && (lp += logpdf(Exponential(0.2), phi) + q[1])
    lp += sum(logpdf.(law.(lam[d.pos_pub]), d.y_pub))
    lp += sum(logpdf(law(sum(lam[d.sup .& (d.g .== c)])), r) for (c, r) in zip(d.cells, d.R))
    lp
end

@stestset "SBBRMI: the density of the published rows and the cells' sums" begin
    d = subset_data()
    for (model, family) in ((poisson_model, :poisson), (nb_model, :nb))
        c = compiled(model(d), string(family))
        @test length(c.names) == (family === :nb) + 2 + 1 + 4
        # One sum per cell reads the predictor at that cell's suppressed rows.
        @test occursin("sum(", BRM.stan_code(c.sb))
        rng = Xoshiro(3)
        for _ in 1:4
            q = 0.3 .* randn(rng, length(c.names))
            @test BridgeStan.log_density(c.stan.model, q; propto=false) ≈ oracle(d, q, family) atol=1e-8 rtol=1e-12
        end
        # A frozen replay onto the same rows keeps the density.
        again = reprocess(c.sb, columns(d))
        stan = BRM.stan_instantiate(again; path=joinpath(FOLDER, "replay_$(family)_" *
            string(hash(BRM.stan_code(again)); base=16) * ".stan"))
        q = 0.2 .* randn(Xoshiro(4), length(c.names))
        @test BridgeStan.log_density(stan.model, q; propto=false) ≈
            BridgeStan.log_density(c.stan.model, q; propto=false) atol=1e-10
    end
end

@stestset "RKBRMI agrees with SBBRMI: value and gradient" begin
    d = subset_data()
    for (model, label) in ((poisson_model, "rk_poisson"), (nb_model, "rk_nb"))
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
