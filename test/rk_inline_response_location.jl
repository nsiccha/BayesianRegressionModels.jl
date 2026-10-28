# test/rk_inline_response_location.jl — a response whose location is computed
# inline from model values, `y ~ NegativeBinomial2(exp(dev) * (…), phi)`,
# against the same location assigned first, `mu = exp(dev) * (…);
# y ~ NegativeBinomial2(mu, phi)`.
#
# Both spellings are one model: SBBRMI emits the same density for them. RKBRMI
# lowers an inline computed location on its value route, as it lowers an
# assigned one; the formula route reads a linear predictor only bare or inside
# its own link and shape spellings. ReactiveKernelsPPL before 0.1.2 (RK main
# `f9a5c9c41`, ReactiveKernels snag rkppl-logistic-a-b6e04821) refuses an inline
# expression in a count family's location slot (`SurfaceLoweringError: response
# y: broadcast the link`) while it takes the named twin; those checks are
# `@test_broken` until the test environment's RK pin carries that release (snag
# rkbrmi-on-a-resp-017c6f44).
#
# Run: julia --startup-file=no --project=test test/rk_inline_response_location.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random
using LogExpFunctions: logit

function location_data(; seed=1)
    rng = Xoshiro(seed)
    n = 24
    g = repeat(1:4, inner=6)
    x = 0.5 .+ rand(rng, n)
    z = 0.2 .+ rand(rng, n)
    mu = exp.(0.3 .* randn(rng, 4))[g] .* (2.0 .* x .+ z)
    y = [rand(rng, NegativeBinomial(4.0, 4.0 / (4.0 + m))) for m in mu]
    v = mu .+ 0.5 .* randn(rng, n)
    b = Int.(rand(rng, n) .< 1 ./ (1 .+ 0.4 .* mu))
    pos = 0.3 .+ rand(rng, n)
    prop = 0.1 .+ 0.8 .* rand(rng, n)
    (; g, x, z, y, v, b, pos, prop)
end

nb_inline(d) = @brm d begin
    phi ~ Exponential(1)
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    y ~ NegativeBinomial2(exp(dev) * (exp(w) * x + z), phi)
end
nb_assigned(d) = @brm d begin
    phi ~ Exponential(1)
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    mu = exp(dev) * (exp(w) * x + z)
    y ~ NegativeBinomial2(mu, phi)
end
# The reporter's shape: another assignment over a predictor already puts the
# model on the value route, and the location stays inline.
nb_inline_beside(d) = @brm d begin
    phi ~ Exponential(1)
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    e = exp(dev)
    y ~ NegativeBinomial2(e * (exp(w) * x + z), phi)
end
poisson_inline(d) = @brm d begin
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    y ~ Poisson(exp(dev) * (exp(w) * x + z))
end
poisson_assigned(d) = @brm d begin
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    lam = exp(dev) * (exp(w) * x + z)
    y ~ Poisson(lam)
end
normal_inline(d) = @brm d begin
    sigma ~ Exponential(1)
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    v ~ Normal(exp(dev) * (exp(w) * x + z), sigma)
end
normal_assigned(d) = @brm d begin
    sigma ~ Exponential(1)
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    m = exp(dev) * (exp(w) * x + z)
    v ~ Normal(m, sigma)
end
normal_scale_inline(d) = @brm d begin
    sigma ~ Exponential(1)
    m ~ 1 + x
    dev ~ 0 + (1 | g)
    v ~ Normal(m, sigma * exp(dev))
end
normal_scale_assigned(d) = @brm d begin
    sigma ~ Exponential(1)
    m ~ 1 + x
    dev ~ 0 + (1 | g)
    s = sigma * exp(dev)
    v ~ Normal(m, s)
end
# The Bernoulli probability avoids a unary minus: its printed RK source does not
# round-trip yet (todo 1cf1xzb).
bernoulli_inline(d) = @brm d begin
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    b ~ Bernoulli(1 / (1 + exp(dev + w) * x))
end
bernoulli_assigned(d) = @brm d begin
    w ~ Normal(0, 1)
    dev ~ 0 + (1 | g)
    p = 1 / (1 + exp(dev + w) * x)
    b ~ Bernoulli(p)
end

# Families whose inline location RKPPL lowers today, and the count families
# whose location slot it refuses inline.
const LOWERED = (
    (name="normal", inline=normal_inline, assigned=normal_assigned),
    (name="normal_scale", inline=normal_scale_inline, assigned=normal_scale_assigned),
    (name="bernoulli", inline=bernoulli_inline, assigned=bernoulli_assigned),
)
const COUNT = (
    (name="nb2", inline=nb_inline, assigned=nb_assigned),
    (name="nb2_beside", inline=nb_inline_beside, assigned=nb_assigned),
    (name="poisson", inline=poisson_inline, assigned=poisson_assigned),
)

# Inline spellings the formula route used to refuse, each a model value now.
# The inline cloglog's unary minus does not survive the printed-source round
# trip yet (todo 1cf1xzb).
const FORMULA = (
    (name="normal_exp_scale", roundtrip=true, model=d -> @brm d begin
        m ~ 1 + x
        log_sigma ~ 1 + z
        v ~ Normal(m, exp(log_sigma))
    end),
    (name="inverse_gaussian_exp", roundtrip=true, model=d -> @brm d begin
        eta ~ 1 + x
        lam ~ LogNormal(-0.3, 1.0)
        pos ~ InverseGaussian(exp(eta), lam)
    end),
    (name="exponential_exp", roundtrip=true, model=d -> @brm d begin
        eta ~ 1 + x
        pos ~ Exponential(exp(eta))
    end),
    (name="bernoulli_cloglog", roundtrip=false, model=d -> @brm d begin
        eta ~ 1 + x
        b ~ Bernoulli(1 - exp(-exp(eta)))
    end),
    (name="beta_sum_shape", roundtrip=true, model=d -> @brm d begin
        logit(mu) ~ 1 + x
        kappa ~ Gamma(2.0, 5.0)
        prop ~ Beta(mu + kappa, (1 - mu) * kappa)
    end),
)

const FOLDER = joinpath(tempdir(), "brm-rk-inline-response-location")
function compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    mkpath(FOLDER)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(FOLDER,
        label * "_" * string(hash(code); base=16) * ".stan"))
    (; sb, stan, names=BridgeStan.param_unc_names(stan.model))
end

# The RK backend's value and gradient against the compiled Stan program at a
# few unconstrained points, through the coordinate transport.
function rk_agrees(brmi, c; roundtrip=true)
    rk = RKBRMI(brmi)
    problem = rk_logdensity_problem(roundtrip ? check_rk_source_roundtrip(rk) : rk;
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    t = brm_coordinate_transport(rk, c.sb, String.(c.names))
    rng = Xoshiro(17)
    all(1:3) do _
        u = 0.3 .* randn(rng, LogDensityProblems.dimension(problem))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(c.stan.model,
            brm_rk_to_stan(t, u); propto=false, jacobian=true)
        isapprox(value, stan_value; atol=1e-10, rtol=1e-11) &&
            isapprox(gradient, brm_stan_to_rk(t, stan_gradient); atol=1e-9, rtol=1e-10)
    end
end

@stestset "SBBRMI: the inline and the assigned location are one density" begin
    d = location_data()
    for p in (LOWERED..., COUNT...)
        inline = compiled(p.inline(d), p.name * "_inline")
        assigned = compiled(p.assigned(d), p.name * "_assigned")
        @test inline.names == assigned.names
        rng = Xoshiro(3)
        for _ in 1:4
            q = 0.3 .* randn(rng, length(inline.names))
            @test BridgeStan.log_density(inline.stan.model, q; propto=false) ≈
                BridgeStan.log_density(assigned.stan.model, q; propto=false) atol=1e-10 rtol=1e-12
        end
    end
end

@stestset "RKBRMI: the assigned location agrees with SBBRMI" begin
    d = location_data()
    for p in (LOWERED..., COUNT...)
        brmi = p.assigned(d)
        @test rk_agrees(brmi, compiled(brmi, p.name * "_assigned"))
    end
end

@stestset "RKBRMI: an inline location agrees with SBBRMI" begin
    d = location_data()
    for p in LOWERED
        brmi = p.inline(d)
        @test rk_agrees(brmi, compiled(brmi, p.name * "_inline"))
    end
end

@stestset "RKBRMI: formula spellings it used to refuse agree with SBBRMI" begin
    d = location_data()
    for p in FORMULA
        brmi = p.model(d)
        @test rk_agrees(brmi, compiled(brmi, p.name); p.roundtrip)
    end
end

@stestset "RKBRMI: an inline count location agrees with SBBRMI" begin
    d = location_data()
    for p in COUNT
        brmi = p.inline(d)
        c = compiled(brmi, p.name * "_inline")
        # RKPPL before 0.1.2 refuses an inline expression in a count family's
        # location slot (`broadcast the link`); @test once the RK pin has 0.1.2.
        @test_broken rk_agrees(brmi, c)
    end
end
