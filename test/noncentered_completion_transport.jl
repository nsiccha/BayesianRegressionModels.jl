# test/noncentered_completion_transport.jl — the coordinate transport of
# completions RKBRMI samples non-centered (`RKBRMI(brmi; completions =
# :noncentered)`): RK's standardized draws `<column>.z_mis` against SBBRMI's
# centred entries `<column>_y_mis`, through `s = location + scale * z` with
# the law's location and scale at the point (`s` the entry under `Normal`, its
# log under `LogNormal`). Points both ways, the log density with the
# log-Jacobian `sum(log, scale)`, the gradient pulled back through the
# location and scale, and the physical check, for a law of data columns, a law
# of parameters, a LogNormal law and a law reading another completion (snag
# coordinate-trans-23993804).
#
# Run: julia --startup-file=no --project=test test/noncentered_completion_transport.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

# 24 rows; `x` misses 5 entries and `v` 4, row 5 both.
function completion_data(; seed=3)
    rng = Xoshiro(seed)
    n = 24
    w = randn(rng, n)
    x = 0.5 .+ 0.8 .* w .+ 0.3 .* randn(rng, n)
    v = exp.(1.0 .+ 0.3 .* x .+ 0.2 .* randn(rng, n))
    y = 1.0 .+ 0.7 .* x .+ 0.4 .* log.(v) .+ 0.2 .* randn(rng, n)
    gap_x = falses(n); gap_x[[2, 5, 9, 14, 20]] .= true
    gap_v = falses(n); gap_v[[3, 5, 11, 17]] .= true
    blank(values, gap) = Union{Missing,Float64}[g ? missing : t for (g, t) in zip(gap, values)]
    # Each row's prior location and scale of `x`, as data.
    x_loc = 0.5 .+ 0.8 .* w
    x_sd = 0.25 .+ 0.1 .* abs.(w)
    (; w, x=blank(x, gap_x), v=blank(v, gap_v), y, x_loc, x_sd, gap_x, gap_v)
end
columns(d) = (; d.w, d.x, d.v, d.y, d.x_loc, d.x_sd)

# The law's location and scale are data columns. An assignment reads the
# completion, so it lowers to RK's value plan, which non-centers completions.
data_law(d) = @brm columns(d) begin
    sigma ~ Exponential(1)
    mi(x) ~ Normal(x_loc, x_sd)
    cx = x - 0.5
    m ~ 1 + cx
    y ~ Normal(m, sigma)
end
# A location predictor and a scale parameter.
parameter_law(d) = @brm columns(d) begin
    sigma_x ~ Exponential(1)
    mu_x ~ 1 + w
    mi(x) ~ Normal(mu_x, sigma_x)
    cx = x - 0.5
    m ~ 1 + cx
    y ~ Normal(m, 0.5)
end
# SB samples the logs of LogNormal entries.
lognormal_law(d) = @brm columns(d) begin
    sigma_v ~ Exponential(1)
    mu_v ~ 1 + w
    mi(v) ~ LogNormal(mu_v, sigma_v)
    lv = log(v)
    m ~ 1 + lv
    y ~ Normal(m, 0.5)
end
# The second law's location reads the first completion.
chained_law(d) = @brm columns(d) begin
    sigma_x ~ Exponential(1)
    mu_x ~ 1 + w
    mi(x) ~ Normal(mu_x, sigma_x)
    sigma_v ~ Exponential(1)
    mu_v ~ 1 + x
    mi(v) ~ LogNormal(mu_v, sigma_v)
    lv = log(v)
    m ~ 1 + x + lv
    y ~ Normal(m, 0.5)
end

const FOLDER = joinpath(tempdir(), "brm-noncentered-completion-transport")
function compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    mkpath(FOLDER)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(FOLDER, label * "_" * string(hash(code); base=16) * ".stan"))
    (; sb, stan, names=String.(BridgeStan.param_unc_names(stan.model)))
end
constrained(c, u) = Dict(zip(BridgeStan.param_names(c.stan.model),
    BridgeStan.param_constrain(c.stan.model, u)))
positions(names, stem) = findall(n -> startswith(String(n), stem * "."), names)
message(f) = try f(); "" catch e; sprint(showerror, e) end

# RK's value and native Enzyme gradient against Stan's through the transport,
# both point maps, and the physical check, at `points` random RK points.
function check_transport(brmi, label; points=3, scale=0.3)
    rk = RKBRMI(brmi; completions=:noncentered)
    problem = rk_logdensity_problem(check_rk_source_roundtrip(rk);
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    c = compiled(brmi, label)
    t = brm_coordinate_transport(rk, c.sb, c.names)
    @test LogDensityProblems.dimension(problem) == length(c.names) == length(t)
    rng = Xoshiro(29)
    us = [scale .* randn(rng, length(c.names)) for _ in 1:points]
    for u in us
        before = copy(u)
        stan_u = brm_rk_point_to_stan(t, u)
        @test brm_stan_point_to_rk(t, stan_u) ≈ u atol=1e-12 rtol=1e-12
        stan_value, stan_gradient = BridgeStan.log_density_gradient(c.stan.model,
            stan_u; propto=false, jacobian=true)
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test value ≈ brm_stan_logdensity_to_rk(t, u, stan_value) atol=1e-10 rtol=1e-11
        @test gradient ≈ brm_stan_gradient_to_rk(t, u, stan_gradient) atol=1e-9 rtol=1e-10
        receipt = brm_check_coordinate_transport(t, rk, c.stan.model, u)
        @test receipt.completions == length(t.completions)
        @test u == before
    end
    (; rk, c, t, us)
end

@stestset "a law of data columns: s = location + scale * z" begin
    d = completion_data()
    (; c, t, us) = check_transport(data_law(d), "data_law")
    @test only(t.completions).target === :x
    pairs = [p for p in t.pairs if p.relation === :completion]
    @test [p.rk for p in pairs] == [Symbol("x.z_mis.", i) for i in 1:5]
    @test [p.stan for p in pairs] == ["x_y_mis.$i" for i in 1:5]
    # Against the data alone: the missing rows' entries.
    missing_rows = findall(d.gap_x)
    rk_z = [findfirst(==(p.rk), t.rk_names) for p in pairs]
    stan_y = [findfirst(==(p.stan), t.stan_names) for p in pairs]
    for u in us
        @test brm_rk_point_to_stan(t, u)[stan_y] ≈
            d.x_loc[missing_rows] .+ d.x_sd[missing_rows] .* u[rk_z] rtol=1e-14
    end
    # The point maps leave the permutation helpers and the constant offset.
    @test occursin("standardized completion", message(() -> brm_rk_to_stan(t, us[1])))
    @test occursin("brm_stan_logdensity_to_rk", message(() -> t.logdensity_offset))
    @test occursin("1 standardized completions", sprint(show, t))
end

@stestset "a law of parameters: the location predictor and scale at the point" begin
    d = completion_data()
    (; c, t, us) = check_transport(parameter_law(d), "parameter_law")
    missing_rows = findall(d.gap_x)
    rk_z = positions(t.rk_names, "x.z_mis")
    for u in us
        values = constrained(c, brm_rk_point_to_stan(t, u))
        beta = [values["pop_mu_x_beta_pop.1"], values["pop_mu_x_beta_pop.2"]]
        location = beta[1] .+ beta[2] .* d.w[missing_rows]
        @test [values["x_y_mis.$i"] for i in 1:5] ≈
            location .+ values["sigma_x"] .* u[rk_z] rtol=1e-12
    end
end

@stestset "a LogNormal law: Stan's entries are the logs" begin
    d = completion_data()
    (; c, t, us) = check_transport(lognormal_law(d), "lognormal_law")
    missing_rows = findall(d.gap_v)
    rk_z = positions(t.rk_names, "v.z_mis")
    @test only(t.completions).law === LogNormal
    for u in us
        values = constrained(c, brm_rk_point_to_stan(t, u))
        beta = [values["pop_mu_v_beta_pop.1"], values["pop_mu_v_beta_pop.2"]]
        @test [values["v_y_mis.$i"] for i in 1:4] ≈
            exp.(beta[1] .+ beta[2] .* d.w[missing_rows] .+ values["sigma_v"] .* u[rk_z]) rtol=1e-12
    end
end

@stestset "a law reading another completion: solved in dependency order" begin
    d = completion_data()
    (; t) = check_transport(chained_law(d), "chained_law")
    @test sort([b.target for b in t.completions]) == [:v, :x]
end
