# test/derived_columns.jl — a data-only model-scope assignment is read where a
# raw column is accepted (user decision `1tb1jrg`): population terms,
# interactions, fitted transforms, factors, random-effect groupings, monotonic
# effects, offsets, weights, grouped HSGPs and `groups(key => cells)` keys, on
# SBBRMI and RKBRMI. Each derived model is the same model as one given that
# column as data: SBBRMI emits the same program and data, and RKBRMI has the
# same density, computing the column in its graph from its own inputs. A
# `constant(...)` such an assignment reads is computed in the RK graph too and
# keeps its fitted value on a replay.
#
# Run: julia --project=test test/derived_columns.jl [testset substrings...]
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random, Statistics

derived_shift(x::AbstractVector) = x .+ 0.5
derived_codes(g::AbstractVector) = [findfirst(==(c), ["a", "b", "c", "d"]) for c in g]
derived_same(k::AbstractVector) = copy(k)
derived_mean(x) = mean(filter(isfinite, x))
derived_sd(x) = std(filter(isfinite, x))
# A covariate standardized over its finite values, with a missing value at
# the mean (0 after standardization).
derived_standardize(x::AbstractVector, location, scale) =
    (replace(x, NaN => location) .- location) ./ scale

rng = MersenneTwister(20261026)
const N = 24
const G = ["a", "b", "c", "d"]
flat = (; t=randn(rng, N), x=randn(rng, N), g=G[mod1.(1:N, 4)],
    y=randn(rng, N))
host_codes = derived_codes(flat.g)

# RKBRMI densities and gradients of two models agree at fixed points, over
# the same coordinate names.
function rk_same_density(derived, host)
    rd, pd = consumer_problem(derived)
    rh, ph = consumer_problem(host)
    names = coordinate_names(rd.model.layout)
    @test names == coordinate_names(rh.model.layout)
    n = length(names)
    for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
        vd, gd = LogDensityProblems.logdensity_and_gradient(pd, u)
        vh, gh = LogDensityProblems.logdensity_and_gradient(ph, u)
        @test vd ≈ vh atol=1e-10 rtol=1e-10
        @test gd ≈ gh atol=1e-9 rtol=1e-9
    end
    rd, rh
end

# SBBRMI resolves the assignment into a data field: the same program and data.
function sb_same_program(derived, host)
    sd, sh = SBBRMI(derived; mod=@__MODULE__), SBBRMI(host; mod=@__MODULE__)
    @test sprint(show, stan_code(sd)) == sprint(show, stan_code(sh))
    @test stan_data(sd) == stan_data(sh)
    sd
end

# RKBRMI computes `name` in its graph and binds its inputs, not its value.
function rk_in_graph(rk, name, inputs)
    program = BRM._rk_retained_program(rk)
    @test any(s -> Meta.isexpr(s, :(=)) && first(s.args) === name, program.main.args)
    bound = rk_bound_data(rk)
    @test !haskey(bound, name)
    for input in inputs
        @test haskey(bound, input)
    end
end

# A later design binds its raw columns into the retained program, as a full
# replan of it would.
function rk_binds_raw(rk, design1, design2)
    binder = rk_design_binder(rk)
    binder(design1)
    bound = binder(design2)
    @test all(c -> isempty(c.derived), values(binder.classes))
    replayed = rk_bound_data(reprocess(rk, design2))
    @test Set(keys(bound)) == Set(keys(replayed))
    for key in keys(bound)
        @test isequal(bound[key], replayed[key])
    end
end

@stestset "a population term reads a data-only assignment" begin
    derived = @brm flat begin
        sigma ~ Exponential(1)
        tt = derived_shift(t)
        mu ~ 1 + tt + tt & x
        effect(mu, tt) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, sigma)
    end
    host = @brm merge(flat, (; tt=flat.t .+ 0.5)) begin
        sigma ~ Exponential(1)
        mu ~ 1 + tt + tt & x
        effect(mu, tt) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, sigma)
    end
    sb_same_program(derived, host)
    rk, _ = rk_same_density(derived, host)
    rk_in_graph(rk, :tt, (:t,))
    rows(r) = (; t=flat.t[r], x=flat.x[r], g=flat.g[r], y=flat.y[r])
    rk_binds_raw(rk, rows(1:18), rows(7:24))
end

@stestset "a fitted transform of a data-only assignment fits in the graph" begin
    derived = @brm flat begin
        sigma ~ Exponential(1)
        tt = derived_shift(t)
        mu ~ 1 + standardize(tt)
        y ~ Normal(mu, sigma)
    end
    host_data = merge(flat, (; tt=flat.t .+ 0.5))
    host = @brm host_data begin
        sigma ~ Exponential(1)
        mu ~ 1 + standardize(tt)
        y ~ Normal(mu, sigma)
    end
    sb_same_program(derived, host)
    rk, rkh = rk_same_density(derived, host)
    # The fit-role twin is a definition over the input's fit-role copy.
    program = BRM._rk_retained_program(rk)
    twin = only(s for s in program.main.args if Meta.isexpr(s, :(=)) &&
        first(s.args) === :tt_fit)
    @test :t_fit in BRM._rk_source_symbols!(Set{Symbol}(), last(twin.args))
    # A replay keeps the training mean and SD of `tt`.
    design = (; t=flat.t[1:12] .+ 2.0, x=flat.x[1:12], g=flat.g[1:12], y=flat.y[1:12])
    replay = reprocess(rk, design)
    replay_host = reprocess(rkh, merge(design, (; tt=design.t .+ 0.5)))
    _, pd = consumer_problem(replay)
    u = fill(0.1, length(coordinate_names(replay.model.layout)))
    ph = rk_logdensity_problem(replay_host; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test LogDensityProblems.logdensity(pd, u) ≈ LogDensityProblems.logdensity(ph, u) atol=1e-10
end

@stestset "factors, groupings and monotonic effects read a data-only assignment" begin
    for (label, derived, host) in (
        ("factor", (@brm flat begin
            sigma ~ Exponential(1); lev = derived_codes(g)
            mu ~ 1 + factor(lev); y ~ Normal(mu, sigma) end),
         (@brm merge(flat, (; lev=host_codes)) begin
            sigma ~ Exponential(1); mu ~ 1 + factor(lev); y ~ Normal(mu, sigma) end)),
        ("random intercept", (@brm flat begin
            sigma ~ Exponential(1); lev = derived_codes(g)
            mu ~ 1 + x + (1 | lev); y ~ Normal(mu, sigma) end),
         (@brm merge(flat, (; lev=host_codes)) begin
            sigma ~ Exponential(1); mu ~ 1 + x + (1 | lev); y ~ Normal(mu, sigma) end)),
        ("monotonic", (@brm flat begin
            sigma ~ Exponential(1); lev = derived_codes(g)
            mu ~ 1 + mo(lev); y ~ Normal(mu, sigma) end),
         (@brm merge(flat, (; lev=host_codes)) begin
            sigma ~ Exponential(1); mu ~ 1 + mo(lev); y ~ Normal(mu, sigma) end)),
        ("grouped hsgp", (@brm flat begin
            sigma ~ Exponential(1); lev = derived_codes(g)
            mu ~ 1 + hsgp(x; k = 4, by = lev); y ~ Normal(mu, sigma) end),
         (@brm merge(flat, (; lev=host_codes)) begin
            sigma ~ Exponential(1); mu ~ 1 + hsgp(x; k = 4, by = lev)
            y ~ Normal(mu, sigma) end)))
        @testset "$label" begin
            sb_same_program(derived, host)
            rk, _ = rk_same_density(derived, host)
            rk_in_graph(rk, :lev, (:g,))
        end
    end
end

@stestset "offsets and weights read a data-only assignment" begin
    for (label, derived, host) in (
        ("offset", (@brm flat begin
            sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + x + offset(tt); y ~ Normal(mu, sigma) end),
         (@brm merge(flat, (; tt=flat.t .+ 0.5)) begin
            sigma ~ Exponential(1); mu ~ 1 + x + offset(tt); y ~ Normal(mu, sigma) end)),
        ("weights", (@brm flat begin
            sigma ~ Exponential(1); w = derived_shift(abs(t))
            mu ~ 1 + x; y ~ weighted(Normal(mu, sigma), aweights(w)) end),
         (@brm merge(flat, (; w=abs.(flat.t) .+ 0.5)) begin
            sigma ~ Exponential(1); mu ~ 1 + x
            y ~ weighted(Normal(mu, sigma), aweights(w)) end)))
        @testset "$label" begin
            sb_same_program(derived, host)
            rk_same_density(derived, host)
        end
    end
end

# Flat rows of several subjects, read per subject through `groups` bindings.
const NS = 6
obs_subject = repeat(1:NS, inner=4)
pt_subject = repeat(1:NS, inner=3)
panel = (; subject=collect(1:NS), obs_subject, y=randn(rng, length(obs_subject)),
    pt_raw=0.3 .* randn(rng, length(pt_subject)), pt_subject)

@stestset "a groups(key => cells) key is a data-only assignment" begin
    derived = @brm panel begin
        sigma ~ Exponential(1)
        m ~ 1 + (1 | subject)
        key = derived_same(obs_subject)
        rows = groups(key => subject)
        @plate for i in eachindex(m)
            y[rows[i]] ~ Normal(m[i], sigma)
        end
    end
    host = @brm merge(panel, (; key=copy(obs_subject))) begin
        sigma ~ Exponential(1)
        m ~ 1 + (1 | subject)
        rows = groups(key => subject)
        @plate for i in eachindex(m)
            y[rows[i]] ~ Normal(m[i], sigma)
        end
    end
    sb_same_program(derived, host)
    rk, _ = rk_same_density(derived, host)
    rk_in_graph(rk, :key, (:obs_subject,))
end

@stestset "a predictor on a derived axis, gathered through a derived key" begin
    derived = @brm panel begin
        sigma ~ Exponential(1)
        m ~ 1 + (1 | subject)
        pts = derived_shift(pt_raw)
        key = derived_same(pt_subject)
        h ~ 0 + hsgp(pts; k = 3, domain = (-1.5, 1.5))
        rows = groups(key => subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(m)
            y[obs[i]] ~ Normal(m[i] + sum(h[rows[i]]), sigma)
        end
    end
    host = @brm merge(panel, (; pts=panel.pt_raw .+ 0.5, key=copy(pt_subject))) begin
        sigma ~ Exponential(1)
        m ~ 1 + (1 | subject)
        h ~ 0 + hsgp(pts; k = 3, domain = (-1.5, 1.5))
        rows = groups(key => subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(m)
            y[obs[i]] ~ Normal(m[i] + sum(h[rows[i]]), sigma)
        end
    end
    sb_same_program(derived, host)
    rk, _ = rk_same_density(derived, host)
    rk_in_graph(rk, :pts, (:pt_raw,))
    rk_in_graph(rk, :key, (:pt_subject,))
end

@stestset "constants a data-only assignment reads are computed in the graph" begin
    covariate = [randn(rng, N - 2); NaN; NaN]
    data = merge(flat, (; z=covariate))
    derived = @brm data begin
        sigma ~ Exponential(1)
        z_loc = constant(derived_mean(z))
        z_scale = constant(derived_sd(z))
        z_std = derived_standardize(z, z_loc, z_scale)
        mu ~ 1 + z_std
        y ~ Normal(mu, sigma)
    end
    loc, scale = derived_mean(covariate), derived_sd(covariate)
    host = @brm merge(data, (; z_std=derived_standardize(covariate, loc, scale))) begin
        sigma ~ Exponential(1)
        mu ~ 1 + z_std
        y ~ Normal(mu, sigma)
    end
    sb_same_program(derived, host)
    rk, rkh = rk_same_density(derived, host)
    rk_in_graph(rk, :z_std, (:z,))
    rk_in_graph(rk, :z_loc, ())
    program = BRM._rk_retained_program(rk)
    defined = only(s for s in program.main.args if Meta.isexpr(s, :(=)) &&
        first(s.args) === :z_loc)
    @test BRM._rk_source_symbols!(Set{Symbol}(), last(defined.args)) ⊆ Set([:z_fit])
    # A replay design with another mean keeps the fitted location and scale,
    # by reprocess and from a data-free fitted state alike.
    design = (; z=covariate[1:12] .+ 3.0, t=flat.t[1:12], x=flat.x[1:12],
        g=flat.g[1:12], y=flat.y[1:12])
    expected = merge(design, (; z_std=derived_standardize(design.z, loc, scale)))
    u = fill(0.1, length(coordinate_names(rk.model.layout)))
    density(backend) = LogDensityProblems.logdensity(
        rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
    reference = density(reprocess(rkh, expected))
    @test density(reprocess(rk, design)) ≈ reference atol=1e-10
    state = brm_fitted_state(rk)
    builder = @brm begin
        sigma ~ Exponential(1)
        z_loc = constant(derived_mean(z))
        z_scale = constant(derived_sd(z))
        z_std = derived_standardize(z, z_loc, z_scale)
        mu ~ 1 + z_std
        y ~ Normal(mu, sigma)
    end
    @test density(RKBRMI(builder(design), state)) ≈ reference atol=1e-10
end
