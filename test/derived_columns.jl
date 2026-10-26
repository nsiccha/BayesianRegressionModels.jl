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
using LogExpFunctions: logistic

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

# RKBRMI computes `name` in its graph: the program defines it, and binds its
# inputs rather than its value.
function rk_in_graph(rk, name, inputs)
    program = BRM._rk_retained_program(rk)
    @test any(s -> Meta.isexpr(s, :(=)) && first(s.args) === name, program.main.args)
    bound = BRM._rk_source_data_columns(rk.plan, program)
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
    u = fill(0.1, length(coordinate_names(replay.model.layout)))
    problem(backend) = rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    @test LogDensityProblems.logdensity(problem(replay), u) ≈
        LogDensityProblems.logdensity(problem(replay_host), u) atol=1e-10
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
            mu ~ 1 + x; y ~ weighted(Normal(mu, sigma), weights(w)) end),
         (@brm merge(flat, (; w=abs.(flat.t) .+ 0.5)) begin
            sigma ~ Exponential(1); mu ~ 1 + x
            y ~ weighted(Normal(mu, sigma), weights(w)) end)))
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
    # The grouped response is gathered in Julia, as for a raw key (todo
    # `0v3rk7u`), so neither program reads the key.
    rk_same_density(derived, host)
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

@stestset "a plate's labels and a vector its cells read are data-only assignments" begin
    cells = merge(panel, (; xo=collect(range(-1.0, 1.0; length=length(obs_subject)))))
    derived = @brm cells begin
        sigma ~ Exponential(1)
        b ~ Normal(0, 1)
        cell = derived_same(subject)
        m ~ 1 + (1 | cell)
        key = derived_same(obs_subject)
        xs = derived_shift(xo)
        rows = groups(key => cell)
        @plate for i in eachindex(m)
            y[rows[i]] ~ Normal(m[i] + b * sum(xs[rows[i]]), sigma)
        end
    end
    host = @brm merge(cells, (; cell=copy(cells.subject), key=copy(obs_subject),
            xs=cells.xo .+ 0.5)) begin
        sigma ~ Exponential(1)
        b ~ Normal(0, 1)
        m ~ 1 + (1 | cell)
        rows = groups(key => cell)
        @plate for i in eachindex(m)
            y[rows[i]] ~ Normal(m[i] + b * sum(xs[rows[i]]), sigma)
        end
    end
    sb_same_program(derived, host)
    rk_same_density(derived, host)
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
    # It reads the fit-role copy of `z`, never the design's `z`.
    reads = BRM._rk_source_symbols!(Set{Symbol}(), last(defined.args))
    @test :z_fit in reads && !(:z in reads)
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

@stestset "a column assignment recomputes a scalar assignment it reads" begin
    # `m` is a model value a prior could read by name; `tc` is a column.
    derived = @brm flat begin
        sigma ~ Exponential(1)
        m = mean(t)
        tc = t .- m
        mu ~ 1 + tc
        y ~ Normal(mu, sigma)
    end
    host = @brm merge(flat, (; tc=flat.t .- mean(flat.t))) begin
        sigma ~ Exponential(1)
        m = mean(t)
        mu ~ 1 + tc
        y ~ Normal(mu, sigma)
    end
    sb = sb_same_program(derived, host)
    rk, rkh = rk_same_density(derived, host)
    rk_in_graph(rk, :tc, (:t,))
    # A replay design's mean, as `t .- mean(t)` written inline would read it.
    design = (; t=flat.t[1:12] .+ 3.0, x=flat.x[1:12], g=flat.g[1:12], y=flat.y[1:12])
    expected = merge(design, (; tc=design.t .- mean(design.t)))
    @test stan_data(reprocess(sb, design))[:tc] ≈ expected.tc
    u = fill(0.1, length(coordinate_names(rk.model.layout)))
    density(backend) = LogDensityProblems.logdensity(
        rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
    @test density(reprocess(rk, design)) ≈ density(reprocess(rkh, expected)) atol=1e-10
end

# Every other site that reads a raw column, each against the same model given
# the column as data.
derived_side(h::AbstractVector) = [c == "u" ? 1 : 2 for c in h]
derived_label(v::AbstractVector) = string.(v) .* "_x"
sites = let n = N
    (; t=flat.t, x=flat.x, g=flat.g, y=flat.y, h=["u", "v"][mod1.(1:n, 2)],
        lo_raw=fill(-3.0, n), ny=rand(rng, 3:8, n), yp=exp.(randn(rng, n)),
        time=collect(1.0:n), age=randn(rng, 4), sid=mod1.(1:n, 4),
        rm=Float64.(flat.x .> 0), pid=repeat(1:6, inner=4),
        day=repeat(4.0:-1.0:1.0, outer=6), from=[1, 2, 3], to=[2, 3, 4])
end
sites = merge(sites, (; ky=[rand(rng, 0:v) for v in sites.ny]))
site_lev = derived_codes(sites.g)
with(extra) = merge(sites, extra)

@stestset "every other raw-column site reads a data-only assignment" begin
    cases = [
        ("uncorrelated ||",
         (@brm sites begin sigma ~ Exponential(1); lev = derived_codes(g)
            mu ~ 1 + (1 + x || lev); y ~ Normal(mu, sigma) end),
         (@brm with((; lev=site_lev)) begin sigma ~ Exponential(1)
            mu ~ 1 + (1 + x || lev); y ~ Normal(mu, sigma) end)),
        ("shared |ID|",
         (@brm sites begin lev = derived_codes(g); mu ~ 1 + (1 | p | lev)
            log(sigma) ~ 1 + (1 | p | lev); y ~ Normal(mu, sigma) end),
         (@brm with((; lev=site_lev)) begin mu ~ 1 + (1 | p | lev)
            log(sigma) ~ 1 + (1 | p | lev); y ~ Normal(mu, sigma) end)),
        ("stratum of gr(g, by = ...)",
         (@brm sites begin sigma ~ Exponential(1); side = derived_side(h)
            mu ~ 1 + (1 | gr(g, by = side)); y ~ Normal(mu, sigma) end),
         (@brm with((; side=derived_side(sites.h))) begin sigma ~ Exponential(1)
            mu ~ 1 + (1 | gr(g, by = side)); y ~ Normal(mu, sigma) end)),
        ("group of gr(g, by = ...)",
         (@brm sites begin sigma ~ Exponential(1); lev = derived_codes(g)
            mu ~ 1 + (1 | gr(lev, by = h)); y ~ Normal(mu, sigma) end),
         (@brm with((; lev=site_lev)) begin sigma ~ Exponential(1)
            mu ~ 1 + (1 | gr(lev, by = h)); y ~ Normal(mu, sigma) end)),
        ("multi-membership",
         (@brm sites begin sigma ~ Exponential(1); lev = derived_codes(g)
            lev2 = derived_same(reverse(lev)); mu ~ 1 + (1 | mm(lev, lev2))
            y ~ Normal(mu, sigma) end),
         (@brm with((; lev=site_lev, lev2=reverse(site_lev))) begin sigma ~ Exponential(1)
            mu ~ 1 + (1 | mm(lev, lev2)); y ~ Normal(mu, sigma) end)),
        ("spline",
         (@brm sites begin sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + s(tt); y ~ Normal(mu, sigma) end),
         (@brm with((; tt=sites.t .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + s(tt); y ~ Normal(mu, sigma) end)),
        ("interaction",
         (@brm sites begin sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + x * tt; y ~ Normal(mu, sigma) end),
         (@brm with((; tt=sites.t .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + x * tt; y ~ Normal(mu, sigma) end)),
        ("center and zscale",
         (@brm sites begin sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + center(tt) + zscale(tt) & x; y ~ Normal(mu, sigma) end),
         (@brm with((; tt=sites.t .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + center(tt) + zscale(tt) & x; y ~ Normal(mu, sigma) end)),
        ("tensor spline",
         (@brm sites begin sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + t2(tt, x); y ~ Normal(mu, sigma) end),
         (@brm with((; tt=sites.t .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + t2(tt, x); y ~ Normal(mu, sigma) end)),
        ("exact gp",
         (@brm sites begin sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + gp(tt); y ~ Normal(mu, sigma) end),
         (@brm with((; tt=sites.t .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + gp(tt); y ~ Normal(mu, sigma) end)),
        ("hsgp data axis",
         (@brm sites begin sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + hsgp(tt; k = 5); y ~ Normal(mu, sigma) end),
         (@brm with((; tt=sites.t .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + hsgp(tt; k = 5); y ~ Normal(mu, sigma) end)),
        ("hsgp rows mask",
         (@brm sites begin sigma ~ Exponential(1); keep = derived_same(rm)
            mu ~ 1 + hsgp(x; k = 4, rows = keep); y ~ Normal(mu, sigma) end),
         (@brm with((; keep=copy(sites.rm))) begin sigma ~ Exponential(1)
            mu ~ 1 + hsgp(x; k = 4, rows = keep); y ~ Normal(mu, sigma) end)),
        ("measurement error",
         (@brm sites begin sigma ~ Exponential(1); tt = derived_shift(t)
            mu ~ 1 + me(tt, 0.2); y ~ Normal(mu, sigma) end),
         (@brm with((; tt=sites.t .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + me(tt, 0.2); y ~ Normal(mu, sigma) end)),
        ("ar time",
         (@brm sites begin sigma ~ Exponential(1); tm = derived_same(time)
            mu ~ 1 + ar(tm; p = 1); y ~ Normal(mu, sigma) end),
         (@brm with((; tm=copy(sites.time))) begin sigma ~ Exponential(1)
            mu ~ 1 + ar(tm; p = 1); y ~ Normal(mu, sigma) end)),
        ("dar time",
         (@brm sites begin sigma ~ Exponential(1); tm = derived_same(time)
            mu ~ 1 + dar(tm; p = 1); y ~ Normal(mu, sigma) end),
         (@brm with((; tm=copy(sites.time))) begin sigma ~ Exponential(1)
            mu ~ 1 + dar(tm; p = 1); y ~ Normal(mu, sigma) end)),
        ("areal field",
         (@brm sites begin sigma ~ Exponential(1); area = derived_codes(g)
            mu ~ 1 + car(area; edges = (from, to), type = :icar); y ~ Normal(mu, sigma) end),
         (@brm with((; area=site_lev)) begin sigma ~ Exponential(1)
            mu ~ 1 + car(area; edges = (from, to), type = :icar); y ~ Normal(mu, sigma) end)),
        ("indexed standardize",
         (@brm sites begin sigma ~ Exponential(1); gi = derived_same(sid)
            mu ~ 1 + x + standardize(age)[gi]; y ~ Normal(mu, sigma) end),
         (@brm with((; gi=copy(sites.sid))) begin sigma ~ Exponential(1)
            mu ~ 1 + x + standardize(age)[gi]; y ~ Normal(mu, sigma) end)),
        ("factor reference",
         (@brm sites begin sigma ~ Exponential(1); lev = derived_codes(g)
            mu ~ 1 + factor(lev; ref = 2); y ~ Normal(mu, sigma) end),
         (@brm with((; lev=site_lev)) begin sigma ~ Exponential(1)
            mu ~ 1 + factor(lev; ref = 2); y ~ Normal(mu, sigma) end)),
        ("bare categorical",
         (@brm sites begin sigma ~ Exponential(1); cat = derived_label(g)
            mu ~ 0 + cat; y ~ Normal(mu, sigma) end),
         (@brm with((; cat=derived_label(sites.g))) begin sigma ~ Exponential(1)
            mu ~ 0 + cat; y ~ Normal(mu, sigma) end)),
        ("censoring bound",
         (@brm sites begin sigma ~ Exponential(1); lo = derived_shift(lo_raw)
            mu ~ 1 + x; y ~ censored(Normal(mu, sigma); lower = lo) end),
         (@brm with((; lo=sites.lo_raw .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + x; y ~ censored(Normal(mu, sigma); lower = lo) end)),
        ("truncation bound",
         (@brm sites begin sigma ~ Exponential(1); lo = derived_shift(lo_raw)
            mu ~ 1 + x; y ~ truncated(Normal(mu, sigma), lo, 10.0) end),
         (@brm with((; lo=sites.lo_raw .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + x; y ~ truncated(Normal(mu, sigma), lo, 10.0) end)),
        ("interval bound",
         (@brm sites begin sigma ~ Exponential(1); up = derived_shift(y)
            mu ~ 1 + x; y ~ interval_censored(Normal(mu, sigma); upper = up) end),
         (@brm with((; up=sites.y .+ 0.5)) begin sigma ~ Exponential(1)
            mu ~ 1 + x; y ~ interval_censored(Normal(mu, sigma); upper = up) end)),
        ("trials",
         (@brm sites begin nt = derived_same(ny); eta ~ 1 + x
            ky ~ Binomial(nt, logistic(eta)) end),
         (@brm with((; nt=copy(sites.ny))) begin eta ~ 1 + x
            ky ~ Binomial(nt, logistic(eta)) end)),
        ("observed response",
         (@brm sites begin sigma ~ Exponential(1); ly = log(yp)
            mu ~ 1 + x; ly ~ Normal(mu, sigma) end),
         (@brm with((; ly=log.(sites.yp))) begin sigma ~ Exponential(1)
            mu ~ 1 + x; ly ~ Normal(mu, sigma) end)),
        ("plate over groups(g)",
         (@brm sites begin sigma ~ Exponential(1); q = derived_same(pid); mu ~ 1 + x
            @plate for p in groups(q); y[p] ~ Normal(mu[p], sigma); end end),
         (@brm with((; q=copy(sites.pid))) begin sigma ~ Exponential(1); mu ~ 1 + x
            @plate for p in groups(q); y[p] ~ Normal(mu[p], sigma); end end)),
        ("plate order",
         (@brm sites begin sigma ~ Exponential(1); dd = derived_same(day); mu ~ 1 + x
            @plate for p in groups(pid; order = dd); y[p] ~ Normal(mu[p], sigma); end end),
         (@brm with((; dd=copy(sites.day))) begin sigma ~ Exponential(1); mu ~ 1 + x
            @plate for p in groups(pid; order = dd); y[p] ~ Normal(mu[p], sigma); end end)),
    ]
    for (label, derived, host) in cases
        @testset "$label" begin
            sb_same_program(derived, host)
            rk_same_density(derived, host)
        end
    end
end

@stestset "grouping options and replays over a data-only grouping" begin
    derived = @brm sites begin
        sigma ~ Exponential(1)
        lev = derived_codes(g)
        mu ~ 1 + x + (1 | lev)
        y ~ Normal(mu, sigma)
    end
    host = @brm with((; lev=site_lev)) begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 | lev)
        y ~ Normal(mu, sigma)
    end
    for options in ((; cv_groups=(:lev,)), (; centered_groups=(:lev,)),
            (; total_groups=()), (; s2z_groups=(:lev,), s2z_rho=0.5))
        a, b = SBBRMI(derived; mod=@__MODULE__, options...), SBBRMI(host; mod=@__MODULE__, options...)
        @test sprint(show, stan_code(a)) == sprint(show, stan_code(b))
        @test stan_data(a) == stan_data(b)
    end
    design = (; t=flat.t[1:8], x=flat.x[1:8], g=["d", "c", "b", "a", "a", "b", "c", "d"],
        y=flat.y[1:8], h=sites.h[1:8])
    host_design = merge(design, (; lev=derived_codes(design.g)))
    replayed = reprocess(SBBRMI(derived; mod=@__MODULE__), design)
    expected = reprocess(SBBRMI(host; mod=@__MODULE__), host_design)
    @test sprint(show, stan_code(replayed)) == sprint(show, stan_code(expected))
    @test stan_data(replayed) == stan_data(expected)
    rk, rkh = RKBRMI(derived), RKBRMI(host)
    u = fill(0.1, rk.model.layout.total)
    density(backend) = LogDensityProblems.logdensity(
        rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)
    @test density(reprocess(rk, design; resample_groups=(:lev,))) ≈
        density(reprocess(rkh, host_design; resample_groups=(:lev,))) atol=1e-10
end

@stestset "a replay that cannot evaluate a data-only assignment" begin
    derived = @brm sites begin
        sigma ~ Exponential(1)
        tt = derived_shift(t)
        mu ~ 1 + x + tt
        y ~ Normal(mu, sigma)
    end
    replay = (; x=sites.x[1:5], y=sites.y[1:5])
    message = try
        reprocess(SBBRMI(derived; mod=@__MODULE__), replay)
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("cannot evaluate data-only assignment `tt`", message)
end

# A function with only a Stan definition has no Julia method for the data, so
# its assignment stays a model value that Stan computes.
StanBlocks.@deffun derived_stan_shift(x::vector[n])::vector[n] = x + rep_vector(0.5, n)

@stestset "a Stan-only function keeps its assignment a model value" begin
    model = @brm sites begin
        sigma ~ Exponential(1)
        tt = derived_stan_shift(t)
        mu ~ 1 + x
        y ~ Normal(mu + tt, sigma)
    end
    sb = SBBRMI(model; mod=@__MODULE__)
    @test occursin("derived_stan_shift(t)", sprint(show, stan_code(sb)))
    @test !haskey(stan_data(sb), :tt)
end
