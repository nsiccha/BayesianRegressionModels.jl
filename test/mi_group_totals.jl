# test/mi_group_totals.jl — a completion whose missing entries have known group
# sums: `mi(x)::shares(total, rows) ~ Law(…)` with `rows = groups(g => labels)`.
#
# For every label `k`, the completed `x[rows[k]]` sums to `total[k]`. Per group
# the missing entries are the residual (the total minus the observed entries)
# times shares, the bijection of a `T::shares(total)` cell latent: the law's
# density at the values plus `sum(log.(x[missing])) - log(residual)`. A group
# with one missing entry holds the residual; missing entries of no group stay
# free (snag mi-completion-wi-54222432, design decision 0guq37f). On RKBRMI
# with `completions = :noncentered`, the grouped coordinates are standard
# normal draws and each group observes the rest of the density: the centred
# coordinates and density (snag nlme-view-shares-f1cfaa71).
#
# Run: julia --startup-file=no --project=test test/mi_group_totals.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

# 24 rows in 6 groups of 4. Group 1 misses 2 entries, group 2 misses 3, group 4
# misses 1, group 3 and 5 are complete; group 6 has no total, so its missing
# row 22 is free.
function totals_data(; seed=2)
    rng = Xoshiro(seed)
    n = 24
    g = repeat(1:6, inner=4)
    x = randn(rng, n)
    truth = exp.(1.5 .+ 0.4 .* x .+ 0.3 .* randn(rng, n))
    gap = falses(n); gap[[1, 2, 6, 7, 8, 13, 22]] .= true
    cells = [1, 2, 3, 4, 5]
    tot = [sum(truth[g .== k]) for k in cells]
    v = Union{Missing,Float64}[m ? missing : t for (m, t) in zip(gap, truth)]
    y = 1.0 .+ 0.5 .* log.(truth) .+ 0.1 .* randn(rng, n)
    (; x, g, cells, tot, v, y, truth, gap)
end
columns(data) = (; data.x, data.g, data.cells, data.tot, data.v, data.y)

# The completion read by a later formula; the rows by a binding.
totals_model(data) = @brm columns(data) begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    rows = groups(g => cells)
    mi(v)::shares(tot, rows) ~ LogNormal(mu, sigma)
    m ~ 1 + log(v)
    y ~ Normal(m, 0.5)
end
# The same model with the completion's log as an assignment, the spelling
# RKBRMI takes (a formula transform of a completed column is refused there,
# todo `2026-10-28T12-58-58-902-1m9civj`).
totals_assigned(data) = @brm columns(data) begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    rows = groups(g => cells)
    mi(v)::shares(tot, rows) ~ LogNormal(mu, sigma)
    lv = log(v)
    m ~ 1 + lv
    y ~ Normal(m, 0.5)
end
# The same model with the rows written in place.
totals_inline(data) = @brm columns(data) begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    mi(v)::shares(tot, groups(g => cells)) ~ LogNormal(mu, sigma)
    m ~ 1 + log(v)
    y ~ Normal(m, 0.5)
end
# Another positive family, with a scale per row.
totals_gamma(data) = @brm columns(data) begin
    mu ~ 1 + x
    rows = groups(g => cells)
    mi(v)::shares(tot, rows) ~ Gamma(4.0, exp(mu) / 4.0)
    lv = log(v)
    m ~ 1 + lv
    y ~ Normal(m, 0.5)
end

# Nothing reads the completion: the free entry integrates out and is a
# generated draw, while the grouped entries still carry what the residuals say
# about the law.
unread_model(data) = @brm columns(data) begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    rows = groups(g => cells)
    mi(v)::shares(tot, rows) ~ LogNormal(mu, sigma)
end

const FOLDER = joinpath(tempdir(), "brm-mi-group-totals")
function compiled(brmi, label)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    mkpath(FOLDER)
    code = BRM.stan_code(sb)
    stan = BRM.stan_instantiate(sb; path=joinpath(FOLDER, label * "_" * string(hash(code); base=16) * ".stan"))
    (; sb, stan, names=BridgeStan.param_unc_names(stan.model))
end
constrained(c, u) = Dict(zip(BridgeStan.param_names(c.stan.model; include_tp=true),
    BridgeStan.param_constrain(c.stan.model, u; include_tp=true)))
vector_of(values, name, n) = [values["$name.$i"] for i in 1:n]

# An independent reading of the model at the constrained point: the priors with
# their transforms' Jacobians; the law at the observed, free and grouped
# entries; each group's `sum(log.(x)) - log(residual)` at its missing entries;
# and the observation of `y`.
function oracle(data, values, law)
    n = length(data.v)
    sigma = get(values, "sigma", nothing)
    beta = [values["pop_mu_beta_pop.1"], values["pop_mu_beta_pop.2"]]
    mu = beta[1] .+ beta[2] .* data.x
    v = vector_of(values, "v", n)
    lp = 0.0
    sigma === nothing || (lp += logpdf(Exponential(1), sigma) + log(sigma))
    lp += sum(logpdf.(Normal(), beta))
    lp += sum(logpdf.(law.(mu, sigma === nothing ? 0.0 : sigma), v))
    # Free missing entries are positive parameters: log transform.
    free = [i for i in findall(data.gap) if !(data.g[i] in data.cells)]
    lp += sum(log.(v[free]); init=0.0)
    for (k, label) in enumerate(data.cells)
        rows = findall(==(label), data.g)
        missing_rows = filter(i -> data.gap[i], rows)
        isempty(missing_rows) && continue
        residual = data.tot[k] - sum(v[i] for i in rows if !data.gap[i]; init=0.0)
        lp += sum(log.(v[missing_rows])) - log(residual)
    end
    gamma = [values["pop_m_beta_pop.1"], values["pop_m_beta_pop.2"]]
    lp += sum(logpdf.(Normal(), gamma))
    lp += sum(logpdf.(Normal.(gamma[1] .+ gamma[2] .* log.(v), 0.5), data.y))
    lp
end

@stestset "@brm: the typed completion and its refusals" begin
    data = totals_data()
    m = totals_model(data)
    @test m isa BRM.BRMI
    plan = BRM._brm_missing_response_plan(first(BRM.getargs(parent(m.operations.v))); prefix="test")
    @test plan.totals isa BRM._BRMMissingTotals
    @test (plan.totals.group, plan.totals.cells, plan.totals.total) == (:g, :cells, :tot)
    # Free row 22 first, then groups 1, 2 and 4 in label order.
    @test plan.missing_indices == [22, 1, 2, 6, 7, 8, 13]
    @test plan.totals.nfree == 1
    @test plan.totals.ends == [2, 5, 6]
    @test plan.totals.residuals ≈ [sum(data.truth[[1, 2]]), sum(data.truth[[6, 7, 8]]),
        data.truth[13]] rtol=1e-12
    inline = BRM._brm_missing_response_plan(first(BRM.getargs(parent(totals_inline(data).operations.v))); prefix="test")
    @test inline.missing_indices == plan.missing_indices
    @test inline.totals.residuals == plan.totals.residuals
    # The completion's grouping and totals are inputs of the model, not responses.
    @test issubset((:g, :cells, :tot), brm_columns(BRM.brm(quote
        sigma ~ Exponential(1)
        mu ~ 1 + x
        rows = groups(g => cells)
        mi(v)::shares(tot, rows) ~ LogNormal(mu, sigma)
    end; mod=@__MODULE__)))

    message(f) = try f(); "" catch e; sprint(showerror, e) end
    refusal(body) = message(() -> BRM.brm(body; mod=@__MODULE__)(columns(data)))
    @test occursin("names no bijection", refusal(quote
        mu ~ 1 + x; rows = groups(g => cells); mi(v)::total(tot, rows) ~ LogNormal(mu, 1.0)
    end))
    @test occursin("takes the data vector of the groups' totals and their rows", refusal(quote
        mu ~ 1 + x; mi(v)::shares(tot) ~ LogNormal(mu, 1.0)
    end))
    @test occursin("is no model-scope binding", refusal(quote
        mu ~ 1 + x; mi(v)::shares(tot, rows) ~ LogNormal(mu, 1.0)
    end))
    @test occursin("names one data column", refusal(quote
        mu ~ 1 + x; rows = groups(g => cells); mi(log(v))::shares(tot, rows) ~ LogNormal(mu, 1.0)
    end))
end

@stestset "plan: residuals, free entries and data refusals" begin
    data = totals_data()
    spec(; g=data.g, cells=data.cells, tot=data.tot) = (; group=(; name=:g, values=g),
        cells=(; name=:cells, values=cells), total=(; name=:tot, values=tot))
    missing_rows = findall(data.gap)
    message(f) = try f(); "" catch e; sprint(showerror, e) end
    totals(s) = BRM._brm_missing_totals(:v, data.v, missing_rows, s; prefix="test")
    # A complete group must already sum to its total.
    wrong = copy(data.tot); wrong[3] += 1.0
    @test occursin("group 3 has no missing entry", message(() -> totals(spec(; tot=wrong))))
    # A residual that is not positive.
    low = copy(data.tot); low[1] = sum(data.truth[[3, 4]])
    @test occursin("leaves 0.0 of its total", message(() -> totals(spec(; tot=low))))
    @test occursin("has 4 totals for the 5 groups", message(() -> totals(spec(; tot=data.tot[1:4]))))
    @test occursin("has 23 entries for the 24", message(() -> totals(spec(; g=data.g[1:23]))))
    @test occursin("is listed twice", message(() -> totals(spec(; cells=[1, 2, 2, 4, 5]))))
    # A total of a label no row holds is a complete group summing to 0.
    ordered, t = totals(spec(; cells=[1, 2, 3, 4, 5, 9], tot=[data.tot; 0.0]))
    @test t.ends == [2, 5, 6]
    @test occursin("group 9 has no missing entry", message(() ->
        totals(spec(; cells=[1, 2, 3, 4, 5, 9], tot=[data.tot; 1.0]))))
    # The Julia maps of the grouped entries.
    rng = Xoshiro(5)
    y = randn(rng, 3)
    x = BRM._brm_grouped_shares_values(y, [2, 5, 6], [3.0, 4.0, 2.5])
    @test length(x) == 6
    @test [sum(x[1:2]), sum(x[3:5]), x[6]] ≈ [3.0, 4.0, 2.5] rtol=1e-12
    @test BRM._brm_grouped_shares_coordinates(x, [2, 5, 6]) ≈ y rtol=1e-10
end

@stestset "SBBRMI: the density at the constrained values, every group at its total" begin
    data = totals_data()
    for (model, label, law) in ((totals_model, "lognormal", (m, s) -> LogNormal(m, s)),
            (totals_assigned, "assigned", (m, s) -> LogNormal(m, s)),
            (totals_inline, "inline", (m, s) -> LogNormal(m, s)),
            (totals_gamma, "gamma", (m, _) -> Gamma(4.0, exp(m) / 4.0)))
        c = compiled(model(data), label)
        # One free entry, and K - 1 coordinates for groups of 2, 3 and 1.
        @test count(n -> startswith(n, "v_y_free"), c.names) == 1
        @test count(n -> startswith(n, "v_y_shares"), c.names) == 1 + 2 + 0
        rng = Xoshiro(3)
        for _ in 1:6
            u = 0.4 .* randn(rng, length(c.names))
            values = constrained(c, u)
            v = vector_of(values, "v", length(data.v))
            @test v[.!data.gap] == data.v[.!data.gap]
            @test [sum(v[data.g .== k]) for k in data.cells] ≈ data.tot rtol=1e-12
            @test v[13] ≈ data.tot[4] - sum(data.v[14:16]) rtol=1e-12
            value, gradient = BridgeStan.log_density_gradient(c.stan.model, u; propto=false)
            @test value ≈ oracle(data, values, law) atol=1e-8 rtol=1e-12
            @test all(isfinite, gradient)
        end
    end
end

@stestset "SBBRMI: an unread completion keeps the residuals in the density" begin
    data = totals_data()
    c = compiled(unread_model(data), "unread")
    @test !any(n -> startswith(n, "v_y_free"), c.names)
    @test count(n -> startswith(n, "v_y_shares"), c.names) == 3
    rng = Xoshiro(7)
    for _ in 1:4
        u = 0.4 .* randn(rng, length(c.names))
        values = constrained(c, u)
        sigma = values["sigma"]
        mu = values["pop_mu_beta_pop.1"] .+ values["pop_mu_beta_pop.2"] .* data.x
        shares = [values["v_y_shares.$i"] for i in 1:3]
        residuals = BRM.stan_data(c.sb)[:v_shares_residuals]
        grouped = BRM._brm_grouped_shares_values(shares, [2, 5, 6], residuals)
        rows = [1, 2, 6, 7, 8, 13]
        observed = findall(.!data.gap)
        expected = logpdf(Exponential(1), sigma) + log(sigma) +
            sum(logpdf.(Normal(), [values["pop_mu_beta_pop.1"], values["pop_mu_beta_pop.2"]])) +
            sum(logpdf.(LogNormal.(mu[observed], sigma), data.v[observed])) +
            sum(logpdf.(LogNormal.(mu[rows], sigma), grouped)) + sum(log.(grouped)) - sum(log.(residuals))
        @test BridgeStan.log_density(c.stan.model, u; propto=false) ≈ expected atol=1e-8 rtol=1e-12
        # A predictive draw of the residuals, and of the completion with its free entry.
        draw = Dict(zip(BridgeStan.param_names(c.stan.model; include_tp=true, include_gq=true),
            BridgeStan.param_constrain(c.stan.model, u; include_tp=true, include_gq=true,
                rng=BridgeStan.StanRNG(c.stan.model, 11))))
        @test all(k -> draw["v_shares_residuals_gen.$k"] > 0, 1:3)
        v = vector_of(draw, "v", length(data.v))
        @test v[22] > 0
        @test [sum(v[data.g .== k]) for k in data.cells] ≈ data.tot rtol=1e-12
    end
end

@stestset "descriptor, description and brm_start" begin
    data = totals_data()
    m = totals_assigned(data)
    sb = SBBRMI(m; mod=@__MODULE__, total_groups=())
    d = brm_descriptor(sb)
    # The completed column, its free entry and the grouped entries' coordinates.
    @test !isnothing(brm_output(d, :v))
    @test any(o -> o.name === :v_y_shares && o.kind === :parameter, d.outputs)
    @test any(o -> o.name === :v_y_free && o.kind === :parameter, d.outputs)
    r = brm_description(d)
    @test r.complete
    @test any(p -> occursin("known total", p), r.prose)
    @test brm_description(d; texts=description_texts(:de)).complete
    # The grouped entries keep their base coordinates: zero, equal shares of each residual.
    stan = BRM.stan_instantiate(sb; path=joinpath(FOLDER, "start_" * string(hash(BRM.stan_code(sb)); base=16) * ".stan"))
    names = BridgeStan.param_unc_names(stan.model)
    u = brm_start(sb, stan.model, (; sigma = 0.5))
    @test u[findfirst(==("sigma"), names)] ≈ log(0.5)
    @test all(iszero, u[findall(n -> startswith(n, "v_y_shares"), names)])
    rk = RKBRMI(m)
    urk = brm_start(rk, (; sigma = 0.5))
    rk_names = String.(coordinate_names(rk.model.layout))
    @test length(urk) == length(rk_names)
    @test all(iszero, urk[findall(n -> startswith(n, "v.y_shares"), rk_names)])
end

@stestset "VBRMI does not lower a completion with group totals" begin
    # A backend without the lowering must refuse rather than drop the totals.
    @test_throws Exception VBRMI(totals_model(totals_data()))
end

@stestset "SBBRMI: a real-valued law is refused" begin
    data = totals_data()
    message(f) = try f(); "" catch e; sprint(showerror, e) end
    m = @brm columns(data) begin
        mu ~ 1 + x
        rows = groups(g => cells)
        mi(v)::shares(tot, rows) ~ Normal(mu, 1.0)
    end
    @test occursin("declares a positive latent", message(() -> SBBRMI(m; mod=@__MODULE__)))
end

@stestset "SBBRMI: reprocess keeps the groups and reads the new totals" begin
    data = totals_data()
    c = compiled(totals_model(data), "replay")
    u = 0.3 .* randn(Xoshiro(6), length(c.names))
    same = reprocess(c.sb, columns(data))
    @test BRM.stan_data(same)[:v_shares_residuals] == BRM.stan_data(c.sb)[:v_shares_residuals]
    # New observed values and totals, the same mask: the same coordinates, new residuals.
    fresh = merge(columns(data), (; v=Union{Missing,Float64}[m ? missing : 1.1 * t
        for (m, t) in zip(data.gap, data.truth)], tot=1.1 .* data.tot))
    moved = reprocess(c.sb, fresh)
    @test BRM.stan_data(moved)[:v_shares_residuals] ≈ 1.1 .* BRM.stan_data(c.sb)[:v_shares_residuals] rtol=1e-12
    message(f) = try f(); "" catch e; sprint(showerror, e) end
    other = copy(data.v); other[3] = missing
    @test occursin("same fitted missing-row positions",
        message(() -> reprocess(c.sb, merge(columns(data), (; v=other)))))
end

# RKBRMI's value and native Enzyme gradient against SBBRMI's through the
# coordinate transport, and the printed RK source replays. Points, densities and
# gradients move through the general maps, which a non-centered completion's
# free entries need (`t.completions`) and which reduce to the permutation
# otherwise.
function check_against_stan(brmi, label; points=3, scale=0.3, gradient=true,
        completions=:centered)
    rk = RKBRMI(brmi; completions)
    problem = rk_logdensity_problem(check_rk_source_roundtrip(rk);
        ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    c = compiled(brmi, label)
    t = brm_coordinate_transport(rk, c.sb, String.(c.names))
    rng = Xoshiro(17)
    n = LogDensityProblems.dimension(problem)
    @test n == length(c.names)
    for _ in 1:points
        u = scale .* randn(rng, n)
        stan_value, stan_gradient = BridgeStan.log_density_gradient(c.stan.model,
            brm_rk_point_to_stan(t, u); propto=false, jacobian=true)
        if gradient
            value, g = LogDensityProblems.logdensity_and_gradient(problem, u)
            @test g ≈ brm_stan_gradient_to_rk(t, u, stan_gradient) atol=1e-9 rtol=1e-10
        else
            value = LogDensityProblems.logdensity(problem, u)
        end
        @test value ≈ brm_stan_logdensity_to_rk(t, u, stan_value) atol=1e-10 rtol=1e-11
        @test brm_check_coordinate_transport(t, rk, c.stan.model, u) isa NamedTuple
    end
    (; rk, c, t, problem)
end

@stestset "RKBRMI agrees with SBBRMI: value, gradient, transport" begin
    data = totals_data()
    for (model, label) in ((totals_assigned, "rk_lognormal"), (unread_model, "rk_unread"))
        check_against_stan(model(data), label)
    end
    # A Gamma law. Registered Enzyme (0.13.216) fails its native Reverse
    # gradient on Enzyme's readonly proof (EnzymeMutabilityException), as for a
    # plain `mi(v) ~ Gamma(…)` completion without totals; the test env's Enzyme
    # dogfood carries the repair (`4c73c40`, Enzyme snag
    # `rkbrmi-gradient-519ec090`, enzyme-use), so value and gradient are checked.
    check_against_stan(totals_gamma(data), "rk_gamma")
end

# Non-centered, the grouped entries' coordinates are standard-normal draws and
# each group observes the rest of their density: the same coordinates and the
# same density as the centred joint latent, so the transport pairs them with
# Stan's `y_shares` (snag nlme-view-shares-f1cfaa71).
@stestset "RKBRMI non-centered: the centred density, paired with SBBRMI" begin
    data = totals_data()
    # Every missing entry in a group with a total (row 22 observed).
    v = copy(data.v); v[22] = data.truth[22]
    grouped = merge(data, (; v))
    for (model, label) in ((totals_assigned, "rk_nc_lognormal"), (unread_model, "rk_nc_unread"))
        nc = check_against_stan(model(grouped), label; completions=:noncentered)
        centred = rk_logdensity_problem(RKBRMI(model(grouped));
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        names = BRM._rk_layout_coordinate_names(nc.rk)
        @test replace.(String.(BRM._rk_layout_coordinate_names(RKBRMI(model(grouped)))),
            "y_shares" => "z_shares") == String.(names)
        # Groups of 2, 3 and 1 missing entries: 1 + 2 + 0 coordinates.
        @test count(n -> startswith(String(n), "v.z_shares."), names) == 3
        u = 0.3 .* randn(Xoshiro(23), length(names))
        a, ga = LogDensityProblems.logdensity_and_gradient(centred, u)
        b, gb = LogDensityProblems.logdensity_and_gradient(nc.problem, u)
        @test a ≈ b rtol=1e-12
        @test ga ≈ gb atol=1e-10
        # One observed term per group, at the law's density plus the log
        # Jacobian less the draws' standard normal density.
        pointwise = BRM._rk_pointwise_values(BRM._rk_pointwise_query(nc.rk), u)
        @test length(pointwise.v_shares_weights) == 3
    end
    # A Gamma law with every entry grouped (value and gradient, on the Enzyme
    # dogfood as above).
    check_against_stan(totals_gamma(grouped), "rk_nc_gamma"; completions=:noncentered)
    # Free entries are standardized by their law's location and scale, as for
    # any non-centered completion: a Gamma law has none.
    message(f) = try f(); "" catch e; sprint(showerror, e) end
    @test occursin("needs a location-scale law", message(() ->
        RKBRMI(totals_gamma(data); completions=:noncentered)))
    # With free entries, the free draws `v.z_free` pair with Stan's entries
    # `v_y_free` through their law's location and scale (snag
    # coordinate-trans-23993804).
    free = check_against_stan(totals_assigned(data), "rk_nc_free"; completions=:noncentered)
    @test [b.rk for b in free.t.completions] == [[Symbol("v.z_free.1")]]
    @test [b.stan for b in free.t.completions] == [["v_y_free.1"]]
end
