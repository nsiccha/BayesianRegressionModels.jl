include(joinpath(@__DIR__, "nlme_fixtures.jl"))
include(joinpath(@__DIR__, "nlme_svi_combine_fixture.jl"))
include(joinpath(@__DIR__, "nlme_svi_owned_fixture.jl"))
using LinearAlgebra, Serialization, Statistics

const SLICE_AD = AutoEnzyme(; mode=Enzyme.Reverse)
slice_model(data) = brm_nlme_model(RKBRMI(svi_slice_model(data)); ad_backend=SLICE_AD,
    SVI_SLICE_GROUPS...)
slice_quick(x; seed=1) = brm_svi_fit(x; rng=Xoshiro(seed), SVI_SLICE_QUICK...)

# The expected product for one key: the slices' informed marginals over the
# projected prior (`m0`, `sd0`), as in the formula, computed independently.
function slice_expected(states, key, m0, sd0)
    p0 = isfinite(sd0) ? 1 / sd0^2 : 0.0
    P, η = p0, p0 * m0
    sites = 0
    for st in states
        k = findfirst(isequal(key), st.q.keys)
        if k !== nothing && st.q.informed[k]
            μs, vs = st.q.μ[k], exp(2st.q.ρ[k]) + sum(abs2, st.q.Β[k, :])
        else
            hit = [(u, i) for u in values(st.units) for (i, kk) in enumerate(u.semantic) if isequal(kk, key)]
            isempty(hit) && continue
            u, i = only(hit)
            μs, vs = u.M[i], inv(u.Λ)[i, i]
        end
        sites += 1
        1 / vs < p0 && continue
        P += 1 / vs - p0
        η += μs / vs - p0 * m0
    end
    sites, η / P, 1 / sqrt(P)
end

# A slice's mean of the global whose packed name contains `pattern`, and the
# start's.
slice_mean(state::BRMSVIState, pattern) = state.q.μ[only(findall(k -> hasproperty(k, :coordinate) &&
    occursin(pattern, string(k.coordinate)), state.q.keys))]
slice_mean(start::BRMSVIStart, pattern) = start.globals.μ[only(findall(k ->
    hasproperty(k, :coordinate) && occursin(pattern, string(k.coordinate)), start.globals.keys))]

# The carried product for a draw of a one-margin shared block: slice s's site
# (λ, h) on z_s is (C_s² λ, C_s h) on z_c, with C_s = τ_c / τ_s.
function slice_expected_carried(states, key, m0, sd0, C)
    p0 = isfinite(sd0) ? 1 / sd0^2 : 0.0
    P, η = p0, p0 * m0
    for (s, st) in enumerate(states)
        k = findfirst(isequal(key), st.q.keys)
        k !== nothing && st.q.informed[k] || continue
        μs, vs = st.q.μ[k], exp(2st.q.ρ[k]) + sum(abs2, st.q.Β[k, :])
        1 / vs < p0 && continue
        P += C(s)^2 * (1 / vs - p0)
        η += C(s) * (μs / vs - p0 * m0)
    end
    η / P, 1 / sqrt(P)
end

# A one-draw unit factor (M, Λ) carried by z_s = C z_c + d: its site moves,
# its N(0, 1) prior is applied once.
# The combination with every unit carried, whatever its density says. Factors set by hand
# are not what a unit's data would give, so they test the map and not the choice.
function slice_all_carried(full, states)
    prepared, _ = BRM._brm_svi_checkout!(full; global_coordinates=nothing, previous=nothing)
    start = BRM._brm_svi_combine(prepared.target, states; carry_all=true)
    BRM._brm_svi_checkin!(full, prepared)
    start
end

function slice_carried_unit(M, Λ, C, d)
    Λc = C^2 * (Λ - 1) + 1
    (C * (Λ * M - (Λ - 1) * d) / Λc, Λc)
end

@stestset "slice combination: product of sites, prior counted once, units carried" begin
    data = svi_slice_data(Xoshiro(51), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 4)
    regions = first.(SVI_SLICE_REGIONS)
    full = slice_model(data)
    fits = [slice_quick(svi_slice_model(svi_slice_region(data, r))) for r in regions]
    # Recognizable slice factors, one slice wider than the prior somewhere.
    for (s, fit) in enumerate(fits)
        q = fit.diagnostics.q
        q.μ .= 0.1s .+ 0.05 .* eachindex(q.μ)
        q.ρ .= log(0.05 + 0.02s)
        s == 2 && (q.ρ[1] = log(5.0))
        for i in eachindex(q.M)
            q.M[i] .= 0.3s + 0.1i
            q.Λ[i] .= 4.0 + s + i
            q.U[i] .= sqrt.(q.Λ[i])
        end
    end
    states = brm_svi_checkpoint.(fits)
    @test all(st -> length(st.q.keys) == length(st.q.μ) == length(st.q.informed), states)
    start = slice_all_carried(full, states)
    r = start.receipt
    @test r.slices == 3
    # The projection of this Gaussian global prior: a ~ Normal(0, 2), the log
    # scales Normal(log 0.7, 0.4) and Normal(log 0.5, 0.4), the class draws N(0, 1).
    names = string.(r.globals.coordinates)
    truth = map(names) do n
        n == "a" ? (0.0, 2.0) : occursin("subject.tau", n) ? (log(0.7), 0.4) :
            occursin("class.tau", n) ? (log(0.5), 0.4) : (0.0, 1.0)
    end
    @test all(abs(r.globals.prior_mean[k] - first(truth[k])) < 0.05last(truth[k]) for k in eachindex(truth))
    @test all(abs(r.globals.prior_sd[k] / last(truth[k]) - 1) < 0.1 for k in eachindex(truth))
    @test isempty(r.prior.flat)
    # k1: north and east, k2: north and south, k3: south and east, k4: east.
    @test r.globals.informed == [3, 3, 3, 2, 2, 2, 1]
    @test r.globals.clamped[1] == 1       # slice 2's intercept factor is wider than the prior
    @test sum(r.globals.clamped) == 1
    @test isnan(r.globals.spread[end]) && all(isfinite, r.globals.spread[1:end-1])
    full_keys = BRM._brm_svi_coordinate_keys(full)
    gkeys = full_keys[BRM._brm_svi_globals(full)]
    # The class draws' sites are carried from each slice's class scale to the
    # combined one before their product; the other globals multiply as they are.
    C_class(s) = exp(slice_mean(start, "class.tau") - slice_mean(states[s], "class.tau"))
    @test r.globals.carried == [false, false, false, true, true, true, true]
    for (k, key) in enumerate(start.globals.keys)
        j = findfirst(isequal(key), gkeys)
        sites, μ, σ = slice_expected(states, key, r.globals.prior_mean[j], r.globals.prior_sd[j])
        @test sites == r.globals.informed[j]
        r.globals.carried[j] && ((μ, σ) = slice_expected_carried(states, key,
            r.globals.prior_mean[j], r.globals.prior_sd[j], C_class))
        @test start.globals.μ[k] ≈ μ
        @test start.globals.σ[k] ≈ σ
    end
    # Units take their one slice's factor, carried from that slice's subject
    # scale to the combined one (`mu` has no intercept to mu-reference).
    @test r.units.copied == 9 && isempty(r.units.partial) && isempty(r.units.cold)
    @test r.units.carried == 9 && isempty(r.units.dropped) && isempty(r.units.mapped)
    for (s, st) in enumerate(states), (label, u) in st.units
        C = exp(slice_mean(start, "subject.tau") - slice_mean(st, "subject.tau"))
        M, Λ = slice_carried_unit(only(u.M), only(u.Λ), C, 0.0)
        @test only(start.units[label].M) ≈ M
        @test only(start.units[label].Λ) ≈ Λ
    end
    # The fit starts there: globals by key, units by label, rank part zero.
    fit = brm_svi_fit(full; start, max_iterations=0, ndraws=0, family=LowRankGlobal(2))
    d = fit.diagnostics
    @test d.start.applied == (; globals=7, units=9)
    order = [findfirst(isequal(key), start.globals.keys) for key in gkeys]
    @test d.q.μ == start.globals.μ[order]
    @test d.q.ρ ≈ log.(start.globals.σ[order])
    @test iszero(d.q.Β)
    for (i, label) in enumerate(full.view.levels)
        @test d.q.M[i] == start.units[label].M
        @test d.q.Λ[i] ≈ start.units[label].Λ
        @test d.q.U[i] ≈ cholesky(Symmetric(start.units[label].Λ)).L
    end
    # The shorthand combines inline with the same defaults.
    inline = brm_svi_fit(full; start=fits, max_iterations=0, ndraws=0).diagnostics.q
    public = brm_svi_fit(full; start=brm_svi_combine(full, states), max_iterations=0, ndraws=0).diagnostics.q
    @test inline.μ == public.μ == d.q.μ && inline.M == public.M
    # A start is data-only: it serializes and starts a fit in place of the original.
    io = IOBuffer()
    serialize(io, start)
    loaded = deserialize(seekstart(io))
    @test brm_svi_fit(full; start=loaded, max_iterations=0, ndraws=0).diagnostics.q.μ == d.q.μ
    # Another start leaves the receipt empty.
    @test brm_svi_fit(full; max_iterations=0, ndraws=0).diagnostics.start === nothing
end

@stestset "slice combination: cold units and uninformed coordinates keep the default" begin
    data = svi_slice_data(Xoshiro(52), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 4)
    full = slice_model(data)
    fits = [slice_quick(svi_slice_model(svi_slice_region(data, r))) for r in ("north", "south")]
    start = brm_svi_combine(full, fits)
    @test Set(start.receipt.units.cold) == Set(["e1", "e2", "e3", "e4"])
    @test start.receipt.globals.informed == [2, 2, 2, 1, 2, 1, 0]   # k4 is east's only
    fit = brm_svi_fit(full; start, max_iterations=0, ndraws=0, global_scale=0.3, unit_scale=2.0)
    d = fit.diagnostics
    @test d.start.applied == (; globals=6, units=5)
    @test d.q.ρ[end] == log(0.3)
    point = BRM._brm_svi_start(full, nothing, nothing)
    for (i, label) in enumerate(full.view.levels)
        startswith(label, "e") || continue
        @test d.q.M[i] == point[d.units[i]]
        @test d.q.Λ[i] == fill(0.25, 1, 1)
    end
    # A frozen-preprocessing slice keeps the full class table; classes its rows
    # never read are not informed, and its draws are labelled from that table.
    rk_full = RKBRMI(svi_slice_model(data))
    frozen = slice_quick(RKBRMI(svi_slice_model(svi_slice_region(data, "south")),
        brm_fitted_state(rk_full); resample_groups=(:subject,)))
    state = brm_svi_checkpoint(frozen)
    levels = [k.level for k in state.q.keys if hasproperty(k, :level)]
    @test levels == ["k1", "k2", "k3", "k4"]
    @test state.q.informed == [true, true, true, false, true, true, false]
    mixed = brm_svi_combine(full, [fits[1], state])
    @test mixed.receipt.globals.informed == start.receipt.globals.informed
end

@stestset "slice combination: a global inside slices' units combines their marginals" begin
    # Class k5 is read only by n1 in the north slice and only by w1 in the west
    # slice, so each slice's unit owns it, and in the full model both read it.
    regions = ["north" => ["n1", "n2"], "west" => ["w1", "w2"]]
    classes = Dict("north" => ["k1"], "west" => ["k1"])
    data = svi_slice_data(Xoshiro(53), regions, classes, 4)
    data.class[data.subject .== "n1"] .= "k5"
    data.class[data.subject .== "w1"] .= "k5"
    full = slice_model(data)
    fits = [slice_quick(svi_slice_model(svi_slice_region(data, r))) for r in first.(regions)]
    for (s, fit) in enumerate(fits)
        d = fit.diagnostics
        i = findfirst(==(s == 1 ? "n1" : "w1"), d.model.view.levels)
        @test length(d.units[i]) == 2                 # its draw and the k5 draw
        d.q.M[i] .= [0.2s, -0.4s]
        d.q.Λ[i] .= [3.0 1.0; 1.0 2.0 + s]
        d.q.U[i] .= cholesky(Symmetric(d.q.Λ[i])).L
    end
    states = brm_svi_checkpoint.(fits)
    start = brm_svi_combine(full, states)
    gkeys = BRM._brm_svi_coordinate_keys(full)[BRM._brm_svi_globals(full)]
    k5 = findfirst(k -> hasproperty(k, :level) && k.level == "k5", gkeys)
    @test k5 !== nothing                              # global in the full model
    r = start.receipt.globals
    @test r.informed[k5] == 2
    # k5's sites are the two units' marginals, carried to the combined scale.
    C(s, name) = exp(slice_mean(start, name) - slice_mean(states[s], name))
    p0, m0 = 1 / r.prior_sd[k5]^2, r.prior_mean[k5]
    P, η = p0, p0 * m0
    for (s, Λ) in enumerate(([3.0 1.0; 1.0 3.0], [3.0 1.0; 1.0 4.0]))
        v = inv(Λ)[2, 2]
        P += C(s, "class.tau")^2 * (1 / v - p0)
        η += C(s, "class.tau") * (-0.4s / v - p0 * m0)
    end
    k = findfirst(isequal(gkeys[k5]), start.globals.keys)
    @test r.carried[k5] && start.globals.μ[k] ≈ η / P && start.globals.σ[k] ≈ 1 / sqrt(P)
    # n1 takes its own draw's marginal from the joint factor, carried.
    n1 = start.units["n1"]
    M, Λ = slice_carried_unit(0.2, 1 / inv([3.0 1.0; 1.0 3.0])[1, 1], C(1, "subject.tau"),
        0.0)
    @test only(n1.M) ≈ M && only(n1.Λ) ≈ Λ
    @test start.receipt.units.copied == 4 && isempty(start.receipt.units.partial)
end

@stestset "slice combination: packed-name unit coordinates match only inside their unit" begin
    # `z[c]` entries are unit-owned and named by position. The second slice
    # numbers its coefficients from 1, so its `z.1` is s4's, not s1's.
    full = brm_nlme_model(RKBRMI(svi_owned_model(svi_owned_data(1:6))); ad_backend=SLICE_AD)
    quick(data, seed) = brm_svi_fit(svi_owned_model(data); rng=Xoshiro(seed),
        max_iterations=0, ndraws=0)
    first_slice = quick(svi_owned_data(1:3), 1)
    local_slice = quick(merge(svi_owned_data(4:6), (; c1=[1, 3, 5], c2=[2, 4, 6], ncoeff=6)), 2)
    for (s, fit) in enumerate((first_slice, local_slice)), M in fit.diagnostics.q.M
        M .= 0.1 .* eachindex(M) .+ s
    end
    start = brm_svi_combine(full, [first_slice, local_slice])
    @test start.receipt.units.copied == 3
    @test Set(start.receipt.units.partial) == Set(["s4", "s5", "s6"])
    first_state = brm_svi_checkpoint(first_slice)
    coefficient(key) = hasproperty(key, :coordinate)
    for label in ("s1", "s2", "s3")
        u, v = start.units[label], first_state.units[label]
        @test u.keys == v.semantic && length(u.M) == 3    # its draw and two coefficients
        # The coefficients are no draws: only the draw is carried.
        @test u.M[coefficient.(u.keys)] == v.M[coefficient.(v.semantic)]
    end
    for label in ("s4", "s5", "s6")
        @test length(start.units[label].M) == 1 &&        # its draw only
            !coefficient(only(start.units[label].keys))
    end
    # With the full model's numbering every unit is copied whole, and the
    # entries the second slice's rows never read are not its sites.
    global_slice = quick(svi_owned_data(4:6), 3)
    whole = brm_svi_combine(full, [first_slice, global_slice])
    @test whole.receipt.units.copied == 6
    state = brm_svi_checkpoint(global_slice)
    informed = Dict(zip(string.(state.globals), state.q.informed))
    @test informed["tau"] && informed["sigma"]
    @test sort([name for name in keys(informed) if startswith(name, "z.")]) == ["z.$j" for j in 1:6]
    @test !any(informed["z.$j"] for j in 1:6)
end

# The class block with an explicit ID, so brm_start can address its effects.
function svi_slice_id_model(data)
    @brm data begin
        a ~ Normal(0, 2)
        mu ~ 0 + (1 | subject) + (1 | cls | class)
        sd(mu, subject) ~ LogNormal(log(0.7), 0.4)
        sd(mu, cls) ~ LogNormal(log(0.5), 0.4)
        mean_y = a + mu
        y ~ Normal(mean_y, 0.5)
    end
end

@stestset "frozen replay: a grouping it does not resample keeps the training's level rows" begin
    # The south rows read classes k2 and k3 only, while the replay keeps the
    # training's four class draws. Before this fix the rows came from the
    # replayed column, so `"k3"` addressed draw 2 (the training's k2).
    data = svi_slice_data(Xoshiro(56), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 4)
    rk = RKBRMI(svi_slice_id_model(data))
    frozen = RKBRMI(svi_slice_id_model(svi_slice_region(data, "south")), brm_fitted_state(rk);
        resample_groups=(:subject,))
    record = only(r for r in BRM._brm_nlme_records(frozen.plan) if r.kind === :ranef &&
        r.group === :class)
    @test BRM._rk_ranef_record_levels(frozen, record) == ["k1", "k2", "k3", "k4"]
    @test BRM._rk_ranef_record_levels(rk, record) == ["k1", "k2", "k3", "k4"]
    class_z(x) = findall(n -> startswith(string(n), "b_cls_class.z."),
        BRM._rk_layout_coordinate_names(x))
    @test length(class_z(frozen)) == length(class_z(rk)) == 4
    inits = (; cls=(; effects=["k3" => 0.7]))
    @test brm_start(frozen, inits)[class_z(frozen)] == brm_start(rk, inits)[class_z(rk)] ==
        [0, 0, 0.7, 0]
    # A resampled grouping still reads the replayed rows' levels.
    subject = only(r for r in BRM._brm_nlme_records(frozen.plan) if r.kind === :ranef &&
        r.group === :subject)
    @test BRM._rk_ranef_record_levels(frozen, subject) == ["s1", "s2"]
end

@stestset "slice combination refuses what it cannot match" begin
    data = svi_slice_data(Xoshiro(54), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 4)
    regions = first.(SVI_SLICE_REGIONS)
    full = slice_model(data)
    north = slice_quick(svi_slice_model(svi_slice_region(data, "north")))
    @test_throws "comes from another model" brm_svi_combine(full,
        [slice_quick(svi_slice_scaled_model(svi_slice_region(data, "north")))])
    @test_throws "has factors in slices 1, 2" brm_svi_combine(full, [north, north])
    rk_scaled = RKBRMI(svi_slice_scaled_model(data))
    scaled = brm_nlme_model(rk_scaled; ad_backend=SLICE_AD, SVI_SLICE_GROUPS...)
    @test_throws "preprocessing changed: fitted value" brm_svi_combine(scaled,
        [slice_quick(svi_slice_scaled_model(svi_slice_region(data, r))) for r in regions])
    frozen = [slice_quick(RKBRMI(svi_slice_scaled_model(svi_slice_region(data, r)),
        brm_fitted_state(rk_scaled); resample_groups=(:subject,))) for r in regions]
    @test brm_svi_combine(scaled, frozen).receipt.globals.informed == [3, 3, 3, 2, 2, 2, 1, 3]
    @test_throws "not a brm_svi_fit result" brm_svi_combine(full, [(; q=1)])
    @test_throws "as a vector" brm_svi_combine(full, north)
    st = brm_svi_checkpoint(north)
    old = BRMSVIState(st.format, st.preprocessing, st.source_id, st.globals,
        BRM._brm_svi_global_factor(st.q),
        Dict{Any,NamedTuple}(k => (; v.keys, v.M, v.Λ, v.U) for (k, v) in st.units),
        st.group, st.nested, st.shared, st.fixed, st.unit_partition, st.updates)
    @test_throws "a BRM without brm_svi_combine wrote it" brm_svi_combine(full, [old])
    @test_throws "factor-level start" brm_svi_fit(full; laplace=GaussianBlockLaplace(),
        start=[north])
    @test_throws "not both" brm_svi_fit(full; start=[north], inits=(; a=0.0),
        max_iterations=0, ndraws=0)
end

# The joint product over `keys` (no shared level's draws among them): each
# slice's factor covariance diag(e^{2ρ}) + ΒΒᵀ over the keys its data reached,
# inverted, less the prior's projection, with its negative part dropped.
function slice_expected_joint(states, keys, m0, p0)
    P, η = Matrix(Diagonal(p0)), p0 .* m0
    for st in states
        J = [findfirst(isequal(key), st.q.keys) for key in keys]
        R = [r for r in eachindex(keys) if J[r] !== nothing && st.q.informed[J[r]]]
        isempty(R) && continue
        Js = Int.(J[R])
        Ps = inv(Symmetric(Matrix(Diagonal(exp.(2 .* st.q.ρ[Js])) + st.q.Β[Js, :] * st.q.Β[Js, :]')))
        E = eigen(Symmetric(Ps - Diagonal(p0[R])))
        V, λ = E.vectors[:, E.values .> 0], E.values[E.values .> 0]
        P[R, R] .+= V * Diagonal(λ) * V'
        η[R] .+= V * (V' * (Ps * st.q.μ[Js] - p0[R] .* m0[R]))
    end
    P \ η, P
end

@stestset "slice combination: a low-rank slice's sites are joint" begin
    data = svi_slice_data(Xoshiro(55), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 4)
    full = slice_model(data)
    fit = brm_svi_fit(svi_slice_model(svi_slice_region(data, "east")); rng=Xoshiro(2),
        family=LowRankGlobal(1), SVI_SLICE_QUICK...)
    q = fit.diagnostics.q
    # Narrower than the prior everywhere, so all of them are in the joint site.
    q.ρ .= log(0.1)
    q.Β .= 0.05 .* (1:length(q.μ))
    state = brm_svi_checkpoint(fit)
    start = brm_svi_combine(full, [state])
    r = start.receipt.globals
    @test r.joint === :joint && start.receipt.slice_rank == [1]
    gkeys = BRM._brm_svi_coordinate_keys(full)[BRM._brm_svi_globals(full)]
    # `a` and the two scales combine jointly; the class draws are carried.
    joined = findall(j -> !r.carried[j] && r.informed[j] > 0, eachindex(gkeys))
    @test length(joined) == 3
    μ, P = slice_expected_joint([state], gkeys[joined], r.prior_mean[joined],
        1 ./ r.prior_sd[joined] .^ 2)
    order = [findfirst(isequal(key), start.globals.keys) for key in gkeys[joined]]
    @test start.globals.μ[order] ≈ μ
    @test start.globals.σ[order] ≈ 1 ./ sqrt.(diag(P))
    block = only(start.globals.correlated)
    @test block.positions == order && block.precision ≈ P
    @test !isdiag(P)
    C_class(s) = exp(slice_mean(start, "class.tau") - slice_mean(state, "class.tau"))
    for j in findall(r.carried)
        k = findfirst(isequal(gkeys[j]), start.globals.keys)
        k === nothing && continue
        μc, σc = slice_expected_carried([state], gkeys[j], r.prior_mean[j], r.prior_sd[j], C_class)
        @test start.globals.μ[k] ≈ μc && start.globals.σ[k] ≈ σc
    end
    # A mean-field fit starts at the mean-field part; a rank-1 fit's rank part
    # takes the direction in which P is widest relative to it.
    globals = BRM._brm_svi_globals(full)
    rows = [findfirst(isequal(key), gkeys) for key in gkeys[joined]]
    meanfield = brm_svi_fit(full; start, max_iterations=0, ndraws=0).diagnostics.q
    @test meanfield.ρ[rows] ≈ log.(1 ./ sqrt.(diag(P)))
    lowrank = brm_svi_fit(full; start, max_iterations=0, ndraws=0,
        family=LowRankGlobal(1)).diagnostics.q
    d = 1 ./ sqrt.(diag(P))
    E = eigen(Symmetric(d .* P .* d'))
    ω, u = E.values[1], E.vectors[:, 1]
    @test ω < 1
    Σq = Diagonal(exp.(2 .* lowrank.ρ[rows])) + lowrank.Β[rows, :] * lowrank.Β[rows, :]'
    @test Σq ≈ Diagonal(d .^ 2) + (1 / ω - 1) * (d .* u) * (d .* u)'
    @test iszero(lowrank.Β[setdiff(eachindex(globals), rows), :])
    # Along that direction the start's variance is the product's.
    @test (u ./ d)' * Σq * (u ./ d) ≈ 1 / ω
end

# A slice's factor over named coordinates, without a model.
slice_source(keys, mean, scale, Β) = BRM._BRMSVISource(collect(Any, keys), mean,
    scale .^ 2 .+ vec(sum(abs2, Β; dims=2)), scale, Β, fill(true, length(keys)), Dict{Any,NamedTuple}())
# The exact posterior of one linear-Gaussian row under a diagonal prior, as a rank-1 factor.
function slice_ridge_source(keys, row, y, σy, p0)
    Σ = inv(Symmetric(Diagonal(p0) + row * row' / σy^2))
    E = eigen(Symmetric(Matrix(Σ)))
    floor_ = minimum(E.values) * 0.999
    Β = sqrt(E.values[end] - floor_) * E.vectors[:, end:end]
    slice_source(keys, Σ * row * y / σy^2, fill(sqrt(floor_), length(keys)), Β)
end
function slice_joint(sources, p0, m0; limit=4096)
    keys = collect(Any, first(sources).keys)
    index = BRM._brm_svi_source_index(sources)
    T = [BRM._brm_svi_heterogeneity(index, key, p0[k], m0[k]) for (k, key) in enumerate(keys)]
    merge(BRM._brm_svi_joint_product(index, keys, collect(eachindex(keys)), p0, m0, T, limit),
        (; heterogeneity=T))
end

# The joint product a plain reading gives: `slice_expected_joint`'s formula on sources.
function slice_plain_joint(sources, p0, m0)
    P, η = Matrix(Diagonal(p0)), p0 .* m0
    for s in sources
        Ps = inv(Symmetric(Diagonal(s.scale .^ 2) + s.Β * s.Β'))
        E = eigen(Symmetric(Ps - Diagonal(p0)))
        V, λ = E.vectors[:, E.values .> 0], E.values[E.values .> 0]
        P .+= V * Diagonal(λ) * V'
        η .+= V * (V' * (Ps * s.mean - p0 .* m0))
    end
    P \ η
end

# A slice whose fit left its ridge narrower than the prior: sharp across the ridge, with
# the rank part carrying the ridge, as `LowRankGlobal` fits do.
function slice_svi_ridge_source(keys, row, y; ridge_sd=0.8, sharp=0.04)
    along = [row[2], -row[1]] ./ sqrt(sum(abs2, row))
    slice_source(keys, row .* y ./ sum(abs2, row), fill(sharp, 2), (ridge_sd * sqrt(2)) .* along[:, :])
end

@stestset "slice combination: a site's factor, its widening and the between-slice variance" begin
    # The Paule-Mandel estimate has a closed form at equal site variances: the sample
    # variance of the means, less that variance.
    c = [1.0, 2.0, 3.0, 8.0]
    @test BRM._brm_svi_paule_mandel([(x, 0.1) for x in c]) ≈ var(c) - 0.1
    @test BRM._brm_svi_paule_mandel([(1.0, 0.1), (9.0, 0.1)]) ≈ 32 - 0.1
    # Below the contradiction level it is 0, though Q is above K - 1 (Q(0) = 4).
    @test BRM._brm_svi_paule_mandel([(1.0, 0.5), (3.0, 0.5)]) == 0
    # So where SVI's sds being a quarter of the posterior's would explain the gap: Q(0) = 64,
    # far past the level at these sds, but 4 at four times them.
    @test BRM._brm_svi_paule_mandel([(1.0, 0.5), (9.0, 0.5)]) == 0
    @test BRM._brm_svi_paule_mandel([(1.0, 1e-4)]) == 0
    @test BRM._brm_svi_paule_mandel(Tuple{Float64,Float64}[]) == 0

    rng = Xoshiro(3)
    n, Β = 6, 0.15 .* randn(rng, 6, 2)
    scale, mean, m0 = 0.1 .+ 0.2 .* rand(rng, n), randn(rng, n), randn(rng, n)
    p0 = [0.5, 1.0, 0.25, 2.0, 0.0, 1.5]  # one flat coordinate
    clamped = zeros(Int, n)
    site = BRM._brm_svi_site_factor(collect(1:n), mean, scale, Β, p0, m0, clamped)
    Ps = inv(Symmetric(Diagonal(scale .^ 2) + Β * Β'))
    S, h = Ps - Diagonal(p0), Ps * mean - p0 .* m0
    @test isposdef(Symmetric(S)) && iszero(clamped)
    # The plain site at T = 0, and the same site with T added to its variance.
    plain = BRM._brm_svi_site_widened(site, zeros(n), m0)
    @test plain.S ≈ S
    @test plain.h ≈ h
    T = [0.1, 0.0, 2.0, 0.5, 0.0, 0.3]
    wide = BRM._brm_svi_site_widened(site, T, m0)
    @test wide.S ≈ inv(inv(Symmetric(S)) + Diagonal(T))
    @test wide.h ≈ wide.S * (S \ h)
    # A rescaling of the coordinates rescales the site, also with a direction dropped:
    # a prior sd of 1e-6 beside one of 1 does not decide the directions.
    Β2 = copy(Β)
    Β2[1, :] .*= 10
    d = [1e-6, 1.0, 1e3, 1.0, 1.0, 1e-3]
    a = BRM._brm_svi_site_factor(collect(1:n), mean, scale, Β2, p0, m0, clamped)
    @test sum(clamped) == 1
    b = BRM._brm_svi_site_factor(collect(1:n), d .* mean, d .* scale, d .* Β2, p0 ./ d .^ 2,
        d .* m0, zeros(Int, n))
    @test size(a.W, 2) == n - 1 == size(b.W, 2)
    @test b.W * b.W' ≈ (a.W * a.W') ./ (d * d') rtol = 1e-8
    # What it drops is the site's negative part, so what it keeps is above the site.
    S2 = inv(Symmetric(Diagonal(scale .^ 2) + Β2 * Β2')) - Diagonal(p0)
    @test !isposdef(Symmetric(S2))
    @test all(>=(-1e-8), eigvals(Symmetric(a.W * a.W' - S2)))
    # A slice wider than the prior in every direction has no site left, and its
    # coordinate stays at the prior.
    none = BRM._brm_svi_site_factor([1], [0.3], [2.0], zeros(1, 1), [1.0], [0.0], zeros(Int, 1))
    @test size(none.W, 2) == 0 && isempty(none.d)
    @test iszero(BRM._brm_svi_site_widened(none, [0.5], [0.0]).S)
    r = slice_joint([slice_source([:x], [0.3], [2.0], [1.0;;])], [1.0], [0.0])
    @test r.status === :joint && r.μ ≈ [0.0] && r.diagonal ≈ [1.0]
    # Who pins a coordinate: narrowed and not carried by the rank part, or far below the prior.
    @test BRM._brm_svi_pins(0.5, 0.1) && BRM._brm_svi_pins(0.001, 0.99)
    @test !BRM._brm_svi_pins(0.5, 0.99) && !BRM._brm_svi_pins(0.9, 0.1)
end

@stestset "slice combination: slices that contradict do not move coordinates they leave at the prior" begin
    # A sharply informed coordinate on which the slices disagree by hundreds of their
    # sds, and coordinates no slice informs, with the small rank rows SVI's noise
    # leaves on them (their marginals scatter around the prior's, as a fit's do).
    rng = Xoshiro(11)
    d, K = 40, 8
    keys = [:l; [Symbol(:z, j) for j in 1:d]]
    p0, m0 = ones(d + 1), zeros(d + 1)
    locations = range(-1, 1; length=K)
    sources = map(1:K) do s
        Β = zeros(d + 1, 2)
        Β[1, 1] = 0.004
        Β[2:end, :] .= 0.15 .* randn(rng, d, 2)
        scale = [0.01; sqrt.(max.(1 .+ 0.06 .* randn(rng, d) .- vec(sum(abs2, Β[2:end, :]; dims=2)), 0.01))]
        slice_source(keys, [locations[s]; 0.05 .* randn(rng, d)], scale, Β)
    end
    plain = slice_plain_joint(sources, p0, m0)
    r = slice_joint(sources, p0, m0)
    @test r.status === :joint
    # The positive control: the plain product carries the contradiction far out.
    @test maximum(abs, plain[2:end]) > 5
    @test maximum(abs, r.μ[2:end]) < 1
    @test median(abs.(r.μ[2:end])) < 0.25
    # The block is what the slices narrow, and the contradiction has its variance.
    @test r.block == [1] && size(r.precision) == (1, 1)
    @test r.heterogeneity[1] ≈ var(locations) - 0.0108^2 rtol = 0.02
    @test all(iszero, r.heterogeneity[2:end])
    @test minimum(locations) < r.μ[1] < maximum(locations)
    # It sits among the slices and stays wide: a product of random effects, not of widths.
    @test 1 / sqrt(r.diagonal[1]) ≈ sqrt(1 / (1 + K / r.heterogeneity[1])) rtol = 0.05
    # So does a coordinate whose narrow marginal the rank part carries.
    carried = map(sources) do s
        Β = copy(s.Β)
        Β[1, 1] = 0.01
        slice_source(s.keys, s.mean, [0.001; s.scale[2:end]], Β)
    end
    @test sum(abs2, carried[1].Β[1, :]) / carried[1].var[1] > 0.8
    c = slice_joint(carried, p0, m0)
    @test c.heterogeneity[1] > 0.3 && maximum(abs, c.μ[2:end]) < 1
    @test 1 / sqrt(c.diagonal[1]) > 0.1
end

@stestset "slice combination: ridges that reconcile cross exactly, wherever the truth sits" begin
    keys = [:a, :b]
    ridge(truth, sd) = let p0 = 1 ./ sd .^ 2
        (p0, [slice_ridge_source(keys, row, row' * truth, 0.05, p0) for row in ([1.0, 1.0], [1.0, -1.0])])
    end
    # Each slice pins one direction, so where they cross neither has a residual, also
    # where the slices' marginals read as a contradiction: truth (2, 5) under N(0, 3²)
    # gives `a` marginals at 3.5 and -1.5, each of sd 2.1.
    for truth in ([2.0, 1.5], [2.0, 5.0], [6.0, 5.0], [9.0, -7.0])
        p0, sources = ridge(truth, [3.0, 3.0])
        r = slice_joint(sources, p0, zeros(2))
        @test r.status === :joint
        @test r.μ ≈ slice_plain_joint(sources, p0, zeros(2)) atol = 1e-3
        @test r.μ ≈ truth atol = 0.05
        @test iszero(r.heterogeneity)
    end
    # So with a fit's ridge narrower than the prior: its marginals are narrow enough to
    # read as a contradiction, but the rank part carries them.
    for truth in ([2.0, 1.5], [2.0, 5.0], [6.0, 5.0], [9.0, -7.0])
        p0 = fill(1 / 9, 2)
        sources = [slice_svi_ridge_source(keys, row, row' * truth) for row in ([1.0, 1.0], [1.0, -1.0])]
        r = slice_joint(sources, p0, zeros(2))
        @test r.status === :joint && iszero(r.heterogeneity)
        @test r.μ ≈ truth atol = 0.05
    end
    # A ridge's narrow-prior side is in the site though the slice barely narrows it:
    # its variance is along the ridge.
    for truth in ([0.0, 2.0], [3.0, 2.0], [1.0, -1.0])
        p0, sources = ridge(truth, [10.0, 1.0])
        @test sources[1].var[2] * p0[2] > 0.95
        r = slice_joint(sources, p0, zeros(2))
        @test r.status === :joint && r.block == [1, 2] && iszero(r.heterogeneity)
        @test r.μ ≈ truth atol = 0.05
    end
    # A third slice contradicting one of them along its own ridge: the mean sits between
    # what the slices say.
    p0, sources = ridge([2.0, 5.0], [3.0, 3.0])
    push!(sources, slice_ridge_source(keys, [1.0, 1.0], 9.0, 0.05, p0))
    r = slice_joint(sources, p0, zeros(2))
    @test r.status === :joint
    @test 7.1 < sum(r.μ) < 8.9
end

@stestset "slice combination: contradicting mean-field slices stay wide, in a fit's start" begin
    data = svi_slice_ridge_data(Xoshiro(7))
    groups = (; subject_group=:subject)
    full = brm_nlme_model(RKBRMI(svi_slice_ridge_model(data)); ad_backend=SLICE_AD, groups...)
    states = [brm_svi_checkpoint(brm_svi_fit(svi_slice_ridge_model(svi_slice_rows(data, data.region .== r));
        rng=Xoshiro(10 + k), family=MeanFieldGlobal(), max_iterations=0, ndraws=0, batch_size=4, groups...))
        for (k, r) in enumerate(("plus", "minus"))]
    at = findfirst(==("a"), String.(first(states).globals))
    # Two slices that each pin `a`, 200 of their sds apart, both with confidence.
    for (state, location) in zip(states, (-1.0, 1.0))
        state.q.μ[at] = location
        state.q.ρ[at] = log(0.01)
    end
    start = brm_svi_combine(full, states)
    r = start.receipt.globals
    @test r.joint === :diagonal
    name = String.(r.coordinates)
    k = findfirst(==("a"), name)
    # Between the slices, with their disagreement for a variance (T = 2 less 0.01², two
    # sites), and starting as wide as a cold fit, where the product's sd is 0.9.
    @test r.heterogeneity[k] ≈ 2 atol = 0.01
    key = BRM._brm_svi_coordinate_keys(full)[BRM._brm_svi_globals(full)][k]
    g = findfirst(isequal(key), start.globals.keys)
    @test abs(start.globals.μ[g]) < 0.05
    @test start.globals.σ[g] ≈ 0.1
    @test r.spread[k] > 50
    # A fit from it takes those.
    q = brm_svi_fit(full; start, max_iterations=0, ndraws=0).diagnostics.q
    @test exp(q.ρ[k]) ≈ start.globals.σ[g]
end

@stestset "slice combination: a unit keeps its slice's draws where carrying them costs it" begin
    # Slices that disagree on the intercept their subjects' block absorbs, whose scale is
    # pinned near 0: a unit's total could only be kept by a draw far beyond its prior.
    data = svi_slice_data(Xoshiro(57), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 4)
    regions = first.(SVI_SLICE_REGIONS)
    full = brm_nlme_model(RKBRMI(svi_slice_intercept_model(data)); ad_backend=SLICE_AD,
        SVI_SLICE_GROUPS...)
    fits = [slice_quick(svi_slice_intercept_model(svi_slice_region(data, r))) for r in regions]
    for (s, fit) in enumerate(fits)
        q = fit.diagnostics.q
        names = string.(fit.diagnostics.model.view.coordinates[fit.diagnostics.globals])
        q.μ[findfirst(==("mu_Intercept"), names)] = Float64(s) - 2
        q.μ[findfirst(n -> occursin("subject.tau", n), names)] = log(1e-6)
        q.ρ .= log(0.05)
        for i in eachindex(q.M)    # units with a site of their own, or nothing moves
            q.M[i] .= 0.0
            q.Λ[i] .= 10.0
            q.U[i] .= sqrt(10.0)
        end
    end
    states = brm_svi_checkpoint.(fits)
    # The positive control: carried regardless, the outer regions' draws run to a million.
    @test maximum(abs, reduce(vcat, [u.M for u in values(slice_all_carried(full, states).units)])) > 1e5
    start = brm_svi_combine(full, states)
    r = start.receipt.units
    # The middle region's intercept is the combination's, so it carries as it would have;
    # the others' are not, so they keep their draws.
    @test !isempty(r.uncarried) && r.carried > 0
    @test r.carried + length(r.uncarried) == 9
    for st in states, (label, u) in st.units
        label in r.uncarried || continue
        @test start.units[label].M ≈ u.M
        @test start.units[label].Λ ≈ BRM._brm_svi_saved_precision(u)
    end
    @test maximum(abs, reduce(vcat, [u.M for u in values(start.units)])) < 100
    # The density reads no value as none, and a draw far out as worse than one near.
    prepared, _ = BRM._brm_svi_checkout!(full; global_coordinates=nothing, previous=nothing)
    t = prepared.target
    g = BRM._brm_svi_start(full, nothing, nothing)[t.globals]
    near, far = BRM._brm_svi_unit_density(t, 1, g, [0.0]), BRM._brm_svi_unit_density(t, 1, g, [1e6])
    @test isfinite(near) && near > far
    @test BRM._brm_svi_unit_density(t, 1, g, [NaN]) == -Inf
    BRM._brm_svi_checkin!(full, prepared)
end

@stestset "slice combination: a joint mean nothing supports falls back" begin
    data = svi_slice_ridge_data(Xoshiro(7))
    groups = (; subject_group=:subject)
    full = brm_nlme_model(RKBRMI(svi_slice_ridge_model(data)); ad_backend=SLICE_AD, groups...)
    states = [brm_svi_checkpoint(brm_svi_fit(svi_slice_ridge_model(svi_slice_rows(data, data.region .== r));
        rng=Xoshiro(10 + k), family=LowRankGlobal(1), max_iterations=0, ndraws=0, batch_size=4, groups...))
        for (k, r) in enumerate(("plus", "minus"))]
    ab = [findfirst(==(c), String.(first(states).globals)) for c in ("a", "b")]
    prior_sd = 3.0
    # Exact slices pinning two nearly parallel directions consistently, at a crossing
    # the priors make implausible: neither the slices' marginals nor their sites
    # disagree, so only the range catches it.
    far = [-30.0, 30.0]
    for (state, row) in zip(states, ([1.0, 1.0], [1.0, 1.01]))
        p0 = fill(1 / prior_sd^2, 2)
        source = slice_ridge_source([:a, :b], row, row' * far, 0.002, p0)
        state.q.μ[ab] .= source.mean
        state.q.ρ[ab] .= log.(source.scale)
        state.q.Β[ab, :] .= source.Β
    end
    start = brm_svi_combine(full, states)
    r = start.receipt.globals
    @test r.joint === :outlying
    names = String.(r.coordinates)
    @test sort(names[r.outlying]) == ["a", "b"]
    @test isempty(start.globals.correlated)
    @test all(iszero, r.heterogeneity)
    # What stands is the coordinate-by-coordinate product: it matches the same slices
    # with the rank parts folded into their marginals.
    fold(state) = let s = deepcopy(state)
        s.q.ρ .= log.(sqrt.(exp.(2 .* s.q.ρ) .+ vec(sum(abs2, s.q.Β; dims=2))))
        s.q.Β .= 0
        s
    end
    diagonal = brm_svi_combine(full, fold.(states))
    @test diagonal.receipt.globals.joint === :diagonal
    gkeys = BRM._brm_svi_coordinate_keys(full)[BRM._brm_svi_globals(full)]
    for c in ("a", "b")
        key = gkeys[findfirst(==(c), names)]
        μs, μd = (s.globals.μ[findfirst(isequal(key), s.globals.keys)] for s in (start, diagonal))
        @test μs ≈ μd
        @test abs(μs) < 3prior_sd
    end
    # `joint_limit` bounds what the joint sites hold, not every informed global.
    limited = brm_svi_combine(full, states; joint_limit=1)
    @test limited.receipt.globals.joint === :limit
    @test brm_svi_combine(full, states; joint_limit=0).receipt.globals.joint === :limit
end

@stestset "slice combination: joint sites land where the slices' ridges cross" begin
    data = svi_slice_ridge_data(Xoshiro(7))
    groups = (; subject_group=:subject)
    full = brm_nlme_model(RKBRMI(svi_slice_ridge_model(data)); ad_backend=SLICE_AD, groups...)
    opts = (; batch_size=4, ndraws=0)
    # A slice's rank part steps in units of its coordinates' scales (BRM 0.2.242),
    # here a few hundredths along a ridge whose sd is near 0.8, so the ridge needs a
    # longer exploration than the default 40 windows: at the default the slices'
    # ridge sds read 0.18-0.24 and the joint start lands 0.3-0.76 sds off.
    slices = [brm_svi_fit(svi_slice_ridge_model(svi_slice_rows(data, data.region .== r));
        rng=Xoshiro(10 + k), family=LowRankGlobal(2), min_windows=200, opts..., groups...)
        for (k, r) in enumerate(("plus", "minus"))]
    # Each slice is a ridge: "plus" pins a + b, "minus" pins a - b.
    for (fit, sign) in zip(slices, (-1, 1))
        Σ = BRM._brm_svi_global_covariance(fit.diagnostics.q)
        @test sign * Σ[1, 2] / sqrt(Σ[1, 1] * Σ[2, 2]) > 0.95
    end
    # The coordinate-by-coordinate product: each slice's rank part folded into
    # its marginal variances.
    fold(fit) = let state = deepcopy(brm_svi_checkpoint(fit))
        state.q.ρ .= log.(sqrt.(diag(BRM._brm_svi_global_covariance(state.q))))
        state.q.Β .= 0
        state
    end
    joint, diagonal = brm_svi_combine(full, slices), brm_svi_combine(full, fold.(slices))
    @test joint.receipt.globals.joint === :joint && joint.receipt.slice_rank == [2, 2]
    @test diagonal.receipt.globals.joint === :diagonal && isempty(diagonal.globals.correlated)
    reference = brm_svi_fit(full; rng=Xoshiro(99), family=LowRankGlobal(2), opts...).diagnostics
    gkeys = BRM._brm_svi_coordinate_keys(full)[reference.globals]
    sd = sqrt.(diag(BRM._brm_svi_global_covariance(reference.q)))
    distance(start) = abs.(start.globals.μ[[findfirst(isequal(key), start.globals.keys)
        for key in gkeys]] .- reference.q.μ) ./ sd
    # The joint product lands where the ridges cross; the diagonal one between
    # the slices' means, many posterior sds off (measured 0.12, 0.05 against
    # 19.3, 13.2).
    @test all(distance(joint)[1:2] .< 0.5)
    @test all(distance(diagonal)[1:2] .> 5)
    first_window(start, family) = brm_svi_fit(full; rng=Xoshiro(80), start, family,
        max_iterations=50, window=50, opts...).diagnostics.elbo_windows_trimmed[1]
    for family in (MeanFieldGlobal(), LowRankGlobal(2))
        cold, warm, diag_warm = first_window(nothing, family), first_window(joint, family),
            first_window(diagonal, family)
        @info "first-window trimmed ELBO" family cold joint = warm diagonal = diag_warm
        @test warm > diag_warm && warm > cold
    end
    # A fit from the start is itself a slice of its data, so the combination
    # nests (sub-slices -> slice -> whole): alone, it returns the start's means.
    again = brm_svi_combine(full, [brm_svi_fit(full; start=joint, family=LowRankGlobal(2),
        max_iterations=0, ndraws=0)])
    @test again.receipt.globals.joint === :joint
    @test again.globals.keys == joint.globals.keys && again.globals.μ ≈ joint.globals.μ
end

@stestset "slice combination: real slice fits start the full fit near its optimum" begin
    data = svi_slice_data(Xoshiro(61), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 8)
    regions = first.(SVI_SLICE_REGIONS)
    full = slice_model(data)
    opts = (; batch_size=4, ndraws=0)
    slices = [brm_svi_fit(svi_slice_model(svi_slice_region(data, r)); rng=Xoshiro(70 + k),
        opts..., SVI_SLICE_GROUPS...) for (k, r) in enumerate(regions)]
    start = brm_svi_combine(full, slices)
    cold = brm_svi_fit(full; rng=Xoshiro(80), opts...)
    warm = brm_svi_fit(full; rng=Xoshiro(80), start, opts...)
    q = cold.diagnostics.q
    gkeys = BRM._brm_svi_coordinate_keys(full)[cold.diagnostics.globals]
    order = [findfirst(isequal(key), start.globals.keys) for key in gkeys]
    # The start lies within about one posterior sd of the converged globals,
    # with scales of the same size, and its first window scores above a cold start.
    @test maximum(abs.(start.globals.μ[order] .- q.μ) ./ exp.(q.ρ)) < 2
    @test all(0.5 .< start.globals.σ[order] ./ exp.(q.ρ) .< 2)
    @test warm.diagnostics.elbo_windows_trimmed[1] > cold.diagnostics.elbo_windows_trimmed[1]
    @test maximum(abs.(warm.diagnostics.q.μ .- q.μ) ./ exp.(q.ρ)) < 0.5
end

# The full model's packed point holding global means and unit means, each by
# semantic key.
function slice_full_point(m, global_keys, global_means, units)
    u = BRM._brm_svi_start(m, nothing, nothing)
    keys = BRM._brm_svi_coordinate_keys(m)
    at = Dict(k => j for (j, k) in enumerate(keys))
    for (key, x) in zip(global_keys, global_means)
        haskey(at, key) && (u[at[key]] = x)
    end
    for (_, unit) in units, (key, x) in zip(unit.keys, unit.M)
        u[at[key]] = x
    end
    u
end
slice_full_point(m, state::BRMSVIState) = slice_full_point(m, state.q.keys, state.q.μ,
    Dict(label => (; keys=u.semantic, u.M) for (label, u) in state.units))
slice_full_point(m, start::BRMSVIStart) = slice_full_point(m, start.globals.keys,
    start.globals.μ, start.units)

@stestset "slice combination: a unit's draws keep its totals under the combined coefficients" begin
    data = svi_slice_data(Xoshiro(57), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 4)
    regions = first.(SVI_SLICE_REGIONS)
    full = brm_nlme_model(RKBRMI(svi_slice_intercept_model(data)); ad_backend=SLICE_AD,
        SVI_SLICE_GROUPS...)
    @test length(full.view.mu_references) == 1        # mu_Intercept into the draw
    fits = [slice_quick(svi_slice_intercept_model(svi_slice_region(data, r))) for r in regions]
    # The regions disagree: intercepts 1, 2, 3; subject scales e^-1, e^-0.5, 1.
    for (s, fit) in enumerate(fits)
        q = fit.diagnostics.q
        names = string.(fit.diagnostics.model.view.coordinates[fit.diagnostics.globals])
        q.μ .= 0.0
        q.μ[findfirst(==("mu_Intercept"), names)] = Float64(s)
        q.μ[findfirst(n -> occursin("subject.tau", n), names)] = 0.5s - 1.5
        q.ρ .= log(0.05)
        for i in eachindex(q.M)
            q.M[i] .= 0.4i - 1.0
            q.Λ[i] .= s == 3 && i == 1 ? 0.5 : 2.0 + i      # one site wider than the prior
            q.U[i] .= sqrt.(q.Λ[i])
        end
    end
    states = brm_svi_checkpoint.(fits)
    start = slice_all_carried(full, states)
    b_c, logτ_c = slice_mean(start, "mu_Intercept"), slice_mean(start, "subject.tau")
    for (s, st) in enumerate(states), (label, u) in st.units
        b_s, logτ_s = slice_mean(st, "mu_Intercept"), slice_mean(st, "subject.tau")
        M, Λ = slice_carried_unit(only(u.M), only(u.Λ), exp(logτ_c - logτ_s),
            (b_c - b_s) / exp(logτ_s))
        @test only(start.units[label].M) ≈ M
        @test only(start.units[label].Λ) ≈ Λ
    end
    # East's first unit has a negative site (Λ = 0.5); carried, it stays
    # positive definite, so nothing is dropped.
    @test start.receipt.units.carried == 9 && isempty(start.receipt.units.dropped)
    # A negative site that the map would make indefinite is dropped instead.
    M, Λ, how = BRM._brm_svi_carry_unit([0.3], fill(0.5, 1, 1), fill(3.0, 1, 1), [0.0], [true])
    @test how === :dropped && M ≈ [0.0] && Λ ≈ fill(1.0, 1, 1)
    # A data-rich unit keeps its total intercept b + τ z exactly.
    for fit in fits, i in eachindex(fit.diagnostics.q.M)
        fit.diagnostics.q.Λ[i] .= 1e8
        fit.diagnostics.q.U[i] .= 1e4
    end
    rich = brm_svi_checkpoint.(fits)
    start = slice_all_carried(full, rich)
    b_c, τ_c = slice_mean(start, "mu_Intercept"), exp(slice_mean(start, "subject.tau"))
    for st in rich, (label, u) in st.units
        total = slice_mean(st, "mu_Intercept") + exp(slice_mean(st, "subject.tau")) * only(u.M)
        @test b_c + τ_c * only(start.units[label].M) ≈ total rtol = 1e-6
    end
end

@stestset "slice combination: correlated draws keep their totals, read back by the NLME view" begin
    data = svi_slice_data(Xoshiro(58), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 6)
    regions = first.(SVI_SLICE_REGIONS)
    full = brm_nlme_model(RKBRMI(svi_slice_slope_model(data)); ad_backend=SLICE_AD,
        subject_group=:subject)
    @test length(full.view.mu_references) == 2        # intercept and slope
    fits = [brm_svi_fit(svi_slice_slope_model(svi_slice_region(data, r)); rng=Xoshiro(s),
        max_iterations=0, ndraws=0, subject_group=:subject) for (s, r) in enumerate(regions)]
    for (s, fit) in enumerate(fits)
        q = fit.diagnostics.q
        q.μ .= [0.3s * (-1)^k + 0.1k for k in eachindex(q.μ)]
        q.ρ .= log(0.05)
        for i in eachindex(q.M)
            q.M[i] .= [0.5 - 0.2i, 0.1i]
            q.Λ[i] .= [1e8 0.0; 0.0 1e8]
            q.U[i] .= cholesky(Symmetric(q.Λ[i])).L
        end
    end
    states = brm_svi_checkpoint.(fits)
    start = slice_all_carried(full, states)
    @test start.receipt.units.carried == 9
    # The natural effects BRM's NLME view reads at a point, plus the
    # mu-referenced coefficients, at each slice's point and at the start's.
    rk, view = full.backend, full.view
    families = BRM._brm_nlme_families(rk, view)
    totals(u) = BRM._brm_nlme_natural(rk, view, families, u).eta .+
        [u[ref.population] for ref in sort(view.mu_references; by=r -> r.subject)]
    combined = totals(slice_full_point(full, start))
    for st in states
        mine = [findfirst(==(label), view.levels) for label in keys(st.units)]
        @test combined[:, mine] ≈ totals(slice_full_point(full, st))[:, mine] rtol = 1e-6
    end
    # A block stratified by kind holds one family per stratum, so the view is
    # ragged and exposes no mu-references; the intercept still adds into every
    # stratum's draws.
    data = svi_slice_kinds(data)
    @test all(r -> length(unique(data.kind[data.region .== r])) == 2, regions)
    full = brm_nlme_model(RKBRMI(svi_slice_stratified_model(data)); ad_backend=SLICE_AD,
        subject_group=:subject)
    rk, view = full.backend, full.view
    @test view.ragged && isempty(view.mu_references)
    fits = [brm_svi_fit(svi_slice_stratified_model(svi_slice_region(data, r)); rng=Xoshiro(s),
        max_iterations=0, ndraws=0, subject_group=:subject) for (s, r) in enumerate(regions)]
    for (s, fit) in enumerate(fits)
        q = fit.diagnostics.q
        q.μ .= [0.3s * (-1)^k + 0.1k for k in eachindex(q.μ)]
        # In a region with one subject of a kind, that kind's scale is read by
        # one unit, which holds it beside its draw.
        for i in eachindex(q.M)
            q.M[i] .= 0.5 - 0.2i
            n = length(q.M[i])
            q.Λ[i] .= Matrix(1e8I, n, n)
            q.U[i] .= Matrix(1e4I, n, n)
        end
    end
    @test any(fit -> any(M -> length(M) == 2, fit.diagnostics.q.M), fits)
    states = brm_svi_checkpoint.(fits)
    start = slice_all_carried(full, states)
    @test start.receipt.units.carried == 9
    families = BRM._brm_nlme_families(rk, view)
    intercept = only(findall(==(:mu_Intercept), view.coordinates))
    totals_r(u) = [only(η) + u[intercept] for η in BRM._brm_nlme_natural(rk, view, families, u).eta]
    combined = totals_r(slice_full_point(full, start))
    for st in states
        mine = [findfirst(==(label), view.levels) for label in keys(st.units)]
        @test combined[mine] ≈ totals_r(slice_full_point(full, st))[mine] rtol = 1e-6
    end
end

@stestset "slice combination: slices that disagree on a scale start better carried than copied" begin
    data = svi_slice_rate_data(Xoshiro(62), 20)
    groups = (; subject_group=:subject)
    full = brm_nlme_model(RKBRMI(svi_slice_rate_model(data)); ad_backend=SLICE_AD, groups...)
    # Each region's factors as a fit on its rows would place them: its own
    # scale (0.03 and 1.2), the shared intercept, every unit's draw at its
    # effect over that scale, tightly.
    effect(s) = (startswith(s, "c") ? 0.03 : 1.2) * sin(3 * sum(Int, codeunits(s)))
    slices = map(("calm" => 0.03, "wild" => 1.2)) do (r, τ)
        fit = brm_svi_fit(svi_slice_rate_model(svi_slice_rows(data, data.region .== r));
            rng=Xoshiro(1), max_iterations=0, ndraws=0, groups...)
        d = fit.diagnostics
        names = string.(d.model.view.coordinates[d.globals])
        d.q.μ[findfirst(==("mu_Intercept"), names)] = 1.0
        d.q.μ[findfirst(n -> occursin("subject.tau", n), names)] = log(τ)
        d.q.ρ .= log(0.05)
        for (i, label) in enumerate(d.model.view.levels)
            d.q.M[i] .= effect(label) / τ
            d.q.Λ[i] .= 1e8
            d.q.U[i] .= 1e4
        end
        brm_svi_checkpoint(fit)
    end
    start = brm_svi_combine(full, slices)
    τ_c = exp(slice_mean(start, "subject.tau"))
    @test 0.1 < τ_c < 0.4                             # between the regions' scales
    # The same start with every unit's factor copied from its slice, as before.
    copied = BRMSVIStart(start.format, start.source_id, start.preprocessing, start.group,
        start.nested, start.shared, start.globals,
        Dict{Any,NamedTuple}(label => (; keys=collect(Any, u.semantic), u.M, u.Λ)
            for st in slices for (label, u) in st.units), start.receipt)
    # Carried, every unit keeps its total 1 + effect; copied, a wild unit's
    # effect shrinks by τ_c / 1.2 and a calm one's grows by τ_c / 0.03.
    b_c = slice_mean(start, "mu_Intercept")
    for st in slices, (label, u) in st.units
        # to the shrink of a finite precision: (C²(Λ − 1) + 1)⁻¹ ≤ 1e-6 of z
        @test b_c + τ_c * only(start.units[label].M) ≈ 1 + effect(label) atol = 1e-5
        startswith(label, "w") &&
            @test abs(b_c + τ_c * only(copied.units[label].M) - 1 - effect(label)) > 0.5
    end
    first_window(start) = brm_svi_fit(full; rng=Xoshiro(81), start, max_iterations=50,
        window=50, batch_size=3, ndraws=0).diagnostics.elbo_windows_trimmed[1]
    cold, carried, copied_elbo = first_window(nothing), first_window(start), first_window(copied)
    @info "first-window trimmed ELBO" cold carried copied = copied_elbo
    @test carried > copied_elbo
    @test carried > cold
end

@stestset "slice combination: a shared level held in a unit keeps its total" begin
    data = svi_slice_devices(svi_slice_data(Xoshiro(59), SVI_SLICE_REGIONS, SVI_SLICE_CLASSES, 6))
    regions = first.(SVI_SLICE_REGIONS)
    groups = (; subject_group=:subject, shared_groups=(:device,))
    full = brm_nlme_model(RKBRMI(svi_slice_device_model(data)); ad_backend=SLICE_AD, groups...)
    device_key(key) = hasproperty(key, :level) && startswith(string(key.level), "d")
    fits = [brm_svi_fit(svi_slice_device_model(svi_slice_region(data, r)); rng=Xoshiro(s),
        max_iterations=0, ndraws=0, groups...) for (s, r) in enumerate(regions)]
    for (s, fit) in enumerate(fits)
        q = fit.diagnostics.q
        names = string.(fit.diagnostics.model.view.coordinates[fit.diagnostics.globals])
        q.μ .= 0.0
        q.μ[findfirst(==("mu_Intercept"), names)] = Float64(s)
        q.μ[findfirst(n -> occursin("device.tau", n), names)] = 0.5s - 1.5
        q.ρ .= log(0.05)
        for i in eachindex(q.M)
            n = length(q.M[i])
            q.M[i] .= 0.3i .- 0.1 .* (1:n)
            q.Λ[i] .= Matrix(1e8I, n, n)
            q.U[i] .= Matrix(1e4I, n, n)
        end
    end
    states = brm_svi_checkpoint.(fits)
    start = slice_all_carried(full, states)
    # Every unit holds its device's draw, which is no global.
    @test all(u -> count(device_key, u.keys) == 1, values(start.units))
    @test !any(device_key, start.globals.keys)
    @test start.receipt.units.carried == 9
    total(b, logτ, units, label) = b + exp(logτ) *
        only(u.M[k] for u in (units[label],) for (k, key) in enumerate(u.keys) if device_key(key))
    for st in states, label in keys(st.units)
        units = Dict(l => (; keys=u.semantic, u.M) for (l, u) in st.units)
        @test total(slice_mean(start, "mu_Intercept"), slice_mean(start, "device.tau"),
            start.units, label) ≈ total(slice_mean(st, "mu_Intercept"),
            slice_mean(st, "device.tau"), units, label) rtol = 1e-6
    end
end
