# test/nlme_offaxis_fixture.jl — a regional panel with a fragment on rows of
# its own, for test/nlme_offaxis_observations.jl and
# test/nlme_offaxis_observations_fit.jl: regions (the units) ⊃ districts ⊃
# municipalities. The panel's rows are municipality-years with a Normal
# response. The fragment counts something else per municipality-year over
# more years: its published counts read its log expectation through an index
# (`exp(log_f[f_obs] + …)`), and its suppressed ones are known only as
# district-year residuals, a plate over its own groups. The published counts'
# expectation also reads the panel's predictor on the municipality's row, so
# they read the region's random effects through two blocks. A count over rows
# of its own reads no random effect at all.

using Random

# `span = true` hides one more municipality-year in a district-year of the
# region before it, so that residual's cell reads rows of two regions.
function offaxis_panel(; seed=4, nregion=5, ndistrict=2, nmuni=3, nyear=2, fyears=3,
        span=false)
    rng = Xoshiro(seed)
    region, district, muni, z = Int[], Int[], Int[], Float64[]
    for r in 1:nregion, d in 1:ndistrict, m in 1:nmuni, y in 1:nyear
        k = (r - 1) * ndistrict + d
        push!(region, r); push!(district, k); push!(muni, (k - 1) * nmuni + m)
        push!(z, randn(rng))
    end
    n = length(region)
    region_effect, district_effect = 0.5 .* randn(rng, nregion), 0.3 .* randn(rng, maximum(district))
    mu = 1.0 .+ 0.5 .* z .+ region_effect[region] .+ district_effect[district]
    y = mu .+ 0.3 .* randn(rng, n)
    # The fragment's municipality-years, the panel's row of each municipality
    # (its first year), and which ones are suppressed: in every other
    # district-year two, in district-year 3 three.
    rows = [(m, y) for m in 1:maximum(muni) for y in 1:fyears]
    main = [findfirst(==(m), muni) for (m, _) in rows]
    dy = [(district[main[q]] - 1) * fyears + y for (q, (_, y)) in enumerate(rows)]
    hidden = falses(length(rows))
    for c in unique(dy)
        qs = findall(==(c), dy)
        isodd(c) && (hidden[qs[1:2]] .= true)
        c == 3 && (hidden[qs[3]] = true)
    end
    gem_effect = 0.3 .* randn(rng, maximum(muni))
    log_lam = [log(4.0) + 0.4 * mu[main[q]] + gem_effect[m] for (q, (m, _)) in enumerate(rows)]
    counts = [rand(rng, Poisson(exp(l))) for l in log_lam]
    label = [hidden[q] ? "dy$(dy[q])" : "" for q in eachindex(rows)]
    if span
        # A published municipality-year of region 2 joins district-year 1's
        # residual (region 1).
        q = findfirst(q -> !hidden[q] && region[main[q]] == 2, eachindex(rows))
        hidden[q] = true; label[q] = "dy1"
    end
    obs, hid = findall(!, hidden), findall(hidden)
    r = [obs; hid]
    cells = sort(unique(label[hid]))
    f_hid = [fill("", length(obs)); label[hid]]
    (; region, district, z, y,
       f_muni=[m for (m, _) in rows][r], f_region=region[main][r],
       f_obs=collect(eachindex(obs)), f_obs_main=main[obs], f_n=counts[obs], f_hid, f_ky=cells,
       f_resid=[sum(counts[hid][label[hid] .== c]) for c in cells],
       c_n=rand(rng, Poisson(3.0), 4))
end

offaxis_model(data) = @brm data begin
    sigma ~ Exponential(0.5)
    mu ~ 1 + z + (1 | region) + (1 | district)
    y ~ Normal(mu, sigma)
    log_f ~ 1 + (1 | fg | f_muni)
    f_off = 0.4 * mu[f_obs_main]
    f_mu = exp(log_f[f_obs] + f_off)
    f_n ~ Poisson(f_mu)
    f_h = groups(f_hid => f_ky)
    @plate for kk in eachindex(f_resid)
        lam = exp.(log_f[f_h[kk]])
        f_resid[kk] ~ Poisson(sum(lam))
    end
    c_log ~ Normal(1.0, 0.5)
    c_n ~ Poisson(exp(c_log))
end

offaxis_groups(data) =
    (; subject_group=:region, nested_groups=(district=:region, f_muni=data.f_region))

# Each fragment density's region, from the data: a published count's row, and
# the rows of a residual's district-year.
function offaxis_owners(data)
    published = data.f_region[data.f_obs]
    residual = map(data.f_ky) do c
        owners = unique(data.f_region[data.f_hid .== c])
        length(owners) == 1 ? only(owners) : owners
    end
    (; f_n=published, f_resid=residual)
end
