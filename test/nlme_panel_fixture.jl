# test/nlme_panel_fixture.jl — a regional panel for the NLME unit tests
# (test/nlme_units_panel.jl, test/nlme_units_panel_fit.jl): regions (the units)
# ⊃ districts ⊃ municipalities and district-years; a completed response with a
# size-stratified municipality effect, and a quantity observed only as
# district-year totals.

using Random

function panel_data(; seed=1, nregion=3, ndistrict=2, nmuni=4, nyear=3, pmissing=0.25)
    rng = Xoshiro(seed)
    region, district, muni, sclass, district_year, year = Int[], Int[], Int[], Int[], Int[], Float64[]
    for r in 1:nregion, d in 1:ndistrict, m in 1:nmuni, y in 1:nyear
        k = (r - 1) * ndistrict + d
        push!(region, r); push!(district, k); push!(muni, (k - 1) * nmuni + m)
        push!(sclass, 1 + m % 2); push!(district_year, (k - 1) * nyear + y); push!(year, y - 2.0)
    end
    n = length(region)
    z = randn(rng, n)
    x1 = Vector{Union{Missing,Float64}}(exp.(4.0 .+ 0.3 .* z .+ 0.2 .* randn(rng, n)))
    x1[rand(rng, n) .< pmissing] .= missing
    cells = sort!(unique(district_year))
    total = [sum(exp.(2.0 .+ 0.5 .* log.(coalesce.(x1[district_year .== c], 55.0))))
        for c in cells] .* exp.(0.02 .* randn(rng, length(cells)))
    (; region, district, muni, sclass, district_year, year, z, x1, c0=4.0,
        dy_obs=cells, total)
end

panel(data) = @brm data begin
    sigma_x ~ Exponential(0.2)
    mu_x ~ 1 + z + (1 + z || region) + (1 | d | district) + (1 | gr(muni, by = sclass))
    sd(mu_x, region) ~ truncated(Normal(0, 0.5); lower = 0)
    sd(mu_x, muni) ~ LogNormal(-1, 0.75)
    mi(x1) ~ LogNormal(mu_x, sigma_x)
    c_x = log(x1) - c0
    log_w ~ 1 + c_x + year + (1 + c_x || region) + (1 | d | district) +
        (1 | m | muni) + (1 | dy | district_year)
    sd(log_w, region) ~ truncated(Normal(0, 0.5); lower = 0)
    sd(:, m) ~ truncated(Normal(0, 0.5); lower = 0)
    sd(:, dy) ~ LogNormal(-3, 1)
    sd(:, d) ~ LogNormal(-1.5, 0.75)
    sigma_t ~ Exponential(0.03)
    rows_t = groups(district_year => dy_obs)
    @plate for k in eachindex(total)
        total[k] ~ Normal(sum(exp.(log_w[rows_t[k]])), sigma_t * total[k])
    end
end

const PANEL = panel_data()
const PANEL_NESTED = (district=:region, muni=:region, district_year=:region)
