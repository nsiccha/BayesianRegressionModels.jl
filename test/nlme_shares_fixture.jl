# test/nlme_shares_fixture.jl — a regional panel whose suppressed cells have
# known district-year sums, `mi(x)::shares(total, rows)`, for
# test/nlme_shares_units.jl and test/nlme_shares_units_fit.jl: regions (the
# units) ⊃ districts ⊃ district-years ⊃ municipality-years, and a law that
# reads the regions' and districts' random effects.

using Random

# District-years of four municipality-years; two suppressed cells in every
# other district-year, three in district-year 3. `free = true` also suppresses
# a cell of a district-year with no known total, a free entry.
function shares_panel(; seed=3, nregion=3, ndistrict=2, nmuni=4, nyear=2, free=false)
    rng = Xoshiro(seed)
    region, district, dy, z = Int[], Int[], Int[], Float64[]
    for r in 1:nregion, d in 1:ndistrict, y in 1:nyear, m in 1:nmuni
        k = (r - 1) * ndistrict + d
        push!(region, r); push!(district, k); push!(dy, (k - 1) * nyear + y)
        push!(z, randn(rng))
    end
    n = length(region)
    truth = exp.(3.0 .+ 0.4 .* z .+ 0.3 .* randn(rng, n))
    x = Vector{Union{Missing,Float64}}(truth)
    for c in unique(dy)
        rows = findall(==(c), dy)
        isodd(c) && (x[rows[1:2]] .= missing)
        c == 3 && (x[rows[3]] = missing)
    end
    cells = sort(unique(dy[ismissing.(x)]))
    free && (x[findfirst(==(2), dy)] = missing)
    tot = [sum(truth[dy .== c]) for c in cells]
    w = 0.5 .* log.(truth) .+ 0.1 .* randn(rng, n)
    (; region, district, dy, z, x, cells, tot, w)
end

shares_model(data) = @brm data begin
    sigma ~ Exponential(0.5)
    mu ~ 1 + z + (1 | region) + (1 | district)
    rows = groups(dy => cells)
    mi(x)::shares(tot, rows) ~ LogNormal(mu, sigma)
    lx = log(x)
    m ~ 1 + lx + (1 | region)
    w ~ Normal(m, 0.5)
end

const SHARES_NESTED = (; subject_group=:region, nested_groups=(district=:region,))
