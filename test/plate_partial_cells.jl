# test/plate_partial_cells.jl — a plate whose cells list only some labels of
# the vector a `groups(g => labels)` binding reads.
#
# Run: julia --startup-file=no --project=test test/plate_partial_cells.jl
#
# Unit-year predictors are observed as district-year totals, but only for the
# first two years: the third year's unit rows stay in the predictors because
# a unit-year response observes them, and its totals are forecast. The cells
# list only the observed district-years; an entry whose label names no cell is
# read by no cell (decision `0z97g0b`, snag `a-plate-over-kre-9fcaaaf0`). The
# fit equals the same model with every district-year a cell and the unobserved
# ones at likelihood weight 0, without placeholder totals or weights. A
# response observed through a binding still needs every entry in a cell: a fit
# receives only the observations it conditions on (`0q8hcwq`).

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random: Xoshiro

const PARTIAL = let rng = Xoshiro(20261025)
    units, years, observed_years = 24, 3, 2
    district = string.("d", 1 .+ (0:units-1) .÷ 6)
    year = repeat(1:years; inner=units)
    cell(d, t) = string(d, "-", t)
    all_cells = [cell(d, t) for t in 1:years for d in unique(district)]
    n_observed = observed_years * length(unique(district))
    frame = (; unit=repeat(string.("u", 1:units), years),
        unit_district=repeat(district, years), t=Float64.(year .- 2),
        x=randn(rng, units * years), y=randn(rng, units * years),
        unit_cell=[cell(district[(r-1)%units+1], year[r]) for r in 1:units*years])
    (; frame, all_cells, n_observed,
        total_a=100 .+ 10 .* rand(rng, length(all_cells)),
        total_b=rand(rng, 5:20, length(all_cells)))
end

partial_builder = @brm begin
    sigma_y ~ Exponential(1)
    mu ~ 1 + x + (1 | unit)
    y ~ Normal(mu, sigma_y)
    log_a ~ 1 + t + x + (1 | dk | unit_district)
    log_b ~ 1 + t + (1 | dk | unit_district)
    cv_a ~ LogNormal(-2.5, 1)
    cv_b ~ LogNormal(-2.5, 1)
    rows = groups(unit_cell => cell)
    @plate for k in eachindex(total_a)
        m = sum(exp.(log_a[rows[k]]))
        total_a[k] ~ Normal(m, cv_a * m)
        total_b[k] ~ NegativeBinomial2(sum(exp.(log_b[rows[k]])), 1.0 / cv_b^2)
    end
end

# Every district-year a cell; the unobserved ones at likelihood weight 0.
weighted_builder = @brm begin
    sigma_y ~ Exponential(1)
    mu ~ 1 + x + (1 | unit)
    y ~ Normal(mu, sigma_y)
    log_a ~ 1 + t + x + (1 | dk | unit_district)
    log_b ~ 1 + t + (1 | dk | unit_district)
    cv_a ~ LogNormal(-2.5, 1)
    cv_b ~ LogNormal(-2.5, 1)
    rows = groups(unit_cell => cell)
    @plate for k in eachindex(total_a)
        m = sum(exp.(log_a[rows[k]]))
        total_a[k] ~ weighted(Normal(m, cv_a * m), weights(w[k]))
        total_b[k] ~ weighted(NegativeBinomial2(sum(exp.(log_b[rows[k]])), 1.0 / cv_b^2),
            weights(w[k]))
    end
end

const OBSERVED = 1:PARTIAL.n_observed
partial_data = merge(PARTIAL.frame, (; cell=PARTIAL.all_cells[OBSERVED],
    total_a=PARTIAL.total_a[OBSERVED], total_b=PARTIAL.total_b[OBSERVED]))
weighted_data = merge(PARTIAL.frame, (; cell=PARTIAL.all_cells,
    total_a=PARTIAL.total_a, total_b=PARTIAL.total_b,
    w=Float64.(eachindex(PARTIAL.all_cells) .<= PARTIAL.n_observed)))
points(n) = (zeros(n), collect(range(-0.4, 0.4; length=n)), 0.3 .* randn(Xoshiro(2), n))

@testset "groups(g => labels) leaves an unlisted entry in no group" begin
    rows = groups(PARTIAL.frame.unit_cell => PARTIAL.all_cells[OBSERVED])
    @test length(rows) == PARTIAL.n_observed
    # The first two years' rows, each in its district-year; the third's in none.
    @test sort!(reduce(vcat, rows)) == findall(<=(0), PARTIAL.frame.t)
end

@testset "SBBRMI: the subset plate is the weight-0 plate" begin
    sb = SBBRMI(partial_builder(partial_data); mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    partial = consumer_stan(partial_builder(partial_data), "plate_partial_cells")
    weighted = consumer_stan(weighted_builder(weighted_data), "plate_partial_cells_weighted")
    @test BridgeStan.param_unc_names(partial.model) ==
          BridgeStan.param_unc_names(weighted.model)
    for u in points(BridgeStan.param_unc_num(partial.model))
        a, ga = BridgeStan.log_density_gradient(partial.model, u; propto=false, jacobian=true)
        b, gb = BridgeStan.log_density_gradient(weighted.model, u; propto=false, jacobian=true)
        @test isfinite(a)
        @test a ≈ b atol=1e-10 rtol=1e-12
        @test ga ≈ gb atol=1e-10 rtol=1e-12
    end
end

@testset "RKBRMI: the subset plate is the weight-0 plate" begin
    partial_backend, partial = consumer_problem(partial_builder(partial_data))
    weighted_backend, weighted = consumer_problem(weighted_builder(weighted_data))
    @test coordinate_names(partial_backend.model.layout) ==
          coordinate_names(weighted_backend.model.layout)
    for u in points(LogDensityProblems.dimension(partial))
        a, ga = LogDensityProblems.logdensity_and_gradient(partial, u)
        b, gb = LogDensityProblems.logdensity_and_gradient(weighted, u)
        @test isfinite(a)
        @test a ≈ b atol=1e-10 rtol=1e-12
        @test ga ≈ gb atol=1e-10 rtol=1e-12
    end
end

@testset "the fit replays onto every cell as a forecast" begin
    sb = SBBRMI(partial_builder(partial_data); mod=@__MODULE__, total_groups=())
    forecast = reprocess(sb, merge(PARTIAL.frame, (; cell=PARTIAL.all_cells)))
    @test forecast.held_out == Set([:total_a, :total_b])
    @test StanBlocks.stanc_check(BRM.stan_code(forecast); warn_pedantic=false).ok
    d = brm_descriptor(forecast)
    for response in (:total_a, :total_b)
        @test brm_output(d, response; role=:posterior_predictive).dims ==
              (length(PARTIAL.all_cells),)
    end
    # Replayed onto the fitted cells it is the fit's own program.
    @test BRM.stan_code(reprocess(sb, partial_data)) == BRM.stan_code(sb)
end

response_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | unit)
    rows = groups(unit_cell => cell)
    @plate for k in eachindex(shift)
        y[rows[k]] ~ normal(mu[rows[k]] .+ shift[k], sigma)
    end
end

@testset "a response observed through a binding needs every entry in a cell" begin
    all_cells = merge(PARTIAL.frame, (; cell=PARTIAL.all_cells,
        shift=zeros(length(PARTIAL.all_cells))))
    some_cells = merge(PARTIAL.frame, (; cell=PARTIAL.all_cells[OBSERVED],
        shift=zeros(PARTIAL.n_observed)))
    sb = SBBRMI(response_builder(all_cells); mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    _, problem = consumer_problem(response_builder(all_cells))
    @test isfinite(LogDensityProblems.logdensity(problem,
        zeros(LogDensityProblems.dimension(problem))))
    # refused: the third year's `y` would be carried into the fit without
    # conditioning on it, the per-row mask `1bwqnxs` removed (`0q8hcwq`).
    @test_throws "belong to no cell" SBBRMI(response_builder(some_cells);
        mod=@__MODULE__, total_groups=())
    @test_throws "belong to no cell" RKBRMI(response_builder(some_cells))
    # A replay onto rows with uncovered responses is refused the same way.
    @test_throws "belong to no cell" reprocess(sb, some_cells)
end
