# test/plate_mvnormal_observation.jl — a plate cell observing its slice
# jointly, `y[rows[i]] ~ MvNormal(location, covariance)`, with the covariance
# computed in the cell.
#
# Run: julia --startup-file=no --project=test test/plate_mvnormal_observation.jl
#
# The model is an aggregate benchmark: units nested in districts, one cell per
# sector. Each cell sums its units' means and parameter-dependent variances to
# district totals, observes the sector's grand total, and observes the sector's
# observed district totals jointly given that total (snag
# `brm-use-declares-abc94afd`). SBBRMI lowers the in-cell `MvNormal` to
# `brm_mvnormal`: Stan's `multi_normal`, with Julia's log density 0 for an
# empty slice, which Stan's `multi_normal` refuses. `Diagonal(v)` and
# `diagm(v)` lower to `diag_matrix(v)`. The density must equal a plain-Julia
# joint `MvNormal` over (observed district totals, grand total) whose
# covariance is built entry by entry. RKBRMI evaluates the cell as written and
# must give the same density and gradient.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra, Random

# Units nested in districts, several sectors. `n1`, `n2`, `x` are unit
# features on one row per (unit, sector), sector-major; `A` is the districts ×
# units incidence matrix; `y` holds the observed district totals, one row per
# (district, sector) cell that is not suppressed; `land` the sector totals.
# With `empty_sector`, sector 3 has no observed district total.
function aggregate_data(; U=30, D=4, S=3, seed=1, empty_sector=true)
    rng = Xoshiro(seed)
    unit_district = vcat(collect(1:D), rand(rng, 1:D, U - D))
    cyc(v) = repeat(v, cld(S, 3))[1:S]
    rho = cyc([-0.4, -0.6, -0.5]); delta = 0.3; w = cyc([0.2, 0.0, 0.5])
    tau = cyc([0.4, 0.3, 0.5]); g = cyc([1.02, 0.98, 1.0]); sigma_reg = 0.05
    n1 = rand(rng, 0:20, U, S) .* 1.0; n2 = rand(rng, 0:5, U, S) .* 1.0; x = rand(rng, U) .* 10
    mu = n1 .* exp.(rho') .+ n2 .* exp.(rho' .+ delta) .+ x .* w'
    unit_total = mu .* exp.(tau' .* randn(rng, U, S) .- tau' .^ 2 ./ 2)
    A = [unit_district[u] == d ? 1.0 : 0.0 for d in 1:D, u in 1:U]
    T = A * unit_total
    land = vec(sum(unit_total; dims=1)) .* (1 .+ 0.002 .* randn(rng, S))
    observed = [(d, s) for s in 1:S for d in 1:D
        if !(empty_sector && s == 3) && !(s == 3 && d == 2) && !(s == 1 && d == 3) && !(s == 2 && d == 1)]
    row_unit = repeat(1:U; outer=S)
    (; n_sector=S, sector=collect(1:S), land,
       row_sector=repeat(1:S; inner=U), row_unit, n1=vec(n1), n2=vec(n2), x=x[row_unit], A,
       cell_sector=[s for (d, s) in observed], cell_district=[d for (d, s) in observed],
       y=[g[s] * T[d, s] * (1 + sigma_reg * randn(rng)) for (d, s) in observed])
end

# The same data with the unit-level sums precomputed per (sector, district),
# one flat row each, district order within a sector: `F_j` sums feature j,
# `G_jk` sums feature j times feature k.
function district_sums(data)
    S, D = data.n_sector, size(data.A, 1)
    F = zeros(S * D, 3); G = zeros(S * D, 3, 3)
    for (r, s) in enumerate(data.row_sector)
        d = findfirst(==(1.0), data.A[:, data.row_unit[r]])
        f = [data.n1[r], data.n2[r], data.x[r]]
        F[(s - 1) * D + d, :] .+= f
        G[(s - 1) * D + d, :, :] .+= f * f'
    end
    (; data..., ds_sector=repeat(1:S; inner=D), F1=F[:, 1], F2=F[:, 2], F3=F[:, 3],
       G11=G[:, 1, 1], G12=G[:, 1, 2], G13=G[:, 1, 3], G22=G[:, 2, 2], G23=G[:, 2, 3], G33=G[:, 3, 3])
end

# The priors every variant shares: one coefficient per sector through
# cell-means formula lines, read in the cell as `rho[s]`.
const AGGREGATE_PRIORS = quote
    rho ~ 0 + factor(sector)
    effect(rho, sector) ~ Normal(-0.5, 1.0)
    delta ~ Normal(0.0, 0.5)
    w ~ 0 + factor(sector)
    effect(w, sector) ~ Normal(0.0, 0.5)
    log_tau ~ 0 + factor(sector)
    effect(log_tau, sector) ~ Normal(log(0.4), 0.5)
    log_g ~ 0 + factor(sector)
    effect(log_g, sector) ~ Normal(0.0, 0.1)
    sigma_reg ~ restricted(Normal(0.0, 0.1); lower=0.0)
end

# The cell's observations, given district totals `M` and variances `V`: the
# joint law of (observed district totals, grand total) factors exactly into
# the total's marginal and the observed totals' conditional given it.
aggregate_observations(diagonal) = quote
    g = exp(log_g[s])
    s_ll = sum(V) + (0.002 * land[s])^2
    Mc = M[cell_district[cells[s]]]
    Vc = V[cell_district[cells[s]]]
    land[s] ~ Normal(sum(M), sqrt(s_ll))
    cmean = g .* Mc .+ g .* Vc .* (land[s] - sum(M)) ./ s_ll
    y[cells[s]] ~ MvNormal(cmean, $diagonal(g^2 .* Vc .+ (sigma_reg .* y[cells[s]]) .^ 2) .- (g .* Vc) * (g .* Vc)' ./ s_ll)
end

# District totals summed from the units in the cell.
unit_level_builder(diagonal) = @eval @brm begin
    $(AGGREGATE_PRIORS.args...)
    units = groups(row_sector => sector)
    cells = groups(cell_sector => sector)
    @plate for s in eachindex(land)
        mu = n1[units[s]] .* exp(rho[s]) .+ n2[units[s]] .* exp(rho[s] + delta) .+ x[units[s]] .* w[s]
        v = mu .^ 2 .* (exp(exp(2 * log_tau[s])) - 1)
        M = A * mu
        V = A * v
        $(aggregate_observations(diagonal).args...)
    end
end
const UNIT_LEVEL = unit_level_builder(:Diagonal)
const UNIT_LEVEL_DIAGM = unit_level_builder(:diagm)

# The same law with the covariance bound to a cell local first.
const UNIT_LEVEL_LOCAL = @eval @brm begin
    $(AGGREGATE_PRIORS.args...)
    units = groups(row_sector => sector)
    cells = groups(cell_sector => sector)
    @plate for s in eachindex(land)
        mu = n1[units[s]] .* exp(rho[s]) .+ n2[units[s]] .* exp(rho[s] + delta) .+ x[units[s]] .* w[s]
        v = mu .^ 2 .* (exp(exp(2 * log_tau[s])) - 1)
        M = A * mu
        V = A * v
        g = exp(log_g[s])
        s_ll = sum(V) + (0.002 * land[s])^2
        Mc = M[cell_district[cells[s]]]
        Vc = V[cell_district[cells[s]]]
        land[s] ~ Normal(sum(M), sqrt(s_ll))
        cmean = g .* Mc .+ g .* Vc .* (land[s] - sum(M)) ./ s_ll
        ccov = Diagonal(g^2 .* Vc .+ (sigma_reg .* y[cells[s]]) .^ 2) .- (g .* Vc) * (g .* Vc)' ./ s_ll
        y[cells[s]] ~ MvNormal(cmean, ccov)
    end
end

# District totals from sums precomputed per district: the same law, without a
# pass over the units per evaluation.
const DISTRICT_LEVEL = @eval @brm begin
    $(AGGREGATE_PRIORS.args...)
    districts = groups(ds_sector => sector)
    cells = groups(cell_sector => sector)
    @plate for s in eachindex(land)
        t1 = exp(rho[s])
        t2 = exp(rho[s] + delta)
        t3 = w[s]
        M = F1[districts[s]] .* t1 .+ F2[districts[s]] .* t2 .+ F3[districts[s]] .* t3
        Q = G11[districts[s]] .* t1^2 .+ G22[districts[s]] .* t2^2 .+ G33[districts[s]] .* t3^2 .+
            2 .* (G12[districts[s]] .* (t1 * t2) .+ G13[districts[s]] .* (t1 * t3) .+ G23[districts[s]] .* (t2 * t3))
        V = Q .* (exp(exp(2 * log_tau[s])) - 1)
        $(aggregate_observations(:Diagonal).args...)
    end
end

# Priors, then per sector one joint MvNormal over (observed district totals,
# grand total), at Stan's unconstrained coordinates (cell-means coefficients in
# sector order, `sigma_reg` on the log scale).
function aggregate_reference(data, u)
    S = data.n_sector
    rho = u[1:S]; delta = u[S+1]; w = u[S+2:2S+1]; log_tau = u[2S+2:3S+1]; log_g = u[3S+2:4S+1]
    sigma_reg = exp(u[4S+2])
    lp = sum(logpdf.(Normal(-0.5, 1.0), rho)) + logpdf(Normal(0, 0.5), delta) +
         sum(logpdf.(Normal(0, 0.5), w)) + sum(logpdf.(Normal(log(0.4), 0.5), log_tau)) +
         sum(logpdf.(Normal(0, 0.1), log_g)) + logpdf(Normal(0, 0.1), sigma_reg) + u[4S+2]
    for s in 1:S
        r = data.row_sector .== s
        mu = data.n1[r] .* exp(rho[s]) .+ data.n2[r] .* exp(rho[s] + delta) .+ data.x[r] .* w[s]
        v = mu .^ 2 .* (exp(exp(2 * log_tau[s])) - 1)
        M = data.A * mu; V = data.A * v; g = exp(log_g[s])
        c = findall(==(s), data.cell_sector); d = data.cell_district[c]; yc = data.y[c]
        n = length(c); L = data.land[s]
        Σ = zeros(n + 1, n + 1)
        for j in 1:n
            Σ[j, j] = g^2 * V[d[j]] + (sigma_reg * yc[j])^2
            Σ[j, n + 1] = Σ[n + 1, j] = g * V[d[j]]
        end
        Σ[n + 1, n + 1] = sum(V) + (0.002 * L)^2
        lp += logpdf(MvNormal(vcat(g .* M[d], sum(M)), Symmetric(Σ)), vcat(yc, L))
    end
    lp
end

const AGGREGATE_NAMES = vcat(
    ["cat_rho_sector_beta.$s" for s in 1:3], ["delta"],
    ["cat_w_sector_beta.$s" for s in 1:3], ["cat_log_tau_sector_beta.$s" for s in 1:3],
    ["cat_log_g_sector_beta.$s" for s in 1:3], ["sigma_reg"])

@stestset "the in-cell MvNormal density is the joint reference" begin
    for empty in (false, true)
        data = aggregate_data(; empty_sector=empty)
        sb = SBBRMI(UNIT_LEVEL(data); mod=@__MODULE__)
        code = BRM.stan_code(sb)
        @test occursin("brm_mvnormal", code)
        @test occursin("diag_matrix", code)
        stan = consumer_stan(UNIT_LEVEL(data), "plate_mvnormal_unit_$(empty)")
        @test BridgeStan.param_unc_names(stan.model) == AGGREGATE_NAMES
        n = length(AGGREGATE_NAMES)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            value, gradient = BridgeStan.log_density_gradient(stan.model, u;
                propto=false, jacobian=true)
            @test value ≈ aggregate_reference(data, u) atol=1e-10 rtol=1e-12
            step = 1e-5
            independent = map(eachindex(u)) do j
                plus, minus = copy(u), copy(u)
                plus[j] += step; minus[j] -= step
                (aggregate_reference(data, plus) - aggregate_reference(data, minus)) / (2step)
            end
            @test gradient ≈ independent atol=1e-5 rtol=1e-6
        end
        # One aggregate pointwise log likelihood per sector; the empty slice adds 0.
        names = BridgeStan.param_names(stan.model; include_tp=false, include_gq=true)
        draws = BridgeStan.param_constrain(stan.model, zeros(n); include_tp=false,
            include_gq=true, rng=BridgeStan.StanRNG(stan.model, 1))
        pointwise = [v for (nm, v) in zip(names, draws) if occursin("_y_ragged_likelihood", nm)]
        predictive = [v for (nm, v) in zip(names, draws) if occursin("_y_ragged_gen", nm)]
        @test length(pointwise) == 3
        @test length(predictive) == length(data.y)
        @test all(isfinite, predictive)
        @test (pointwise[3] == 0.0) == empty
    end
end

@stestset "Diagonal and diagm both spell diag_matrix" begin
    data = aggregate_data()
    diagonal = BRM.stan_code(SBBRMI(UNIT_LEVEL(data); mod=@__MODULE__))
    @test BRM.stan_code(SBBRMI(UNIT_LEVEL_DIAGM(data); mod=@__MODULE__)) == diagonal
end

@stestset "a covariance bound to a cell local gives the same density" begin
    for empty in (false, true)
        data = aggregate_data(; empty_sector=empty)
        stan = consumer_stan(UNIT_LEVEL_LOCAL(data), "plate_mvnormal_local_$(empty)")
        @test BridgeStan.param_unc_names(stan.model) == AGGREGATE_NAMES
        n = length(AGGREGATE_NAMES)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            @test BridgeStan.log_density(stan.model, u; propto=false, jacobian=true) ≈
                aggregate_reference(data, u) atol=1e-10 rtol=1e-12
        end
    end
end

@stestset "unit-level and district-sum spellings give one density" begin
    data = district_sums(aggregate_data())
    unit = consumer_stan(UNIT_LEVEL(data), "plate_mvnormal_unit_sums")
    district = consumer_stan(DISTRICT_LEVEL(data), "plate_mvnormal_district_sums")
    @test BridgeStan.param_unc_names(district.model) == AGGREGATE_NAMES
    n = length(AGGREGATE_NAMES)
    for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
        a = BridgeStan.log_density(unit.model, u; propto=false)
        b = BridgeStan.log_density(district.model, u; propto=false)
        @test b ≈ a atol=1e-10 rtol=1e-12
        @test b ≈ aggregate_reference(data, u) atol=1e-10 rtol=1e-12
    end
end

# RK's coordinates of the same model, by Stan's name.
const RK_TO_STAN = vcat([:delta => "delta", :sigma_reg => "sigma_reg"],
    [Symbol("$(p)_sector.$s") => "cat_$(p)_sector_beta.$s"
     for p in (:rho, :w, :log_tau, :log_g) for s in 1:3])

@stestset "RKBRMI: the same model gives the same joint density" begin
    # One declaration, one law on both backends, an empty slice included. The
    # Stan program is checked against the plain-Julia reference above; RK's
    # gradient is native Enzyme.
    for empty in (false, true)
        data = aggregate_data(; empty_sector=empty)
        backend, problem = consumer_problem(UNIT_LEVEL(data))
        stan = consumer_stan(UNIT_LEVEL(data), "plate_mvnormal_rk_$(empty)")
        n = length(AGGREGATE_NAMES)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            check_consumer_stan(problem, stan, RK_TO_STAN, backend, u)
        end
    end
end
