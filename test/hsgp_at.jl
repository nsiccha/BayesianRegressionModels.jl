using Test, Random
using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, LogDensityProblems, StanBlocks
using DifferentiationInterface: AutoEnzyme

const mapped = @brm begin
    mu ~ 0 + hsgp(x, z; k=(3, 4), domain=((-2.0, 2.0), (-2.0, 2.0)),
                  orthogonal_to=:intercept, at=pos)
    y ~ Normal(mu, 1.0)
end
const explicit = @brm begin
    field ~ 0 + hsgp(x, z; k=(3, 4), domain=((-2.0, 2.0), (-2.0, 2.0)),
                     orthogonal_to=:intercept)
    mu = field[pos]
    y ~ Normal(mu, 1.0)
end

const periodic_mapped = @brm begin
    mu ~ 0 + hsgp(month; k=3, cov=:periodic, period=12.0, at=month_pos)
    y ~ Normal(mu, 1.0)
end
const periodic_explicit = @brm begin
    field ~ 0 + hsgp(month; k=3, cov=:periodic, period=12.0)
    mu = field[month_pos]
    y ~ Normal(mu, 1.0)
end
const multiplied_mapped = @brm begin
    mu ~ 0 + hsgp(x, z; k=(3, 4), domain=((-2.0, 2.0), (-2.0, 2.0)),
                  orthogonal_to=:intercept, at=pos, multiply=wave)
    y ~ Normal(mu, 1.0)
end
const multiplied_explicit = @brm begin
    field ~ 0 + hsgp(x, z; k=(3, 4), domain=((-2.0, 2.0), (-2.0, 2.0)),
                     orthogonal_to=:intercept)
    mu = field[pos] .* wave
    y ~ Normal(mu, 1.0)
end

function compare_mapped(a, b, data)
    ra, rb = RKBRMI(a(data)), RKBRMI(b(data))
    @test ra.model.layout.total == rb.model.layout.total
    qa = prepare_query(ra.model, rk_bound_plan(ra), (:mu,))
    qb = prepare_query(rb.model, rk_bound_plan(rb), (:mu,))
    pa = rk_logdensity_problem(ra; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    pb = rk_logdensity_problem(rb; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(ra.model.layout.total), 0.1 .* randn(Xoshiro(7), ra.model.layout.total))
        @test qa(u) ≈ qb(u) rtol=1e-13 atol=1e-13
        la, ga = LogDensityProblems.logdensity_and_gradient(pa, u)
        lb, gb = LogDensityProblems.logdensity_and_gradient(pb, u)
        @test la ≈ lb rtol=1e-12 atol=1e-10
        @test ga ≈ gb rtol=1e-11 atol=1e-9
    end
    sa = SBBRMI(a(data); mod=@__MODULE__)
    sb = SBBRMI(b(data); mod=@__MODULE__)
    @test occursin("[pos]", StanBlocks.stan_code(sa.model)) ||
          occursin("[month_pos]", StanBlocks.stan_code(sa.model))
    basis_name = haskey(data, :x) ? :PHI_hsgp_x_z : :PHI_hsgp_month
    axis_size = haskey(data, :x) ? length(data.x) : length(data.month)
    @test size(sa.data[basis_name], 1) == axis_size
    @test size(sa.data[basis_name]) == size(sb.data[basis_name])
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sa.model);
                                 warn_pedantic=false).ok
end

@testset "HSGP position map preserves field density and gradient" begin
    data = (; x=[-1.0, -0.2, 0.4, 1.1], z=[0.5, -0.7, 0.9, -0.2],
        pos=[1, 2, 1, 4, 3, 2, 4], y=zeros(7))
    compare_mapped(mapped, explicit, data)
    bad = merge(data, (; pos=[1, 5]))
    @test_throws ErrorException RKBRMI(mapped(bad))
    @test_throws ErrorException SBBRMI(mapped(bad); mod=@__MODULE__)
end

@testset "periodic HSGP position map" begin
    data = (; month=[1.0, 4.0, 7.0, 10.0], month_pos=[1, 2, 4, 3, 1, 3], y=zeros(6))
    compare_mapped(periodic_mapped, periodic_explicit, data)
end

@testset "observation multiplier preserves field density and gradient" begin
    data = (; x=[-1.0, -0.2, 0.4, 1.1], z=[0.5, -0.7, 0.9, -0.2],
        pos=[1, 2, 1, 4, 3, 2, 4],
        wave=cos.(2pi .* [1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0] ./ 12),
        y=zeros(7))
    compare_mapped(multiplied_mapped, multiplied_explicit, data)
end

const one_formula = @brm begin
    mu ~ 1 + hsgp(x, z; k=(3, 4), domain=((-2.0, 2.0), (-2.0, 2.0)),
                   orthogonal_to=:intercept, at=pos) +
             hsgp(month; k=3, cov=:periodic, period=12.0) +
             hsgp(year; k=3) + (1 | district_row)
    y ~ Normal(mu, 1.0)
end
const split_formula = @brm begin
    place ~ 0 + hsgp(x, z; k=(3, 4), domain=((-2.0, 2.0), (-2.0, 2.0)),
                      orthogonal_to=:intercept) + (1 | district)
    when ~ 1 + hsgp(month; k=3, cov=:periodic, period=12.0) + hsgp(year; k=3)
    mu = when + place[pos]
    y ~ Normal(mu, 1.0)
end

@testset "one observation formula keeps district-field density" begin
    district = string.("d", 1:4)
    pos = [1, 2, 1, 4, 3, 2, 4]
    data = (; x=[-1.0, -0.2, 0.4, 1.1], z=[0.5, -0.7, 0.9, -0.2],
        district, pos, district_row=district[pos],
        month=[1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
        year=[-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.4], y=zeros(7))
    a, b = RKBRMI(one_formula(data)), RKBRMI(split_formula(data))
    @test a.model.layout.total == b.model.layout.total
    qa = prepare_query(a.model, rk_bound_plan(a), (:mu,))
    qb = prepare_query(b.model, rk_bound_plan(b), (:mu,))
    pa = rk_logdensity_problem(a; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    pb = rk_logdensity_problem(b; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(a.model.layout.total), 0.1 .* randn(Xoshiro(17), a.model.layout.total))
        @test qa(u) ≈ qb(u) rtol=1e-12 atol=1e-12
        la, ga = LogDensityProblems.logdensity_and_gradient(pa, u)
        lb, gb = LogDensityProblems.logdensity_and_gradient(pb, u)
        @test la ≈ lb rtol=1e-11 atol=1e-9
        @test ga ≈ gb rtol=1e-10 atol=1e-8
    end
    sb = SBBRMI(one_formula(data); mod=@__MODULE__)
    @test size(sb.data[:PHI_hsgp_x_z], 1) == 4
    @test occursin("[pos]", StanBlocks.stan_code(sb.model))
end
