using Test, Random, LinearAlgebra
using BayesianRegressionModels, Distributions, ReactiveKernels, ReactiveKernelsPPL
using Enzyme, LogDensityProblems, StanBlocks
using DifferentiationInterface: AutoEnzyme
using SpecialFunctions: besselix

const BRM = BayesianRegressionModels

const space_periodic = @brm begin
    mu ~ 0 + hsgp(east, north, month; periodic=:month, period=12.0,
                  k=(2, 3, 2), domain=((-2.0, 2.0), (-2.0, 2.0)),
                  orthogonal_to=:intercept, at=cell_pos)
    y ~ Normal(mu, 1.0)
end

@testset "separable spatial × periodic HSGP" begin
    cell_pos = [1, 2, 3, 1, 2, 3, 2]
    data = (; east=[-0.8, 0.1, 0.9], north=[0.2, -0.6, 0.7],
        month=[1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], cell_pos,
        y=zeros(length(cell_pos)))
    rk = RKBRMI(space_periodic(data))
    @test rk.model.layout.total == 27
    q = prepare_query(rk.model, rk_bound_plan(rk), (:mu,))
    p = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(rk.model.layout.total),
              0.1 .* randn(Xoshiro(7), rk.model.layout.total))
        @test length(q(u)) == length(cell_pos)
        lp, grad = LogDensityProblems.logdensity_and_gradient(p, u)
        @test isfinite(lp) && all(isfinite, grad)
        draw = ReactiveKernelsPPL.constrain(rk.model.layout, u).hsgp_east_north_month
        space = BRM._brm_hsgp_basis_state(
            (data.east, data.north), (2, 3), :exp_quad, true, nothing;
            fits=((0.0, 2.0), (0.0, 2.0)), orthogonal=:intercept)
        periodic = BRM._brm_hsgp_basis_state(
            (data.month,), (2,), :periodic, true, 12.0)
        frequency = vec(sum(space.omega2; dims=2))
        space_scale = draw.sigma *
            (draw.rho_1 * sqrt(2pi))^(size(space.omega2, 2) / 2) .*
            exp.(-0.25 .* draw.rho_1^2 .* frequency)
        period_scale = sqrt.(2 .* besselix.(periodic.harmonics,
            inv(draw.rho_2^2)))
        scales = vec(space_scale .* transpose(period_scale))
        fields = space.PHI * reshape(scales .* draw.beta_raw,
            size(space.PHI, 2), size(periodic.PHI, 2))
        expected = vec(sum(fields[cell_pos, :] .* periodic.PHI; dims=2))
        @test q(u) ≈ expected rtol=1e-12 atol=1e-12
        block = only(BRM._rk_adaptive_hsgp_blocks(rk))
        @test length(block.effects) == length(scales)
        @test [BRM._adaptive_hsgp_log_scale(u, block, b)
               for b in eachindex(scales)] ≈ log.(scales) rtol=1e-12 atol=1e-10
        Kspace = space.PHI * Diagonal(space_scale .^ 2) * space.PHI'
        Kperiod = periodic.PHI * Diagonal(period_scale .^ 2) * periodic.PHI'
        expanded = hcat([space.PHI[cell_pos, b] .* periodic.PHI[:, j]
            for j in 1:size(periodic.PHI, 2) for b in 1:size(space.PHI, 2)]...)
        Kexpanded = (expanded .* transpose(scales)) *
            transpose(expanded .* transpose(scales))
        @test Kexpanded ≈ Kspace[cell_pos, cell_pos] .* Kperiod rtol=1e-12
    end
    sb = SBBRMI(space_periodic(data); mod=@__MODULE__)
    @test size(sb.data[:PHI_space_hsgp_east_north_month]) == (3, 6)
    @test size(sb.data[:PHI_period_hsgp_east_north_month]) == (7, 4)
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model);
        warn_pedantic=false).ok
    replay_data = merge(data, (; month=data.month .+ 0.5,
        cell_pos=reverse(data.cell_pos)))
    replay = reprocess(sb, replay_data)
    @test replay.data[:PHI_space_hsgp_east_north_month] ==
        sb.data[:PHI_space_hsgp_east_north_month]
    @test replay.data[:PHI_period_hsgp_east_north_month] !=
        sb.data[:PHI_period_hsgp_east_north_month]
    @test replay.data[:cell_pos] == replay_data.cell_pos
end
