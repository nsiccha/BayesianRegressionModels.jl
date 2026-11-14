# A non-connected SVI target whose units own `@plate` cell latents evaluates its
# units on a partition that declares that ownership to RKPPL (snag
# brm-svi-fit-cost-568aea45). The NLME view leaves cell latents shared, and
# RKPPL computes shared values into one prefix that every unit's call reuses
# while its inputs stay put. With a unit's own latents among those inputs, each
# of its draws invalidated the prefix: every unit call recomputed and pulled
# back a prefix over all units, and an iteration's cost grew with the number of
# units. The model is the reporter's TOY3 (nlme_svi_events_fixture.jl).
# RUN: julia --project=test test/nlme_svi_cell_latent_partition.jl
using Test
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using ADTypes: AutoEnzyme
import Enzyme, Random
include("testset_filter.jl")
include("nlme_svi_events_fixture.jl")
const BRM = BayesianRegressionModels

@stestset "a unit's own cell latents stay out of the partition's shared prefix" begin
    rk = RKBRMI(TOY3(toy3_data(; S=6, T=40, seed=3)))
    m = brm_nlme_model(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse), subject_group=:unit)
    t = BRM._brm_svi_target(m)
    names = string.(m.view.coordinates)
    latent(j) = any(prefix -> startswith(names[j], prefix), ("z_gain.", "z_offset.", "z_sigma."))
    latents = filter(latent, eachindex(names))
    @test !isempty(latents)
    # The view leaves the cell latents shared; SVI gives each to the unit whose cells read it.
    @test all(j -> m.view.subject[j] == 0, latents)
    @test all(j -> t.ownership.owner[j] > 0, latents)
    # The model's own partition, as the target used before 0.2.299.
    model_plan = BRM._brm_nlme_partition_plan(m.cell_gradient, m.unit, t.globals, m.view)
    point = m.unit .+ 0.1 .* sin.(eachindex(m.unit))
    for i in eachindex(t.units)
        footprint = t.plan.footprints[i]
        # Every coordinate of the unit that its cells read is local to its partition,
        # so none is an input of the prefix all units' calls share.
        @test Set(t.plan.local_footprints[i]) == Set(intersect(footprint, t.units[i]))
        @test any(latent, t.plan.local_footprints[i])
        # The positive control: the model's partition keeps them shared.
        @test !any(latent, model_plan.local_footprints[i])
        # The same reads, density and gradient.
        @test footprint == model_plan.footprints[i]
        a, b = zeros(length(footprint)), zeros(length(footprint))
        va = first(BRM._rk_nlme_partition_value_and_gradient!(t.plan.sampler, a, point, i))
        vb = first(BRM._rk_nlme_partition_value_and_gradient!(model_plan.sampler, b, point, i))
        @test isfinite(va)
        @test va ≈ vb rtol=1e-12
        @test a ≈ b rtol=1e-10
    end
    # A model whose units own only their effects keeps the model's partition.
    plain = BRM._brm_svi_view_partition(m, merge(t.ownership, (; owner=m.view.subject)))
    @test first(plain) === m.cell_gradient
end
