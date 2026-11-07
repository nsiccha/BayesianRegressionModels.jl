using Test, BayesianRegressionModels
const BRM = BayesianRegressionModels

# Independent fixture inventory: a shared hyperparent, one scientific effect
# and one ordinary coefficient per subject. Each hub edge reads the leaf's
# coefficient. The hub's unused coefficient is a prior-only global.
function regroup_hub(n)
    supports = [[1, 2, i+1, n+i+1] for i in 2:n]
    view = (; subject_levels=["s$i" for i in 1:n],
        coordinate_subject=vcat(0, collect(1:n), zeros(Int, n)),
        factor_supports=Dict(:y => supports))
    factors = (; owned=[[j] for j in 1:2n+1],
        support=vcat([[j] for j in 1:n+1], [[1, j] for j in n+2:2n+1]))
    estimated = trues(2n+1)
    axes = [[j] for j in 1:2n+1]
    inventory = BRM._brm_svi_factor_inventory(view, 2n+1)
    evaluate = promoted -> BRM._brm_svi_regroup_layout(view, inventory,
        factors, estimated, axes, promoted)
    (; view, factors, estimated, axes, inventory, evaluate)
end

@testset "separator regrouping closes authored ownership" begin
    n = 13
    fixture = regroup_hub(n)
    before = deepcopy(fixture.view)
    connected = fixture.evaluate(Int[])
    @test length(connected.units) == 1
    @test connected.globals == [1, n+2]
    split = fixture.evaluate([2])
    @test split.globals == [1, 2, n+2]
    @test split.units == vcat([Int[]], [[i+1, n+i+1] for i in 2:n])
    @test split.levels == [("s$i",) for i in 1:n]
    @test only(split.attribution).second == collect(2:n)
    @test fixture.view == before
    @test all(i -> all(j -> split.owner[j] in (0, i), split.footprints[i]), 1:n)
    # Shared hyperparents do not force their independently owned children
    # global. Every authored factor belongs to at most one local owner.
    @test all(reads -> length(unique(filter(!iszero, split.owner[reads]))) <= 1,
        fixture.factors.support)
    @test all(>(0), split.work[2:end])
    incidence = zeros(Int, 2n+1)
    for reads in fixture.inventory.supports, j in reads
        incidence[j] += 1
    end
    automatic = BRM._brm_svi_separator_plan(fixture.evaluate, fixture.axes, incidence;
        max_unit_coordinates=2, max_promoted_coordinates=1)
    @test automatic.layout.units == split.units
    @test automatic.policy.status === :satisfied
    # A coupled transform forces both scientific coordinates global, so a
    # one-coordinate automatic budget must refuse that otherwise useful cut.
    axes = vcat([[2, 3]], fixture.axes)
    coupled = promoted -> BRM._brm_svi_regroup_layout(fixture.view, fixture.inventory,
        fixture.factors, fixture.estimated, axes, promoted)
    explicit = coupled([2])
    @test all(in(explicit.globals), [2, 3])
    limited = BRM._brm_svi_separator_plan(coupled, [[2, 3]], incidence;
        max_unit_coordinates=2, max_promoted_coordinates=1)
    @test limited.policy.status === :promotion_budget
    @test isempty(limited.policy.promotions)
end

@testset "retained regroup metadata scales with structural support" begin
    small = regroup_hub(32).evaluate([2])
    large = regroup_hub(128).evaluate([2])
    @test length(large.units) == 4length(small.units)
    @test Base.summarysize(large) < 5Base.summarysize(small)
    @test sum(length, large.footprints) < 5sum(length, small.footprints)
end
