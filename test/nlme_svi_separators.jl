# Pure policy checks. Native ownership/partition checks exercise its adapter.
using Test
include(joinpath(@__DIR__, "..", "src", "nlme_svi_separators.jl"))

# Independent breadth-first oracle on a small pairwise likelihood graph.
function separator_oracle(n, edges; atomic=[collect(i:i) for i in 1:n])
    function evaluate(requested)
        globals = Set(requested)
        for block in atomic
            any(in(globals), block) && union!(globals, block)
        end
        unseen = setdiff(Set(1:n), globals)
        units = Vector{Int}[]
        while !isempty(unseen)
            unit = [minimum(unseen)]
            delete!(unseen, first(unit))
            k = 1
            while k <= length(unit)
                j = unit[k]
                for (a, b) in edges
                    neighbor = a == j ? b : b == j ? a : 0
                    neighbor in unseen || continue
                    delete!(unseen, neighbor)
                    push!(unit, neighbor)
                end
                k += 1
            end
            push!(units, sort!(unit))
        end
        (; globals=sort!(collect(globals)), units,
            work=[length(u) + count(e -> any(in(u), e), edges) for u in units])
    end
end

@testset "bounded structural separator policy" begin
    n = 33
    blocks = [collect(i:i) for i in 1:n]
    hub = [(1, j) for j in 2:n]
    incidence = [count(e -> j in e, hub) for j in 1:n]
    evaluate = separator_oracle(n, hub)
    fit = _brm_svi_separator_plan(evaluate, blocks, incidence;
        max_unit_coordinates=4, max_promoted_coordinates=1)
    @test fit.layout.globals == [1]
    @test maximum(length, fit.layout.units) == 1
    @test fit.policy.status === :satisfied
    @test fit.policy.trials <= 16
    @test only(fit.policy.promotions).reason === :unit_budget
    @test fit == _brm_svi_separator_plan(evaluate, blocks, incidence;
        max_unit_coordinates=4, max_promoted_coordinates=1)
    none = _brm_svi_separator_plan(evaluate, blocks, incidence; separator=:none)
    @test only(none.layout.units) == 1:n
    @test none.policy.trials == 0
    manual = _brm_svi_separator_plan(evaluate, blocks, incidence;
        separator=:none, global_coordinates=[1])
    @test manual.layout.units == fit.layout.units
    # A bounded number of interior cuts breaks a chain without a clique or
    # a quadratic matrix. All candidates have the same degree except its ends.
    edges = [(j, j+1) for j in 1:n-1]
    incidence = [count(e -> j in e, edges) for j in 1:n]
    evaluate = separator_oracle(n, edges)
    chain = _brm_svi_separator_plan(evaluate, blocks, incidence;
        max_unit_coordinates=8, max_promoted_coordinates=4)
    @test maximum(length, chain.layout.units) <= 8
    @test chain.policy.status === :satisfied
    @test chain.policy.trials <= 8 * 16
    @test length(chain.layout.globals) <= 4
    carried = _brm_svi_separator_plan(evaluate, blocks, incidence;
        global_coordinates=[1], max_unit_coordinates=4,
        max_promoted_coordinates=4, preserve_globals=true)
    @test carried.layout.globals == [1]
    @test carried.policy.status === :carried
    @test carried.policy.trials == 0
    @test carried.policy.max_promoted_coordinates == 4
    @test carried.policy.after > 1
    # Promotion cost includes the closure, not only the requested coordinate.
    evaluate = separator_oracle(n, hub; atomic=vcat([[1, 2]], blocks[3:end]))
    coupled = _brm_svi_separator_plan(evaluate, [[1]], ones(Int, n);
        max_unit_coordinates=4, max_promoted_coordinates=1)
    @test isempty(coupled.layout.globals)
    @test coupled.policy.status === :promotion_budget
    explicit = _brm_svi_separator_plan(evaluate, [[1]], ones(Int, n);
        separator=:none, global_coordinates=[1], max_promoted_coordinates=0)
    @test explicit.layout.globals == [1, 2]
    @test_throws ArgumentError _brm_svi_separator_plan(evaluate, blocks, incidence;
        max_unit_coordinates=0)
end
