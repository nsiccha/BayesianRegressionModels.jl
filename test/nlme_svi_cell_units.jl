using Test, BayesianRegressionModels
const BRM = BayesianRegressionModels

# Pure ownership: where cells that read no subject's coordinate go. Packed
# layout: 1 global, 2:3 the two subjects, 4:5 and 6:7 two sensors' effects,
# 8 a sample's effect, 9 read by one subject's cell and one sensor's rows.
function cell_units_view()
    coordinate_subject = vcat(0, 1, 2, zeros(Int, 6))
    supports = Dict{Symbol,Vector{Vector{Int}}}(
        :site => [[1, 2, 9], [1, 3]],
        :sensor => [[1, 4, 5], [1, 6, 7, 9], [1, 4, 5]],
        :sample => [[1, 8]],
        :constant => [[1] for _ in 1:9])
    (; subject_levels=["a", "b"], coordinate_subject,
       coordinates=[Symbol("c", j) for j in 1:9], factor_supports=supports)
end

function cell_units_layout(view; owned=[[j] for j in 1:9], support=owned,
        promoted=[1], cell_units=:components)
    n = length(view.coordinate_subject)
    inventory = BRM._brm_svi_factor_inventory(view, n)
    BRM._brm_svi_regroup_layout(view, inventory, (; owned, support), trues(n),
        [[j] for j in 1:n], promoted; cell_units)
end

cell_units_of(layout, name) = Dict(layout.attribution)[name]

# Every observation cell counted exactly once, reading only its unit and globals.
function check_exact_split(view, layout)
    @test all(pair -> all(>(0), pair.second), layout.attribution)
    for (name, owners) in layout.attribution, (cell, unit) in enumerate(owners)
        @test all(j -> layout.owner[j] in (0, unit), view.factor_supports[name][cell])
    end
end

@testset "subjectless cells form their own inference units" begin
    view = cell_units_view()
    layout = cell_units_layout(view)
    check_exact_split(view, layout)
    # Scientific units keep their order and labels; cell units are appended.
    @test layout.levels[1:2] == [("a",), ("b",)]
    @test layout.subject_unit == [1, 2]
    @test layout.unit_members[1:2] == [[1], [2]]
    @test all(isempty, layout.unit_members[3:end])
    @test cell_units_of(layout, :site) == [1, 2]
    # A sensor's effects and rows are one unit; the two sensors stay apart.
    sensor = cell_units_of(layout, :sensor)
    @test sensor[1] == sensor[3] != sensor[2]
    @test layout.units[sensor[1]] == [4, 5]
    @test layout.units[sensor[2]] == [6, 7]
    sample = only(cell_units_of(layout, :sample))
    @test layout.units[sample] == [8]
    @test only(layout.levels[sample]) == (:cells, :c8)
    # A coordinate a subject's cell also reads has two readers: global, never a link.
    @test 9 in layout.globals
    @test layout.units[1] == [2]
    # Nine cells reading only globals: three units of three, no local coordinates.
    constant = cell_units_of(layout, :constant)
    @test sort(unique(constant)) == collect(length(layout.units)-2:length(layout.units))
    @test all(i -> count(==(i), constant) == 3, unique(constant))
    @test all(i -> isempty(layout.units[i]), unique(constant))
    @test allunique(layout.levels)
    # Each label holds one member, as the streaming membership check needs.
    @test all(level -> length(level) == 1, layout.levels)
end

@testset "`:first_unit` keeps the scientific layout" begin
    view = cell_units_view()
    layout = cell_units_layout(view; cell_units=:first_unit)
    check_exact_split(view, layout)
    @test layout.levels == [("a",), ("b",)]
    @test all(==(1), cell_units_of(layout, :sensor))
    @test all(==(1), cell_units_of(layout, :sample))
    @test all(==(1), cell_units_of(layout, :constant))
    # Coordinate 9 is local only because its two readers both landed in unit 1.
    @test sort(layout.units[1]) == [2, 4, 5, 6, 7, 8, 9]
    @test BRM._brm_svi_cell_units_option(nothing, true) === :components
    @test BRM._brm_svi_cell_units_option(nothing, false) === :first_unit
    @test BRM._brm_svi_cell_units_option(:first_unit, false) === :first_unit
    @test_throws ArgumentError BRM._brm_svi_cell_units_option(:components, false)
    @test_throws ArgumentError BRM._brm_svi_cell_units_option(:each_row, true)
end

@testset "what may and may not join cells" begin
    base = cell_units_view()
    # A joint prior block moves together, so it joins its readers' cells.
    joint = cell_units_layout(base; owned=[[1], [2], [3], [4, 6], [5], [7], [8], [9]])
    sensor = cell_units_of(joint, :sensor)
    @test length(unique(sensor)) == 1
    check_exact_split(base, joint)
    # A hub only subjectless cells read joins them until it is promoted.
    hub = merge(base, (; factor_supports=merge(base.factor_supports,
        Dict(:sensor => [[1, 4, 5, 8], [1, 6, 7, 8], [1, 4, 5, 8]],
             :sample => [[1, 8]]))))
    joined = cell_units_layout(hub)
    check_exact_split(hub, joined)
    @test length(unique(cell_units_of(joined, :sensor))) == 1
    split = cell_units_layout(hub; promoted=[1, 8])
    check_exact_split(hub, split)
    @test 8 in split.globals
    @test length(unique(cell_units_of(split, :sensor))) == 2
    @test only(cell_units_of(split, :sample)) ∉ cell_units_of(split, :sensor)
    # A prior makes coordinate 8 global once two units read through it, and the
    # next refinement pass stops it joining the sensors' cells. Without that
    # refinement they would stay one unit.
    prior = cell_units_layout(hub; owned=[[1], [2], [3], [4], [5], [6], [7], [8], [9]],
        support=[[1], [2, 8], [3, 8], [4], [5], [6], [7], [8], [9]])
    check_exact_split(hub, prior)
    @test 8 in prior.globals
    @test length(unique(cell_units_of(prior, :sensor))) == 2
    # A coordinate of no subject's cell that is fixed joins nothing.
    n = 9
    inventory = BRM._brm_svi_factor_inventory(hub, n)
    estimated = trues(n)
    estimated[8] = false
    fixed = BRM._brm_svi_regroup_layout(hub, inventory,
        (; owned=[[j] for j in 1:n], support=[[j] for j in 1:n]), estimated,
        [[j] for j in 1:n], [1]; cell_units=:components)
    @test length(unique(cell_units_of(fixed, :sensor))) == 2
    @test all(==(0), fixed.owner[[8]])
end

# The mask other coordinate classes clear: a cleared coordinate never joins cells.
@testset "a cleared coordinate never joins cells" begin
    supports = [[1, 2], [1, 3], [4], [4], [5]]
    mapping = BRM._brm_nlme_connected_unit_map(1, [1, 0, 0, 0, 0], [[1, 2]])
    subjectless = [false, true, true, true, true]
    mapping = (; mapping.subject_unit, mapping.members,
        factor_unit=[1, 1, 1, 1, 1], unit_factors=[collect(1:5)])
    linkable = BitVector([false, true, false, true, true])
    joined = BRM._brm_svi_split_subjectless_cells(mapping, supports, subjectless,
        [[j] for j in 1:5], linkable)
    @test joined.factor_unit[1] == 1
    # Coordinate 3 is cleared, so its cell reads nothing linkable: it is alone.
    @test joined.factor_unit[3] == joined.factor_unit[4]
    @test joined.factor_unit[5] != joined.factor_unit[3]
    @test joined.factor_unit[2] > 1
    cleared = copy(linkable)
    cleared[4] = false
    apart = BRM._brm_svi_split_subjectless_cells(mapping, supports, subjectless,
        [[j] for j in 1:5], cleared)
    @test length(apart.members) > 1
    @test all(>(0), apart.factor_unit)
end

@testset "cell units cost is linear in the supports" begin
    # Half a million cells over two hundred thousand coordinates, one subject.
    n, m = 200_000, 500_000
    coordinate_subject = vcat(1, zeros(Int, n - 1))
    supports = [[2 + (i % (n - 1))] for i in 1:m]
    mapping = (; subject_unit=[1], members=[[1]], factor_unit=fill(1, m),
        unit_factors=[collect(1:m)])
    subjectless = fill(true, m)
    linkable = BitVector(j > 1 for j in 1:n)
    seconds = @elapsed split = BRM._brm_svi_split_subjectless_cells(mapping, supports,
        subjectless, [[j] for j in 1:n], linkable)
    @test length(split.members) == n
    @test seconds < 30
end
