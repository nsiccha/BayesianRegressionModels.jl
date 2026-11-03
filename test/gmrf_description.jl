include("gmrf_preparation.jl")

function gmrf_description_fixture(side)
    axis = collect(range(0.0, 1.0; length=side))
    xy = hcat(repeat(axis; outer=side), repeat(axis; inner=side))
    triangles = reduce(vcat, ([i + side*j i + 1 + side*j i + side*(j+1);
        i + 1 + side*j i + 1 + side*(j+1) i + side*(j+1)]
        for j in 0:(side-2) for i in 1:(side-1)))
    input = merge(data, (; xy, triangles))
    model = @brm input begin
        field ~ 0 + gmrf((east, north), time, channel;
            mesh=(vertices=xy, cells=triangles), cor=Omega)
        sd(field, gmrf((east, north), time, channel)) = 1.2
        length_scale(field, gmrf((east, north), time, channel), space) = 0.4
        length_scale(field, gmrf((east, north), time, channel), time) ~ LogNormal(0.5, 0.7)
        y ~ Normal(field, 0.2)
    end
    term = only(only(values(BRM._brm_term_address_map(model, :field))))
    context = BRM._brm_backend_context(model)
    prepared = BRM._brm_prepare_term(term, :field, context)
    metadata = BRM._brm_prepare_term(term, :field, context; geometry=false)
    (; prepared, metadata, input)
end

@testset "GMRF description retains exact support with linear metadata size" begin
    sizes = Int[]
    for side in (9, 17, 33)
        fixture = gmrf_description_fixture(side)
        snapshot = BRM._brmd_snapshot(fixture.prepared)
        @test snapshot == BRM._brmd_snapshot(fixture.metadata)
        @test snapshot.source == (:east, :north, :time, :channel)
        @test snapshot.state.vertices.size == size(fixture.input.xy)
        @test snapshot.state.vertices.values == Tuple(fixture.input.xy)
        @test snapshot.state.cells.values == Tuple(fixture.input.triangles)
        @test snapshot.state.layout.times == (0.0, 0.3, 1.7)
        @test snapshot.state.layout.channels == ("a", "b")
        @test snapshot.state.layout.ncoordinates == side^2 * 2 * 3
        push!(sizes, Base.summarysize(snapshot))
        @test sizes[end] < 1000 * side^2
        # Snapshots retain immutable support values, not the mutable fitted arrays.
        saved = first(snapshot.state.vertices.values)
        fixture.prepared.state.vertices[1, 1] += 0.25
        @test first(snapshot.state.vertices.values) == saved
        println((; vertices=side^2, snapshot_bytes=sizes[end]))
    end
    @test all(sizes[2:end] .< 5 .* sizes[1:end-1])
end
