# AlgebraOfVega specifications evaluated by docs/src/spatial-fields.md.
# No image or specification is written to the repository.
using AlgebraOfVega

function numeric_rows(name; root=joinpath(@__DIR__, "results"))
    lines = readlines(joinpath(root, name * ".tsv"))
    names = Tuple(Symbol.(split(first(lines), '\t')))
    [NamedTuple{names}(Tuple(parse.(Float64, split(line, '\t')))) for line in lines[2:end]]
end

time_label(t) = "Time $(Int(t))"

function field_maps(grid=numeric_rows("field_grid"), observations=numeric_rows("observations"))
    times = time_label.(sort(unique(r.time for r in grid)))
    panels = ["Generating field", "Posterior median"]
    cells = [(; r.east, r.north, time=time_label(r.time), panel,
        value=panel == first(panels) ? r.truth : r.q50) for r in grid for panel in panels]
    stations = unique([(; r.east, r.north, time=time_label(r.time), panel=first(panels))
        for r in observations])
    bound = ceil(2maximum(abs(r.value) for r in cells)) / 2
    field = data(cells) * mapping(:east => "East (km)", :north => "North (km)",
        :value => "Field value"; row=:time => sorter(times), col=:panel => sorter(panels)) * visual(Heatmap)
    points = data(stations) * mapping(:east => "East (km)", :north => "North (km)";
        row=:time => sorter(times), col=:panel => sorter(panels)) *
        visual(Scatter; color="#0b0b0b", markersize=8, strokewidth=2, strokecolor=:white)
    (field + points) * config(width=230, height=230,
        axis=(; xlabel="East (km)", ylabel="North (km)", limits=((0, 100), (0, 100))),
        scales=scales(Color=(; colormap=:blueorange, colorrange=(-bound, bound))))
end

function uncertainty_maps(grid=numeric_rows("field_grid"))
    times = time_label.(sort(unique(r.time for r in grid)))
    cells = [(; r.east, r.north, time=time_label(r.time), r.sd) for r in grid]
    bound = ceil(10maximum(r.sd for r in cells)) / 10
    data(cells) * mapping(:east => "East (km)", :north => "North (km)", :sd => "Field SD";
        col=:time => sorter(times)) * visual(Heatmap) *
        config(width=165, height=165,
            axis=(; xlabel="East (km)", ylabel="North (km)", limits=((0, 100), (0, 100))),
            scales=scales(Color=(; colormap=:blues, colorrange=(0, bound))))
end

function field_slices(grid=numeric_rows("field_grid"))
    times = time_label.(sort(unique(r.time for r in grid)))
    center = (minimum(r.north for r in grid) + maximum(r.north for r in grid)) / 2
    rows = [(; r..., time=time_label(r.time)) for r in grid if r.north == center]
    bands = data(rows) * mapping(:east => "East (km)", :q50 => "Field value";
        col=:time => sorter(times)) * lineribbon(bands=[:q025 => :q975, :q25 => :q75]) *
        visual(; color="#2a78d6")
    truth = data(rows) * mapping(:east => "East (km)", :truth => "Field value";
        col=:time => sorter(times)) * visual(Lines; color="#eb6834", linestyle=:dash, linewidth=2)
    (bands + truth) * config(width=165, height=180,
        axis=(; xlabel="East (km)", ylabel="Field value", limits=((0, 100), nothing)))
end
