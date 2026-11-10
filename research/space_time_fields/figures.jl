# julia --project=research/adaptive_centering/plots research/space_time_fields/figures.jl
using AlgebraOfVega, CairoMakie, CSV, Tables
import AlgebraOfGraphics

function spatial_figures()
    results = joinpath(@__DIR__, "results")
    grid = collect(Tables.namedtupleiterator(CSV.File(joinpath(results, "field_grid.tsv"))))
    observations = collect(Tables.namedtupleiterator(CSV.File(joinpath(results, "observations.tsv"))))
    times = sort(unique(r.time for r in grid))
    output = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "space-time-fields"))
    mkpath(output)
    set_theme!(Theme(; fontsize=17, textcolor="#30343b", backgroundcolor=:white,
        Axis=(; xgridvisible=false, ygridvisible=false, topspinevisible=false, rightspinevisible=false)))

    extent = maximum(max(abs(r.truth), abs(r.q50)) for r in grid)
    bound = ceil(2extent) / 2
    sdmax = ceil(10maximum(r.sd for r in grid)) / 10
    fig = Figure(; size=(1120, 1110))
    Label(fig[0, 1:3], "A moving field, recovered from fixed stations";
        fontsize=25, font=:bold, tellwidth=false)
    for (row, time) in enumerate(times)
        cells = filter(r -> r.time == time, grid)
        points = unique([(east=r.east, north=r.north) for r in observations if r.time == time])
        for (col, (value, title)) in enumerate(((:truth, "Truth and stations"), (:q50, "Posterior median"), (:sd, "Posterior SD")))
            scale = value == :sd ? (colorrange=(0, sdmax), colormap=:Blues) :
                                  (colorrange=(-bound, bound), colormap=:vik)
            spec = data(cells) * mapping(:east => "East (km)", :north => "North (km)", value => "") * visual(Heatmap)
            if value == :truth
                spec += data(points) * mapping(:east, :north) *
                    visual(Scatter; color="#30343b", markersize=8, strokewidth=2, strokecolor=:white)
            end
            spec *= config(axis=(; title=string(title, " · time ", Int(time)), aspect=DataAspect(),
                xlabel="East (km)", ylabel="North (km)",
                limits=(0, 100, 0, 100), xticks=[0, 50, 100], yticks=[0, 50, 100]),
                scales=scales(Color=scale))
            sdraw!(fig[row, col], spec)
        end
    end
    Colorbar(fig[4, 1:2]; colormap=:vik, limits=(-bound, bound), vertical=false,
        label="Field value · shared scale for truth and median", height=16)
    Colorbar(fig[4, 3]; colormap=:Blues, limits=(0, sdmax), vertical=false,
        label="Posterior SD", height=16)
    rowgap!(fig.layout, 15)
    save(joinpath(output, "field-maps.png"), fig; px_per_unit=1.5)

    section = Figure(; size=(1200, 390))
    Label(section[0, 1:3], "A slice through the maps at north = 50 km";
        fontsize=24, font=:bold, tellwidth=false)
    for (i, time) in enumerate(times)
        rows = filter(r -> r.time == time && r.north == 50, grid)
        band = data(rows) * mapping(:east => "East (km)", :q50 => "Field value") *
            lineribbon(bands=[:q025 => :q975, :q25 => :q75]) * visual(; color="#2a78d6")
        truth = data(rows) * mapping(:east, :truth) * visual(Lines; color="#30343b", linestyle=:dash, linewidth=2)
        sdraw!(section[1, i], (band + truth) * config(axis=(; title="Time $(Int(time))",
            xlabel="East (km)", ylabel="Field value", limits=(0, 100, -bound, bound))))
    end
    Label(section[2, 1:3], "Solid: posterior median   ·   Shading: 50% and 95% intervals   ·   Dashed: truth";
        fontsize=16, tellwidth=false)
    save(joinpath(output, "field-slices.png"), section; px_per_unit=1.5)
    println("FIGURES ", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    spatial_figures()
end
