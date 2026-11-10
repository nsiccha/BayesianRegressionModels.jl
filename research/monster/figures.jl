# The figure of docs/src/monster.md, drawn from the tables fit.jl writes to results/.
#
#   julia --project=research/adaptive_centering/plots research/monster/figures.jl [outdir]
#
# outdir defaults to docs/src/assets/monster. Colours are the reference palette
# of the dataviz skill (light surface): blue for venous blood, orange for
# exhaled air, ink for the observations.

using AlgebraOfVega, CairoMakie
import AlgebraOfGraphics

const RESULTS = joinpath(@__DIR__, "results")
const BLUE = "#2a78d6"
const ORANGE = "#eb6834"
const INK = "#0b0b0b"

function table(name)
    lines = readlines(joinpath(RESULTS, name))
    header = Symbol.(split(lines[1], '\t'))
    cell(s) = something(tryparse(Int, s), tryparse(Float64, s), String(s))
    [NamedTuple{Tuple(header)}(Tuple(cell.(split(line, '\t')))) for line in lines[2:end]]
end

"Person × exposure panels: both channels' posterior predictive bands and the measurements, on a log scale."
function ppc(outdir)
    rows = [(; r..., channel=startswith(r.response, "venous") ? "venous blood" : "exhaled air",
               panel=string(r.source_subject, ", ", endswith(r.response, "72") ? "72" : "144", " ppm"))
            for r in table("ppc.tsv")]
    # Facet on the label in natural order: the ribbon layer facets on the raw
    # column values, so a `sorter` on the point layer would not merge with it.
    xlabel = "hours after the start of exposure"
    ylabel = "tetrachloroethylene (mg/L)"
    facet = :panel => ""
    layers = mapreduce(+, [("venous blood", BLUE), ("exhaled air", ORANGE)]) do (which, colour)
        sub = [r for r in rows if r.channel == which]
        data(sub) * mapping(:time => xlabel, :q50 => ylabel; layout=facet) *
            lineribbon(bands=[:q025 => :q975, :q25 => :q75]) * visual(; color=colour) +
        data(sub) * mapping(:time => xlabel, :observed => ylabel; layout=facet) *
            visual(Scatter; markersize=8, color=INK)
    end
    # Both axes logarithmic: the samples span a week, the concentrations three
    # orders of magnitude. The scale rides on the spec (aov-use §1a).
    layers = layers * config(scales=scales(X=(; scale=log10), Y=(; scale=log10)))
    fig = Figure(; size=(1500, 950), fontsize=14)
    Label(fig[0, 1], "Posterior predictive bands (median, 50 % and 95 %) and the measurements, per person and exposure";
          fontsize=18, font=:bold, tellwidth=false)
    sdraw!(fig[1, 1], layers)
    Legend(fig[1, 2], [LineElement(; color=BLUE, linewidth=3), LineElement(; color=ORANGE, linewidth=3),
                       MarkerElement(; color=INK, marker=:circle, markersize=10)],
           ["venous blood", "exhaled air", "measured"])
    save(joinpath(outdir, "ppc.png"), fig; px_per_unit=1.5)
    println("FIGURE ppc")
end

function main(outdir = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "monster"))
    mkpath(outdir)
    ppc(outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main((length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "monster")))
end
