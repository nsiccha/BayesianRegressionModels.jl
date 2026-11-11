# The figure of docs/src/monster.md: an AlgebraOfVega spec over the posterior
# predictive quantiles fit.jl writes to results/ppc.tsv. The docs build renders
# it in the page through Main.BRMDocsComparisons.figure (Vega-Lite, light and
# dark); no rendered file is written to the repository.
#
#   julia --project=research/adaptive_centering/plots research/monster/figures.jl [outdir]
#
# writes a PNG preview with vsave (vl-convert); outdir defaults to a scratch
# directory, never the repo.

using AlgebraOfVega

const RESULTS = joinpath(@__DIR__, "results")
const BLUE = "#2a78d6"
const ORANGE = "#eb6834"

"The rows of one results table, numbers parsed."
function table(name)
    lines = readlines(joinpath(RESULTS, name))
    header = Symbol.(split(lines[1], '\t'))
    cell(s) = something(tryparse(Int, s), tryparse(Float64, s), String(s))
    [NamedTuple{Tuple(header)}(Tuple(cell.(split(line, '\t')))) for line in lines[2:end]]
end

"""
One panel per person and exposure: the posterior predictive median with its
50 % and 95 % bands at the sampling times, venous blood in blue and exhaled
air in orange, and the measurements. Both axes are logarithmic.
"""
function ppc_spec(; panel_width=150, panel_height=110)
    rows = map(table("ppc.tsv")) do r
        channel, exposure = split(r.response, '_')
        (; r..., channel=channel == "venous" ? "venous blood" : "exhaled air",
           panel="$(r.source_subject), $(exposure) ppm")
    end
    isempty(rows) && error("results/ppc.tsv has no rows")
    exposure(p) = parse(Int, last(split(p.response, '_')))
    panels = unique(p.panel for p in sort(rows; by=p -> (p.source_subject, exposure(p))))
    # Short axis titles: Vega-Lite repeats them on every facet row and column.
    xlabel = "hours after exposure start"
    ylabel = "concentration"
    facet = :panel => sorter(panels) => ""
    colour = :channel => ""
    bands = data(rows) * mapping(:time => xlabel, :q50 => ylabel; color=colour, layout=facet) *
            lineribbon(bands=[:q025 => :q975, :q25 => :q75])
    observed = data(rows) * mapping(:time => xlabel, :observed => ylabel; color=colour, layout=facet) *
               visual(Scatter; markersize=6)
    (bands + observed) * config(width=panel_width, height=panel_height,
        scales=scales(X=(; scale=log10), Y=(; scale=log10),
                      Color=(; categories=["venous blood", "exhaled air"], palette=[BLUE, ORANGE])))
end

const CAPTION = "Posterior predictive median, 50 % and 95 % bands and the measurements, per person and exposure: venous blood (blue) and exhaled air (orange), both in mg/L, both axes logarithmic."

function main(outdir = mktempdir())
    path = joinpath(outdir, "ppc.png")
    vsave(ppc_spec(), path; scale=2)
    println("FIGURE ", path)
    outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(length(ARGS) > 0 ? ARGS[1] : mktempdir())
end
