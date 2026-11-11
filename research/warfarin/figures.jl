# Figures of docs/src/warfarin.md: AlgebraOfVega specs over the tables fit.jl
# writes to results/<fit>/. The docs build renders them in the page through
# Main.BRMDocsComparisons.figure (Vega-Lite, light and dark); no rendered file
# is written to the repository. Colours are the reference palette of the
# dataviz skill: blue for a fit, orange for the joint model beside it, the
# page's ink for the observations.
#
#   julia --project=research/adaptive_centering/plots research/warfarin/figures.jl [outdir]
#
# writes PNG previews of the four figures with vsave (vl-convert) for a look
# before a docs build; outdir defaults to a scratch directory, never the repo.

using AlgebraOfVega

const RESULTS = joinpath(@__DIR__, "results")
const BLUE = "#2a78d6"
const ORANGE = "#eb6834"

"The rows of one table of one fit (results/<fit>/<name>), numbers parsed."
function table(fit, name)
    lines = readlines(joinpath(RESULTS, fit, name))
    header = Symbol.(split(lines[1], '\t'))
    cell(s) = something(tryparse(Int, s), tryparse(Float64, s), String(s))
    [NamedTuple{Tuple(header)}(Tuple(cell.(split(line, '\t')))) for line in lines[2:end]]
end

"""
One panel per subject: the posterior predictive median with its 50 % and 95 %
bands at the sampling times, and the observations. `subjects` restricts the
panels (source subject ids); the panels wrap to the page width.
"""
function ppc_spec(fit, response; ylabel, subjects=nothing, panel_width=140, panel_height=100)
    rows = [(; r..., panel="subject $(r.source_subject)")
            for r in table(fit, "ppc.tsv") if r.response == response]
    isnothing(subjects) || (rows = [r for r in rows if r.source_subject in subjects])
    isempty(rows) && error("$fit/ppc.tsv has no rows for $response")
    panels = ["subject $s" for s in sort(unique(r.source_subject for r in rows))]
    xlabel = "time after the dose (h)"
    facet = :panel => sorter(panels) => ""
    bands = data(rows) * mapping(:time => xlabel, :q50 => ylabel; layout=facet) *
            lineribbon(bands=[:q025 => :q975, :q25 => :q75]) * visual(; color=BLUE)
    observed = data(rows) * mapping(:time => xlabel, :observed => ylabel; layout=facet) *
               visual(Scatter; markersize=6, color=:ink)
    (bands + observed) * config(width=panel_width, height=panel_height)
end

"""
Every population quantity in its own panel: the two-stage estimate (blue)
beside the joint model's (orange), median and 95 % interval, each panel on
its own scale.
"""
function population_spec()
    rows = vcat([(; r..., stage="two-stage") for r in table("pk", "population.tsv")],
                [(; r..., stage="two-stage") for r in table("pd", "population.tsv")],
                [(; r..., stage="joint") for r in table("joint", "population.tsv")])
    quantities = unique(r.quantity for r in rows)
    facet = :quantity => sorter(quantities) => ""
    x = :stage => sorter(["two-stage", "joint"]) => ""
    colour = :stage => ""
    intervals = data(rows) * mapping(x, :q025 => "", :q975 => ""; color=colour, layout=facet) *
                visual(Rangebars; linewidth=2)
    medians = data(rows) * mapping(x, :median => ""; color=colour, layout=facet) *
              visual(Scatter; markersize=9)
    (intervals + medians) * config(width=80, height=110, facet=(; linkyaxes=:none),
        scales=scales(Color=(; categories=["two-stage", "joint"], palette=[BLUE, ORANGE])))
end

"The four figures of the page, in page order, with their captions."
function figures()
    joint_subjects = unique(r.source_subject for r in table("joint", "ppc.tsv"))[1:8]
    [
        "ppc-pk" => (ppc_spec("pk", "pk_dv"; ylabel="warfarin (mg/L)"),
                     "Stage 1, population PK: posterior predictive median, 50 % and 95 % bands (blue) and the observed concentrations, one panel per subject."),
        "ppc-pd" => (ppc_spec("pd", "pd_dv"; ylabel="prothrombin complex activity (%)"),
                     "Stage 2, turnover PD on the fixed PK medians: posterior predictive bands and the observed activity, one panel per subject."),
        "ppc-joint-pk" => (ppc_spec("joint", "pk_dv"; ylabel="warfarin (mg/L)", subjects=joint_subjects),
                           "Joint model, PK channel, first eight subjects: posterior predictive bands and the observed concentrations."),
        "ppc-joint-pd" => (ppc_spec("joint", "pd_dv"; ylabel="prothrombin complex activity (%)", subjects=joint_subjects),
                           "Joint model, PD channel, the same eight subjects: posterior predictive bands and the observed activity, from the same posterior."),
        "population" => (population_spec(),
                         "Population estimates of the two-stage fits (blue) and the joint model (orange): median and 95 % interval, one panel per quantity."),
    ]
end

function main(outdir = mktempdir())
    for (name, (spec, caption)) in figures()
        path = joinpath(outdir, name * ".png")
        vsave(spec, path; scale=2)
        println("FIGURE ", path, "  ", caption)
    end
    outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(length(ARGS) > 0 ? ARGS[1] : mktempdir())
end
