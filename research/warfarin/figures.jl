# Figures of docs/src/warfarin.md, drawn from the tables fit.jl writes to results/.
#
#   julia --project=research/adaptive_centering/plots research/warfarin/figures.jl [outdir]
#
# outdir defaults to docs/src/assets/warfarin. Colours are the reference palette
# of the dataviz skill (light surface): blue for a fit, orange for the joint
# model beside it, ink for the observations.

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

function save_figure(outdir, name, fig)
    save(joinpath(outdir, name * ".png"), fig; px_per_unit=1.5)
    println("FIGURE ", name)
end

"One panel per subject: the posterior predictive median with its 50 % and 95 % bands, and the observations."
function ppc(outdir, name; model, response, ylabel, title, subjects=nothing, size=(1500, 1100))
    rows = [r for r in table("ppc.tsv") if r.model == model && r.response == response]
    isnothing(subjects) || (rows = [r for r in rows if r.source_subject in subjects])
    order = AlgebraOfGraphics.sorter(unique(r.source_subject for r in rows))
    xlabel = "time after the dose (h)"
    facet = :source_subject => order => "subject"
    band = data(rows) * mapping(:time => xlabel, :q50 => ylabel; layout=facet) *
           lineribbon(bands=[:q025 => :q975, :q25 => :q75]) * visual(; color=BLUE)
    observed = data(rows) * mapping(:time => xlabel, :observed => ylabel; layout=facet) *
               visual(Scatter; markersize=7, color=INK)
    fig = Figure(; size, fontsize=14)
    Label(fig[0, 1], title; fontsize=18, font=:bold, tellwidth=false)
    sdraw!(fig[1, 1], band + observed)
    save_figure(outdir, name, fig)
end

"Every population quantity: the two-stage estimate beside the joint one (median, 50 % and 95 % intervals)."
function population(outdir)
    rows = table("population.tsv")
    stage(model) = model == "joint" ? "joint" : "two-stage"
    quantities = unique(r.quantity for r in rows)
    points = [(; quantity=r.quantity, stage=stage(r.model), median=r.median,
                 q025=r.q025, q975=r.q975) for r in rows]
    facet = :quantity => AlgebraOfGraphics.sorter(quantities) => ""
    x = :stage => AlgebraOfGraphics.sorter(["two-stage", "joint"]) => ""
    layers = mapreduce(+, [("two-stage", BLUE), ("joint", ORANGE)]) do (which, colour)
        sub = [p for p in points if p.stage == which]
        data(sub) * mapping(x, :q025 => "", :q975 => ""; layout=facet) *
            visual(Rangebars; linewidth=2, color=colour) +
        data(sub) * mapping(x, :median => ""; layout=facet) *
            visual(Scatter; markersize=11, color=colour)
    end
    fig = Figure(; size=(1500, 900), fontsize=14)
    Label(fig[0, 1], "Population estimates of the two-stage fits (blue) and the joint model (orange): median and 95 % interval";
          fontsize=18, font=:bold, tellwidth=false)
    sdraw!(fig[1, 1], layers; facet=(; linkyaxes=:none))
    save_figure(outdir, "population", fig)
end

function main(outdir = joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "warfarin"))
    mkpath(outdir)
    ppc(outdir, "ppc-pk"; model="two-stage PK", response="pk_dv",
        ylabel="warfarin (mg/L)", title="Stage 1, population PK: posterior predictive bands and the observed concentrations")
    ppc(outdir, "ppc-pd"; model="two-stage PD", response="pd_dv",
        ylabel="prothrombin complex activity (%)", title="Stage 2, turnover PD on the fixed PK medians: posterior predictive bands and the observations")
    subjects = unique(r.source_subject for r in table("ppc.tsv"))[1:8]
    ppc(outdir, "ppc-joint-pk"; model="joint", response="pk_dv", subjects,
        ylabel="warfarin (mg/L)", title="Joint model, PK channel, first eight subjects", size=(1500, 420))
    ppc(outdir, "ppc-joint-pd"; model="joint", response="pd_dv", subjects,
        ylabel="prothrombin complex activity (%)", title="Joint model, PD channel, first eight subjects", size=(1500, 420))
    population(outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main((length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "warfarin")))
end
