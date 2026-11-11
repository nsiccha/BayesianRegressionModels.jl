# Included by the documentation build. No fitting or rendered-file writes.
using AlgebraOfVega

const BLUE = "#2a78d6"
const ORANGE = "#eb6834"

function filtering_table(name)
    lines = readlines(joinpath(@__DIR__, "results", name * ".tsv"))
    columns = Tuple(Symbol.(split(first(lines), '\t')))
    cell(s) = something(tryparse(Float64, s), String(s))
    [NamedTuple{columns}(Tuple(cell.(split(line, '\t')))) for line in lines[2:end]]
end

function filtered_states()
    observations = Dict((r.day, r.unit) => r.state for r in filtering_table("observations"))
    rows = [(; r..., unit="Unit $(Int(r.unit))", truth=observations[(r.day, r.unit)])
        for r in filtering_table("filtered_state")]
    facets = (; row=:unit => "")
    band90 = data(rows) * mapping(:day, :q05, :q95; facets...) *
        visual(Band; color=BLUE, alpha=0.16)
    band50 = data(rows) * mapping(:day, :q25, :q75; facets...) *
        visual(Band; color=BLUE, alpha=0.32)
    median = data(rows) * mapping(:day => "Day", :q50 => "Latent log-rate deviation"; facets...) *
        visual(Lines; color=BLUE, linewidth=2)
    truth = data(rows) * mapping(:day, :truth; facets...) *
        visual(Lines; color=ORANGE, linestyle=:dash, linewidth=1.5)
    (band90 + band50 + median + truth) * config(width=510, height=125,
        axis=(; xgridvisible=false), config=Dict("axis" => Dict("labelFontSize" => 12, "titleFontSize" => 13)))
end

function marginal_comparison()
    methods = ["MCMC", "Batch SMC 1", "Batch SMC 2", "Batch SMC 3",
        "Cumulative SMC", "Carried windows", "Learning filter"]
    names = Dict("level" => "Level", "rho" => "Persistence ρ", "tau" => "Innovation SD τ")
    rows = [(; r..., parameter=names[r.parameter],
        family=r.method in ("Carried windows", "Learning filter") ? "Gaussian boundary" : "Full history")
        for r in filtering_table("marginals")]
    facets = (; col=:parameter => sorter([names[p] for p in ["level", "rho", "tau"]]) => "", color=:family => "Target")
    order = :method => sorter(methods) => ""
    outer = data(rows) * mapping(order, :q05, :q95; facets...) *
        visual(Rangebars; direction=:x, linewidth=1.5, whiskerwidth=0)
    inner = data(rows) * mapping(order, :q25, :q75; facets...) *
        visual(Rangebars; direction=:x, linewidth=5, whiskerwidth=0)
    median = data(rows) * mapping(:q50 => "Parameter value", order; facets...) *
        visual(Scatter; markersize=7)
    (outer + inner + median) * config(width=135, height=240,
        facet=(; linkxaxes=:none),
        scales=scales(Color=(; categories=["Full history", "Gaussian boundary"], palette=[BLUE, ORANGE])),
        config=Dict("axis" => Dict("labelFontSize" => 12, "titleFontSize" => 13),
            "legend" => Dict("orient" => "bottom")))
end
