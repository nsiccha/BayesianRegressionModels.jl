module BRMGuideFigures

using AlgebraOfVega

const ROOT = normpath(joinpath(@__DIR__, ".."))

function rows(path)
    lines = readlines(joinpath(ROOT, path))
    names = Tuple(Symbol.(split(first(lines), '\t')))
    [NamedTuple{names}(Tuple(String.(split(line, '\t')))) for line in lines[2:end]]
end

function centering_efficiency()
    source = rows("research/centering_refresh/results/eight/efficiency.tsv")
    selected = filter(r -> r.metric == "Relative total efficiency", source)
    isempty(selected) && error("No full-workflow efficiency summary")
    values = [(method=r.method, efficiency=parse(Float64, r.value),
        kind=startswith(r.method, "Online") || startswith(r.method, "Post-hoc") ?
            "Adaptive" : "Fixed") for r in selected]
    data(values) * mapping(:method => "Coordinates", :efficiency => "Relative ESS per total gradient";
        color=:kind => "Method") * visual(BarPlot) * config(width=540, height=260,
        scales=scales(Color=(palette=["#2a78d6", "#eb6834"], categories=["Fixed", "Adaptive"])))
end

function temporal_correlation()
    values = [(gap=gap, correlation=exp(-gap / ell), length="ell_t = $(ell)")
        for ell in (0.5, 2.0) for gap in range(0, 6; length=121)]
    data(values) * mapping(:gap => "Time gap", :correlation => "Prior temporal correlation";
        color=:length => "OU correlation time") * visual(Lines) *
        config(width=500, height=260,
            scales=scales(Color=(palette=["#2a78d6", "#eb6834"],
                categories=["ell_t = 0.5", "ell_t = 2.0"])))
end

end
