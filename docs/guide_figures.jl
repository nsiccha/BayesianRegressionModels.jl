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

function field_slice()
    source = rows("research/space_time_fields/results/field_grid.tsv")
    records = [NamedTuple{keys(r)}(Tuple(parse(Float64, v) for v in values(r))) for r in source]
    norths = sort!(unique(r.north for r in records))
    times = sort!(unique(r.time for r in records))
    north = norths[cld(length(norths), 2)]
    time = times[cld(length(times), 2)]
    slice = sort!(filter(r -> r.north == north && r.time == time, records); by=r -> r.east)
    length(slice) > 1 || error("The field summary has no east–west slice")
    ribbon = data(slice) * mapping(:east, :q025, :q975) * visual(Band; color="#2a78d6", alpha=0.2)
    posterior = data(slice) * mapping(:east => "East (km)", :q50 => "Latent field") * visual(Lines; color="#2a78d6")
    truth = data(slice) * mapping(:east, :truth) * visual(Lines; color="#eb6834")
    (ribbon + posterior + truth) * config(width=540, height=260,
        title="North = $(north) km, time = $(time)")
end

end
