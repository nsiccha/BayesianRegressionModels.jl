module SealFigures

using AlgebraOfVega

const RESULTS = joinpath(@__DIR__, "results")
const BLUE = "#2a78d6"
const ORANGE = "#d65f2e"

function rows(filename)
    lines = readlines(joinpath(RESULTS, filename))
    keys = Tuple(Symbol.(split(first(lines), '\t')))
    [NamedTuple{keys}(Tuple(parse.(Float64, split(line, '\t')))) for line in lines[2:end]]
end

function density_response()
    values = [(; r..., capacity_label="K = $(Int(r.capacity ÷ 1000)) thousand", series="Birth probability")
        for r in rows("density_response.tsv")]
    at_capacity = [merge(r, (; series="Replacement at K")) for r in values if r.population == r.capacity]
    curve = data(values) * mapping(:population => "Population (seals)",
        :birth_rate => "Birth probability", col=:capacity_label => "Carrying capacity", color=:series => "") *
        visual(Lines; linewidth=2)
    equilibrium = data(at_capacity) * mapping(:population, :birth_rate,
        col=:capacity_label, color=:series => "") * visual(Scatter; markersize=9)
    (curve + equilibrium) * config(width=175, height=220,
        config=(; legend=(; orient="top")),
        scales=scales(Col=(; categories=["K = 30 thousand", "K = 60 thousand", "K = 120 thousand"]),
            Color=(; categories=["Birth probability", "Replacement at K"], palette=[BLUE, ORANGE])))
end

function aerial_simulations()
    values = [(; r..., chain_label="Chain $(Int(r.chain))", series="Saved-state simulations")
        for r in rows("aerial_simulations.tsv")]
    envelope = data(values) * mapping(:year => "Fixture year",
        :q50 => "Aerial count (log scale)", col=:chain_label => "Rejected chain", color=:series => "") *
        lineribbon(bands=[:q025 => :q975, :q25 => :q75])
    observed = data([merge(r, (; series="Synthetic observation")) for r in values]) *
        mapping(:year, :observed, col=:chain_label, color=:series => "") * visual(Scatter; markersize=9)
    (envelope + observed) * config(width=130, height=240,
        config=(; legend=(; orient="top"), axisX=(; values=[1, 2, 3], format="d")),
        scales=scales(Y=(; scale=log10),
            Color=(; categories=["Saved-state simulations", "Synthetic observation"], palette=[BLUE, ORANGE])))
end

end
