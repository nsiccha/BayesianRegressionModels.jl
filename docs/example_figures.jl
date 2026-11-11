module BRMExampleFigures

using AlgebraOfVega

const ROOT = dirname(@__DIR__)
const BLUE = "#2a78d6"
const ORANGE = "#eb6834"

"Read the small numeric trajectory tables written by the cached fits."
function numeric_rows(path)
    lines = readlines(joinpath(ROOT, path))
    fields = Tuple(Symbol.(split(first(lines), '\t')))
    [NamedTuple{fields}(Tuple(parse.(Float64, split(line, '\t')))) for line in lines[2:end]]
end

function trajectory(path; xlabel, ylabel, observed=false, truth=false, select=(_ -> true))
    rows = filter(select, numeric_rows(path))
    isempty(rows) && error("no figure rows selected from $path")
    bands = [(; r.time, r.q025, r.q25, r.q50, r.q75, r.q975) for r in rows]
    spec = data(bands) * mapping(:time => xlabel, :q50 => ylabel) *
        lineribbon(bands=[:q025 => :q975, :q25 => :q75]) * visual(; color=BLUE)
    if truth
        truth_rows = [(; r.time, r.truth) for r in rows if isfinite(r.truth)]
        spec += data(truth_rows) * mapping(:time => xlabel, :truth => ylabel) * visual(Lines; color=ORANGE)
    end
    if observed
        obs = [(; r.time, r.observed) for r in rows if isfinite(r.observed)]
        spec += data(obs) * mapping(:time => xlabel, :observed => ylabel) * visual(Scatter; color=ORANGE, markersize=7)
    end
    spec * config(width=640, height=300)
end

end
