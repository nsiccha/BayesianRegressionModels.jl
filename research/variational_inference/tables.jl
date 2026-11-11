# Helpers the `@eval` blocks of docs/src/variational-inference.md use to read the
# tables in results/ and to state their numbers with no more digits than they carry.
# Each block `include`s this file, because Documenter evaluates every block in a
# module of its own.
using Markdown

vi_results(name) = joinpath(@__DIR__, "results", name)

function vi_table(name)
    lines = readlines(vi_results(name))
    header = Symbol.(split(lines[1], '\t'))
    cell(s) = isempty(s) ? missing : something(tryparse(Int, s), tryparse(Float64, s), String(s))
    [NamedTuple{Tuple(header)}(Tuple(cell.(split(l, '\t'; keepempty=true)))) for l in lines[2:end]]
end

vi_rows(name, predicate) = filter(predicate, vi_table(name))
vi_row(name, predicate) = only(vi_rows(name, predicate))

# A number with `digits` significant digits, without a trailing `.0`.
function vi_sig(x, digits=2)
    s = string(round(x; sigdigits=digits))
    endswith(s, ".0") ? s[1:end-2] : s
end
vi_int(x) = string(round(Int, x))
vi_seconds(x) = x < 100 ? string(vi_int(x), " s") : string(vi_sig(x / 60, 2), " min")
vi_interval(r) = string("[", vi_sig(r.q025, 3), ", ", vi_sig(r.q975, 3), "]")

function vi_markdown_table(header, rows)
    lines = ["| " * join(header, " | ") * " |", "|" * join(fill(":--", length(header)), "|") * "|"]
    for row in rows
        push!(lines, "| " * join(string.(row), " | ") * " |")
    end
    Markdown.parse(join(lines, "\n"))
end
