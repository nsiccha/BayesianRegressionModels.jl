module BRMMacroWeb

using HTMXObjects
using Random
using Chairmarks
using DataFrames
using FiniteDifferences: FiniteDifferences, central_fdm

# The @brm macro and the VBRMI implementation live alongside this module so
# Revise tracks them. The scripts/ entry points (parsing.jl, Benchmarking,
# StanBlocksImpl) include them via relative paths into here.
include("macro.jl")
include("vimpl.jl")

# ── Default formula + synthetic data ────────────────────────────────────────

default_formula() = """loc1 ~ 1 + a + c1 + (1 + b + c1 | g1) + (1 | g2)
log(err1) ~ 1 + d
y1 ~ Normal(loc1, err1)

log_rate ~ 1 + a + (1 | g3)
k1 ~ Poisson(exp(log_rate))

log_odds_bin ~ 1 + c2 + (1 | g2)
bin_succ ~ Binomial(bin_n, logistic(log_odds_bin))

log_odds_b ~ 1 + b
bin_y ~ Bernoulli(logistic(log_odds_b))
"""

# Synthetic data shaped roughly like the benchmarking example: continuous
# covariates plus a categorical `group` column. vimpl.jl currently materializes
# population-level terms only, so the default formula stays scalar — but the
# `group` column is kept around so users can experiment with `(1 | group)` once
# vimpl.jl supports it.
function synthetic_df(; n=64, seed=1)
    rng = Xoshiro(seed)
    # Continuous covariates
    a = randn(rng, n)
    b = randn(rng, n)
    c = randn(rng, n)
    d = randn(rng, n)
    # Grouping factors with different numbers of levels — use these on the
    # right-hand side of `(... | gN)` to test multiple random-effects blocks.
    g1 = repeat(1:8, inner=cld(n, 8))[1:n]   # 8 levels
    g2 = repeat(1:4, inner=cld(n, 4))[1:n]   # 4 levels
    g3 = rand(rng, 1:6, n)                   # 6 levels, unordered
    # Categorical predictors (treatment-coded) with small level counts.
    c1 = rand(rng, 1:3, n)
    c2 = rand(rng, 1:2, n)
    c3 = rand(rng, 1:4, n)
    # Continuous outcomes
    eta1 = 0.5 .+ 1.2 .* a .- 0.7 .* b .+ 0.3 .* c .+ 0.1 .* d
    y1 = eta1 .+ 0.3 .* randn(rng, n)
    y2 = -0.2 .+ 0.6 .* a .+ 0.4 .* b .+ 0.2 .* randn(rng, n)
    # Non-negative integer (count) outcomes — Poisson likelihoods
    k1 = rand.(rng, Distributions.Poisson.(exp.(0.5 .* eta1)))
    k2 = rand.(rng, Distributions.Poisson.(exp.(0.3 .+ 0.4 .* a)))
    # Binomial-likelihood pair: variable trial counts plus successes
    # (the `trials(size)` brms sidecar is unnecessary in our DSL — `size`
    # is just another positional argument to `Binomial`).
    bin_n = rand(rng, 5:30, n)
    bin_p_true = @. 1 / (1 + exp(-(0.2 + 0.5 * a)))
    bin_succ = [rand(rng, Distributions.Binomial(n_i, p_i))
                for (n_i, p_i) in zip(bin_n, bin_p_true)]
    # Bernoulli-likelihood column (0/1) — for hierarchical-Bernoulli models
    # like Kruschke's `therapeutic_touch`.
    bin_y = [rand(rng, Distributions.Bernoulli(p_i)) ? 1 : 0 for p_i in bin_p_true]
    DataFrame(; a, b, c, d, g1, g2, g3, c1, c2, c3,
                y1, y2, k1, k2, bin_n, bin_succ, bin_y)
end

# ── Pipeline stages ─────────────────────────────────────────────────────────
#
# Each stage is computed lazily on demand so the user can stop at any
# intermediate step (parsing, transforming, wrapping, eval'ing, materializing,
# benchmarking) and inspect the result without paying for the later stages.

const STAGES = (
    :parse,      # Meta.parse(formula)
    :transform,  # parse!(...) — rewrites = and ~ into @n/@x macro calls
    :wrap,       # _brm(formula; df) — full let-block ready to eval
    :brmi,       # eval(...) — BRMI value
    :vbrmi,      # VBRMI(brmi) — materialized action with blocks/dim
    :bench,      # Chairmarks @be primal logdensity
)

stage_index(s::Symbol) = something(findfirst(==(s), STAGES), length(STAGES))

# Run the pipeline up to (and including) `stage`. Returns a NamedTuple
# carrying every intermediate value computed so far.
function pipeline(formula::AbstractString, stage::Symbol)
    s = stage_index(stage)
    df = synthetic_df()
    out = (; df)

    s >= 1 || return out
    raw = Meta.parse("begin\n$formula\nend")
    out = merge(out, (; raw))

    s >= 2 || return out
    alllocals = OrderedDict{Symbol,Symbol}()
    transformed = parse!(deepcopy(raw); info=(;alllocals))
    out = merge(out, (; transformed, alllocals))

    s >= 3 || return out
    wrapped = _brm(formula; df)
    out = merge(out, (; wrapped))

    s >= 4 || return out
    brmi = eval(wrapped)
    out = merge(out, (; brmi))

    s >= 5 || return out
    vbrmi = VBRMI(brmi)
    dim = LogDensityProblems.dimension(vbrmi)
    x0 = randn(Xoshiro(0), dim)
    ldp = try
        string(LogDensityProblems.logdensity(vbrmi, x0))
    catch e
        "error: " * sprint(showerror, e)
    end
    grad = try
        FiniteDifferences.grad(
            central_fdm(5, 1),
            Base.Fix1(LogDensityProblems.logdensity, vbrmi),
            x0,
        )[1]
    catch e
        e
    end
    out = merge(out, (; vbrmi, dim, ldp, x0, grad))

    s >= 6 || return out
    bench = try
        @be randn(dim) LogDensityProblems.logdensity($vbrmi, _)
    catch e
        e
    end
    merge(out, (; bench))
end

# ── Rendering helpers ───────────────────────────────────────────────────────

_section(title, body) = (h.h3(title), h.pre(body))

function render_output(formula::AbstractString; stage::Symbol=:vbrmi)
    sections = Vector{Any}[]  # one entry per stage; rendered most-recent-first
    try
        out = pipeline(formula, stage)

        # Synthetic data always pinned at the top, collapsed by default so the
        # macro pipeline output stays the focus.
        data_section = Any[
            h.details(
                h.summary("Synthetic data ($(nrow(out.df)) rows × $(ncol(out.df)) cols: " *
                    join(string.(names(out.df)), ", ") * ") — click to expand"),
                render_table(out.df; sortable=false),
            ),
        ]

        if haskey(out, :raw)
            push!(sections, Any[_section("1. Meta.parse — raw Julia AST",
                sprint(show, out.raw))...])
        end
        if haskey(out, :transformed)
            push!(sections, Any[
                _section("2. parse! — rewritten AST (= → @n/@x assign, ~ → @n/@x ~)",
                    sprint(show, out.transformed))...,
                h.h3("    locals classified by parse!"),
                h.pre(sprint(show, out.alllocals)),
            ])
        end
        if haskey(out, :wrapped)
            push!(sections, Any[_section("3. _brm — full let-block (df spliced as a literal)",
                sprint(show, out.wrapped))...])
        end
        if haskey(out, :brmi)
            push!(sections, Any[_section("4. eval — BRMI value (parsed model)",
                sprint(show, out.brmi))...])
        end
        if haskey(out, :vbrmi)
            vbrmi_children = Any[
                _section("5. VBRMI — materialized action (blocks, dim, columns)",
                    sprint(show, out.vbrmi))...,
                h.h3("    logdensity at a fixed random point"),
                h.p("dim = ", string(out.dim), ", logdensity = ", out.ldp),
                h.h3("    finite-difference gradient sanity check"),
            ]
            if out.grad isa Exception
                push!(vbrmi_children,
                    h.pre("gradient error: " * sprint(showerror, out.grad)))
            else
                tol = 1e-8
                live = findall(>(tol) ∘ abs, out.grad)
                dead = findall(<=(tol) ∘ abs, out.grad)
                summary_color = isempty(dead) ? "green" : "crimson"
                push!(vbrmi_children, h.p(
                    "active params: ", h.strong("$(length(live))/$(out.dim)"),
                    " — ",
                    h.span(; style="color:$summary_color")(
                        isempty(dead) ?
                            "all parameters influence the logdensity ✓" :
                            "$(length(dead)) dead param(s) at indices $(dead)"
                    ),
                ))
                push!(vbrmi_children,
                    h.pre(sprint(show, MIME"text/plain"(), out.grad)))
            end
            push!(sections, vbrmi_children)
        end
        if haskey(out, :bench)
            bench_body = out.bench isa Exception ?
                h.pre("benchmark error: " * sprint(showerror, out.bench)) :
                h.pre(sprint(show, MIME"text/plain"(), out.bench))
            push!(sections, Any[h.h3("6. Chairmarks @be — primal logdensity"), bench_body])
        end

        # Stages render most-recent-first; synthetic data sits at the very top.
        children = reduce(vcat, reverse(sections); init=Any[])
        prepend!(children, data_section)
        return h.div(; id="brm-macro-output")(children...)
    catch e
        return h.div(; id="brm-macro-output")(
            h.h3("Error"),
            h.pre(sprint(showerror, e, catch_backtrace())),
        )
    end
end

# ── Routes ──────────────────────────────────────────────────────────────────

_stage_button(label, stage) = h.button(label; type="button",
    hx_get="/stage/$stage",
    hx_include="#brm-macro-form",
    hx_target="#brm-macro-output",
    hx_swap="outerHTML")

@htmx struct AppContext
    req = nothing

    @get index(; formula::String=default_formula()) = htmx(h.main(class="container")(
        h.h1("BRM macro action"),
        h.p(
            "Enter a ", h.code("@brm"), " formula and step through the macro pipeline: ",
            h.code("Meta.parse"), " → ", h.code("parse!"), " → ", h.code("_brm"),
            " let-block → ", h.code("eval"), " → ", h.code("VBRMI"), " action → ",
            h.code("Chairmarks"), " benchmark.",
        ),
        h.form(; id="brm-macro-form")(
            h.label("Formula")(
                h.textarea(formula;
                    name="formula", rows=8,
                    style="width:100%;font-family:monospace"),
            ),
            h.fieldset(; class="grid")(
                _stage_button("1. Parse",     :parse),
                _stage_button("2. Transform", :transform),
                _stage_button("3. Wrap",      :wrap),
                _stage_button("4. BRMI",      :brmi),
                _stage_button("5. VBRMI",     :vbrmi),
                _stage_button("6. Benchmark", :bench),
            ),
        ),
        render_output(formula),
    ); pico_version="2", extra_head=(
        h.title("BRM macro action"),
        h.style(":root { font-size: 87.5%; }"),
    ))

    @get stage(name::AbstractString; formula::String=default_formula()) =
        render_output(formula; stage=Symbol(name))
end

function __init__()
    route!(AppContext())
end

end # module
