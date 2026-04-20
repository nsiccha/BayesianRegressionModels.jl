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
    # Positive exposure column — for Poisson-with-offset patterns where the
    # log-rate gets a row-specific shift `+ log(exposure)` inside the
    # likelihood expression directly (no `offset(...)` wrapper needed).
    exposure = 0.5 .+ rand(rng, n)
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
    DataFrame(; a, b, c, d, g1, g2, g3, c1, c2, c3, exposure,
                y1, y2, k1, k2, bin_n, bin_succ, bin_y)
end

# ── Formula safety whitelist ────────────────────────────────────────────────
#
# The formula textarea accepts arbitrary text that gets `Meta.parse`d then
# `eval`'d. To prevent arbitrary code execution when the web app is shared,
# we walk the parsed AST *before* any eval and reject any function call or
# expression type that isn't on the allowlist. The allowlist is deliberately
# generous for formula-writing (math, distributions, data-column references)
# but blocks I/O, shell, eval, include, ccall, macros, etc.

const _ALLOWED_CALLS = Set{Symbol}([
    # DSL operators
    :~, :(+), :(-), :(*), :(/), :(^), :(|), :(||),
    # Comparison (may appear in ifelse-style expressions)
    :(==), :(!=), :(<), :(>), :(<=), :(>=),
    # Math
    :log, :log2, :log10, :log1p, :exp, :exp2, :expm1,
    :sqrt, :cbrt, :abs, :abs2, :sign, :floor, :ceil, :round,
    :sin, :cos, :tan, :asin, :acos, :atan,
    :min, :max, :clamp, :mod, :rem, :div,
    :logistic, :logit, :softmax, :logsumexp,
    :log_abs_tanh, :log_square_tanh,
    # Distributions (Type constructors — the pass-through handles these)
    :Normal, :Poisson, :Binomial, :Bernoulli, :Beta, :Gamma,
    :Exponential, :Cauchy, :StudentT, :LogNormal, :Weibull,
    :NegativeBinomial, :Geometric, :Laplace, :Uniform,
    :MvNormal, :MixtureModel, :Dirichlet,
    :InverseGamma, :InverseGaussian, :VonMises, :Pareto,
    :OrderedLogistic, :Categorical,
    # TODO stubs (not yet implemented but syntactically valid)
    :scale, :center, :standardize, :factor, :offset,
    :s, :bs, :t2, :gp, :ar, :ar1, :mo,
    :cbind, :mvbind, :mm, :gr, :dp, :me, :centered,
    :Horseshoe, :ZeroInflatedPoisson, :weighted,
    # Data helpers
    :length, :unique, :sort, :size, :eltype, :nrow, :ncol,
])

# Expression heads that are safe in a formula AST (literals, blocks, calls, …)
const _SAFE_HEADS = Set{Symbol}([
    :block, :call, :., :(=), :(||), :tuple, :vect, :ref,
    :kw, :parameters, :(...),
    # Comparison chains
    :comparison, :&&,
])

struct FormulaSecurityError <: Exception
    msg::String
end
Base.showerror(io::IO, e::FormulaSecurityError) = print(io, "FormulaSecurityError: ", e.msg)

function _check_formula_safety!(x)
    # Literals, symbols, line numbers — always safe
    x isa Union{Number, AbstractString, Symbol, LineNumberNode, Nothing, Bool, QuoteNode} && return
    x isa Expr || return

    # Reject dangerous expression types outright
    if x.head == :macrocall
        throw(FormulaSecurityError("macro calls are not allowed in formulas (got $(x.args[1]))"))
    elseif x.head in (:cmd, :string)
        throw(FormulaSecurityError("`\$(x.head)` expressions are not allowed in formulas"))
    elseif x.head == :quote || x.head == :$
        throw(FormulaSecurityError("quote/interpolation expressions are not allowed in formulas"))
    end

    # For :call expressions, check the function name is on the allowlist
    if x.head == :call
        fname = x.args[1]
        if fname isa Symbol && fname ∉ _ALLOWED_CALLS
            throw(FormulaSecurityError(
                "function `$fname` is not in the formula allowlist. " *
                "Allowed: arithmetic, math, distributions, DSL operators. " *
                "See _ALLOWED_CALLS in BRMMacroWeb.jl for the full list."))
        end
        # Also allow Type{...} constructors if the type name is allowed
        if fname isa Expr && fname.head == :curly
            tname = fname.args[1]
            tname isa Symbol && tname ∉ _ALLOWED_CALLS &&
                throw(FormulaSecurityError("type constructor `$tname` is not in the formula allowlist"))
        end
    end

    # Check the expression head is expected
    if x.head ∉ _SAFE_HEADS
        throw(FormulaSecurityError(
            "expression type `:$(x.head)` is not allowed in formulas"))
    end

    # Recurse into children
    for arg in x.args
        _check_formula_safety!(arg)
    end
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
    # Safety check: reject any AST node that isn't in the formula whitelist
    # before handing the expression to eval. This blocks arbitrary code
    # execution from the formula textarea.
    _check_formula_safety!(raw)
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

# ── Styled HTML rendering for BRMI / VBRMI cards ───────────────────────────
#
# Each symbol gets a deterministic color, data columns are bold, parameters
# are italic, and likelihood statements are underlined. The tree walker
# (_html_expr) converts an ExprColumn AST into a nest of <span>s.
#
# TODO: HTMX.jl should grow a generic `htmx_node(x)::Node` extension point
# so downstream packages can overload once and have every consumer dispatch
# automatically. For now these card functions are wired in by hand.

# Deterministic HSL color per symbol (golden-ratio spread for visual variety).
_symbol_color(name::Symbol) = "hsl($(mod(hash(name) * 137, 360)), 60%, 40%)"

# A colored <span> with role-based font styling.
# Data columns are normal weight; parameters (latent/sampled) are bold.
_styled_name(name::Symbol, role::Symbol) = begin
    s = "color:$(_symbol_color(name));"
    role == :parameter && (s *= "font-weight:bold;")
    h.span(string(name); style=s)
end

# ── _html_expr: recursive ExprColumn → styled HTML ─────────────────────────

_html_expr(x::NamedColumn{<:Any, <:DataColumn})    = _styled_name(name(x), :data)
_html_expr(x::NamedColumn{<:Any, MissingColumn})    = _styled_name(name(x), :parameter)
_html_expr(x::NamedColumn)                           = _styled_name(name(x), :derived)
_html_expr(x::Int)                                   = h.span(string(x); style="color:#666")
_html_expr(x::Float64)                               = h.span(string(x); style="color:#666")
_html_expr(x::Number)                                = h.span(string(x); style="color:#666")
_html_expr(x::DataColumn) = h.span("data($(eltype(parent(x))))"; style="color:#999")
_html_expr(x::MaterializedColumn) = _html_expr(getbroadcast(x))
_html_expr(x::LikelihoodColumn)   = h.span(
    _html_expr(parent(x)), h.span(" .~ "; style="color:#333"), _html_expr(rhs(x)))

# Infix operators: always parenthesized so inner expressions like (1 + b | g1)
# keep their grouping. Top-level callers (_html_brmi_row) use _html_infix
# directly to skip the outermost parens.
_html_expr(x::ExprColumn{<:Union{typeof.((~,*,+,|,doublepipe,assign))...}}) = begin
    h.span("(", _html_infix(x), ")")
end

_html_infix(x::ExprColumn) = begin
    op_str = " $(getop(x)) "
    args = getargs(x)
    parts = Any[]
    for (i, arg) in enumerate(args)
        i > 1 && push!(parts, h.span(op_str; style="color:#555"))
        push!(parts, _html_expr(arg))
    end
    h.span(parts...)
end

# Function-call style: fname(args...; kwargs...)
_html_expr(x::ExprColumn) = begin
    fname = getf(x) isa Function ? nameof(getf(x)) :
            getf(x) isa Type     ? nameof(getf(x)) : string(getf(x))
    args = getargs(x)
    kw = getkwargs(x)
    parts = Any[h.span(string(fname); style="color:#777"), "("]
    for (i, arg) in enumerate(args)
        i > 1 && push!(parts, ", ")
        push!(parts, _html_expr(arg))
    end
    if length(kw) > 0
        push!(parts, "; ")
        for (i, (k, v)) in enumerate(pairs(kw))
            i > 1 && push!(parts, ", ")
            push!(parts, "$k=", _html_expr(v))
        end
    end
    push!(parts, ")")
    h.span(parts...)
end

# Broadcasted objects (from VBRMI materialization): walk their inner structure
_html_expr(x::Base.Broadcast.Broadcasted) = begin
    fname = x.f isa Function ? nameof(x.f) :
            x.f isa Type     ? nameof(x.f) : string(x.f)
    args = x.args
    # Infix for common operators
    if x.f in (+, -, *, /)
        parts = Any[]
        for (i, arg) in enumerate(args)
            i > 1 && push!(parts, h.span(" $(x.f) "; style="color:#555"))
            push!(parts, _html_expr(arg))
        end
        return h.span(parts...)
    end
    parts = Any[h.span(string(fname); style="color:#777"), "("]
    for (i, arg) in enumerate(args)
        i > 1 && push!(parts, ", ")
        push!(parts, _html_expr(arg))
    end
    push!(parts, ")")
    h.span(parts...)
end

# Arrays / views from block parameter slots: show as a compact shape description
_html_expr(x::SubArray) = h.span("param[$(join(size(x), "×"))]";
    style="font-style:italic;color:#888")
_html_expr(x::AbstractVector{<:Number}) = h.span("vec[$(length(x))]";
    style="font-weight:bold;color:#888")
_html_expr(x::Base.RefValue) = _html_expr(x[])
_html_expr(x::AbstractMatrix) = h.span("mat[$(join(size(x), "×"))]";
    style="font-style:italic;color:#888")

# Catch-all
_html_expr(x) = h.span(sprint(show, x; context=:compact=>true); style="color:#999")

# ── BRMI card ───────────────────────────────────────────────────────────────

function brmi_card(brmi::BRMI)
    rows = [_html_brmi_row(key, parent(value)) for (key, value) in pairs(brmi.operations)]
    h.article(; style="margin:0.5rem 0")(
        h.header(h.strong("BRMI"),
            h.small(" — $(length(brmi.operations)) operations")),
        h.div(; style="font-family:monospace;font-size:1.15em;line-height:1.8;padding:0.3rem 0")(
            rows...),
    )
end

_leaf_column(x::NamedColumn) = x
_leaf_column(x::ExprColumn) = _leaf_column(getargs(x)[1])
_leaf_column(x) = x

function _html_brmi_row(key, op::ExprColumn{typeof(~)})
    lhs_leaf = _leaf_column(getargs(op)[1])
    is_likelihood = lhs_leaf isa NamedColumn && parent(lhs_leaf) isa DataColumn
    content = _html_infix(op)  # no outer parens at top level
    style = is_likelihood ?
        "text-decoration:underline;text-decoration-color:#aaa;text-underline-offset:3px;" : ""
    h.div(content; style)
end
_html_brmi_row(key, op::ExprColumn{typeof(assign)}) = h.div(_html_infix(op))
_html_brmi_row(key, op) = h.div(
    _styled_name(key, :data),
    h.span(": "; style="color:#666"),
    h.span(sprint(show, op); style="color:#999"),
)

# ── VBRMI card ──────────────────────────────────────────────────────────────

function vbrmi_card(vbrmi::VBRMI)
    brmi = getfield(vbrmi, :parent)
    (; meta) = vbrmi
    n_dim = LogDensityProblems.dimension(vbrmi)
    n_mat = length(meta.materialized)
    n_blocks = length(meta.blocks)

    # Materialized columns: show the BRMI-level symbolic expression (styled)
    # plus a compact description of the materialized value type.
    mat_rows = [begin
        nc = get(brmi.operations, key, nothing)
        inner = nc !== nothing ? parent(nc) : nothing
        if inner isa DataColumn
            # Data column: styled name + eltype
            h.div(_styled_name(key, :data),
                h.span(": data($(eltype(parent(inner))))"; style="color:#999"))
        elseif value isa LikelihoodColumn
            # Likelihood: full expression, underlined
            expr = inner !== nothing ? _html_infix(inner) : _styled_name(key, :data)
            h.div(expr; style="text-decoration:underline;text-decoration-color:#aaa;text-underline-offset:3px")
        elseif inner !== nothing
            # Materialized (sampled/derived): symbolic expression + shape
            expr = inner isa ExprColumn{<:Union{typeof(~),typeof(assign)}} ?
                _html_infix(inner) : _html_expr(inner)
            shape = "$(eltype(parent(value)))[$(length(parent(value)))]"
            h.div(expr, h.span(" → $shape"; style="color:#999;font-size:0.85em"))
        else
            h.div(_styled_name(key, :derived),
                h.span(": $(sprint(show, value))"; style="color:#999"))
        end
    end for (key, value) in pairs(meta.materialized)]

    # Blocks: each block is a tuple of parts; print the block key + one line per part.
    blocks_rows = [begin
        role = key === :__population__ ? :derived : :data
        h.div(
            _styled_name(key, role),
            [h.div(; style="color:#999;margin-left:1.5rem")(sprint(show, part))
                for part in parts]...,
        )
    end for (key, parts) in pairs(meta.blocks)]

    h.article(; style="margin:0.5rem 0")(
        h.header(h.strong("VBRMI"),
            h.small(" — dim $n_dim, $n_mat materialized, $n_blocks blocks")),
        h.h6(; style="margin-bottom:0.2rem")("materialized"),
        h.div(; style="font-family:monospace;font-size:1.15em;line-height:1.8;margin-left:1rem")(
            mat_rows...),
        h.h6(; style="margin-bottom:0.2rem")("blocks"),
        h.div(; style="font-family:monospace;font-size:1.15em;line-height:1.8;margin-left:1rem")(
            blocks_rows...),
    )
end


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
            push!(sections, Any[
                h.h3("4. eval — BRMI value (parsed model)"),
                brmi_card(out.brmi),
            ])
        end
        if haskey(out, :vbrmi)
            # Build the FD check summary for the <details> toggle line
            fd_summary = if out.grad isa Exception
                h.span("logdensity + FD check: error"; style="color:crimson")
            else
                tol = 1e-8
                n_dead = count(<=(tol) ∘ abs, out.grad)
                if n_dead == 0
                    h.span("logdensity + FD check: $(out.dim)/$(out.dim) active ✓";
                        style="color:green")
                else
                    h.span("logdensity + FD check: $(n_dead) dead param(s)";
                        style="color:crimson")
                end
            end

            fd_body = if out.grad isa Exception
                h.pre("gradient error: " * sprint(showerror, out.grad))
            else
                tol = 1e-8
                dead = findall(<=(tol) ∘ abs, out.grad)
                h.div(
                    h.p("dim = ", string(out.dim), ", logdensity = ", out.ldp),
                    isempty(dead) ? "" :
                        h.p(; style="color:crimson")(
                            "dead param indices: ", string(dead)),
                    h.pre(sprint(show, MIME"text/plain"(), out.grad)),
                )
            end

            push!(sections, Any[
                h.h3("5. VBRMI — materialized action (blocks, dim, columns)"),
                vbrmi_card(out.vbrmi),
                h.details(h.summary(fd_summary), fd_body),
            ])
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
    id="stage-$stage",
    hx_get="/stage/$stage",
    hx_include="#brm-macro-form",
    hx_target="#brm-macro-output",
    hx_swap="outerHTML")

# Pre-canned formulas. The ones above the divider exercise individual features
# in isolation; the last one stacks everything into a single multi-likelihood
# model. Click loads the formula into the textarea — user still hits "Render"
# (or any stage button) to advance the pipeline.
function presets()
    [
        "min"      => "loc ~ 1\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "linear"   => "loc ~ 1 + a\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "multi-lin" => "loc ~ 1 + a + b + c + d\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "categorical" => "loc ~ 1 + c1\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "random intercept" => "loc ~ 1 + a + (1 | g1)\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "random slope" => "loc ~ 1 + (1 + a | g1)\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "categorical random slope" => "loc ~ 1 + (1 + c1 | g1)\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "multiple groups" => "loc ~ 1 + a + (1 | g1) + (1 | g2)\nlog(err) ~ 1\ny1 ~ Normal(loc, err)\n",
        "distributional" => "loc ~ 1 + a\nlog(err) ~ 1 + b\ny1 ~ Normal(loc, err)\n",
        "Poisson" => "log_rate ~ 1 + a + (1 | g1)\nk1 ~ Poisson(exp(log_rate))\n",
        "Binomial" => "log_odds ~ 1 + a + (1 | g1)\nbin_succ ~ Binomial(bin_n, logistic(log_odds))\n",
        "Bernoulli" => "log_odds ~ 1 + a + (1 | g1)\nbin_y ~ Bernoulli(logistic(log_odds))\n",
        # Joint smoke test for Peter's verified non-Normal examples:
        # brms::cbpp_binomial → categorical + random intercept + Binomial with
        # per-row trial counts; kruschke::therapeutic_touch → hierarchical
        # Bernoulli. Both share the same grouping factor here so they hit the
        # cross-likelihood block-sharing path too.
        "cbpp + therapeutic touch" => """log_odds_bin ~ 1 + c1 + (1 | g1)
bin_succ ~ Binomial(bin_n, logistic(log_odds_bin))

log_odds_b ~ 1 + (1 | g1)
bin_y ~ Bernoulli(logistic(log_odds_b))
""",
        "everything" => default_formula(),
    ]
end

_preset_button(label, formula) = h.button(label;
    type="button",
    data_formula=formula,
    onclick="document.querySelector('textarea[name=formula]').value = this.dataset.formula; document.getElementById('stage-vbrmi').click()",
    style="font-size:0.8em;padding:0.2rem 0.5rem;margin:0")

_index_body(formula::String) = h.div(
    h.h1("BRM macro pipeline"),
    h.p(
        "Enter a ", h.code("@brm"), " formula and step through the macro pipeline: ",
        h.code("Meta.parse"), " → ", h.code("parse!"), " → ", h.code("_brm"),
        " let-block → ", h.code("eval"), " → ", h.code("VBRMI"), " action → ",
        h.code("Chairmarks"), " benchmark.",
    ),
    h.details(
        h.summary(h.small("Allowed functions in formulas")),
        h.p(h.small(
            join(sort(collect(string.(s) for s in _ALLOWED_CALLS)), ", "),
        )),
    ),
    h.form(; id="brm-macro-form")(
        h.label("Load preset"),
        h.div(; style="display:flex;flex-wrap:wrap;gap:0.3rem;margin-bottom:0.6rem")(
            [_preset_button(label, body) for (label, body) in presets()]...,
        ),
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
)

@htmx struct AppContext
    

    # HTMXObjects auto-uses `__page__` to wrap any route's return value into a
    # full page on direct browser navigation, while returning just the fragment
    # for HTMX requests (see `_resolve_response` in HTMXObjects.jl). The
    # sidebar's `hx-get` swaps target `#content` directly.
    __page__(content) = htmx(
        h.div(; style="display:flex;gap:1rem;align-items:flex-start")(
            nav_sidebar([
                "Pipeline" => "/",
                "TODO list" => "/todo",
            ]),
            h.main(; class="container", style="flex:1;min-width:0")(
                h.div(; id="content")(content),
            ),
        );
        pico_version="2",
        extra_head=(
            h.title("BRM macro action"),
            h.style(":root { font-size: 87.5%; }"),
        ),
    )

    @get index(; formula::String=default_formula(), label::String="") = begin
        # If a TODO form posted us a (label, formula) pair, persist the edited
        # formula to that TODO's .jl file so the next visit to the TODO page
        # shows the user's edits instead of the seed default.
        isempty(label) || _save_todo!(label; new_formula=formula)
        _index_body(formula)
    end

    @get mark(; label::String="", state::String="") = begin
        isempty(label) && return ""
        target = Symbol(state)
        entry = _find_todo(label)
        entry === nothing && return ""
        next_status = entry.status == target ? :open : target
        updated = _save_todo!(label; new_status=next_status)
        # Re-render the whole card so the border + collapse state update
        # together with the pill text.
        _todo_card(updated)
    end

    @get stage(name::AbstractString; formula::String=default_formula(), label::String="") = begin
        # When called from a TODO card's form, persist the (possibly edited)
        # formula back to the todo's .jl file before rendering.
        isempty(label) || _save_todo!(label; new_formula=formula)
        render_output(formula; stage=Symbol(name))
    end

    @get todo = begin
        todos = _load_todos(refresh=true)
        h.div(
            h.h1("TODO — what's missing for full BRM coverage"),
            h.p("Sorted by last modified. Each item has a sketch of what it is, why it matters, how to implement, and how to verify. Sourced from .jl files under ", h.code("web-macro/todos/"), "; status edits and edited formulas are written back to disk."),
            [_todo_card(t) for t in todos]...,
        )
    end
end

# ── TODO content (file-backed) ──────────────────────────────────────────────
#
# Each TODO is a `.jl` file under `web-macro/todos/`. File format:
#
#     # label: 1.1 verify Bernoulli/Binomial — done
#     # tier: 1
#     # status: open
#     #=
#     **Markdown body** with whatever explanation text you want.
#     =#
#
#     <raw formula text — runs through @brm when "Try in pipeline" is clicked>
#
# Header lines (`# key: value`) carry metadata. The `#= ... =#` block is the
# markdown body. Everything after the body block is the formula. The web app
# loads + parses these files on every render of the TODO page, and writes them
# back when the user toggles status or submits an edited formula. Reopening the
# server picks up exactly where the user left off — no in-memory state.

struct TodoEntry
    path::String
    label::String
    tier::Int
    status::Symbol               # :open | :done | :deprioritized
    body::String                 # markdown
    formula::Union{String,Nothing}
end

_todos_dir() = joinpath(dirname(@__DIR__), "todos")
_slug(label::AbstractString) = lowercase(strip(replace(label, r"[^\w.]+" => "-"), '-'))

const _todos_cache = Ref{Vector{TodoEntry}}()

function _load_todos(; refresh::Bool=false)
    if refresh || !isassigned(_todos_cache)
        dir = _todos_dir()
        isdir(dir) || _migrate_todos!()
        files = sort(filter(endswith(".jl"), readdir(dir; join=true)); by=mtime, rev=true)
        _todos_cache[] = TodoEntry[_parse_todo_file(f) for f in files]
    end
    _todos_cache[]
end

function _find_todo(label::AbstractString)
    for t in _load_todos()
        t.label == label && return t
    end
    nothing
end

function _parse_todo_file(path::String)
    lines = readlines(path)
    header = Dict{String,String}()
    i = 1

    # Header: leading lines matching `# key: value`. Stop at the first
    # non-matching line. Line-based avoids any UTF-8 byte-index footguns
    # (labels routinely contain multi-byte characters like `✓`).
    while i <= length(lines)
        m = match(r"^# (\w+):\s*(.*)$", lines[i])
        m === nothing && break
        header[m[1]] = m[2]
        i += 1
    end

    # Body: optional `#= ... =#` block. Both delimiters live on their own
    # lines (the writer guarantees this), so a simple line scan suffices.
    body_lines = String[]
    if i <= length(lines) && strip(lines[i]) == "#="
        i += 1
        while i <= length(lines) && strip(lines[i]) != "=#"
            push!(body_lines, lines[i])
            i += 1
        end
        i <= length(lines) && (i += 1)  # consume `=#`
    end
    body = join(body_lines, '\n')

    # Formula: everything that's left, stripped.
    formula_text = strip(join(lines[i:end], '\n'))
    formula = isempty(formula_text) ? nothing : String(formula_text)

    label = get(header, "label", basename(path))
    tier = parse(Int, get(header, "tier", "1"))
    status = Symbol(get(header, "status", "open"))
    TodoEntry(path, label, tier, status, body, formula)
end

function _write_todo_file(todo::TodoEntry)
    io = IOBuffer()
    println(io, "# label: ", todo.label)
    println(io, "# tier: ", todo.tier)
    println(io, "# status: ", todo.status)
    if !isempty(todo.body)
        println(io, "#=")
        println(io, todo.body)
        println(io, "=#")
    end
    if todo.formula !== nothing && !isempty(todo.formula)
        println(io)
        print(io, todo.formula)
        endswith(todo.formula, "\n") || println(io)
    end
    write(todo.path, take!(io))
end

function _save_todo!(label::AbstractString;
                     new_status::Union{Symbol,Nothing}=nothing,
                     new_formula::Union{String,Nothing}=nothing)
    todo = _find_todo(label)
    todo === nothing && return nothing
    updated = TodoEntry(
        todo.path, todo.label, todo.tier,
        something(new_status, todo.status),
        todo.body,
        new_formula === nothing ? todo.formula : new_formula,
    )
    _write_todo_file(updated)
    _load_todos(refresh=true)
    updated
end

# One-shot migration: dumps the in-source `_tier1()`/`_tier2()`/`_tier3()`
# seed lists to files on first run. Skips files that already exist, so user
# edits to existing files survive. Once everything is migrated, the in-source
# seed functions are dead code that can eventually be deleted.
function _migrate_todos!()
    dir = _todos_dir()
    isdir(dir) || mkpath(dir)
    for (tier, items) in [(1, _tier1()), (2, _tier2()), (3, _tier3())]
        for item in items
            label, body, formula = item isa Pair ?
                (first(item), last(item), nothing) :
                (item.label, item.body, item.formula)
            slug = _slug(label)
            path = joinpath(dir, slug * ".jl")
            isfile(path) && continue
            _write_todo_file(TodoEntry(path, label, tier, :open, body, formula))
        end
    end
end

# ── Rendering: one Pico CSS article per TODO with status-colored border ────

const _TIER_LABELS = (
    "T1",  # tier 1
    "T2",  # tier 2
    "T3",  # tier 3
)
const _TIER_COLORS = ("#4a7c59", "#5a6a8c", "#8c5a5a")

_tier_pill(tier::Int) = h.span(
    get(_TIER_LABELS, tier, "T$tier");
    style="font-size:0.7em;padding:0.1rem 0.4rem;border-radius:1rem;" *
          "color:white;background:$(get(_TIER_COLORS, tier, "#888"));" *
          "vertical-align:middle;font-weight:normal",
)

const _STATUS_COLORS = (
    open = "#888",
    done = "#2e7d32",
    deprioritized = "#a05a2c",
)
_status_color(s::Symbol) = get(_STATUS_COLORS, s, "#888")

function _todo_card(todo::TodoEntry)
    border_color = _status_color(todo.status)
    body_children = Any[HTMXObjects.md_to_node(todo.body)]
    if todo.formula !== nothing
        push!(body_children, _formula_form(todo.label, todo.formula))
        # Inline pipeline-result target — the form's hx_get fills this div
        # with `render_output(formula; stage=:vbrmi)` so the user sees the
        # VBRMI/finite-difference output right inside the card.
        push!(body_children, h.div(; id="todo-result-$(hash(todo.label))",
                                     style="margin-top:0.5rem"))
    end
    # `:open` status → expanded; `:done`/`:deprioritized` → collapsed by default.
    # Pills sit inside the <summary> so they're always reachable, but their
    # onclick stops propagation so clicking a pill doesn't also toggle the
    # disclosure.
    h.article(;
        id="todo-card-$(hash(todo.label))",
        style="border-left:6px solid $border_color;margin:0.8rem 0;padding:0.5rem 1rem",
    )(
        h.details(; open=todo.status == :open)(
            h.summary(; style="cursor:pointer;list-style-position:outside")(
                _tier_pill(todo.tier), " ",
                h.strong(todo.label), " ",
                _status_pills(todo.label, todo.status),
            ),
            h.div(; style="margin-top:0.5rem")(body_children...),
        ),
    )
end

function _formula_form(label::String, formula::String)
    h.form(;
        hx_get="/stage/vbrmi",
        hx_target="#todo-result-$(hash(label))",
        hx_swap="innerHTML",
        style="margin:0.5rem 0",
    )(
        h.input(; type="hidden", name="label", value=label),
        h.textarea(formula;
            name="formula",
            rows=max(3, count('\n', formula) + 1),
            style="width:100%;font-family:monospace;font-size:0.85em"),
        h.button("Try in pipeline ▶";
            type="submit",
            style="font-size:0.85em;padding:0.3rem 0.8rem;margin:0.3rem 0 0 0"),
    )
end

# Mutually-exclusive status pills wrapped in a single span so one toggle
# replaces both at once via `outerHTML` targeting `#status-{hash(label)}`.
function _status_pills(label::AbstractString, state::Symbol)
    h.span(; id="status-$(hash(label))",
              style="margin-left:0.5rem;display:inline-flex;gap:0.3rem;vertical-align:middle")(
        _state_pill(label, :done, state, "✓ done", "mark done"),
        _state_pill(label, :deprioritized, state, "✓ deprioritized", "deprioritize"),
    )
end

# Convenience overload that fetches the current status from disk.
_status_pills(label::AbstractString) = _status_pills(label,
    something(_find_todo(label), (status=:open,)).status)

# Pill text and color reflect the *current* state. When the pill's target_state
# is currently active, it shows the active label (e.g. "✓ done") in the active
# color and clicking it toggles back to :open. Otherwise it shows the inactive
# action label (e.g. "mark done") in gray and clicking it sets the target state.
function _state_pill(label, target_state, current_state, active_text, inactive_text)
    is_active = current_state == target_state
    text = is_active ? active_text : inactive_text
    bg = is_active ? _status_color(target_state) : "#888"
    h.button(text;
        type="button",
        hx_get="/mark?label=$(HTTP.URIs.escapeuri(label))&state=$(target_state)",
        hx_target="#todo-card-$(hash(label))",
        hx_swap="outerHTML",
        onclick="event.stopPropagation()",
        style="font-size:0.7em;padding:0.15rem 0.6rem;border:none;border-radius:1rem;" *
              "color:white;background:$bg;cursor:pointer",
    )
end

_tier1() = [
    (label="1.1 verify Bernoulli/Binomial",
     body=raw"""
**Status: done.** ✓ Confirmed that the existing `FBroadcasted{<:Type{<:Distribution}}` pass-through in `vimpl.jl` handles both Bernoulli and Binomial cleanly.

**Verification.** The form below loads the **cbpp + therapeutic touch** model — a faithful translation of `brms::cbpp_binomial` (categorical predictor + random intercept + Binomial with per-row trial counts) and `kruschke::therapeutic_touch` (hierarchical Bernoulli) into one multi-likelihood model. brms's `incidence | trials(size) ~ ...` sidecar collapses to a plain positional argument: `bin_succ ~ Binomial(bin_n, logistic(η))`. If the gradient sanity check stays green every other `Distribution` family (Beta, Gamma, NegBinomial, …) should be free as well.
""",
     formula="""log_odds_bin ~ 1 + c1 + (1 | g1)
bin_succ ~ Binomial(bin_n, logistic(log_odds_bin))

log_odds_b ~ 1 + (1 | g1)
bin_y ~ Bernoulli(logistic(log_odds_b))
"""),

    (label="1.2 offset / fixed exposure — already works without a wrapper",
     body=raw"""
**Status: already works without any new code.** brms needs `offset(z)` because R's formula syntax has no other way to put a "no-coefficient term" into the linear predictor — the only thing on the RHS of `~` is the formula DSL. In our DSL the linear predictor and the likelihood are *separate* `~` lines, and the second one (the likelihood) takes a free-form Julia expression. Anything inside that expression gets evaluated as plain code at materialization time via `vbroadcasted` — function calls dispatch to whatever Julia function the symbol resolves to, and data column references are pulled from the dataframe.

So instead of `count ~ x + offset(log(exposure))`, you write the offset directly inside the likelihood:

```julia
loc ~ 1 + a
k1 ~ Poisson(exp(loc + log(exposure)))
```

The `log(exposure)` here is just `Base.log` applied to the `exposure` data column, broadcasted across rows and added to `loc` (which is the materialized linear predictor). No parameter is allocated for it because `growblock!!` is never called for that branch — there's no `~` on the data side, just an argument to `Poisson(...)`.

**Verification.** Form below loads exactly that model. The VBRMI dim should match the offset-free version (only the population intercept + slope on `a`); the gradient sanity check should stay green; and the materialized `k1` likelihood should incorporate the row-specific exposure shift.
""",
     formula="""loc ~ 1 + a
k1 ~ Poisson(exp(loc + log(exposure)))
"""),

    (label="1.3 I(expr) — likely already works",
     body=raw"""
**What it is.** brms's `I()` is a literal-escape: `I(x^2)` says "compute `x^2` from the data and treat it as a single column". brms needs it because `+`, `*`, `:`, `|`, … all have special meaning inside an R formula.

**Why we probably don't need it.** Our DSL is parsed by Julia first, then walked by `_x`. `_x` recursively wraps every `Expr(:call, f, args...)` in an `ExprColumn`, regardless of whether `f` is special. So `loc ~ a + x^2` becomes `+(a, ^(x, 2))` → `ExprColumn(+, NamedColumn(:a), ExprColumn(^, NamedColumn(:x), 2))`. The `^` is just another function call, no special handling needed.

The only operators that have DSL meaning in our system are `~` (sampling), `=` (assignment), and `|` / `||` inside random-effects specs. Everything else (`^`, `/`, `sqrt`, `log`, `exp`, `mod`, `min`, `max`, …) is a regular function call resolved at materialization time via `vbroadcasted`.

**Verification.** The form below loads a model with three nonlinear terms (`a^2`, `sqrt(abs(b))`, `log(exposure)`) directly as population-level covariates. The VBRMI dim should match the number of distinct terms; the gradient sanity check should be all-active. If it works, that confirms `I()` is unnecessary because Julia function calls are first-class on the formula RHS.
""",
     formula="""loc ~ 1 + a + a^2 + sqrt(abs(b)) + log(exposure)
y1 ~ Normal(loc, 1)
"""),

    "1.4 scale(x) / standardize(x)" => raw"""
**What it is.** brms's `scale(x)` z-transforms a column at parse time: `scale(x) = (x - mean(x)) / std(x)`. The model sees the standardized column. Crucial for default priors (which are scale-invariant only after standardization) and sampler stability (well-conditioned linear predictors).

**Why it matters.** Most brms vignettes do `scale(x)` automatically as a convenience. Without it, every formula has to either manually z-transform the data or accept poorly-scaled coefficients.

**Implementation.**
1. Add `function scale end` (and `function center end`, `function standardize end`) to `macro.jl`.
2. Add a `vmeta_sampling_rhs` overload in `vimpl.jl`:
```julia
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(scale)}; group) = begin
    inner = vbroadcasted(only(getargs(x)); meta)
    materialized = Base.materialize(inner)
    z = (materialized .- Statistics.mean(materialized)) ./ Statistics.std(materialized)
    vmeta_sampling_rhs(meta, z; group)
end
```
The standardization happens once when the BRMI is materialized into a VBRMI. Composes with the existing dense-map caching TODO.
3. Add `Statistics` to `vimpl.jl`'s using-list (or vendor `mean`/`std` inline).

**Verification.** Preset: `loc ~ 1 + scale(a) + scale(b); y1 ~ Normal(loc, 1)`. Compare against the unscaled version: same dim, different posterior geometry. The fitted coefficients should be ≈ the unscaled coefficients × std(x).
""",
    "1.5 zerocorr — independent random effects" => raw"""
**What it is.** brms (via lme4 syntax) lets you opt out of the LKJ correlation between multiple random terms in the same group. `(1 + x || group)` (double bar) says "estimate the random intercept and the random slope independently — don't fit a 2×2 Cholesky factor between them". Useful when there isn't enough data to estimate the correlations, or when you have prior reason to believe the terms are uncorrelated.

**Why it matters.** Multi-term random specs are common, and the LKJ correlation often dominates the prior cost without much identifiability. Letting users skip it is a meaningful sampling speedup and prior simplification.

**Implementation.** Our `_x` walker already wraps `||` as `ExprColumn{typeof(doublepipe)}`. Add a `vmeta_sampling_rhs` overload that splits each term inside the `||` LHS into its own block (with a synthetic per-term key like `Symbol(group_name, :__nocor__, term_index)`):

```julia
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(doublepipe)}; kwargs...) = begin
    lhs, rhs = getargs(x, 2)
    terms = lhs isa ExprColumn{typeof(+)} ? getargs(lhs) : (lhs,)
    foldl(enumerate(terms); init=(meta, ())) do (m, args), (i, term)
        nocor_key = NamedColumn(Symbol(name(rhs), :__nocor__, i), parent(rhs))
        m, arg = vmeta_sampling_rhs(m, term; group=nocor_key)
        m, (args..., arg)
    end |> ((m, args),) -> (m, Base.broadcasted(+, args...))
end
```

Each per-term block ends up as 1×1 with one `log_scale` Cholesky parameter — `lprior!`'s existing single-column path handles this with no changes.

**Verification.** Preset: `loc ~ 1 + (1 + a || g1); y1 ~ Normal(loc, 1)`. Compare its dim against the correlated `(1 + a | g1)` version: the correlated version has 3 Cholesky params (1+2/2 for a 2×2), the uncorrelated version has 2 (one log_scale per term). Same direct-parameter count (2 cols × 8 levels = 16) either way.
""",
    "1.6 cache levels / level_map / dense / gc_idx" => raw"""
**What it is.** Stop rebuilding the dense level mapping (`Dict(level => row_index)`) and the gc_idx vector on every `VBRMI(brmi)` call. Cache them once per source data column.

**Why it matters.** Today every `VBRMI` build re-traces the categorical / grouping columns, sorts unique values, builds a Dict, and walks the column to dense-encode it. For models with many categorical columns or many `VBRMI` rebuilds (e.g. during AD), this adds up.

**Implementation.** Pick a storage layout for the per-column metadata. Two candidates:
- A new `meta.factor` NamedTuple keyed by source column name, holding `(; levels, level_map, dense, gc_idx)` per column. Built lazily on first reference, indexed via `name(column)`.
- Attach the metadata to `meta.materialized[column_name]` directly. More tightly coupled but avoids a parallel NamedTuple.

The TODO already lives at `vimpl.jl:78–86`. Once a layout is picked, refactor `_gc_idx` and the inline dense map in the categorical path to read from the cache, falling back to a build-on-miss helper.

**Verification.** No behavioral change — the gradient sanity check should stay green. Benchmark `VBRMI(brmi)` with Chairmarks before/after and confirm a measurable speedup on a model with multiple categorical/grouping columns.
""",
    "1.7 CategoricalArrays / PooledArrays integration" => raw"""
**What it is.** When the input column is already a `CategoricalVector` or a `PooledArray`, the dense level mapping is already computed and stored in the column's `.refs` field. Use it directly instead of rebuilding via `Dict`.

**Why it matters.** Most real-world DataFrames use `CategoricalArrays.jl` for factor columns. Skipping the rebuild eliminates allocation entirely for the common case and gets us "for free" interop with the standard categorical-data ecosystem.

**Implementation.** Two design choices:
- **Hard dep**: add `CategoricalArrays` to vimpl.jl's deps, dispatch on `CategoricalVector`, read `levelcode.(col)` and `levels(col)` directly.
- **Duck-typed**: sniff for the `.refs` field and `levels` method without importing the package, falling back to the generic Dict path.

Recommend hard dep — it's the standard for tabular Julia code, and the duck-type path is more code with no real win. Same for `PooledArrays`.

The actual integration is small once the design is picked: a method specialization in `_gc_idx` and in the categorical-predictor path. Composes with the caching TODO above.

**Verification.** Preset (or test) that builds a DataFrame with a `CategoricalVector` column and uses it as a grouping factor / categorical predictor. Confirm the gradient sanity check stays green and the per-VBRMI allocation count drops.
""",
]

_tier2() = [
    "2.1 interactions a:b, a*b" => raw"""
**What it is.** brms's `a:b` is the elementwise interaction term (a single coefficient multiplying `a[i] * b[i]`). `a*b` is the "main effects + interaction" shorthand: it desugars to `a + b + a:b`.

**Why it matters.** Interactions are the most commonly missed feature in regression DSLs. Without them, every model that needs `a:b` has to manually create the interaction column in the input DataFrame.

**Implementation.**
1. **Parser side.** Add a `:` case to `_x` so that `a:b` becomes `ExprColumn(:, NamedColumn(:a), NamedColumn(:b))` instead of falling through to a Symbol/Range parse error.
2. **Materialization side.** Add `vmeta_sampling_rhs(meta, x::ExprColumn{typeof(:)}; group)` that elementwise-multiplies the operands and dispatches to the float-vector path. For continuous × continuous it's a single coefficient on `a .* b`; for categorical × continuous it's `(k-1)` coefficients (one per non-reference level of the categorical, multiplied by the continuous); for categorical × categorical it's `(k₁-1)*(k₂-1)` coefficients via a 2D `_cat_lookup`.
3. **`a*b` desugaring.** At parse time in `_x`, rewrite `*` between formula terms as `+(a, b, :(a:b))`. This needs care because `*` also means multiplication elsewhere (e.g. `Normal(0, 2*sigma)`); the rewrite should only apply at formula-RHS top-level.

**Verification.** Presets exercising each interaction type:
- continuous×continuous: `loc ~ 1 + a + b + a:b; y1 ~ Normal(loc, 1)` → dim 4
- continuous×categorical: `loc ~ 1 + a + c1 + a:c1; y1 ~ Normal(loc, 1)` → dim 6 (1 + 1 + 2 + 2)
- categorical×categorical: `loc ~ 1 + c1 + c2 + c1:c2; y1 ~ Normal(loc, 1)` → dim 5 (1 + 2 + 1 + 2)
- shorthand: `loc ~ 1 + a*b; y1 ~ Normal(loc, 1)` should match `loc ~ 1 + a + b + a:b` exactly.
""",
    "2.2 configurable categorical reference level" => raw"""
**What it is.** Currently the reference level for treatment-coded categoricals is `sort(unique(x))[1]`. brms / lme4 let you override this via `factor(x, ref="some_level")` or by reordering the factor's levels.

**Why it matters.** The reference level changes the interpretation of the intercept (it becomes "the mean for the reference level") and of the coefficients (each becomes "the difference from reference"). For some analyses, changing the reference is the only way to make the coefficients directly answer the research question.

**Implementation.**
1. Add `function factor end` to `macro.jl`.
2. Either store the override at parse time (rewrite `factor(x, ref=:level3)` into a wrapper that the materializer recognizes) or at materialization time via a `meta.factor_ref` NamedTuple keyed by column name.
3. The categorical-predictor path's `levels = sort(unique(x))` becomes `levels = sort(unique(x), by=l -> l == ref ? -Inf : l)` so the chosen reference always sorts first.

**Verification.** Preset: `loc ~ 1 + factor(c1, ref=2); y1 ~ Normal(loc, 1)`. Compare the fitted coefficients against the default-reference version — they should differ by the level-2-vs-level-1 mean shift but produce the same logdensity.
""",
    "2.3 per-parameter prior scales" => raw"""
**What it is.** Currently every parameter is `Normal(0, 1)` in `lprior!`. brms / Stan-style models routinely set custom priors per coefficient: `b ~ Normal(0, 0.5)` for tight priors on slopes, `b ~ Cauchy(0, 1)` for heavy-tailed priors, etc.

**Why it matters.** Default `Normal(0, 1)` is fine after standardization but poor on raw scales. Allowing per-parameter prior scales is the prerequisite for spike-and-slab, Horseshoe, and most prior sensitivity analyses. Without it, users have no way to express domain knowledge about parameter magnitudes.

**Implementation.** This is the largest design decision in Tier 2 because it has knock-on effects for every other prior-related TODO.

Two storage candidates:
- **Per-block scales**: extend `meta.block_data[group]` with a per-column scale vector. `lprior!` multiplies the standard-normal draw by the scale before storing. Simple but only handles Normal-with-scale priors.
- **Per-block prior distributions**: store a vector of `Distribution` objects per block. `lprior!` calls `logpdf(prior_i, xi)` for each parameter. More general; handles Cauchy, StudentT, Horseshoe, etc.

Recommend the second — it's strictly more powerful and the runtime cost is identical (one `logpdf` call per parameter). Default value is `Normal(0, 1)` for backward compatibility.

**Implementation sketch.**
1. Extend `meta.block_data` with a `priors` field per block.
2. The macro syntax `b ~ Normal(0, 0.5)` parses as a sampling statement with a Distribution-typed RHS. Currently this is reserved for likelihood declarations; it would need a new "is this a prior or a likelihood?" branch in `vmeta_sampling`. Likelihood: LHS is a data column. Prior: LHS is a maybelocal (parameter).
3. `lprior!` reads the per-column prior and calls `logpdf(prior, value)` instead of the hard-coded `logpdf(Normal(), value)`.

**Verification.** Preset: `loc ~ 1 + a; b ~ Normal(0, 0.1); y1 ~ Normal(loc, 1)` — confirm the gradient is dampened on `b` compared to the default-prior version, and that the dead-param check still passes.
""",
    "2.4 centered / non-centered parameterization toggle" => raw"""
**What it is.** Currently every random-effect block uses non-centered parameterization (we sample standard normals and apply `mul!(vi, C.L, xi)`). brms / Stan let you choose centered (sample directly from `Normal(0, σ)` per group) on a per-factor basis.

**Why it matters.** Non-centered is the default for "weak data per group" cases (Neal's funnel pathology), but for "strong data per group" cases centered samples better. Letting users choose is a meaningful sampling speedup for the latter regime.

**Implementation.** Small change to `lprior!` and `growblock!!`. Add a `centered::Bool` flag to `meta.block_data[group]`. In `lprior!`'s non-population branch, if the block is centered, sample directly from `Normal(0, exp(log_scale))` instead of `Normal(0, 1)` then multiplying by `L`. The Cholesky machinery for off-diagonal correlations still applies in the centered case — just on the column before the variance scaling rather than after.

Once (2.3) is in place, the centered/non-centered choice could be encoded as `(1 | g) ~ Normal(0, σ)` (centered) vs the implicit non-centered default — but for now a per-block kwarg or a wrapper function (e.g. `centered((1 + a | g))`) is simpler.

**Verification.** Same model with both parameterizations should produce the same logdensity at the same parameter values (after the appropriate change of variables). Sampling efficiency on a known-funnel dataset should differ.
""",
    "2.5 grouped random effects (per-factor variance)" => raw"""
**What it is.** Peter's "different variance by diagnosis" pattern: `(1 | subject) gr(diagnosis)` says "the random intercept by subject has a different variance per diagnosis level". In brms this is a custom group structure where the variance hyperparameter itself depends on a second factor.

**Why it matters.** Common in clinical data where treatment groups have intrinsically different between-subject variability. Without this, you have to fit separate models per diagnosis or accept a single pooled variance.

**Implementation.** Bigger than it looks because the variance is no longer a single scalar but a length-`n_levels(diagnosis)` vector that needs its own prior and its own gradient.

Proposed shape:
- A new `gr(group_factor)` wrapper recognized in the `~` RHS via a `function gr end` stub (already exists in `macro.jl`).
- The wrapped block stores `n_levels(group_factor)` log-scale parameters instead of one. `lprior!` walks them, multiplying each subject's random intercept by the diagnosis-specific scale.
- Requires the gc_idx for the inner factor (subject) AND for the outer factor (diagnosis) — both vectors of length N.

This composes naturally with (1.6 caching) and (2.3 per-parameter priors).

**Verification.** Preset against synthetic data with two grouping factors, one nested inside the other, with intentionally different per-outer-level variance. Confirm the fitted scales recover the synthetic values.
""",
    "2.6 multi-membership random effects mm()" => raw"""
**What it is.** brms's `mm(g1, g2, ...)` lets one observation belong to **multiple** levels of the same random factor simultaneously, with weights summing to 1. Standard use: a student belongs to multiple schools across the year, and we want their random effect to be a weighted average of the per-school effects.

**Why it matters.** Standard random effects assume each observation belongs to exactly one group. Multi-membership is the only clean way to handle observations that span groups (mobile students, patients seen by multiple clinicians, etc.).

**Implementation.** `_gc_idx` would have to return a row-of-vectors instead of a single Int per row. Two paths:
- **Sparse design matrix**: replace the `gc_idx` lookup with a sparse `(N × n_levels)` matrix where each row's nonzero entries are the membership weights. The materialized random effect becomes `sparse_membership * random_effects_vector`.
- **Per-row lookup loop**: keep the row-major view but make `_re_lookup` iterate the membership list per row, summing weighted contributions.

The sparse matrix approach is more memory-efficient and SIMD-friendly. Needs a new wrapper in the formula syntax: `(1 | mm(g1, g2; weights=...))`.

**Verification.** Preset against synthetic data where each observation has 2 random group memberships with weights summing to 1. Compare against the equivalent "fully observed in primary group only" model.
""",
    "2.7 se() / weights() for meta-analysis and weighted regression" => raw"""
**What it is.** brms's `y | se(sigma_y) ~ ...` lets each observation have its own known standard error (typical for meta-analysis where each `y` is itself a summary estimate). `y | weights(w) ~ ...` is observation-level weighting (typical for survey data or sample-size correction).

**Why it matters.** Both are extremely common in applied work. Without them, meta-analysis can't be expressed at all in this DSL, and weighted regression has to be hacked via likelihood multiplication.

**Implementation.** Both are sidecar modifiers on the LHS of `~`, so they need parser support similar to brms's `|` syntax. Or, more naturally for our DSL: pass them as positional arguments to the distribution itself.
- `y ~ Normal(mu, se_y)` for the meta-analysis case (already works! `se_y` is just another data column).
- For weights, define a `weighted` likelihood wrapper: `y ~ weighted(Normal(mu, sigma), w)` where `weights` multiplies the per-row logpdf by `w[i]`. Needs a new `vmeta_sampling_rhs` overload and a new `LikelihoodColumn`-like type with a per-row weight.

Meta-analysis is essentially free (already works). Weights need ~15 lines.

**Verification.** Preset for meta-analysis: `y ~ Normal(mu, se_y)` with `se_y` from synthetic data. Preset for weights: `y ~ weighted(Normal(mu, 1), weight); loc ~ 1 + a` confirming the gradient is rescaled per-row by `weight`.
""",
]

_tier3() = [
    "3.1 multivariate outcomes cbind(y1, y2)" => raw"""
**What it is.** brms's `cbind(y1, y2) ~ x + (1 | g)` declares that `y1` and `y2` share the same linear predictor structure but have correlated residuals. The likelihood becomes multivariate normal (or multivariate-t) over `(y1, y2)` with a covariance matrix to estimate.

**Why it matters.** Unblocks `mcelreath::waffle_divorce_multivariate` and any model where multiple outcomes share latent structure (joint pharmacology/efficacy, paired outcomes, mediation analysis).

**Implementation.** Real new infrastructure:
1. New parser support for `cbind(...)` on the LHS of `~`.
2. New `LikelihoodColumn`-like type that holds a tuple of data columns and a multivariate distribution.
3. `llikelihood!` calls `logpdf(MvNormal(loc_vec, Σ), [y1[i], y2[i]])` per row.
4. `Σ` is a new parameter block: a Cholesky factor over the outcomes (separate from the random-effects Cholesky).

**Verification.** Translate `mcelreath::waffle_divorce_multivariate` directly. Compare fitted parameters against the published reference.
""",
    "3.2 inferred predictors / measurement error me()" => raw"""
**What it is.** brms's `me(x_obs, sd_x)` says "the predictor `x_obs` is itself measured with error of size `sd_x`; sample the latent true value during inference". The model sees both the observed value and the latent.

**Why it matters.** Standard regression treats predictors as fixed/known. When predictors are themselves estimates (e.g. from a previous study or a noisy sensor), ignoring measurement error biases the slope estimates toward zero. `me()` is the principled fix.

**Implementation.** Bigger architectural change: predictor columns become latent variables sampled during inference, not data columns evaluated once. Needs:
- A new column type analogous to `MissingColumn` but with an observation-driven prior `Normal(x_obs, sd_x)`.
- The latent column gets a slot in the population block (one parameter per row).
- `lprior!` adds the per-row Normal prior contribution.
- `vbroadcasted` resolves the column to the latent values, not the observed ones.

**Verification.** Preset against synthetic data where the true `x` is known but only a noisy observed version is in the dataframe. Compare slope estimates with and without `me()`.
""",
    "3.3 ordinal predictors mo() (monotonic effects)" => raw"""
**What it is.** brms's `mo(x)` for an ordinal predictor with K levels: instead of `K-1` independent treatment-coded coefficients, fit a single "total effect" β plus a `K-1`-dim simplex of inter-level shape. Forces the effect to be monotonic in the ordering of `x`'s levels.

**Why it matters.** Likert-scale predictors and ordered categorical inputs (e.g. age groups) have a natural ordering that treatment coding ignores. `mo()` enforces the monotonicity prior, dramatically reducing the parameter count and tightening posterior inference.

**Implementation.** New block layout: one β coefficient + one Dirichlet-distributed simplex of length `K-1`. `_cat_lookup`-style materialization but the per-level contribution is `β * cumulative_sum(simplex)[level]` instead of `coefficients[level]`.

Needs a new prior block type (Dirichlet) in `lprior!`, plus new parser support for `mo(x)` and a new `vmeta_sampling_rhs` overload.

**Verification.** Preset against synthetic data where the true effect is monotonic but the levels are unordered in the data. Compare fitted shape parameters against the synthetic monotonic curve.
""",
    "3.4 ordinal outcomes (proportional odds)" => raw"""
**What it is.** When `y` is itself ordered categorical (Likert response, severity grades, …), use a cumulative-link model: `Pr(y ≤ k) = logistic(α_k - η)` where `α_k` are K-1 cutpoints and η is the linear predictor. The likelihood is the difference of consecutive CDFs.

**Why it matters.** Ordinal outcomes are common in survey data, clinical scoring, and any "rating" task. Treating them as continuous is statistically wrong; treating them as nominal categorical loses the ordering information.

**Implementation.** New likelihood family with a vector of cutpoints as additional parameters. `Distributions.jl` has `OrderedLogistic` already — the pass-through path should mostly handle it once the parser knows to extract cutpoints from a `cumulative` wrapper.

The cutpoints need an ordered prior (e.g. ordered transform of unconstrained reals), which means a new prior block type — similar to (3.3)'s simplex.

**Verification.** Preset against synthetic Likert data. Compare cutpoints against `polr` from R's MASS package.
""",
    "3.5 mixture models" => raw"""
**What it is.** brms's `mixture(Normal, Normal)` lets the likelihood be a weighted mixture of K component distributions, with mixing weights estimated as parameters.

**Why it matters.** Heterogeneous populations, latent class analysis, robust regression (Normal + heavy-tailed component), zero-inflated outcomes, …

**Implementation.** Extends the existing `Distribution` pass-through. Need a new `MixtureModel` wrapper that holds component distributions plus a weights parameter block. `llikelihood!` uses `logsumexp(log_weights .+ logpdf.(components, y))` per row.

Composes with (3.4 ordered priors) for the mixing weights' Dirichlet-like prior.

**Verification.** Preset against synthetic two-component-Normal data. Confirm the recovered mixture weights and component parameters.
""",
    "3.6 splines / GP submodels s(), bs(), gp(), t2()" => raw"""
**What it is.** Smoothers in the linear predictor: `s(x)` for a generic spline, `bs(x, knots=...)` for a B-spline basis, `gp(x)` for a Gaussian process, `t2(x, y)` for a tensor-product spline. brms / mgcv use these heavily.

**Why it matters.** Nonlinear effects without committing to a specific functional form. The de facto way to model dose-response curves, time effects, growth curves, spatial trends, …

**Implementation.** Each smoother is a basis-matrix builder that grows a population block by `n_basis` columns and stores the basis matrix as part of `meta`. Function stubs (`s`, `bs`, `t2`, `gp`) already exist in `scripts/parsing.jl` so the parser side is partly done.

For each smoother type, the materialization is `basis_matrix * coefficients` (length-N output). The smoothness prior is a structured prior on the coefficients (typically a Gaussian prior with a banded or 2D-difference penalty matrix), which requires (2.3 per-parameter priors) as a prerequisite.

**Verification.** Preset against synthetic curve data. Compare fitted smoother against `mgcv::gam`.
""",
    "3.7 autoregressive submodels ar(), ar1()" => raw"""
**What it is.** Add an AR(p) structure to the residuals: `y_t = η_t + φ * (y_{t-1} - η_{t-1}) + ε_t`. brms's `ar(time, p=1)` specifies the order and the time variable.

**Why it matters.** Time-series and repeated-measures data routinely have autocorrelated errors. Ignoring AR structure inflates the effective sample size and gives overconfident posteriors.

**Implementation.** Bigger than it looks because the likelihood is no longer per-row independent — it's a chain. Need a new `LikelihoodColumn`-like type that holds the time index and walks the data in time order, accumulating the AR contribution row by row.

Composes with (2.5 grouped random effects) for per-subject AR structure.

**Verification.** Preset against synthetic AR(1) data. Recover φ.
""",
    "3.8 decompositions (QR, orthogonal polar)" => raw"""
**What it is.** Numerical-stability transformations of the population design matrix. brms uses QR decomposition on the design matrix internally so the sampler sees an orthogonal-columns version, then transforms back at the end. Stan does the same.

**Why it matters.** When population covariates are correlated (which is the norm), the unrotated design matrix gives a poorly-conditioned posterior that NUTS struggles with. QR fixes this with no statistical change.

**Implementation.** Mostly orthogonal to the formula DSL — happens at `VBRMI` build time. Add a per-block transform: store both the original design matrix and the QR factor, run sampling on the rotated parameter space, transform back when extracting coefficients.

Could be implemented today without any parser changes, as a `qr_transform=true` flag on `VBRMI`. Lift to a default-on once verified.

**Verification.** Same model with and without QR should produce identical logdensity values, but the gradient should be better-conditioned (smaller condition number on the Hessian).
""",
    "3.9 spike-and-slab / Horseshoe priors" => raw"""
**What it is.** Sparsity-inducing priors for high-dimensional regression. Spike-and-slab puts a delta-spike at zero plus a wide slab; Horseshoe uses a half-Cauchy hyperprior on a per-coefficient scale, producing a heavy-tailed shrinkage prior.

**Why it matters.** Without sparsity priors, high-dimensional regressions overfit. These are the standard solution in Bayesian variable selection and high-dim genomics / finance.

**Implementation.** Depends entirely on (2.3 per-parameter priors). Once that's in place, spike-and-slab is `prior = Mixture(Normal(0, ε), Normal(0, slab_scale))` per coefficient, and Horseshoe is `Normal(0, λ_i * τ)` with `λ_i ~ HalfCauchy(0, 1)` and `τ ~ HalfCauchy(0, 1)` — both expressible in the existing Distribution pass-through once per-coefficient priors are wired up.

**Verification.** Preset on a sparse synthetic regression (mostly-zero true coefficients with a few large ones). Confirm the Horseshoe-fitted coefficients shrink the noise toward zero and preserve the signal.
""",
    "3.10 Dirichlet process / non-parametric models" => raw"""
**What it is.** Models where the number of components / clusters / random-effect levels is itself inferred during sampling, via a Dirichlet process or stick-breaking prior.

**Why it matters.** When you don't know how many clusters are in your data, fixing K is itself a strong assumption. DP priors let the model decide.

**Implementation.** The heaviest item on the list. Needs sampling-time level inference, a stick-breaking parameter block, and a different `growblock!!` that grows during sampling rather than at `VBRMI` build time. Probably requires a different `lprior!` interface entirely.

Defer until everything else is solid.

**Verification.** Preset against synthetic data with an unknown number of latent clusters. Compare recovered K against the truth.
""",
    "3.11 zero-inflated / hurdle likelihoods" => raw"""
**What it is.** ZI Poisson, ZI Negative Binomial, hurdle Poisson, hurdle Gamma, … — likelihoods that mix a point mass at zero (or a separate "is zero" Bernoulli) with a continuous/count distribution for the nonzero values.

**Why it matters.** Count data with excess zeros (insurance claims, species abundance, healthcare utilization) is everywhere. Standard Poisson / NegBin underfits the zero count.

**Implementation.** Should mostly work through the existing `Distribution` pass-through once we use `Distributions.jl`'s ZI distributions (or write small wrappers). The mixing weight needs its own linear predictor, which is just another distributional-regression-style `~` line.

**Verification.** Preset against synthetic ZI Poisson data. Confirm the recovered zero-inflation probability matches the synthetic generator.
""",
]

function __init__()
    route!(AppContext())
end

end # module
