module BRMMacroWeb

using HTMXObjects
using DynamicObjects: fetchindex!, clear_mem_caches!
using Treebars: polling_fetchindex, initialize_progress!,
                prepare_progress!, with_prepared_progress,
                htmx_treebar_styles, htmx_treebar_script,
                ThreadsafeDict
using Random
using Chairmarks
using DataFrames
using Statistics: quantile, median
using FiniteDifferences: FiniteDifferences, central_fdm
using BridgeStan: BridgeStan
using StanLogDensityProblems: StanProblem
using LogDensityProblems
using StanBlocks
using Distributions
using OrderedCollections: OrderedDict
using WarmupHMC: initialize_mcmc, adaptive_warmup_mcmc
using JSON
using AlgebraOfVega: vega_head, vega_controls, auto_remap_node, with_plot_caption,
    config, pointinterval, lineribbon, to_node, ppc_overlay,
    ECDFPlot, VLines, nonnumeric, Scatter
import AlgebraOfGraphics as AoG

# The @brm macro and VBRMI / SBBRMI implementations now live in the main
# `BayesianRegressionModels` package (moved out of web-macro in `ns/devibe`
# so bruno can consume them as a submodule without the web app deps).
using BayesianRegressionModels
# Pull in the types / functions the web app touches by unqualified name.
using BayesianRegressionModels: AbstractColumn, MissingColumn, DataColumn,
    NamedColumn, ExprColumn, LikelihoodColumn, MaterializedColumn,
    Data, MaybeData, BRMI, VBRMI, SBBRMI,
    assign, doublepipe, gr, gp, offset, zscale, center, standardize, protect,
    me, s, ar, OrderedLogistic,
    # Accessors used unqualified by html_expr.jl and stan_compile code.
    name, getargs, getf, getkwargs, getbroadcast, getop,
    # Macro / pipeline entry points called by Formula + stan_code.
    parse!, _brm, stan_code, maybedata,
    # Part machinery and push_parts!! (called by bruno-ext unqualified).
    Part, push_parts!!, vbroadcasted, llikelihood!
# Extension hooks — use `import` (not `using`) so bruno-ext can ADD
# methods to the same function binding rather than creating a shadowing
# local function.
import BayesianRegressionModels: _sb_submodel_rhs!, vmeta_sampling_rhs,
    nparams, lprior!

# Styled HTML rendering for BRMI / VBRMI cards stays web-side (pulls in
# HTMX builders).
include("html_expr.jl")

# Extension hook: extensions (e.g. the gitignored `bruno-ext.jl`) that need
# to contribute auxiliary data which doesn't fit as per-row DataFrame
# columns -- for example `dose_times::Vector{<:AbstractVector}` indexed by
# subject id -- add a method `dataset_extras(::Val{:ns}, df)` returning a
# NamedTuple of extras. The namespace is derived from the example
# label/slug (first dash/space-separated segment), so `bruno-qt-*` examples
# dispatch to `::Val{:bruno}`. Default is no extras.
dataset_extras(::Val, df) = (;)

# DOs in dependency order. Every feature is a focused @dynamicstruct:
# - Backend/data lives on `AppData`: `dataset`, `run(text, ns)` with its pipeline
#   data, `step_chain` / `compute_steps` for polling_fetchindex, `context` for
#   per-request namespace/run bundles.
# - UI/HTML lives on the routes structs: `PipelineRoutes` owns the formula
#   editor page, per-step render dispatch, and `context!`; the `@include
#   examples` sub-struct owns the examples list/detail/mark routes and the
#   inline `example_store` that constructs ExampleEntry instances with the
#   right `__parent__` for URL construction.
struct FormulaSecurityError <: Exception
    msg::String
end
Base.showerror(io::IO, e::FormulaSecurityError) = print(io, "FormulaSecurityError: ", e.msg)

_ALLOWED_CALLS = Set{Symbol}([
    :~, :(+), :(-), :(*), :(/), :(^), :(|), :(||), :(&),
    :(==), :(!=), :(<), :(>), :(<=), :(>=),
    :log, :log2, :log10, :log1p, :exp, :exp2, :expm1,
    :sqrt, :cbrt, :abs, :abs2, :sign, :floor, :ceil, :round,
    :sin, :cos, :tan, :asin, :acos, :atan,
    :min, :max, :clamp, :mod, :rem, :div,
    :logistic, :logit, :softmax, :logsumexp,
    :log_abs_tanh, :log_square_tanh,
    :Normal, :Poisson, :Binomial, :Bernoulli, :BernoulliLogit, :Beta, :Gamma,
    :Exponential, :Cauchy, :StudentT, :LogNormal, :Weibull,
    :NegativeBinomial, :Geometric, :Laplace, :Uniform,
    :MvNormal, :MixtureModel, :Dirichlet,
    :InverseGamma, :InverseGaussian, :VonMises, :Pareto,
    :OrderedLogistic, :Categorical,
    :zscale, :center, :standardize, :factor, :offset, :protect,
    :s, :bs, :t2, :gp, :ar, :ar1, :mo, :mo1,
    :cbind, :mvbind, :mm, :gr, :dp, :me, :centered,
    :Horseshoe, :ZeroInflatedPoisson, :weighted,
    :length, :unique, :sort, :size, :eltype, :nrow, :ncol,
])

_SAFE_HEADS = Set{Symbol}([
    :block, :call, :., :(=), :(||), :tuple, :vect, :ref,
    :kw, :parameters, :(...),
    :comparison, :&&,
])
walk(x) = begin
    x isa Union{Number, AbstractString, Symbol, LineNumberNode,
                Nothing, Bool, QuoteNode} && return nothing
    x isa Expr || return nothing
    x.head == :macrocall &&
        return FormulaSecurityError(
            "macro calls are not allowed in formulas (got $(x.args[1]))")
    x.head in (:cmd, :string) &&
        return FormulaSecurityError(
            "`$(x.head)` expressions are not allowed in formulas")
    (x.head == :quote || x.head == :$) &&
        return FormulaSecurityError(
            "quote/interpolation expressions are not allowed in formulas")
    if x.head == :call
        fname = x.args[1]
        if fname isa Symbol && fname ∉ _ALLOWED_CALLS
            return FormulaSecurityError(
                "function `$fname` is not in the formula allowlist. " *
                "Allowed: arithmetic, math, distributions, DSL operators.")
        end
        if fname isa Expr && fname.head == :curly
            tname = fname.args[1]
            tname isa Symbol && tname ∉ _ALLOWED_CALLS &&
                return FormulaSecurityError(
                    "type constructor `$tname` is not in the formula allowlist")
        end
    end
    x.head ∉ _SAFE_HEADS &&
        return FormulaSecurityError(
            "expression type `:$(x.head)` is not allowed in formulas")
    for arg in x.args
        v = walk(arg); v === nothing || return v
    end
    nothing
end

@dynamicstruct struct Formula
    text::String

    raw = Meta.parse("begin\n$text\nend")

    violation = walk(raw)
    is_safe = violation === nothing

    _t = begin
        local alllocals = OrderedDict{Symbol,Symbol}()
        (; ex=parse!(deepcopy(raw); info=(;alllocals)), alllocals)
    end
    transformed = _t.ex
    alllocals   = _t.alllocals
end
@dynamicstruct struct Dataset
    n::Int = 16
    seed::Int = 1

    df = begin
        rng = Xoshiro(seed)
        a = randn(rng, n)
        b = randn(rng, n)
        c = randn(rng, n)
        d = randn(rng, n)
        # Grouping factors with different numbers of levels -- use these on
        # the right-hand side of `(... | gN)` to test multiple random-effects
        # blocks.
        g1 = repeat(1:8, inner=cld(n, 8))[1:n]
        g2 = repeat(1:4, inner=cld(n, 4))[1:n]
        g3 = rand(rng, 1:6, n)
        c1 = rand(rng, 1:3, n)
        c2 = rand(rng, 1:2, n)
        c3 = rand(rng, 1:4, n)
        exposure = 0.5 .+ rand(rng, n)
        eta1 = 0.5 .+ 1.2 .* a .- 0.7 .* b .+ 0.3 .* c .+ 0.1 .* d
        y1 = eta1 .+ 0.3 .* randn(rng, n)
        y2 = -0.2 .+ 0.6 .* a .+ 0.4 .* b .+ 0.2 .* randn(rng, n)
        k1 = rand.(rng, Distributions.Poisson.(exp.(0.5 .* eta1)))
        k2 = rand.(rng, Distributions.Poisson.(exp.(0.3 .+ 0.4 .* a)))
        bin_n = rand(rng, 5:30, n)
        bin_p_true = @. 1 / (1 + exp(-(0.2 + 0.5 * a)))
        bin_succ = [rand(rng, Distributions.Binomial(n_i, p_i))
                    for (n_i, p_i) in zip(bin_n, bin_p_true)]
        bin_y = [rand(rng, Distributions.Bernoulli(p_i)) ? 1 : 0
                 for p_i in bin_p_true]
        DataFrame(; a, b, c, d, g1, g2, g3, c1, c2, c3, exposure,
                    y1, y2, k1, k2, bin_n, bin_succ, bin_y)
    end

    # NamedTuple view of `df`, merged with namespace-dispatched extras so
    # extensions (e.g. bruno-ext.jl) can splice in `dose_times` etc. without
    # touching macro.jl. `@brm` only needs `hasproperty`/`getproperty` on
    # its data argument, so the NamedTuple stands in for the DataFrame.
    container(namespace=:default) = begin
        cols = (; (Symbol(c) => df[!, c] for c in names(df))...)
        merge(cols, dataset_extras(Val(namespace), df))
    end
end

# Each example is a `.jl` file under `web-macro/examples/`. File format:
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
# loads + parses these files on every render of the Examples page, and writes
# them back when the user toggles status or submits an edited formula.
# Reopening the server picks up exactly where the user left off — no in-memory
# state.

@dynamicstruct struct ExampleEntry
    path::String
    __parent__ = nothing

    _STATUS_COLORS = (open="#888", done="#2e7d32", deprioritized="#a05a2c")
    _TIER_LABELS   = ("T1", "T2", "T3")
    _TIER_COLORS   = ("#4a7c59", "#5a6a8c", "#8c5a5a")

    _parsed = begin
        lines = readlines(path)
        header = Dict{String,String}()
        i = 1
        while i <= length(lines)
            m = match(r"^# (\w+):\s*(.*)$", lines[i])
            m === nothing && break
            header[m[1]] = m[2]
            i += 1
        end
        body_lines = String[]
        if i <= length(lines) && strip(lines[i]) == "#="
            i += 1
            while i <= length(lines) && strip(lines[i]) != "=#"
                push!(body_lines, lines[i])
                i += 1
            end
            i <= length(lines) && (i += 1)  # consume `=#`
        end
        formula_text = strip(join(lines[i:end], '\n'))
        (; header,
           body=join(body_lines, '\n'),
           formula=isempty(formula_text) ? nothing : String(formula_text))
    end
    label   = get(_parsed.header, "label", basename(path))
    tier    = parse(Int, get(_parsed.header, "tier", "1"))
    status  = Symbol(get(_parsed.header, "status", "open"))
    # `# flag: sb|sbbrm|both` header marks a card as needing backend
    # attention. `:sb` targets the StanBlocks-proper agent; `:sbbrm`
    # targets the BRM sbimpl.jl agent; `:both` is both. Default `:none`.
    # Flagged-in-any-way cards sort to the top so agents can triage them.
    flag = Symbol(get(_parsed.header, "flag", "none"))
    body    = _parsed.body
    formula = _parsed.formula
    slug    = replace(basename(path), r"\.jl$" => "")

    # Per-stage pass/fail state, persisted as comma-separated stage names in
    # `# stages_pass:` / `# stages_fail:` header lines. Empty / missing → no
    # known state for that stage (rendered as unknown/gray indicator).
    _parse_stage_set(key) = Set{Symbol}(
        Symbol(strip(s)) for s in split(get(_parsed.header, key, ""), ",")
        if !isempty(strip(s)))
    stages_pass = _parse_stage_set("stages_pass")
    stages_fail = _parse_stage_set("stages_fail")
    _stage_pill_color = (pass="#2e7d32", fail="#a02828", unknown="#888")
    stage_state(name) = name in stages_fail ? :fail :
                        name in stages_pass ? :pass : :unknown

    border_color = get(_STATUS_COLORS, status, "#888")
    tier_label   = get(_TIER_LABELS,   tier,   "T$tier")
    tier_color   = get(_TIER_COLORS,   tier,   "#888")

    # DOM ids derived once — HTMX targets reference these (hx_target= / id=).
    # Hashing the label keeps ids stable across requests without needing to
    # URL-escape the label.
    _label_hash = hash(label)
    card_id     = "example-card-$_label_hash"
    result_id   = "example-result-$_label_hash"
    status_id   = "status-$_label_hash"

    tier_pill = h.span(tier_label;
        class="brm-tier-pill",
        style="background:$tier_color")

    # `__parent__` is the `@include examples` sub-struct (owns /examples/* routes).
    # `__parent__.__parent__.pipeline` is the sibling PipelineRoutes sub-struct
    # (owns /pipeline/* routes). URLs built via `query_url` so request @params
    # auto-propagate and values auto-encode.
    permalink = h.a("🔗";
        href=string(__parent__/slug),
        title="Standalone URL",
        onclick="event.stopPropagation()",
        class="brm-permalink")

    state_pill(target_state, active_text, inactive_text) = begin
        is_active = status == target_state
        bg = is_active ? get(_STATUS_COLORS, target_state, "#888") : "#888"
        h.button(is_active ? active_text : inactive_text;
            type="button",
            class="brm-state-pill",
            hx_get=string(query_url(__parent__/"mark"; label, state=target_state)),
            hx_target="#$card_id",
            hx_swap="outerHTML",
            onclick="event.stopPropagation()",
            style="background:$bg",
        )
    end

    # Flag pill — cycles through none → SB → SBBRM → both → none.
    # `SB` targets the StanBlocks-proper agent; `SBBRM` targets the BRM
    # sbimpl.jl agent; `both` goes to both. Flagged cards sort to the top.
    _flag_label = flag === :sb    ? "⚑ SB" :
                  flag === :sbbrm ? "⚑ SBBRM" :
                  flag === :both  ? "⚑ SB+SBBRM" :
                                    "flag"
    _flag_color = flag === :sb    ? "#c97f0c" :
                  flag === :sbbrm ? "#2e7d32" :
                  flag === :both  ? "#a02828" :
                                    "#888"
    flag_pill = h.button(_flag_label;
        type="button",
        class="brm-state-pill",
        hx_get=string(query_url(__parent__/"flag"; label)),
        hx_target="#$card_id",
        hx_swap="outerHTML",
        onclick="event.stopPropagation()",
        style="background:$_flag_color",
    )

    status_pills = h.span(;
        id=status_id,
        class="brm-status-pills")(
        state_pill(:done,          "✓ done",          "mark done"),
        state_pill(:deprioritized, "✓ deprioritized", "deprioritize"),
        flag_pill,
    )

    # Stage list pulled from AppContext.stage_labels so the order + names
    # stay in sync with the pipeline page.
    _stages = __parent__.__parent__.stage_labels

    # Formula form with stage buttons. `push_url=true` in the standalone
    # detail view makes each button push `/examples/<slug>?stage=<id>` into
    # browser history (shareable, back/forward works); `push_url=false` in
    # the list view leaves the URL at `/examples`.
    formula_form(push_url::Bool) = begin
        stage_btns = [
            h.button(lbl;
                type="button",
                class="brm-branch-btn",
                data_stage_id=string(id),
                hx_get=string(query_url(__parent__.__parent__.pipeline/"stage/$id"; force=true)),
                hx_include="closest form",
                hx_target="#$result_id",
                hx_swap="innerHTML",
                (push_url ? (; hx_push_url="$(__parent__/slug)?stage=$id") : (;))...,
            ) for (id, lbl) in _stages]
        h.form(; class="brm-example-form")(
            h.input(; type="hidden", name="label", value=label),
            h.textarea(formula;
                name="formula",
                rows=max(3, count('\n', formula) + 1),
                class="brm-example-textarea"),
            stage_btns...,
            h.button("SB repro ▶";
                type="submit",
                formaction=string(__parent__.__parent__.pipeline/"sb_repro"),
                class="secondary"),
        )
    end

    # Default card render (no stage pre-loaded; clicks on stage buttons
    # populate the inline result div on demand). `card_with_preload("")`
    # below handles both list-mode and (when called with a non-empty
    # preload_stage) detail-mode.
    card = card_with_preload("")

    save!(; new_status=status, new_formula=formula,
            new_stages_pass=stages_pass, new_stages_fail=stages_fail,
            new_flag=flag) = begin
        io = IOBuffer()
        println(io, "# label: ", label)
        println(io, "# tier: ", tier)
        println(io, "# status: ", new_status)
        new_flag === :none || println(io, "# flag: ", new_flag)
        _fmt_stage_set(s) = join(sort(collect(String.(s))), ",")
        isempty(new_stages_pass) || println(io, "# stages_pass: ", _fmt_stage_set(new_stages_pass))
        isempty(new_stages_fail) || println(io, "# stages_fail: ", _fmt_stage_set(new_stages_fail))
        if !isempty(body)
            println(io, "#=")
            println(io, body)
            println(io, "=#")
        end
        if new_formula !== nothing && !isempty(new_formula)
            println(io)
            print(io, new_formula)
            endswith(new_formula, "\n") || println(io)
        end
        write(path, take!(io))
        # Preserve __parent__ so the reloaded entry can still build route URLs.
        ExampleEntry(path; __parent__)
    end

    toggle_status!(target) =
        save!(; new_status = status == target ? :open : target)

    # Cycle: none → sb → sbbrm → both → none
    _flag_next = Dict(:none => :sb, :sb => :sbbrm, :sbbrm => :both, :both => :none)
    cycle_flag!() = save!(; new_flag = _flag_next[flag])

    # Mark a set of stage names as pass / fail and persist to disk. Passed
    # stages remove from the fail set (and vice versa) so the most recent
    # outcome wins. Returns the reloaded entry.
    mark_stages!(; passed=Symbol[], failed=Symbol[]) = begin
        new_pass = union(setdiff(stages_pass, failed), passed)
        new_fail = union(setdiff(stages_fail, passed), failed)
        save!(; new_stages_pass=new_pass, new_stages_fail=new_fail)
    end

    # Per-stage indicator pills rendered in the card summary. Each links to
    # `/examples/<slug>?stage=<name>`, a shareable GET URL that loads the
    # card with that stage's result pre-populated (without force=true).
    stage_indicators = h.span(; class="brm-stage-indicators")(
        [h.a(;
            href="$(__parent__/slug)?stage=$id",
            title="$lbl -- $(stage_state(id))",
            onclick="event.stopPropagation()",
            class="brm-stage-indicator",
            style="background:$(_stage_pill_color[stage_state(id)])")(lbl)
         for (id, lbl) in __parent__.__parent__.stage_labels]...
    )

    # Card renderer, parameterized by an optional `preload_stage`. When set,
    # the result div is seeded with a `lazy(...)` that fires the stage GET
    # on first view — so `/examples/<slug>?stage=<name>` shows the card with
    # that stage's output already running/rendered (no click needed).
    # Numeric sort attributes used by the client-side sort bar. Higher
    # `brokenness` = more failed stages (fewer passes, more fails); zero when
    # no stage state is recorded yet.
    _sort_mtime      = stat(path).mtime
    _sort_tier       = tier
    _sort_complexity = formula === nothing ? 0 : length(formula)
    _sort_brokenness = length(stages_fail) - length(stages_pass)
    # 0 = open (top), 1 = done / deprioritized (bottom, tied). Ascending
    # surfaces open work first; default-on below.
    _sort_status     = status === :open ? 0 : 1
    _sort_progress   = length(stages_pass)
    # 0=none, 1=sb-only or sbbrm-only, 2=both → "most in need" sorts highest.
    _sort_flagged    = flag === :none ? 0 : flag === :both ? 2 : 1

    card_with_preload(preload_stage::AbstractString; force_open::Bool=false) = begin
        children = Any[HTMXObjects.md_to_node(body)]
        if formula !== nothing
            # Push URL on stage-button clicks only in the detail view
            # (`force_open=true`); list view keeps URL at `/examples`.
            push!(children, formula_form(force_open))
            result_children = Any[]
            isempty(preload_stage) || push!(result_children,
                lazy(query_url(__parent__.__parent__.pipeline/"stage/$preload_stage";
                               formula, label)))
            push!(children, h.div(; id=result_id,
                class="brm-example-result")(result_children...))
        end
        h.article(;
            id=card_id,
            class="brm-example-card",
            style="border-left-color:$border_color",
            data_mtime=string(_sort_mtime),
            data_tier=string(_sort_tier),
            data_complexity=string(_sort_complexity),
            data_brokenness=string(_sort_brokenness),
            data_status=string(_sort_status),
            data_progress=string(_sort_progress),
            data_flagged=string(_sort_flagged),
            data_label=label,
        )(
            h.details(; open=(force_open || status == :open))(
                h.summary(; class="brm-example-summary")(
                    tier_pill, " ",
                    h.strong(label), " ",
                    status_pills, " ",
                    stage_indicators, " ",
                    permalink,
                ),
                h.div(; class="brm-example-body")(children...),
            ),
        )
    end
end
# Element-returning counterpart to `Base.findfirst(pred, coll)` (which returns an
# index or `nothing`). Does the index-then-lookup dance once so callers don't.
findfirstelement(pred, coll) = begin
    i = findfirst(pred, coll)
    isnothing(i) ? nothing : coll[i]
end

# Stan draws → DataFrames plumbing, shared between prior-predictive generation
# and posterior fits (pathfinder / warmup). Keep these as plain module-level
# helpers so callsites inside `@struct stan = …` don't accidentally become IPs.

# Param-constrain each column of `unc_draws` (dim × n) with `include_tp=true,
# include_gq=true`, returning an (m × n) matrix where `m = length(param_names(instance; include_tp=true, include_gq=true))`.
constrain_draws(unc_draws, instance; rng_seed) = begin
    rng = BridgeStan.StanRNG(instance, rng_seed)
    m = length(BridgeStan.param_names(instance; include_tp=true, include_gq=true))
    n = size(unc_draws, 2)
    mat = Matrix{Float64}(undef, m, n)
    for i in 1:n
        mat[:, i] = BridgeStan.param_constrain(
            instance, collect(view(unc_draws, :, i));
            include_tp=true, include_gq=true, rng=rng,
        )
    end
    mat
end

# Build the (long, wide, summary) DataFrame triple from a constrained draws
# matrix `constrained` (m × n) and its matching parameter `names` (length m).
# Splits indexed names on the first `.` into (:param, :index) with :index as Int
# (0 for scalars); summary groups by (:param, :index) with the bands columns
# expected by `pointinterval(bands=…)` / `lineribbon(bands=…)`.
dfs_from_constrained(constrained, names) = begin
    n = size(constrained, 2)
    splits     = [split(nm, '.', limit=2) for nm in names]
    base_names = [String(first(s)) for s in splits]
    parse_idx(s) = (v = tryparse(Int, s); isnothing(v) ? 0 : v)
    indices    = [length(s) > 1 ? parse_idx(String(s[2])) : 0 for s in splits]
    long = DataFrame(
        param = repeat(base_names, inner=n),
        index = repeat(indices, inner=n),
        draw  = repeat(1:n, outer=length(base_names)),
        value = vec(constrained'),
    )
    wide = DataFrame(
        [Symbol(names[i]) => constrained[i, :] for i in eachindex(names)]
    )
    summary = combine(
        groupby(long, [:param, :index]),
        :value => (v -> quantile(v, 0.025)) => :q025,
        :value => (v -> quantile(v, 0.10))  => :q10,
        :value => (v -> quantile(v, 0.25))  => :q25,
        :value => median                    => :median,
        :value => (v -> quantile(v, 0.75))  => :q75,
        :value => (v -> quantile(v, 0.90))  => :q90,
        :value => (v -> quantile(v, 0.975)) => :q975,
    )
    (; long, wide, summary)
end

# Detect a "single-continuous-predictor" PPC pattern in a BRMI. Returns a
# NamedTuple describing what to plot, or `nothing` if the formula doesn't fit.
#
# Supported patterns:
#   1. Linreg:        `loc ~ 1 + a`,  `y ~ Normal(loc, _)`           link = identity
#   2. Linreg + RE:   `loc ~ 1 + a + (1|g)`, `y ~ Normal(loc, _)`    link = identity, group = g
#   3. Poisson:       `eta ~ 1 + a`,  `k ~ Poisson(exp(eta))`        link = exp
#   4. Bernoulli:     `eta ~ 1 + a`,  `b ~ Bernoulli(logistic(eta))` link = logistic
#
# Returns `(; predictor, response, loc, family, link_fn, group)`. `group` is
# `nothing` unless an `(1 | g)` random intercept is present alongside the bare
# predictor (multi-term + RE is allowed). Skips on: multi-bare-predictor,
# non-supported family, link transform on the loc LHS, integer/categorical
# predictor, missing loc op.
_ppc_pattern(brmi::BRMI) = begin
    ops = brmi.operations
    response = nothing
    loc_name = nothing
    family   = nothing
    link_fn  = identity
    for (k, v) in pairs(ops)
        v isa NamedColumn || continue
        op = parent(v)
        op isa ExprColumn || continue
        getf(op) === (~) || continue
        lhs, rh = getargs(op, 2)
        lhs isa NamedColumn && parent(lhs) isa DataColumn || continue
        rh isa ExprColumn || continue
        f = getf(rh)
        # Only detect families we know how to overlay observed values for.
        (f === Normal || f === Poisson || f === Bernoulli) || continue
        args = getargs(rh)
        length(args) >= 1 || continue
        # Accept either a bare loc NamedColumn or a unary link wrap
        # (e.g. `Poisson(exp(eta))`, `Bernoulli(logistic(eta))`).
        a1 = args[1]
        if a1 isa NamedColumn
            loc_name = name(a1)
            link_fn  = identity
        elseif a1 isa ExprColumn && length(getargs(a1)) == 1 &&
               getargs(a1)[1] isa NamedColumn
            link_fn  = getf(a1)
            loc_name = name(getargs(a1)[1])
        else
            continue
        end
        response = k
        family   = f
        break
    end
    isnothing(response) && return nothing
    haskey(ops, loc_name) || return nothing
    loc_v = ops[loc_name]
    loc_v isa NamedColumn || return nothing
    loc_op = parent(loc_v)
    loc_op isa ExprColumn || return nothing
    getf(loc_op) === (~) || return nothing
    lhs, rhs_expr = getargs(loc_op, 2)
    lhs isa NamedColumn || return nothing            # link on loc LHS -> skip
    rhs_expr isa ExprColumn || return nothing
    getf(rhs_expr) === (+) || return nothing
    bare = NamedColumn[]
    group = nothing
    for s in getargs(rhs_expr)
        s isa Number && continue                     # intercept literal
        if s isa NamedColumn
            push!(bare, s)
        elseif s isa ExprColumn && getf(s) === (|)
            # `(... | g)` random-effects term: extract grouping factor name.
            ra = getargs(s)
            if length(ra) >= 2 && ra[end] isa NamedColumn &&
               parent(ra[end]) isa DataColumn
                group = name(ra[end])
            end
        else
            return nothing
        end
    end
    length(bare) == 1 || return nothing
    pcol = parent(bare[1])
    pcol isa DataColumn || return nothing
    elt = eltype(parent(pcol))
    (elt <: Real && !(elt <: Integer)) || return nothing
    (; predictor=name(bare[1]), response, loc=loc_name, family, link_fn, group)
end

# Project the long-format draws (cols: param, index, draw, value) into a
# (x, y, draw, group?) table by filtering for `loc`, mapping `index -> x`
# from the dataset's predictor column, and applying `link_fn` to `value` so
# `y` lives on the response scale (identity for Normal, exp for Poisson,
# logistic for Bernoulli). `df` is the dataset; `group` is the optional
# grouping-factor column name (added as a `:group` column).
_loc_pred_table(long::DataFrame, df::DataFrame, loc::Symbol, predictor::Symbol,
                link_fn::Function, group::Union{Symbol,Nothing}) = begin
    rows = filter(:param => ==(string(loc)), long)
    isempty(rows) && return nothing
    out = DataFrame(
        x     = df[!, predictor][rows.index],
        y     = link_fn.(rows.value),
        draw  = rows.draw,
        index = rows.index,
    )
    isnothing(group) || (out.group = df[!, group][rows.index])
    out
end

# Prior-predictive PPC spec: lineribbon (median + 50/80/95% bands) of `loc[i]`
# (link-transformed onto the response scale) vs the predictor x[i], aggregated
# across draws. NO observation overlay -- the prior and the data shouldn't be
# eyeballed against each other on this stage.
_prior_ppc_spec(long, df, p::NamedTuple; title) = begin
    pred = _loc_pred_table(long, df, p.loc, p.predictor, p.link_fn, p.group)
    isnothing(pred) && return nothing
    base = AoG.data(pred) * AoG.mapping(:x, :y, group=:draw)
    base = isnothing(p.group) ? base : base * AoG.mapping(color=:group => nonnumeric)
    base * lineribbon() * config(title=title)
end

# Posterior-predictive PPC spec: predicted lineribbon + observed scatter via
# AoV's `ppc_overlay`. `obs_y` is the response vector the model was fit to
# (synthetic SBC observations from `fit_data_dict`). For non-identity links
# the response data lives on the response scale already (k for Poisson, 0/1
# for Bernoulli) so it lines up with the link-transformed prediction.
_posterior_ppc_spec(long, df, obs_y, p::NamedTuple; title) = begin
    pred = _loc_pred_table(long, df, p.loc, p.predictor, p.link_fn, p.group)
    isnothing(pred) && return nothing
    obs = DataFrame(x = df[!, p.predictor], y = obs_y)
    layers = ppc_overlay(obs, pred; x=:x, y=:y,
                         group=:draw,
                         color = isnothing(p.group) ? nothing : :group)
    layers * config(title=title)
end

@dynamicstruct struct AppData
    __status__ = initialize_progress!(:state; description="BRM pipeline")
    examples_dir = joinpath(dirname(@__DIR__), "examples")

    dataset = Dataset()

    # Per-stan-path locks for the "ensure .stan written + compile_model"
    # sequence. Without these, two requests for the same model can run `make`
    # concurrently into the same .so target and corrupt it -- subsequent
    # dlopen segfaults Julia (no Julia stack, no OOM, just process gone).
    # Keyed by absolute .stan path. Entries are never reaped; one per distinct
    # compiled model. `get!(ReentrantLock, locks, path)` lazily creates.
    stan_compile_locks = ThreadsafeDict{String,ReentrantLock}()

    namespace_from(label) = isempty(strip(label)) ? :default :
        Symbol(lowercase(first(split(strip(label), r"[\s:\-]+"))))

    # One run of the @brm pipeline for a given (text, namespace) pair. Every
    # stage is a derived property -- accessing `.brmi` triggers parse + eval,
    # `.benches` triggers the benchmark loop, etc. Unused branches don't
    # compute. Safety is enforced at `wrapped`, which refuses to produce Julia
    # code for an unsafe formula.
    # TODO(DO): investigate whether indexed inline structs (@struct) should
    # accept default values for index params. Today `@struct run(text, namespace=:default)`
    # errors with "index param must be a Symbol" because the default becomes
    # a `:kw` Expr. Unclear if there's a sensible meaning for such defaults at
    # all (cache keying, forwarding, etc.) or if we should just continue
    # requiring bare Symbol params.
    @struct run(text, namespace) = begin
        formula = Formula(text)
        df = dataset.df
        container = dataset.container(namespace)

        wrapped = begin
            formula.is_safe || throw(formula.violation)
            _brm(text; df=container)
        end
        brmi = eval(wrapped)

        # ── cimpl branch ──
        vbrmi = VBRMI(brmi)
        dim   = LogDensityProblems.dimension(vbrmi)
        x0    = randn(Xoshiro(0), dim)
        ldp   = string(LogDensityProblems.logdensity(vbrmi, x0))
        grad  = FiniteDifferences.grad(
            central_fdm(5, 1),
            Base.Fix1(LogDensityProblems.logdensity, vbrmi),
            x0,
        )[1]

        tol    = 1e-8
        n_dead = count(<=(tol) ∘ abs, grad)
        dead   = findall(<=(tol) ∘ abs, grad)

        benches = vcat(
            [
                "logdensity (total)" => @be(randn(dim), LogDensityProblems.logdensity($vbrmi, _)),
                "lprior!"            => @be(randn(dim), lprior!($vbrmi, _)),
            ],
            [
                "  lprior!($group_key[$i] $(part))" => @be(randn(nparams(part)), lprior!($part, _))
                for (group_key, parts) in pairs(vbrmi.meta.blocks)
                for (i, part) in enumerate(parts)
            ],
            [
                "llikelihood!($key)" => @be(llikelihood!($m))
                for (key, m) in pairs(vbrmi.meta.materialized)
            ],
        )

        # ── stan branch ──
        sbbrmi = SBBRMI(brmi; mod=@__MODULE__)

        # Everything Stan-related bundled under `r.stan.*`. Nested access via
        # step_chain's tuple-path specs (e.g. `(:stan, :src)`).
        @struct stan = begin
            src = stan_code(sbbrmi)
            # Hash-keyed cache path so identical Stan source reuses the same
            # .stan (and co-located .so) across requests. `BridgeStan.compile_model`
            # invokes make, which skips when the .so is newer than the .stan —
            # so a cache hit resolves in milliseconds instead of re-running
            # the C++ build.
            # Hash folds in the make-args so changing compile flags (e.g.
            # adding STAN_THREADS=true) naturally routes to a fresh path and
            # triggers a rebuild — BridgeStan doesn't support reloading a
            # previously-dlopened .so, so we can't reuse old binaries.
            _make_args = ["STAN_THREADS=true"]
            file = joinpath(tempdir(), "brm_stan",
                            string(hash((src, _make_args))) * ".stan")
            mkpath(dirname(file))
            # Serialize file-write + compile per .stan path. Two threads
            # racing into the same `make` invocation produced a half-written
            # `.so` whose subsequent dlopen segfaulted Julia. The lock is
            # held only for the file-write + compile_model call; once the
            # `.so` is on disk the rest of the pipeline (StanModel /
            # param_constrain / ...) proceeds in parallel as before.
            lib = Base.lock(get!(ReentrantLock, stan_compile_locks, file)) do
                isfile(file) || write(file, src)
                BridgeStan.compile_model(file; make_args=_make_args)
            end
            # SB's `stan_data` walks the SlicModel → StanModel tracing which
            # auto-declares `_n` / `_m` sizes for every vector / matrix, then
            # `bridgestan_data` JSON-serializes with Stan's column-major
            # matrix convention.
            data     = StanBlocks.stan.bridgestan_data(StanBlocks.stan_data(sbbrmi.model))
            instance = BridgeStan.StanModel(lib, data)
            dim      = BridgeStan.param_unc_num(instance)
            # Fixed-rng narrow-normal init — deterministic, cache-friendly.
            init     = 0.1 .* randn(Xoshiro(42), dim)

            # Smoke-test evaluation: log density at the init params. Forces the
            # model-loaded + data-bound path without running Pathfinder.
            log_density = BridgeStan.log_density(instance, init)

            # Synthetic-data generation: sample N unconstrained parameter draws
            # from a narrow zero-mean normal, then `param_constrain` each with
            # `include_tp` + `include_gq` so the output matrix also carries
            # transformed parameters and generated-quantities (the synthetic
            # outcomes `y_sim` live in the GQ block when the model defines
            # one).
            generated_n     = 50
            generated_unc   = 0.1 .* randn(Xoshiro(44), dim, generated_n)
            generated_names = BridgeStan.param_names(instance; include_tp=true, include_gq=true)
            # One row per base parameter name (before any `.`), with the
            # number of indexed entries and Stan-block classification:
            # P = parameter, TP = transformed parameter, GQ = generated
            # quantity. Classification is derived from BridgeStan's
            # param_names called with/without include_tp / include_gq.
            param_shapes_df = begin
                p_names   = BridgeStan.param_names(instance)
                p_tp_names = BridgeStan.param_names(instance; include_tp=true)
                tp_set = Set(setdiff(p_tp_names, p_names))
                gq_set = Set(setdiff(generated_names, p_tp_names))
                classify(name) =
                    name in tp_set ? "TP" : name in gq_set ? "GQ" : "P"
                splits     = [split(nm, '.', limit=2) for nm in generated_names]
                base_names = [String(first(s)) for s in splits]
                kinds      = [classify(nm) for nm in generated_names]
                uniq_base  = unique(base_names)
                uniq_kind  = [kinds[findfirst(==(b), base_names)] for b in uniq_base]
                DataFrame(
                    param = uniq_base,
                    kind = uniq_kind,
                    n_indices = [count(==(p), base_names) for p in uniq_base],
                )
            end
            generated = constrain_draws(generated_unc, instance; rng_seed=45)
            generated_dfs = dfs_from_constrained(generated, generated_names)
            generated_df      = generated_dfs.long
            generated_wide_df = generated_dfs.wide
            generated_summary_df = generated_dfs.summary
            # Ground-truth overlay: the `fit_draw_idx`-th column of the
            # constrained generated matrix — one value per indexed parameter.
            # Same (param, index) layout as `generated_df` so plots can join on it.
            truth_df = begin
                truth_col = view(generated, :, fit_draw_idx)
                splits     = [split(nm, '.', limit=2) for nm in generated_names]
                base_names = [String(first(s)) for s in splits]
                parse_idx(s) = (v = tryparse(Int, s); isnothing(v) ? 0 : v)
                indices    = [length(s) > 1 ? parse_idx(String(s[2])) : 0 for s in splits]
                DataFrame(
                    param = base_names,
                    index = indices,
                    truth = collect(truth_col),
                )
            end
            # Parallel truth-df for fit plots: unconstrained parameter names
            # + `fit_truth_unc` as the ground-truth values. Matches the
            # `posterior_unc_names`-based long/wide/summary DFs on the fit
            # path so the scatter overlay joins correctly.
            truth_unc_df = begin
                unc_names = BridgeStan.param_unc_names(fit_instance)
                splits     = [split(nm, '.', limit=2) for nm in unc_names]
                base_names = [String(first(s)) for s in splits]
                parse_idx(s) = (v = tryparse(Int, s); isnothing(v) ? 0 : v)
                indices    = [length(s) > 1 ? parse_idx(String(s[2])) : 0 for s in splits]
                DataFrame(
                    param = base_names,
                    index = indices,
                    truth = fit_truth_unc,
                )
            end
            # Simulation-based calibration setup: pick one draw from the prior
            # predictive `generated` matrix, extract every `*_gen[.i.j]` entry,
            # and fold them back into the Stan data dict as their `*` (observed)
            # counterparts. The resulting `fit_instance` shares the compiled
            # .so with `instance` but is bound to this synthetic observed data,
            # so Pathfinder / full warmup samples `p(theta | y_sim)` and should
            # recover the ground-truth `generated_unc[:, fit_draw_idx]`.
            fit_draw_idx = 1
            fit_truth_unc = collect(view(generated_unc, :, fit_draw_idx))
            fit_data_dict = begin
                base = StanBlocks.stan_data(sbbrmi.model)
                col  = view(generated, :, fit_draw_idx)
                groups = Dict{Symbol, Vector{Tuple{Vector{Int}, Float64}}}()
                for (i, name) in enumerate(generated_names)
                    m = match(r"^(.+)_gen(?:\.(.+))?$", name)
                    isnothing(m) && continue
                    base_name = Symbol(m.captures[1])
                    idx_str   = m.captures[2]
                    idxs = isnothing(idx_str) ? Int[] :
                           [Base.parse(Int, s) for s in split(idx_str, ".")]
                    push!(get!(Vector{Tuple{Vector{Int}, Float64}}, groups, base_name),
                          (idxs, col[i]))
                end
                # Preserve the original's element type so integer-valued data
                # vars (Bernoulli outcomes, counts) stay integers after the
                # override — Stan's int-typed data variables reject Float64s
                # even when the values are exact (1.0, 0.0, …).
                overrides = Dict{Symbol, Any}()
                for (name, entries) in groups
                    orig = base[name]
                    if length(entries) == 1 && isempty(entries[1][1])
                        overrides[name] = convert(typeof(orig), entries[1][2])
                    else
                        out = similar(orig)  # preserves eltype
                        for (idxs, v) in entries
                            out[idxs...] = v  # implicit convert to eltype(out)
                        end
                        overrides[name] = out
                    end
                end
                merge(base, overrides)
            end
            fit_instance = BridgeStan.StanModel(lib,
                StanBlocks.stan.bridgestan_data(fit_data_dict))

            # IP: Pathfinder init (fast, no MCMC). The `progress=__status__`
            # hook lets Treebars nest the maxiters subtree under whatever
            # node called `fetchindex!(status, …, pathfinder, instance, init)`.
            "Pathfinder(maxiters=$maxiters)"
            pathfinder(instance, init; rng=Xoshiro(42), maxiters=100) =
                initialize_mcmc(StanProblem(instance; nan_on_error=true), init; rng, progress=__status__, maxiters)
            # IP: full Stan + WarmupHMC fit. Same progress-hooking pattern.
            # Returns a rich NamedTuple with `.posterior_position`, `.ess`,
            # `.n_divergent_samples`, etc.
            "WarmupHMC(n_draws=$n_draws)"
            posterior_warmup(instance, init; rng=Xoshiro(42), n_draws=200) =
                adaptive_warmup_mcmc(rng, StanProblem(instance; nan_on_error=true); init, n_draws, progress=__status__)
            # Gaussian-approximation draws from Pathfinder. Reads the IP via
            # `@memo` so if `compute_steps` already computed it with progress
            # nesting, we get the cached value for free.
            posterior_pathfinder = begin
                pf = @memo pathfinder(fit_instance, init)
                pf.position .+ pf.scale * randn(Xoshiro(43), dim, 200)
            end
            # Default. Switch this alias to the warmup draws (`@memo
            # posterior_warmup(instance, init).posterior_position`) to promote
            # the full fit, or expose a toggle via a param later.
            posterior = posterior_pathfinder
            # For fit plots we only want the unconstrained parameters (the
            # raw sampler output) — constrained/TP/GQ adds lots of derived
            # noise (y_gen, y_likelihood) that drowns out the actual
            # parameter posterior. Keep prior-predictive plots on the full
            # constrained set above; fit plots use `param_unc_names` here.
            posterior_unc_names = BridgeStan.param_unc_names(fit_instance)
            posterior_dfs = dfs_from_constrained(posterior, posterior_unc_names)
            posterior_long_df    = posterior_dfs.long
            posterior_wide_df    = posterior_dfs.wide
            posterior_summary_df = posterior_dfs.summary

            # Parallel set for the warmup+MCMC path. The warmup IP cache is
            # warmed by `fetchindex!` in `compute_steps`; `@memo` hits it here.
            posterior_warmup_draws = (@memo posterior_warmup(fit_instance, init)).posterior_position
            posterior_warmup_dfs = dfs_from_constrained(posterior_warmup_draws, posterior_unc_names)
            posterior_warmup_long_df    = posterior_warmup_dfs.long
            posterior_warmup_wide_df    = posterior_warmup_dfs.wide
            posterior_warmup_summary_df = posterior_warmup_dfs.summary
            posterior_warmup_diagnostics = begin
                w = @memo posterior_warmup(fit_instance, init)
                (; w.n_divergent_samples, ess=w.ess)
            end

            # Constrained-with-TP/GQ variants — needed for PPCs that consume
            # the linear-predictor TP (`loc` / `log_rate` / `log_odds`). The
            # base posterior_*_long_df above intentionally excludes TPs/GQs to
            # keep the generic plot tabset focused on raw parameters.
            posterior_constrained_full = constrain_draws(posterior, fit_instance; rng_seed=46)
            posterior_full_long_df = dfs_from_constrained(
                posterior_constrained_full,
                BridgeStan.param_names(fit_instance; include_tp=true, include_gq=true),
            ).long
            posterior_warmup_constrained_full = constrain_draws(posterior_warmup_draws, fit_instance; rng_seed=47)
            posterior_warmup_full_long_df = dfs_from_constrained(
                posterior_warmup_constrained_full,
                BridgeStan.param_names(fit_instance; include_tp=true, include_gq=true),
            ).long
        end

        # Stage-named aliases. `step_chain` / `compute_steps` extract step
        # outputs by looking up these names via `getproperty`.
        parse     = formula.raw
        transform = (; formula.transformed, formula.alllocals)
        wrap      = wrapped
    end

    # Per-request bundle for a (label, formula) pair. Pure construction; the
    # routes side handles persistence before invoking this.
    @struct context(label, formula) = begin
        namespace = namespace_from(label)
        run = __parent__.run(formula, namespace)
    end

    # Expanded DAG of step chains — one NamedTuple per stage target, keyed by
    # step names in dependency order. Built incrementally via `merge`: each
    # stage's chain is its parent's chain plus the one step it adds. Fork
    # points are visible in the `merge` calls (e.g. `slic_model = merge(brmi,
    # …)` forks off of `brmi`, parallel to `vbrmi`). Inner values are one of:
    #   - Symbol                       → `r.<sym>` (top-level property)
    #   - Tuple{Vararg{Symbol}}        → nested access, e.g. `(:stan, :src)` → `r.stan.src`
    #   - NamedTuple of (Symbol|Tuple) → bundle, keys preserved in the result.
    # Top-level key = stage name. Callers either iterate (stage list, indicators)
    # or index by name (`step_chain(:brmi)`).
    step_chain = begin
        parse            = (; parse=:parse)
        transform        = merge(parse,      (; transform=:transform))
        wrap             = merge(transform,  (; wrap=:wrap))
        brmi             = merge(wrap,       (; brmi=:brmi))
        vbrmi            = merge(brmi,       (; vbrmi=(; vbrmi=:vbrmi, dim=:dim, ldp=:ldp, grad=:grad, n_dead=:n_dead, dead=:dead)))
        bench            = merge(vbrmi,      (; bench=:benches))
        slic_model       = merge(brmi,       (; slic_model=:sbbrmi))
        stan_code        = merge(slic_model, (; stan_code=(:stan, :src)))
        stan_compile     = merge(stan_code,  (; stan_compile=(; file=(:stan, :file), lib=(:stan, :lib))))
        stan_instantiate = merge(stan_compile, (; stan_instantiate=(; instance=(:stan, :instance), dim=(:stan, :dim), init=(:stan, :init))))
        stan_eval        = merge(stan_instantiate, (; stan_eval=(:stan, :log_density)))
        stan_shapes      = merge(stan_eval,  (; stan_shapes=(:stan, :param_shapes_df)))
        stan_generate    = merge(stan_shapes, (; stan_generate=(; long=(:stan, :generated_df), wide=(:stan, :generated_wide_df), summary=(:stan, :generated_summary_df), truth=(:stan, :truth_df))))
        # Pathfinder / full warmup are computed via `fetchindex!` in
        # `compute_steps` (special-cased below by step name) so the IP's
        # progress subtree attaches to the step's phase. Chain-level specs
        # read the resulting cached values back out as plain properties.
        stan_fit_pathfinder = merge(stan_generate, (; stan_fit_pathfinder=(; long=(:stan, :posterior_long_df), wide=(:stan, :posterior_wide_df), summary=(:stan, :posterior_summary_df), full_long=(:stan, :posterior_full_long_df), truth=(:stan, :truth_unc_df))))
        stan_fit_warmup     = merge(stan_fit_pathfinder, (; stan_fit_warmup=(; long=(:stan, :posterior_warmup_long_df), wide=(:stan, :posterior_warmup_wide_df), summary=(:stan, :posterior_warmup_summary_df), full_long=(:stan, :posterior_warmup_full_long_df), diagnostics=(:stan, :posterior_warmup_diagnostics), truth=(:stan, :truth_unc_df))))
        (; parse, transform, wrap, brmi, vbrmi, bench, slic_model, stan_code, stan_compile,
           stan_instantiate, stan_eval, stan_shapes, stan_generate, stan_fit_pathfinder, stan_fit_warmup)
    end

    # Fetch target for `polling_fetchindex`. Pre-enumerates each step in the
    # requested chain as a pending progress child (so the whole pipeline is
    # visible up front as dim "pending" nodes), then runs each step under its
    # phase — `with_prepared_progress` handles start/finalize/fail around the
    # property access. Heavy work runs in polling_fetchindex's background task
    # via DO's lazy property cascading. Returns a NamedTuple keyed by step
    # names plus `:data` (the synthetic-data frame for the pipeline top pin).
    "Pipeline($name)"
    compute_steps(text, namespace, name::Symbol) = begin
        r = run(text, namespace)
        chain = step_chain[name]
        phases = [prepare_progress!(__status__; description=string(k)) for k in keys(chain)]
        # Resolve each spec against `r`:
        #   Symbol          → `r.<sym>`
        #   Tuple of Symbol → nested path `r.<a>.<b>...`
        #   NamedTuple      → bundle, recurse per value.
        resolve(s::Symbol) = getproperty(r, s)
        resolve(p::Tuple{Vararg{Symbol}}) = foldl(getproperty, p; init=r)
        resolve(b::NamedTuple) = map(resolve, b)
        vals = map(pairs(chain), phases) do (step_name, spec), phase
            with_prepared_progress(phase) do progress
                if step_name === :stan_fit_pathfinder
                    # Warm the pathfinder IP cache under this phase's progress,
                    # then resolve the (long, wide, summary) bundle (which
                    # reads back the cached value via `@memo`).
                    fetchindex!(progress, r.stan.pathfinder, r.stan.fit_instance, r.stan.init)
                    resolve(spec)
                elseif step_name === :stan_fit_warmup
                    # Warm the warmup IP cache under this phase's progress,
                    # then resolve the (long, wide, summary, diagnostics)
                    # bundle (which reads back the cached value via `@memo`).
                    fetchindex!(progress, r.stan.posterior_warmup, r.stan.fit_instance, r.stan.init)
                    resolve(spec)
                else
                    resolve(spec)
                end
            end
        end
        # Stages render most-recent-first; synthetic data pinned at the top.
        merge((; data=r.df),
              NamedTuple{reverse(keys(chain))}(Tuple(reverse(vals))))
    end
end

APPDATA = AppData(; cache_type=:parallel)

# Pipeline-page routes mounted at /pipeline. The formula editor, stage polling,
# and sbimpl source views all live here. The top-level AppContext just includes
# this struct plus the Examples section and the page chrome.
@htmx struct PipelineRoutes
    (; context, compute_steps) = __appdata__
    (; default_formula) = __parent__
    @param (; formula, label) = __parent__

    # Persist + context. Reaches into the sibling Examples include for the
    # examples store (UI concern: writing the edited formula back to the .jl
    # file corresponding to `label`), then returns the pure run context.
    context!() = begin
        isempty(label) || __parent__.examples.example_store.persist!(label, formula)
        context(label, formula)
    end

    # Per-step HTML rendering. `getproperty(render, step_key)(value)` emits the
    # section for that step; `compute_steps` produces the NamedTuple whose keys
    # drive dispatch here.
    @struct render = begin
        data(df) = h.details(
            h.summary("Synthetic data ($(nrow(df)) rows × $(ncol(df)) cols: " *
                join(names(df), ", ") * ") — click to expand"),
            render_table(df; sortable=false),
        )
        parse(x) = h.section(
            h.h3("1. Meta.parse — raw Julia AST"),
            h.pre(x),
        )
        transform(x) = h.section(
            h.h3("2. parse! — rewritten AST (= → @n/@x assign, ~ → @n/@x ~)"),
            h.pre(x.transformed),
            h.h3("    locals classified by parse!"),
            h.pre(x.alllocals),
        )
        wrap(x) = h.section(
            h.h3("3. _brm — full let-block (df spliced as a literal)"),
            h.pre(x),
        )
        brmi(x) = h.section(
            h.h3("4. eval — BRMI value (parsed model)"),
            brmi_card(x),
        )
        vbrmi(x) = begin
            fd_summary = x.n_dead == 0 ?
                h.span("logdensity + FD check: $(x.dim)/$(x.dim) active ✓"; class="brm-status-ok") :
                h.span("logdensity + FD check: $(x.n_dead) dead param(s)"; class="brm-status-err")
            fd_body = h.div(
                h.p("dim = ", x.dim, ", logdensity = ", x.ldp),
                isempty(x.dead) ? "" :
                    h.p(; class="brm-status-err")("dead param indices: ", x.dead),
                h.pre(x.grad),
            )
            h.section(
                h.h3("5. VBRMI — materialized action (blocks, dim, columns)"),
                vbrmi_card(x.vbrmi),
                h.details(h.summary(fd_summary), fd_body),
            )
        end
        bench(x) = h.section(
            h.h3("6. Chairmarks @be — per-step"),
            [h.article(h.header(lbl), h.pre(b)) for (lbl, b) in x]...,
        )
        slic_model(x) = h.section(
            h.h3("5a. SlicModel — SBBRMI @slic body"),
            h.pre(x.model.model),
            h.p("data keys: ", h.code(sort(collect(keys(x.data))))),
        )
        stan_code(x) = h.section(
            h.h3("5b. StanCode — transpiled Stan source"),
            h.pre(x),
        )
        stan_compile(x) = h.section(
            h.h3("5c. StanCompile — BridgeStan shared library"),
            h.p("stan file: ", h.code(x.file)),
            h.p("compiled .so: ", h.code(x.lib)),
        )
        stan_instantiate(x) = h.section(
            h.h3("6a. StanInstantiate — model bound to data"),
            h.p("param_unc_num = ", x.dim),
            h.p("init (narrow normal, rng=Xoshiro(42)):"),
            h.pre(x.init),
        )
        stan_eval(x) = h.section(
            h.h3("6b. StanEval — log density at init"),
            h.p("log_density = ", x),
        )
        stan_shapes(df) = h.section(
            h.h3("6b'. StanShapes — index count per base parameter (p + tp + gq)"),
            h.p("total indexed entries: ", sum(df.n_indices),
                " across ", nrow(df), " base params"),
            render_table(df; sortable=true),
        )
        # Shared plot-tabset builder used by stan_generate and the fit stages.
        # `kind` goes into tab titles ("prior predictive" / "posterior") and
        # plot ids. Returns the tabset + wide-table details block.
        posterior_plots(long, wide, summary; id_prefix, kind, truth=nothing) = begin
            bands = [:q025 => :q975, :q10 => :q90, :q25 => :q75]
            pi_title   = "$kind (N=$(nrow(wide)) draws)"
            ecdf_title = "$kind — ECDF"
            lr_title   = "$kind — line + ribbon"
            den_title  = "$kind — histogram"
            # Overlay layers: for (x=:index, y=:value) plots, plot truth as
            # filled black dots at (:index, :truth); for (x=:value) plots,
            # overlay vertical rules at truth values, colored by :index to
            # match the base layer's (nominal-sorted) coloring.
            overlay_xy    = isnothing(truth) ? nothing :
                AoG.data(truth) * AoG.mapping(:index, :truth, row=:param) *
                AoG.visual(AoG.Scatter; color=:black)
            overlay_vrule = isnothing(truth) ? nothing :
                AoG.data(truth) * AoG.mapping(:truth; row=:param,
                                               color=:index => nonnumeric) *
                AoG.visual(VLines)
            add(spec, overlay) = isnothing(overlay) ? spec : spec + overlay
            spec_pi = add(AoG.data(summary) *
                          AoG.mapping(:index, :median, row=:param) *
                          pointinterval(; bands, orientation=:vertical),
                          overlay_xy) *
                      config(title=pi_title, facet=(; linkyaxes=:none))
            spec_lr = add(AoG.data(summary) *
                          AoG.mapping(:index, :median, row=:param) *
                          lineribbon(; bands),
                          overlay_xy) *
                      config(title=lr_title, facet=(; linkyaxes=:none))
            spec_hist = add(AoG.data(long) *
                            AoG.mapping(:value; row=:param,
                                        color=:index => nonnumeric) *
                            AoG.visual(ECDFPlot),
                            overlay_vrule) *
                        config(title=ecdf_title, facet=(; linkxaxes=:none))
            spec_den = add(AoG.data(long) *
                           AoG.mapping(:value; row=:param,
                                       color=:index => nonnumeric) *
                           AoG.histogram(; bins=30),
                           overlay_vrule) *
                       config(title=den_title, facet=(; linkxaxes=:none))
            tabs = tabset(
                "Point + Interval" => to_node(spec_pi;   id="$id_prefix-pi"),
                "Line + Ribbon"    => to_node(spec_lr;   id="$id_prefix-lr"),
                "ECDF"             => to_node(spec_hist; id="$id_prefix-ecdf"),
                "Histogram"        => to_node(spec_den;  id="$id_prefix-hist"),
                "Point + Interval (picker)" => with_plot_caption(spec_pi;
                    auto_remap=(; dims=["param" => "Parameter / TP / GQ"]),
                    title=pi_title, plot_id="$id_prefix-pi-pick"),
                "Line + Ribbon (picker)" => with_plot_caption(spec_lr;
                    auto_remap=(; dims=["param" => "Parameter / TP / GQ"]),
                    title=lr_title, plot_id="$id_prefix-lr-pick"),
                "ECDF (picker)" => with_plot_caption(spec_hist;
                    auto_remap=(; dims=["param" => "Parameter / TP / GQ",
                                         "index" => "Index (vector/matrix position)"]),
                    title=ecdf_title, plot_id="$id_prefix-ecdf-pick"),
                "Histogram (picker)" => with_plot_caption(spec_den;
                    auto_remap=(; dims=["param" => "Parameter / TP / GQ",
                                         "index" => "Index (vector/matrix position)"]),
                    title=den_title, plot_id="$id_prefix-hist-pick");
                id="$id_prefix-tabs",
            )
            wide_details = h.details(
                h.summary("Wide-format table (one row per draw, one column per indexed parameter)"),
                render_table(wide; sortable=true),
            )
            (; tabs, wide_details)
        end
        # Helper: build the optional predictive-check section. `which` is
        # `:prior` (prior predictive, no overlay) or `:posterior` (PPC -- pulls
        # observed y from `fit_data_dict[response]` and overlays it).
        build_ppc_section(long, which::Symbol; id_prefix) = begin
            pat = _ppc_pattern(context!().run.brmi)
            isnothing(pat) && return nothing
            df  = context!().run.df
            heading = which === :prior ? "Prior predictive" :
                                          "Posterior predictive check"
            title = "$(pat.response) vs $(pat.predictor) -- $heading"
            spec = if which === :prior
                _prior_ppc_spec(long, df, pat; title)
            else
                obs_y = context!().run.stan.fit_data_dict[Symbol(pat.response)]
                _posterior_ppc_spec(long, df, obs_y, pat; title)
            end
            isnothing(spec) && return nothing
            link_label = pat.link_fn === identity ? string(pat.loc) :
                         "$(pat.link_fn)($(pat.loc))"
            group_label = isnothing(pat.group) ? "" :
                          "; coloured by $(pat.group)"
            cap = which === :prior ?
                "Prior-predictive draws of $link_label vs $(pat.predictor)$group_label (no observation overlay -- $(pat.family) family)" :
                "Posterior draws of $link_label vs $(pat.predictor)$group_label, with observed $(pat.response) overlaid (black dots)"
            h.section(
                h.h4(heading, ": ", h.code(string(pat.response)), " vs ",
                     h.code(string(pat.predictor))),
                with_plot_caption(spec; plot_id="$id_prefix-ppc", title=cap),
            )
        end

        stan_generate(x) = begin
            (; long, wide, summary, truth) = x
            p = posterior_plots(long, wide, summary;
                                id_prefix="brm-plot-generated",
                                kind="Generated data (prior predictive)",
                                truth)
            ppc = build_ppc_section(long, :prior; id_prefix="brm-plot-generated")
            parts = Any[
                h.h3("6c. StanGenerate — synthetic data from narrow-normal prior + param_constrain"),
                h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                    "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
            ]
            isnothing(ppc) || push!(parts, ppc)
            push!(parts, p.tabs, p.wide_details)
            h.section(parts...)
        end
        stan_fit_pathfinder(x) = begin
            (; long, wide, summary, full_long, truth) = x
            p = posterior_plots(long, wide, summary;
                                id_prefix="brm-plot-pf",
                                kind="Pathfinder posterior",
                                truth)
            ppc = build_ppc_section(full_long, :posterior; id_prefix="brm-plot-pf")
            parts = Any[
                h.h3("6d. StanFit (Pathfinder) — variational approximation draws"),
                h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                    "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
            ]
            isnothing(ppc) || push!(parts, ppc)
            push!(parts, p.tabs, p.wide_details)
            h.section(parts...)
        end
        stan_fit_warmup(x) = begin
            (; long, wide, summary, full_long, diagnostics, truth) = x
            p = posterior_plots(long, wide, summary;
                                id_prefix="brm-plot-warmup",
                                kind="Warmup+MCMC posterior",
                                truth)
            ppc = build_ppc_section(full_long, :posterior; id_prefix="brm-plot-warmup")
            parts = Any[
                h.h3("6d'. StanFit (Warmup+MCMC) — full Stan fit"),
                h.p("n_divergent_samples: ", diagnostics.n_divergent_samples,
                    " · min ESS: ", minimum(diagnostics.ess)),
                h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                    "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
            ]
            isnothing(ppc) || push!(parts, ppc)
            push!(parts, p.tabs, p.wide_details)
            h.section(parts...)
        end
    end

    # Pre-canned formulas. The ones above the divider exercise individual
    # features in isolation; the last one stacks everything into a single
    # multi-likelihood model.
    presets = [
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
        "cbpp + therapeutic touch" => """log_odds_bin ~ 1 + c1 + (1 | g1)
bin_succ ~ Binomial(bin_n, logistic(log_odds_bin))

log_odds_b ~ 1 + (1 | g1)
bin_y ~ Bernoulli(logistic(log_odds_b))
""",
        "everything" => default_formula,
    ]

    @struct preset(label, formula) = begin
        button = h.button(label;
            type="button",
            class="brm-preset-btn",
            data_formula=formula,
            onclick="document.querySelector('textarea[name=formula]').value = this.dataset.formula; document.getElementById('stage-vbrmi').click()"
        )
    end

    @struct stage(label, id) = begin
        button = h.button(label;
            type="button",
            id="stage-$id",
            hx_get=string(query_url(__self__/"stage/$id"; force=true)),
            hx_include="#brm-macro-form",
            hx_target="#brm-macro-output",
            # `innerHTML` keeps the `#brm-macro-output` wrapper in the DOM
            # across swaps — including when polling_fetchindex throws and the
            # response is a bare error article with no matching id. Without
            # this, buttons target a gone id after the first failure.
            hx_swap="innerHTML")
    end

    @get index = begin
        # If an example form posted us a (label, formula) pair, persist the
        # edited formula to that example's .jl file so the next visit to the
        # Examples page shows the user's edits instead of the seed default.
        context!()
        h.div(
            h.h1("BRM macro pipeline"),
            h.p(
                "Enter a brms-style ", h.code("@brm"), " formula (one ",
                h.code("y ~ ..."), " line per outcome; predictors on the LHS via ",
                h.code("(name = expr)"), ") and step through the pipeline. ",
                "Stages 1-4 are the macro frontend (",
                h.code("Meta.parse"), " -> ", h.code("parse!"), " -> ", h.code("_brm"),
                " let-block -> ", h.code("eval"), " -> ", h.code("BRMI"),
                "). From the ", h.code("BRMI"), " you can branch into either ",
                h.code("VBRMI"), " (5/6 -- vectorized Julia LogDensityProblem + ",
                h.code("Chairmarks"), " benchmark; mostly a stub right now) or the ",
                "StanBlocks backend (5a-c: ", h.code("SlicModel"), " -> ",
                h.code("StanCode"), " -> ", h.code("StanCompile"),
                "; 6a-d': instantiate / eval / shapes / generate / fit via Pathfinder or full warmup).",
            ),
            h.aside(; style="border-left: 3px solid var(--pico-muted-border-color, #888); padding: 0.4rem 0.8rem; margin: 0.5rem 0; background: var(--pico-card-background-color, transparent);")(
                h.small(
                    h.strong("Errors: "),
                    "runtime exceptions are logged on the server (",
                    h.code("/tmp/htmxo_errors/<uid>.log"),
                    ") for the maintainer to inspect, but the page only surfaces a short ",
                    "message and an error UID -- the stack trace and full cause chain ",
                    "stay server-side. If a stage you expect to work returns an error, ",
                    "ping Niko with the formula text and the stage name (or the UID).",
                ),
            ),
            h.details(
                h.summary(h.small("Supported on the StanBlocks (SB) backend (5a-c / 6a-d')")),
                h.div(
                    h.h4("Likelihoods (RHS of ", h.code("y ~ ..."), ")"),
                    h.ul(
                        h.li(h.code("Normal(loc, sigma)")),
                        h.li(h.code("Bernoulli(p)"), ", ", h.code("BernoulliLogit(eta)")),
                        h.li(h.code("Binomial(N, p)"), " -- needs a trials column"),
                        h.li(h.code("Poisson(lambda)")),
                        h.li(h.code("NegativeBinomial(r, p)"), " -- ",
                             h.em("Distributions.jl parameterization; the emitted Stan model uses Stan's "),
                             h.code("neg_binomial(alpha, beta)"),
                             h.em(", so the posterior is NOT identical to the Julia side. Convert upstream if that matters.")),
                        h.li(h.code("Gamma(alpha, beta)")),
                        h.li(h.code("Beta(alpha, beta)")),
                        h.li(h.code("OrderedLogistic(eta)"), " -- integer outcome with K = ",
                             h.code("max(y)"), " levels; allocates K-1 ",
                             h.code("ordered"), " cutpoints with a ", h.code("std_normal"), " prior"),
                    ),
                    h.h4("Link transforms on the LHS"),
                    h.p(h.small(
                        "Any link whose Julia ", h.code("inverse"),
                        " resolves to a Stan-known function name works. Examples: ",
                        h.code("log(err) ~ ..."), " samples ", h.code("err"),
                        " on the log scale; ", h.code("logit(p) ~ ..."), " on the logit scale. ",
                        "Confirmed working: ", h.code("log"), ", ", h.code("exp"), ", ",
                        h.code("logit"), ", ", h.code("logistic"), ", ", h.code("sqrt"), ", ",
                        h.code("square"), ". Unknown links error at transpile time.",
                    )),
                    h.h4("Population predictors"),
                    h.ul(
                        h.li("Intercept ", h.code("1"), ", continuous covariates, arithmetic ",
                             h.code("+ - * / ^")),
                        h.li("Treatment-coded categoricals: any integer-typed column auto-expands to K-1 free betas (level 1 = reference)"),
                        h.li("Two-way interactions ", h.code("a & b"),
                             " (continuous x continuous only for now -- ",
                             h.code("*"), " full-interaction expansion is NOT implemented)"),
                        h.li("Standardization helpers: ", h.code("offset(x)"), ", ",
                             h.code("zscale(x)"), ", ", h.code("center(x)"), ", ",
                             h.code("standardize(x)"), ", ", h.code("protect(x)")),
                    ),
                    h.h4("Submodel functions"),
                    h.ul(
                        h.li(h.code("mo(c)"), " / ", h.code("mo1(c)"),
                             " -- monotonic effects (Buerkner & Charpentier 2018)"),
                        h.li(h.code("me(x_obs, sd_x)"),
                             " -- measurement error: a latent ", h.code("x_true"),
                             " is sampled with ", h.code("std_normal"),
                             " prior and ", h.code("x_obs ~ Normal(x_true, sd_x)"),
                             " emitted as the observation likelihood"),
                        h.li(h.code("s(x)"),
                             " -- natural cubic spline (truncated-power basis with knots at equally-spaced quantiles, ",
                             h.code("std_normal"),
                             " prior on basis coefficients). No smoothness penalty, no tensor products, no ",
                             h.code("bs"), " / ", h.code("t2"), " / ", h.code("gp"), " yet."),
                        h.li(h.code("ar(time, p=1)"),
                             " -- AR(1) noise (only ", h.code("p=1"),
                             "; rows are assumed already time-ordered, ",
                             h.code("time"), " is used as a length probe)"),
                    ),
                    h.h4("Random effects"),
                    h.ul(
                        h.li(h.code("(1 | g)"),
                             " intercept-only; ", h.code("(1 + x | g)"), ", ", h.code("(0 + x | g)"),
                             ", or any K-term combination -- correlated via LKJ-Cholesky + non-centered z, with marginal scales ",
                             h.code("tau ~ half-N(0,1)")),
                        h.li("Same-", h.code("g"), " blocks merge: ",
                             h.code("(1 | g) + (x | g)"), " is identical to ",
                             h.code("(1 + x | g)"), " (the LKJ + tau are shared by design)"),
                        h.li(h.code("(... | gr(g, by=b))"),
                             " -- stratified: independent LKJ + tau per level of ",
                             h.code("b"), " (errors at transpile time if any ",
                             h.code("g"), " straddles strata)"),
                        h.li(h.code("(e | ID | g)"), " or ", h.code("(... | gr(g, id=ID))"),
                             " -- cross-formula bucket coalescing: sub-formulas sharing an ",
                             h.code("ID"), " draw from one shared correlation block (brms style)"),
                    ),
                ),
            ),
            h.details(
                h.summary(h.small("Allowed functions in formulas (parser allowlist)")),
                h.p(h.small(
                    join(sort(collect(string.(s) for s in _ALLOWED_CALLS)), ", "),
                )),
            ),
            h.form(; id="brm-macro-form")(
                h.label("Load preset"),
                h.div(; class="brm-preset-row")(
                    [preset(lbl, body).button for (lbl, body) in presets]...,
                ),
                h.label("Formula")(
                    h.textarea(formula;
                        name="formula", rows=8,
                        class="brm-formula-textarea"),
                ),
                h.fieldset(; class="grid")(
                    stage("1. Parse",     :parse).button,
                    stage("2. Transform", :transform).button,
                    stage("3. Wrap",      :wrap).button,
                    stage("4. BRMI",      :brmi).button,
                ),
                h.small("Pick a branch:"),
                h.fieldset(; class="grid")(
                    stage("5. VBRMI",     :vbrmi).button,
                    stage("6. Benchmark", :bench).button,
                ),
                h.fieldset(; class="grid")(
                    stage("5a. SlicModel",  :slic_model).button,
                    stage("5b. StanCode",   :stan_code).button,
                    stage("5c. StanCompile", :stan_compile).button,
                ),
                h.fieldset(; class="grid")(
                    stage("6a. StanInstantiate", :stan_instantiate).button,
                    stage("6b. StanEval",        :stan_eval).button,
                    stage("6b'. StanShapes",     :stan_shapes).button,
                    stage("6c. StanGenerate",    :stan_generate).button,
                    stage("6d. StanFit (PF)",    :stan_fit_pathfinder).button,
                    stage("6d'. StanFit (Warmup)", :stan_fit_warmup).button,
                ),
                h.small("Bug-report helper:"),
                h.button("SB repro (current formula)";
                    type="submit",
                    formaction=string(__self__/"sb_repro"),
                    class="secondary"),
            ),
            # Persistent wrapper — buttons swap `innerHTML` into here so the
            # id survives polling/error responses.
            h.div(; id="brm-macro-output")(
                lazy(query_url(__self__/"stage/bench"; formula)),
            ),
        )
    end

    @get stage(name::Symbol; force::Bool=false) = polling_fetchindex(
        compute_steps, formula, context!().namespace, name;
        poll_url=query_url(__self__/"stage/$name"; formula, label),
        label="BRM pipeline - $name",
        force,
    ) do result
        # On successful stage computation, mark this stage + all prerequisite
        # stages as pass on the corresponding ExampleEntry (if label
        # identifies one). `result` keys are `(:data, <stages in reverse>)`;
        # filter :data out to get the stage symbols. No-op if label is empty
        # (main pipeline page without an example context) or label doesn't
        # match any saved example.
        if !isempty(label)
            entry = __parent__.examples.example_store.find(label)
            isnothing(entry) || entry.mark_stages!(;
                passed=[k for k in keys(result) if k !== :data])
        end
        # No id on this wrapper — the outer `#brm-macro-output` div in the
        # form is the persistent target (see buttons' `hx_swap="innerHTML"`);
        # putting the id here too would duplicate ids after a button swap.
        h.div(
            (getproperty(render, k)(v) for (k, v) in pairs(result))...,
        )
    end

    # Focused per-model views of the sbimpl intermediate artifacts. Each
    # route runs the pipeline just far enough and returns the relevant
    # source in `h.pre` (plus markdown_only serves the bare source via
    # `?plain` / `Accept: text/plain`, for piping into agents or curl).
    @get slic = h.pre(context!().run.sbbrmi.model.model)

    @get stan = h.pre(context!().run.stan.src)

    # One-stop bug-report page for the StanBlocks agent. Renders the SlicModel
    # body, generated Stan source, and BridgeStan compile output (success msg
    # or full error) for the current formula. HTMXO's `_resolve_response`
    # auto-converts to markdown when `Accept: text/plain` is requested, so the
    # same URL works for humans (browser) and agents (curl).
    #   curl -H 'Accept: text/plain' 'http://localhost:<port>/pipeline/sb_repro?formula=<url-encoded>'
    # Shared renderer: takes a ready `run` context + the raw formula string and
    # emits the bug-report HTML. Same output whether invoked via POST with an
    # edited formula (`sb_repro`) or via GET by example slug (`sb_repro_example`).
    _sb_repro_html(r, formula_str) = begin
        compile_out = try
            r.stan.lib
            "(compile succeeded — lib at `$(r.stan.lib)`)"
        catch e
            sprint(showerror, e)
        end
        h.div(
            h.h1("StanBlocks bug report"),
            h.h2("Formula"),
            h.pre(formula_str),
            h.h2("SlicModel body"),
            h.p(h.code("r.sbbrmi.model.model")),
            h.pre(r.sbbrmi.model.model),
            h.h2("Generated Stan source"),
            h.p(h.code("r.stan.src")),
            h.pre(r.stan.src),
            h.h2("BridgeStan compile output"),
            h.p(h.code("r.stan.lib")),
            h.pre(compile_out),
        )
    end

    @get sb_repro = _sb_repro_html(context!().run, formula)

    # Saved-example entry point for external agents: GET by slug so curl/agents
    # can reproduce a StanBlocks bug without POSTing a formula. The slug is the
    # URL-safe name used by /examples/<slug>.
    #   curl -H 'Accept: text/plain' 'http://.../pipeline/sb_repro_example?name=<slug>'
    @get sb_repro_example(; name::AbstractString="") = begin
        entry = __parent__.examples.example_store.find_by_slug(name)
        isnothing(entry) && return h.div(
            h.h1("Example not found"),
            h.p("No example with slug ", h.code(name)),
        )
        _sb_repro_html(context(entry.label, entry.formula).run, entry.formula)
    end
end

@htmx struct AppContext
    __appdata__ = APPDATA

    default_formula = """loc ~ 1
log(err) ~ 1
y1 ~ Normal(loc, err)
"""

    # Per-stage display labels, in the order they should appear in the UI
    # (pipeline page buttons, example-card indicators, etc.). The stage
    # symbols (`:parse`, `:transform`, …) are the DAG keys from
    # `__appdata__.step_chain`; labels here are UI-only.
    stage_labels = [
        :parse               => "1. Parse",
        :transform           => "2. Transform",
        :wrap                => "3. Wrap",
        :brmi                => "4. BRMI",
        :vbrmi               => "5. VBRMI",
        :bench               => "6. Benchmark",
        :slic_model          => "5a. SlicModel",
        :stan_code           => "5b. StanCode",
        :stan_compile        => "5c. StanCompile",
        :stan_instantiate    => "6a. StanInstantiate",
        :stan_eval           => "6b. StanEval",
        :stan_shapes         => "6b'. StanShapes",
        :stan_generate       => "6c. StanGenerate",
        :stan_fit_pathfinder => "6d. StanFit (PF)",
        :stan_fit_warmup     => "6d'. StanFit (Warmup)",
    ]

    # Page-level stylesheet read once at construction. Classes are consumed by
    # ExampleEntry.card / html_expr.jl; per-symbol / per-status colors that
    # are data-derived stay inline on the element.
    css = read(joinpath(@__DIR__, "brm-macro.css"), String)

    # HTMXObjects auto-uses `__page__` to wrap any route's return value into a
    # full page on direct browser navigation, while returning just the fragment
    # for HTMX requests (see `_resolve_response` in HTMXObjects.jl). The
    # sidebar's `hx-get` swaps target `#content` directly.
    __page__(content) = htmx(
        h.div(; class="brm-layout")(
            nav_sidebar([
                "Pipeline" => "/pipeline",
                "Examples" => "/examples",
            ])(
                # Site-wide Vega-Embed controls: zoom +/- and an "Actions"
                # toggle that exposes Vega-Embed's per-plot menu (download
                # PNG/SVG, view source, share via spec URL, ...).
                h.div(; style="margin-top:0.6rem; padding:0 0.6rem; font-size:0.85em;")(
                    vega_controls(),
                ),
            ),
            h.main(; class="container brm-main")(
                h.div(; id="content")(content),
            ),
        );
        pico_version="2",
        extra_head=(
            h.title("BRM macro action"),
            h.style(css),
            htmx_treebar_styles(),
            htmx_treebar_script(),
            sortable_table_js(),
            vega_head()...,
        ),
    )

    @param begin
        formula::String = default_formula
        label::String   = ""
    end

    # `/` mirrors the pipeline landing page.
    @get index = __self__.pipeline.index

    # Utility route: clears all in-memory caches on AppData (compute_steps
    # results, nested IP dicts, etc.). Useful after code changes when you
    # want to re-run a stage that's currently returning a stale cached
    # value. Mirrors bruno's OpsRoutes.reset.
    @get reset = begin
        clear_mem_caches!(__appdata__)
        h.p("In-memory caches cleared.")
    end

    @include pipeline = PipelineRoutes()

    # The Examples section mounts at /examples. Both the list view and per-slug
    # detail view share `@get index(slug)` (HTMXO registers `/examples` AND
    # `/examples/{slug}` thanks to slug's default). `@get mark` lives here too
    # since it operates exclusively on ExampleEntry; pill URLs hit /examples/mark.
    @include examples = begin
        (; examples_dir) = __appdata__
        # `label` is auto-forwarded from AppContext's `@param`; no explicit
        # `@param (; label) = __parent__` needed (and declaring it explicitly
        # collides with the auto-forward → "method overwritten" error).

        # Inline examples store. Constructs ExampleEntry with
        # `__parent__=__parent__` (the Examples sub-struct) so each entry's
        # rendering methods can build URLs via its parent chain.
        @struct example_store = begin
            entries() = begin
                isdir(examples_dir) || mkpath(examples_dir)
                files = sort(filter(endswith(".jl"),
                                    readdir(examples_dir; join=true));
                             by=mtime, rev=true)
                ExampleEntry[ExampleEntry(f; __parent__) for f in files]
            end
            find(label)        = findfirstelement(e -> e.label == label, entries())
            find_by_slug(slug) = findfirstelement(e -> e.slug == slug,   entries())
            persist!(label, formula) = begin
                e = find(label)
                e === nothing || e.save!(; new_formula=formula)
            end
        end

        @get mark(; state::Symbol=Symbol("")) =
            example_store.find(label).toggle_status!(state).card

        @get flag = example_store.find(label).cycle_flag!().card

        # List view vs detail view as separate derived-property methods; DO
        # supports multi-method dispatch on a single property name (the route
        # layer doesn't — see TODO below).
        # Sort-bar pill. `key` matches a `data-<key>` attribute on each card.
        # Click cycles off → desc → asc → off; drag between pills reorders
        # sort priority (left = highest). A small inline script below wires
        # this up to the `#brm-examples-list` container's DOM order.
        _sort_pill(key, label) = h.span(label;
            class="brm-sort-pill", draggable=true, data_sort_key=key)

        _sort_script = """
        (() => {
            const bar = document.querySelector('.brm-sort-bar');
            const list = document.querySelector('#brm-examples-list');
            const search = document.querySelector('#brm-examples-search');
            if (!bar || !list) return;
            const pills = Array.from(bar.querySelectorAll('.brm-sort-pill'));
            // Query is treated as a case-insensitive regex. Invalid regex
            // (e.g. while typing `(foo`) falls back to literal substring
            // match so the input never feels broken mid-keystroke.
            const filter = () => {
                const raw = (search && search.value || '').trim();
                let re = null, lit = '';
                if (raw) {
                    try { re = new RegExp(raw, 'i'); }
                    catch (_) { lit = raw.toLowerCase(); }
                }
                if (search) search.classList.toggle('brm-search-literal', !!lit);
                const cards = Array.from(list.querySelectorAll('.brm-example-card'));
                cards.forEach(c => {
                    const hay = c.dataset.label + ' ' + c.textContent;
                    const show = !raw
                        || (re && re.test(hay))
                        || (lit && hay.toLowerCase().includes(lit));
                    c.style.display = show ? '' : 'none';
                });
            };
            if (search) {
                const params = new URLSearchParams(window.location.search);
                const initial = params.get('q');
                if (initial) search.value = initial;
                const syncUrl = () => {
                    const url = new URL(window.location.href);
                    const v = search.value.trim();
                    if (v) url.searchParams.set('q', v);
                    else   url.searchParams.delete('q');
                    history.replaceState(null, '', url);
                };
                search.addEventListener('input', () => { filter(); syncUrl(); });
                window.addEventListener('popstate', () => {
                    const p = new URLSearchParams(window.location.search);
                    search.value = p.get('q') || '';
                    filter();
                });
            }
            // Global stage buttons: dispatch a click to the matching
            // per-card stage button of every currently visible card.
            document.querySelectorAll('.brm-global-btn').forEach(gb => {
                gb.addEventListener('click', () => {
                    const id = gb.dataset.stageId;
                    const cards = Array.from(list.querySelectorAll('.brm-example-card'))
                        .filter(c => c.style.display !== 'none');
                    cards.forEach(c => {
                        const btn = c.querySelector(
                            '.brm-branch-btn[data-stage-id="' + id + '"]');
                        if (btn) btn.click();
                    });
                });
            });
            // Defaults: `flagged` desc (triage candidates first),
            // `status` asc (open first), `brokenness` desc (most-broken
            // first), `complexity` asc (simple first). Other pills off.
            const state = { flagged: -1, status: 1, brokenness: -1, complexity: 1 };  // key -> 1 | -1 | 0
            const render = () => pills.forEach(p => {
                const s = state[p.dataset.sortKey] || 0;
                p.textContent = p.dataset.sortKey + (s === 1 ? ' ↑' : s === -1 ? ' ↓' : '');
                p.classList.toggle('active', s !== 0);
            });
            const resort = () => {
                const orderedPills = Array.from(bar.querySelectorAll('.brm-sort-pill'));
                const active = orderedPills
                    .map(p => [p.dataset.sortKey, state[p.dataset.sortKey] || 0])
                    .filter(([k, s]) => s !== 0);
                if (active.length === 0) return;
                const cards = Array.from(list.querySelectorAll('.brm-example-card'));
                cards.sort((a, b) => {
                    for (const [k, dir] of active) {
                        const aRaw = a.dataset[k], bRaw = b.dataset[k];
                        const aNum = parseFloat(aRaw), bNum = parseFloat(bRaw);
                        const numeric = !isNaN(aNum) && !isNaN(bNum);
                        const av = numeric ? aNum : aRaw;
                        const bv = numeric ? bNum : bRaw;
                        if (av < bv) return -dir;
                        if (av > bv) return dir;
                    }
                    return 0;
                });
                cards.forEach(c => list.appendChild(c));
            };
            pills.forEach(p => p.addEventListener('click', () => {
                const k = p.dataset.sortKey;
                state[k] = ({0:-1, '-1':1, 1:0})[state[k] || 0];
                render(); resort();
            }));
            // Drag-reorder
            let dragged = null;
            pills.forEach(p => {
                p.addEventListener('dragstart', e => { dragged = p; e.dataTransfer.effectAllowed='move'; });
                p.addEventListener('dragover',  e => { e.preventDefault(); });
                p.addEventListener('drop',      e => {
                    e.preventDefault();
                    if (dragged && dragged !== p) bar.insertBefore(dragged, p);
                    dragged = null;
                    resort();
                });
            });
            // When a card's outerHTML gets swapped (e.g. mark done /
            // deprioritize updates its data-status), re-run the sort so the
            // card moves to its new position.
            document.body.addEventListener('htmx:afterSwap', e => {
                if (list.contains(e.detail.target)) { resort(); filter(); }
            });
            // Success/error feedback for stage-button runs. A card gets a
            // green flash on success (auto-clears) or red border on failure
            // (persistent until the next fresh attempt on that card).
            const cardOf = el => el && el.closest && el.closest('.brm-example-card');
            document.body.addEventListener('htmx:beforeRequest', e => {
                const el = e.detail.elt;
                if (!el || !el.classList.contains('brm-branch-btn')) return;
                const card = cardOf(el);
                if (!card) return;
                card.classList.remove('brm-card-success', 'brm-card-error');
            });
            document.body.addEventListener('htmx:afterRequest', e => {
                const el = e.detail.elt;
                if (!el || !el.classList.contains('brm-branch-btn')) return;
                const card = cardOf(el);
                if (!card) return;
                const ok = e.detail.successful && e.detail.xhr.status < 400;
                if (ok) {
                    card.classList.add('brm-card-success');
                    setTimeout(() => card.classList.remove('brm-card-success'), 1200);
                } else {
                    card.classList.add('brm-card-error');
                }
            });
            render(); resort(); filter();
        })();
        """

        _index() = h.div(
            h.h1("Examples - coverage gaps and demos for BRM"),
            h.p("Each item has a sketch of what it is, why it matters, how to implement, and how to verify. Sourced from .jl files under ", h.code("web-macro/examples/"), "; status edits and edited formulas are written back to disk."),
            h.div(; class="brm-search-bar")(
                h.input(;
                    id="brm-examples-search",
                    type="search",
                    placeholder="Search (case-insensitive regex; e.g. interact|bruno)...",
                    autocomplete="off"),
            ),
            h.div(; class="brm-global-bar")(
                h.span("Run stage on all visible:"),
                [h.button(lbl;
                    type="button",
                    class="brm-global-btn",
                    data_stage_id=string(id),
                ) for (id, lbl) in __parent__.stage_labels]...,
            ),
            h.div(; class="brm-sort-bar")(
                h.span("Sort by (click to cycle, drag to reorder):"),
                _sort_pill("flagged",    "flagged"),
                _sort_pill("status",     "status"),
                _sort_pill("brokenness", "brokenness"),
                _sort_pill("complexity", "complexity"),
                _sort_pill("mtime",      "mtime"),
                _sort_pill("tier",       "tier"),
                _sort_pill("progress",   "progress"),
                _sort_pill("label",      "label"),
            ),
            h.div(; id="brm-examples-list")(
                [e.card for e in example_store.entries()]...,
            ),
            h.script(_sort_script),
        )

        _index(slug::AbstractString, stage::AbstractString) = h.div(
            h.p(h.a("<- Back to Examples"; href=__prefix__)),
            example_store.find_by_slug(slug).card_with_preload(stage; force_open=true),
        )

        # TODO(HTMXO): allow two `@get name` methods with distinct arities
        # (e.g. `@get index()` + `@get index(slug::String)`) to be registered
        # as separate paths `/examples` and `/examples/{slug}`. Today DO's
        # meta dict rejects duplicate route property names, so we delegate to
        # the multi-methoded `_index` helper above. `stage` is a query kwarg
        # that pre-populates the card's inline result div with that stage's
        # output -- used by the per-stage indicator pill links.
        @get index(slug::AbstractString=""; stage::AbstractString="") =
            isempty(slug) ? _index() : _index(slug, stage)
    end
end

function __init__()
    route!(AppContext())
end

# Bruno-specific extensions (gitignored); load if present. Adds a
# `dataset_extras(::Val{:bruno}, df)` method and optionally more.
let path = joinpath(@__DIR__, "bruno-ext.jl")
    isfile(path) && include(path)
end

end # module
