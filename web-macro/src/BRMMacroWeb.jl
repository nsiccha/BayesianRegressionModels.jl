module BRMMacroWeb

using HTMXObjects
using Treebars: polling_fetchindex, initialize_progress!
using Random
using Chairmarks
using DataFrames
using FiniteDifferences: FiniteDifferences, central_fdm
using BridgeStan: BridgeStan

# The @brm macro and the VBRMI / SBBRMI implementations live alongside this
# module so Revise tracks them. The scripts/ entry points (parsing.jl,
# Benchmarking, StanBlocksImpl) include them via relative paths into here.
include("macro.jl")
include("vimpl.jl")
include("sbimpl.jl")
# Styled HTML rendering for BRMI / VBRMI cards. Lives in its own file
# because it will eventually move to an ext of the main package.
include("html_expr.jl")

# Extension hook: extensions (e.g. the gitignored `bruno-ext.jl`) that need
# to contribute auxiliary data which doesn't fit as per-row DataFrame
# columns -- for example `dose_times::Vector{<:AbstractVector}` indexed by
# subject id -- add a method `dataset_extras(::Val{:ns}, df)` returning a
# NamedTuple of extras. The namespace is derived from the example
# label/slug (first dash/space-separated segment), so `bruno-qt-*` examples
# dispatch to `::Val{:bruno}`. Default is no extras.
dataset_extras(::Val, df) = (;)

# DOs in dependency order. Every feature is a focused @dynamicstruct: the
# pipeline stages live as derived properties on `BRMRun`, example file I/O
# on `ExampleStore`, synthetic + namespace-merged data on `Dataset`, and
# AST parse/safety/transform on `Formula`. `AppData` is just a thin holder
# of sub-DOs plus the `pipeline_run` polling entry point; `AppContext` is
# the HTMX routes and rendering layer.
struct FormulaSecurityError <: Exception
    msg::String
end
Base.showerror(io::IO, e::FormulaSecurityError) = print(io, "FormulaSecurityError: ", e.msg)

_ALLOWED_CALLS = Set{Symbol}([
    :~, :(+), :(-), :(*), :(/), :(^), :(|), :(||),
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
    :scale, :center, :standardize, :factor, :offset,
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
        alllocals = OrderedDict{Symbol,Symbol}()
        (; ex=parse!(deepcopy(raw); info=(;alllocals)), alllocals)
    end
    transformed = _t.ex
    alllocals   = _t.alllocals
end
@dynamicstruct struct Dataset
    n::Int = 64
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
    body    = _parsed.body
    formula = _parsed.formula
    slug    = replace(basename(path), r"\.jl$" => "")

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

    permalink = h.a("🔗";
        href="/examples/$(HTTP.URIs.escapeuri(slug))",
        title="Standalone URL",
        onclick="event.stopPropagation()",
        class="brm-permalink")

    state_pill(target_state, active_text, inactive_text) = begin
        is_active = status == target_state
        bg = is_active ? get(_STATUS_COLORS, target_state, "#888") : "#888"
        h.button(is_active ? active_text : inactive_text;
            type="button",
            class="brm-state-pill",
            hx_get="/mark?label=$(HTTP.URIs.escapeuri(label))&state=$target_state",
            hx_target="#$card_id",
            hx_swap="outerHTML",
            onclick="event.stopPropagation()",
            style="background:$bg",
        )
    end

    status_pills = h.span(;
        id=status_id,
        class="brm-status-pills")(
        state_pill(:done,          "✓ done",          "mark done"),
        state_pill(:deprioritized, "✓ deprioritized", "deprioritize"),
    )

    formula_form(routes) = h.form(; class="brm-example-form")(
        h.input(; type="hidden", name="label", value=label),
        h.textarea(formula;
            name="formula",
            rows=max(3, count('\n', formula) + 1),
            class="brm-example-textarea"),
        h.button("cimpl (bench) ▶";
            type="button",
            class="brm-branch-btn",
            hx_get=string(query_url(routes/"stage/bench"; force=true)),
            hx_include="closest form",
            hx_target="#$result_id",
            hx_swap="innerHTML"),
        h.button("sbimpl (compile) ▶";
            type="button",
            class="brm-branch-btn",
            hx_get=string(query_url(routes/"stage/stan_compile"; force=true)),
            hx_include="closest form",
            hx_target="#$result_id",
            hx_swap="innerHTML"),
    )

    card(routes) = begin
        children = Any[HTMXObjects.md_to_node(body)]
        if formula !== nothing
            push!(children, formula_form(routes))
            # Inline pipeline-result target — the form's hx_get fills this div
            # with `render_pipeline(out)` so the user sees the
            # VBRMI/finite-difference output right inside the card.
            push!(children, h.div(;
                id=result_id,
                class="brm-example-result"))
        end
        # `:open` status → expanded; `:done`/`:deprioritized` → collapsed by
        # default. Pills sit inside the <summary> so they're always reachable,
        # but their onclick stops propagation so clicking a pill doesn't also
        # toggle the disclosure.
        h.article(;
            id=card_id,
            class="brm-example-card",
            style="border-left-color:$border_color",
        )(
            h.details(; open=(status == :open))(
                h.summary(; class="brm-example-summary")(
                    tier_pill, " ",
                    h.strong(label), " ",
                    status_pills, " ",
                    permalink,
                ),
                h.div(; class="brm-example-body")(children...),
            ),
        )
    end

    write_with!(; new_status=status, new_formula=formula) = begin
        io = IOBuffer()
        println(io, "# label: ", label)
        println(io, "# tier: ", tier)
        println(io, "# status: ", new_status)
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
        ExampleEntry(; path)
    end
end
@dynamicstruct struct ExampleStore
    dir::String

    # `entries` is a method, not a cached field, because `save!` writes to
    # disk and we want subsequent reads to see the new file mtime/content.
    entries() = begin
        isdir(dir) || mkpath(dir)
        files = sort(filter(endswith(".jl"),
                            readdir(dir; join=true));
                     by=mtime, rev=true)
        ExampleEntry[ExampleEntry(; path=f) for f in files]
    end

    find(label) = begin
        for e in entries()
            e.label == label && return e
        end
        nothing
    end
    find_by_slug(slug) = begin
        for e in entries()
            e.slug == slug && return e
        end
        nothing
    end

    save!(label; new_status=nothing, new_formula=nothing) = begin
        e = find(label)
        e === nothing && return nothing
        e.write_with!(;
            new_status = new_status === nothing ? e.status  : new_status,
            new_formula = new_formula === nothing ? e.formula : new_formula,
        )
    end
end
# One run of the @brm pipeline for a given (text, namespace) pair. Every
# stage is a derived property -- accessing `run.brmi` triggers parse + eval,
# accessing `run.benches` triggers vbrmi + finite-difference check + the
# benchmark loop, etc. Unused branches don't compute. Safety is enforced
# at the one point where it matters: `wrapped` refuses to produce Julia
# code for an unsafe formula.

@dynamicstruct struct BRMRun
    text::String
    namespace::Symbol = :default
    (; dataset) = __parent__

    formula = Formula(; text)
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

    benches = begin
        x_rand = randn(dim)
        bs = Pair{String,Any}[]
        push!(bs, "logdensity (total)" =>
            @be randn(dim) LogDensityProblems.logdensity($vbrmi, _))
        push!(bs, "lprior!" =>
            @be randn(dim) lprior!($vbrmi, _))
        # Per-Part lprior! split: the foldl in lprior!(blocks, x) hands
        # each Part a view of exactly nparams(part) reals. Reconstruct
        # those slices here so each Part's contribution can be benched
        # in isolation.
        let pos = 0
            for (group_key, parts) in pairs(vbrmi.meta.blocks)
                for (i, part) in enumerate(parts)
                    n = nparams(part)
                    xi = view(x_rand, pos+1:pos+n)
                    push!(bs, "  lprior!($group_key[$i] $(part))" =>
                        @be lprior!($part, $xi))
                    pos += n
                end
            end
        end
        # llikelihood! splits: each materialized column (either a
        # linear-predictor MaterializedColumn or a LikelihoodColumn).
        _ = lprior!(vbrmi, x_rand)
        for (key, m) in pairs(vbrmi.meta.materialized)
            push!(bs, "llikelihood!($key)" =>
                @be llikelihood!($m))
        end
        bs
    end

    # ── stan branch ──
    sbbrmi    = SBBRMI(brmi)
    stan_src  = stan_code(sbbrmi)
    stan_file = begin
        f = tempname() * ".stan"
        write(f, stan_src)
        f
    end
    stan_lib = BridgeStan.compile_model(stan_file)
end
@dynamicstruct struct AppData
    __status__ = initialize_progress!(:state; description="BRM pipeline")
    examples_dir = joinpath(dirname(@__DIR__), "examples")

    default_formula = """loc1 ~ 1 + a + c1 + (1 + b + c1 | g1) + (1 | g2)
log(err1) ~ 1 + d
y1 ~ Normal(loc1, err1)

log_rate ~ 1 + a + (1 | g3)
k1 ~ Poisson(exp(log_rate))

log_odds_bin ~ 1 + c2 + (1 | g2)
bin_succ ~ Binomial(bin_n, logistic(log_odds_bin))

log_odds_b ~ 1 + b
bin_y ~ Bernoulli(logistic(log_odds_b))
"""

    dataset       = Dataset()
    example_store = ExampleStore(; dir=examples_dir)

    namespace_from(label) = isempty(strip(label)) ? :default :
        Symbol(lowercase(first(split(strip(label), r"[\s:\-]+"))))

    # Ordered pipeline stages. Index gates which BRMRun properties
    # `pipeline_run` touches (and which sections `render_pipeline` shows).
    stages = (:parse, :transform, :wrap, :brmi,
              :vbrmi, :bench,
              :slic_model, :stan_code, :stan_compile)
    stage_index(s) = something(findfirst(==(s), stages), length(stages))

    # Indexable property: `appdata.run[text, ns]` is cached in-memory per key,
    # `appdata.run(text, ns)` is fresh each call.
    run(text, namespace=:default) =
        BRMRun(; __parent__=__self__, text, namespace)

    # Indexable fetch for `polling_fetchindex` (accessed via brackets by the
    # caller). Touches BRMRun properties up through `stage` so the heavy work
    # lands inside the polled task rather than the HTTP response callback.
    pipeline_run(text, stage::Symbol, namespace=:default) = begin
        r = run[text, namespace]
        s = stage_index(stage)
        s >= 1 && r.formula.raw
        s >= 2 && r.formula.transformed
        s >= 3 && r.wrapped
        s >= 4 && r.brmi
        stage === :vbrmi       && r.grad
        stage === :bench       && r.benches
        stage === :slic_model  && r.sbbrmi
        stage === :stan_code   && r.stan_src
        stage === :stan_compile && r.stan_lib
        r
    end
end

APPDATA = AppData(; cache_type=:parallel)
@htmx struct AppContext
    __appdata__ = APPDATA
    (; default_formula, dataset, example_store, namespace_from,
       stages, stage_index, run, pipeline_run) = __appdata__

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
                "Pipeline" => "/",
                "Examples" => "/examples",
            ]),
            h.main(; class="container brm-main")(
                h.div(; id="content")(content),
            ),
        );
        pico_version="2",
        extra_head=(
            h.title("BRM macro action"),
            h.style(__self__.css),
        ),
    )

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

    preset_button(label, formula) = h.button(label;
        type="button",
        class="brm-preset-btn",
        data_formula=formula,
        onclick="document.querySelector('textarea[name=formula]').value = this.dataset.formula; document.getElementById('stage-vbrmi').click()")

    stage_button(label, stage) = h.button(label;
        type="button",
        id="stage-$stage",
        hx_get=string(query_url(__self__/"stage/$stage"; force=true)),
        hx_include="#brm-macro-form",
        hx_target="#brm-macro-output",
        hx_swap="outerHTML")

    render_pipeline(r, stage) = begin
        s = stage_index(stage)
        sections = Vector{Any}[]

        # Synthetic data pinned at top, collapsed by default so the macro
        # pipeline output stays the focus.
        data_section = Any[
            h.details(
                h.summary("Synthetic data ($(nrow(r.df)) rows × $(ncol(r.df)) cols: " *
                    join(string.(names(r.df)), ", ") * ") — click to expand"),
                render_table(r.df; sortable=false),
            ),
        ]

        s >= 1 && push!(sections, Any[
            h.h3("1. Meta.parse — raw Julia AST"),
            h.pre(sprint(show, r.formula.raw)),
        ])
        s >= 2 && push!(sections, Any[
            h.h3("2. parse! — rewritten AST (= → @n/@x assign, ~ → @n/@x ~)"),
            h.pre(sprint(show, r.formula.transformed)),
            h.h3("    locals classified by parse!"),
            h.pre(sprint(show, r.formula.alllocals)),
        ])
        s >= 3 && push!(sections, Any[
            h.h3("3. _brm — full let-block (df spliced as a literal)"),
            h.pre(sprint(show, r.wrapped)),
        ])
        s >= 4 && push!(sections, Any[
            h.h3("4. eval — BRMI value (parsed model)"),
            brmi_card(r.brmi),
        ])

        if stage in (:vbrmi, :bench)
            tol = 1e-8
            n_dead = count(<=(tol) ∘ abs, r.grad)
            fd_summary = n_dead == 0 ?
                h.span("logdensity + FD check: $(r.dim)/$(r.dim) active ✓";
                    class="brm-status-ok") :
                h.span("logdensity + FD check: $(n_dead) dead param(s)";
                    class="brm-status-err")
            dead = findall(<=(tol) ∘ abs, r.grad)
            fd_body = h.div(
                h.p("dim = ", string(r.dim), ", logdensity = ", r.ldp),
                isempty(dead) ? "" :
                    h.p(; class="brm-status-err")(
                        "dead param indices: ", string(dead)),
                h.pre(sprint(show, MIME"text/plain"(), r.grad)),
            )
            push!(sections, Any[
                h.h3("5. VBRMI — materialized action (blocks, dim, columns)"),
                vbrmi_card(r.vbrmi),
                h.details(h.summary(fd_summary), fd_body),
            ])
        end
        if stage === :bench
            bench_rows = [h.div(
                h.strong(lbl), h.br(),
                h.pre(sprint(show, MIME"text/plain"(), b))
            ) for (lbl, b) in r.benches]
            push!(sections, Any[h.h3("6. Chairmarks @be — per-step"), bench_rows...])
        end
        if stage in (:slic_model, :stan_code, :stan_compile)
            push!(sections, Any[
                h.h3("5a. SlicModel — SBBRMI @slic body"),
                h.pre(sprint(show, r.sbbrmi.model.model)),
                h.p("data keys: ",
                    h.code(string(sort(collect(keys(r.sbbrmi.data)))))),
            ])
        end
        if stage in (:stan_code, :stan_compile)
            push!(sections, Any[
                h.h3("5b. StanCode — transpiled Stan source"),
                h.pre(r.stan_src),
            ])
        end
        if stage === :stan_compile
            push!(sections, Any[
                h.h3("5c. StanCompile — BridgeStan shared library"),
                h.p("stan file: ", h.code(r.stan_file)),
                h.p("compiled .so: ", h.code(r.stan_lib)),
            ])
        end

        # Stages render most-recent-first; synthetic data sits at the very top.
        children = reduce(vcat, reverse(sections); init=Any[])
        prepend!(children, data_section)
        h.div(; id="brm-macro-output")(children...)
    end

    index_body(formula) = h.div(
        h.h1("BRM macro pipeline"),
        h.p(
            "Enter a ", h.code("@brm"), " formula and step through the macro pipeline: ",
            h.code("Meta.parse"), " -> ", h.code("parse!"), " -> ", h.code("_brm"),
            " let-block -> ", h.code("eval"), " -> ", h.code("VBRMI"), " action -> ",
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
            h.div(; class="brm-preset-row")(
                [preset_button(lbl, body) for (lbl, body) in presets]...,
            ),
            h.label("Formula")(
                h.textarea(formula;
                    name="formula", rows=8,
                    class="brm-formula-textarea"),
            ),
            h.fieldset(; class="grid")(
                stage_button("1. Parse",     :parse),
                stage_button("2. Transform", :transform),
                stage_button("3. Wrap",      :wrap),
                stage_button("4. BRMI",      :brmi),
            ),
            h.small("Pick a branch:"),
            h.fieldset(; class="grid")(
                stage_button("5. VBRMI",     :vbrmi),
                stage_button("6. Benchmark", :bench),
            ),
            h.fieldset(; class="grid")(
                stage_button("5a. SlicModel",  :slic_model),
                stage_button("5b. StanCode",   :stan_code),
                stage_button("5c. StanCompile",:stan_compile),
            ),
        ),
        lazy(string(query_url(__self__/"stage/bench"; formula)); id="brm-macro-output"),
    )

    @get index(; formula::String=default_formula, label::String="") = begin
        # If an example form posted us a (label, formula) pair, persist the
        # edited formula to that example's .jl file so the next visit to the
        # Examples page shows the user's edits instead of the seed default.
        isempty(label) || example_store.save!(label; new_formula=formula)
        index_body(formula)
    end

    @get mark(; label::String="", state::String="") = begin
        isempty(label) && return ""
        target = Symbol(state)
        entry = example_store.find(label)
        entry === nothing && return ""
        next_status = entry.status == target ? :open : target
        updated = example_store.save!(label; new_status=next_status)
        # Re-render the whole card so the border + collapse state update
        # together with the pill text.
        updated === nothing ? "" : updated.card(__self__)
    end

    @get stage(name::AbstractString; formula::String=default_formula,
               label::String="", force::Bool=false) = begin
        # When called from an example card's form, persist the (possibly
        # edited) formula back to the example's .jl file before rendering.
        isempty(label) || example_store.save!(label; new_formula=formula)
        ns = namespace_from(label)
        stage_sym = Symbol(name)
        polling_fetchindex(pipeline_run,
                           formula, stage_sym, ns;
                           poll_url=string(query_url(__self__/"stage/$name"; formula, label)),
                           label="BRM pipeline - $name",
                           force) do r
            render_pipeline(r, stage_sym)
        end
    end

    # Focused per-model views of the sbimpl intermediate artifacts. Each
    # route runs the pipeline just far enough and returns the relevant
    # source in `h.pre` (plus markdown_only serves the bare source via
    # `?plain` / `Accept: text/plain`, for piping into agents or curl).
    @get slic(; formula::String=default_formula, label::String="") = begin
        isempty(label) || example_store.save!(label; new_formula=formula)
        h.pre(sprint(show, run(formula, namespace_from(label)).sbbrmi.model.model))
    end

    @get stan(; formula::String=default_formula, label::String="") = begin
        isempty(label) || example_store.save!(label; new_formula=formula)
        h.pre(run(formula, namespace_from(label)).stan_src)
    end

    @get examples(slug::String="") = begin
        if !isempty(slug)
            entry = example_store.find_by_slug(slug)
            entry === nothing && return h.div(
                h.p("No example with slug ", h.code(slug), "."),
                h.a("<- Back to Examples"; href="/examples"),
            )
            return h.div(
                h.p(h.a("<- Back to Examples"; href="/examples")),
                entry.card(__self__),
            )
        end
        h.div(
            h.h1("Examples - coverage gaps and demos for BRM"),
            h.p("Sorted by last modified. Each item has a sketch of what it is, why it matters, how to implement, and how to verify. Sourced from .jl files under ", h.code("web-macro/examples/"), "; status edits and edited formulas are written back to disk."),
            [e.card(__self__) for e in example_store.entries()]...,
        )
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
