module BRMMacroWeb

using HTMXObjects
using DynamicObjects: fetchindex!
using Treebars: polling_fetchindex, initialize_progress!,
                prepare_progress!, with_prepared_progress,
                htmx_treebar_styles
using Random
using Chairmarks
using DataFrames
using Statistics: quantile, median
using FiniteDifferences: FiniteDifferences, central_fdm
using BridgeStan: BridgeStan
using StanLogDensityProblems: StanProblem
using WarmupHMC: initialize_mcmc, adaptive_warmup_mcmc
using JSON
using AlgebraOfVega: vega_head, auto_remap_node, with_plot_caption, config, pointinterval, lineribbon, to_node, ECDFPlot, VLines
import AlgebraOfGraphics as AoG

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

    status_pills = h.span(;
        id=status_id,
        class="brm-status-pills")(
        state_pill(:done,          "✓ done",          "mark done"),
        state_pill(:deprioritized, "✓ deprioritized", "deprioritize"),
    )

    formula_form = h.form(; class="brm-example-form")(
        h.input(; type="hidden", name="label", value=label),
        h.textarea(formula;
            name="formula",
            rows=max(3, count('\n', formula) + 1),
            class="brm-example-textarea"),
        h.button("cimpl (bench) ▶";
            type="button",
            class="brm-branch-btn",
            hx_get=string(query_url(__parent__.__parent__.pipeline/"stage/bench"; force=true)),
            hx_include="closest form",
            hx_target="#$result_id",
            hx_swap="innerHTML"),
        h.button("sbimpl (compile) ▶";
            type="button",
            class="brm-branch-btn",
            hx_get=string(query_url(__parent__.__parent__.pipeline/"stage/stan_compile"; force=true)),
            hx_include="closest form",
            hx_target="#$result_id",
            hx_swap="innerHTML"),
    )

    card = begin
        children = Any[HTMXObjects.md_to_node(body)]
        if formula !== nothing
            push!(children, formula_form)
            # Inline pipeline-result target — the form's hx_get fills this div
            # with the pipeline output so the user sees the VBRMI/FD/stan-compile
            # output right inside the card.
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

    save!(; new_status=status, new_formula=formula) = begin
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
        # Preserve __parent__ so the reloaded entry can still build route URLs.
        ExampleEntry(; __parent__, path)
    end

    toggle_status!(target) =
        save!(; new_status = status == target ? :open : target)
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

@dynamicstruct struct AppData
    __status__ = initialize_progress!(:state; description="BRM pipeline")
    examples_dir = joinpath(dirname(@__DIR__), "examples")

    dataset = Dataset()

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
        sbbrmi = SBBRMI(brmi)

        # Everything Stan-related bundled under `r.stan.*`. Nested access via
        # step_chain's tuple-path specs (e.g. `(:stan, :src)`).
        @struct stan = begin
            src = stan_code(sbbrmi)
            # Hash-keyed cache path so identical Stan source reuses the same
            # .stan (and co-located .so) across requests. `BridgeStan.compile_model`
            # invokes make, which skips when the .so is newer than the .stan —
            # so a cache hit resolves in milliseconds instead of re-running
            # the C++ build.
            file = begin
                p = joinpath(tempdir(), "brm_stan", string(hash(src)) * ".stan")
                mkpath(dirname(p))
                isfile(p) || write(p, src)
                p
            end
            lib      = BridgeStan.compile_model(file)
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
                overrides = Dict{Symbol, Any}()
                for (name, entries) in groups
                    if length(entries) == 1 && isempty(entries[1][1])
                        overrides[name] = entries[1][2]
                    else
                        orig = base[name]
                        out  = similar(orig, Float64)
                        for (idxs, v) in entries
                            out[idxs...] = v
                        end
                        overrides[name] = out
                    end
                end
                merge(base, overrides)
            end
            fit_instance = BridgeStan.StanModel(lib,
                StanBlocks.stan.bridgestan_data(fit_data_dict))

            # IP: Pathfinder init (fast, no MCMC). The `progress=__status__`
            # hook lets Treebars nest the 100 maxiters subtree under whatever
            # node called `fetchindex!(status, …, pathfinder, instance, init)`.
            pathfinder(instance, init; rng=Xoshiro(42), maxiters=100) =
                initialize_mcmc(StanProblem(instance), init; rng, progress=__status__, maxiters)
            # IP: full Stan + WarmupHMC fit. Same progress-hooking pattern.
            # Returns a rich NamedTuple with `.posterior_position`, `.ess`,
            # `.n_divergent_samples`, etc.
            posterior_warmup(instance, init; rng=Xoshiro(42), n_draws=200) =
                adaptive_warmup_mcmc(rng, StanProblem(instance); init, n_draws, progress=__status__)
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
            # Constrained posterior draws (+TP+GQ), plus (long, wide, summary).
            posterior_constrained = constrain_draws(posterior, fit_instance; rng_seed=46)
            posterior_dfs = dfs_from_constrained(posterior_constrained, generated_names)
            posterior_long_df    = posterior_dfs.long
            posterior_wide_df    = posterior_dfs.wide
            posterior_summary_df = posterior_dfs.summary

            # Parallel set for the warmup+MCMC path. The warmup IP cache is
            # warmed by `fetchindex!` in `compute_steps`; `@memo` hits it here.
            posterior_warmup_draws = (@memo posterior_warmup(fit_instance, init)).posterior_position
            posterior_warmup_constrained = constrain_draws(posterior_warmup_draws, fit_instance; rng_seed=47)
            posterior_warmup_dfs = dfs_from_constrained(posterior_warmup_constrained, generated_names)
            posterior_warmup_long_df    = posterior_warmup_dfs.long
            posterior_warmup_wide_df    = posterior_warmup_dfs.wide
            posterior_warmup_summary_df = posterior_warmup_dfs.summary
            posterior_warmup_diagnostics = begin
                w = @memo posterior_warmup(fit_instance, init)
                (; w.n_divergent_samples, ess=w.ess)
            end
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

    # DAG of step chains — one NamedTuple per stage target, keyed by step names
    # in dependency order. Built incrementally via `merge`: each stage's chain
    # is its parent's chain plus the one step it adds. The DAG branches are
    # visible in the `merge` calls (e.g. `slic_model = merge(brmi, …)` forks
    # off of `brmi`, parallel to `vbrmi`). Inner values are one of:
    #   - Symbol                       → `r.<sym>` (top-level property)
    #   - Tuple{Vararg{Symbol}}        → nested access, e.g. `(:stan, :src)` → `r.stan.src`
    #   - NamedTuple of (Symbol|Tuple) → bundle, keys preserved in the result.
    step_chain(name::Symbol) = begin
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
        stan_generate    = merge(stan_eval,  (; stan_generate=(; long=(:stan, :generated_df), wide=(:stan, :generated_wide_df), summary=(:stan, :generated_summary_df), truth=(:stan, :truth_df))))
        # Pathfinder / full warmup are computed via `fetchindex!` in
        # `compute_steps` (special-cased below by step name) so the IP's
        # progress subtree attaches to the step's phase. Chain-level specs
        # read the resulting cached values back out as plain properties.
        stan_fit_pathfinder = merge(stan_generate, (; stan_fit_pathfinder=(; long=(:stan, :posterior_long_df), wide=(:stan, :posterior_wide_df), summary=(:stan, :posterior_summary_df), truth=(:stan, :truth_df))))
        stan_fit_warmup     = merge(stan_fit_pathfinder, (; stan_fit_warmup=(; long=(:stan, :posterior_warmup_long_df), wide=(:stan, :posterior_warmup_wide_df), summary=(:stan, :posterior_warmup_summary_df), diagnostics=(:stan, :posterior_warmup_diagnostics), truth=(:stan, :truth_df))))
        (; parse, transform, wrap, brmi, vbrmi, bench, slic_model, stan_code, stan_compile,
           stan_instantiate, stan_eval, stan_generate, stan_fit_pathfinder, stan_fit_warmup)[name]
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
        chain = step_chain(name)
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
        # Shared plot-tabset builder used by stan_generate and the fit stages.
        # `kind` goes into tab titles ("prior predictive" / "posterior") and
        # plot ids. Returns the tabset + wide-table details block.
        posterior_plots(long, wide, summary; id_prefix, kind, truth=nothing) = begin
            bands = [:q025 => :q975, :q10 => :q90, :q25 => :q75]
            pi_title   = "$kind (N=$(nrow(wide)) draws)"
            ecdf_title = "$kind — ECDF"
            lr_title   = "$kind — line + ribbon"
            den_title  = "$kind — histogram"
            indep_x = config(facet=(; linkxaxes=:none))
            indep_y = config(facet=(; linkyaxes=:none))
            # Overlay layers: for (x=:index, y=:value) plots, plot truth as
            # filled black dots at (:index, :truth); for (x=:value) plots,
            # overlay vertical rules at truth values, colored by :index to
            # match the base layer's coloring.
            overlay_xy    = isnothing(truth) ? nothing :
                AoG.data(truth) * AoG.mapping(:index, :truth, row=:param) *
                AoG.visual(AoG.Scatter; color=:black)
            overlay_vrule = isnothing(truth) ? nothing :
                AoG.data(truth) * AoG.mapping(:truth; row=:param, color=:index) *
                AoG.visual(VLines)
            add(spec, overlay) = isnothing(overlay) ? spec : spec + overlay
            spec_pi = add(AoG.data(summary) *
                          AoG.mapping(:index, :median, row=:param) *
                          pointinterval(; bands, orientation=:vertical),
                          overlay_xy) *
                      config(title=pi_title) * indep_y
            spec_lr = add(AoG.data(summary) *
                          AoG.mapping(:index, :median, row=:param) *
                          lineribbon(; bands),
                          overlay_xy) *
                      config(title=lr_title) * indep_y
            spec_hist = add(AoG.data(long) *
                            AoG.mapping(:value; row=:param, color=:index) *
                            AoG.visual(ECDFPlot),
                            overlay_vrule) *
                        config(title=ecdf_title) * indep_x
            spec_den = add(AoG.data(long) *
                           AoG.mapping(:value; row=:param, color=:index) *
                           AoG.histogram(),
                           overlay_vrule) *
                       config(title=den_title) * indep_x
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
        stan_generate(x) = begin
            (; long, wide, summary, truth) = x
            p = posterior_plots(long, wide, summary;
                                id_prefix="brm-plot-generated",
                                kind="Generated data (prior predictive)",
                                truth)
            h.section(
                h.h3("6c. StanGenerate — synthetic data from narrow-normal prior + param_constrain"),
                h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                    "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
                p.tabs, p.wide_details,
            )
        end
        stan_fit_pathfinder(x) = begin
            (; long, wide, summary, truth) = x
            p = posterior_plots(long, wide, summary;
                                id_prefix="brm-plot-pf",
                                kind="Pathfinder posterior",
                                truth)
            h.section(
                h.h3("6d. StanFit (Pathfinder) — variational approximation draws"),
                h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                    "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
                p.tabs, p.wide_details,
            )
        end
        stan_fit_warmup(x) = begin
            (; long, wide, summary, diagnostics, truth) = x
            p = posterior_plots(long, wide, summary;
                                id_prefix="brm-plot-warmup",
                                kind="Warmup+MCMC posterior",
                                truth)
            h.section(
                h.h3("6d'. StanFit (Warmup+MCMC) — full Stan fit"),
                h.p("n_divergent_samples: ", diagnostics.n_divergent_samples,
                    " · min ESS: ", minimum(diagnostics.ess)),
                h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                    "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
                p.tabs, p.wide_details,
            )
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
    @get sb_repro = begin
        r = context!().run
        compile_out = try
            r.stan.lib
            "(compile succeeded — lib at `$(r.stan.lib)`)"
        catch e
            sprint(showerror, e)
        end
        h.div(
            h.h1("StanBlocks bug report"),
            h.h2("Formula"),
            h.pre(formula),
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
end

@htmx struct AppContext
    __appdata__ = APPDATA

    default_formula = """loc ~ 1
log(err) ~ 1
y1 ~ Normal(loc, err)
"""

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
            ]),
            h.main(; class="container brm-main")(
                h.div(; id="content")(content),
            ),
        );
        pico_version="2",
        extra_head=(
            h.title("BRM macro action"),
            h.style(css),
            htmx_treebar_styles(),
            vega_head()...,
        ),
    )

    @param begin
        formula::String = default_formula
        label::String   = ""
    end

    # `/` mirrors the pipeline landing page.
    @get index = __self__.pipeline.index

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
                ExampleEntry[ExampleEntry(; __parent__=__parent__, path=f) for f in files]
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

        # List view vs detail view as separate derived-property methods; DO
        # supports multi-method dispatch on a single property name (the route
        # layer doesn't — see TODO below).
        _index() = h.div(
            h.h1("Examples - coverage gaps and demos for BRM"),
            h.p("Sorted by last modified. Each item has a sketch of what it is, why it matters, how to implement, and how to verify. Sourced from .jl files under ", h.code("web-macro/examples/"), "; status edits and edited formulas are written back to disk."),
            [e.card for e in example_store.entries()]...,
        )

        _index(slug::AbstractString) = h.div(
            h.p(h.a("<- Back to Examples"; href=__prefix__)),
            example_store.find_by_slug(slug).card,
        )

        # TODO(HTMXO): allow two `@get name` methods with distinct arities
        # (e.g. `@get index()` + `@get index(slug::String)`) to be registered
        # as separate paths `/examples` and `/examples/{slug}`. Today DO's
        # meta dict rejects duplicate route property names, so we delegate to
        # the multi-methoded `_index` helper above.
        @get index(slug::String="") = isempty(slug) ? _index() : _index(slug)
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
