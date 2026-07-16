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
        isempty(label) || __parent__.examples.persist!(label, formula)
        context(label, formula)
    end

    # Per-stage bundle: indexed `@include` mounts each `(name)` instance's
    # routes under `/pipeline/stage/<name>/...`. `@get index` is the polling
    # progress page (formerly `@get stage(name; force)`); the per-step
    # renderers (`data`, `parse`, …, `stan_section`) are the dispatch targets
    # `index`'s `render_step` selects from. Single owner of the
    # `(formula, label, name)` axis — replaces the old sibling pair
    # `@include render = …` + `@get stage(name; force) = …`.
    @include stage(name::Symbol) = begin
        @get index(; force::Bool=false) = polling_fetchindex(
            __parent__.compute_steps, formula,
            __parent__.context!().namespace, name;
            poll_url=query_url(__self__; formula, label),
            label="BRM pipeline - $name",
            # Honour `force=true` only when the request actually came from
            # HTMX (button click). A direct browser reload of a pushed URL
            # has no HX-Request header, so we ignore `force` and let the
            # polling_fetchindex IP cache re-attach to whatever's already
            # running / cached for this (formula, name).
            force=force && is_htmx(__req__),
        ) do result
            # On successful stage computation, mark this stage + all
            # prerequisite stages as pass on the corresponding ExampleEntry
            # (if label identifies one). `result` keys are
            # `(:data, <stages in reverse>)`; filter :data out to get the
            # stage symbols. No-op if label is empty (main pipeline page
            # without an example context) or label doesn't match any
            # saved example.
            if !isempty(label)
                entry = __parent__.__parent__.examples.find(label)
                isnothing(entry) || entry.mark_stages!(;
                    passed=[k for k in keys(result) if k !== :data])
            end
            # Step-key dispatch. `compute_steps` already populated the IP
            # cache under progress phases; renderers below are bare
            # properties that read `__parent__.context!().run.<…>` themselves — we
            # only need the keys here. `:stan_generate` routes to
            # `stan_section.generate`; `:stan_fit_{pathfinder,warmup}` to
            # `stan_section.fit.<kind>`; other `stan_*` to `stan.<suffix>`;
            # everything else to `__self__.<key>`.
            render_step(k::Symbol) = begin
                s = String(k)
                if k === :stan_generate
                    stan_section.generate
                elseif k === :stan_fit_pathfinder
                    stan_section.fit.pathfinder
                elseif k === :stan_fit_warmup
                    stan_section.fit.warmup
                elseif startswith(s, "stan_")
                    getproperty(stan, Symbol(s[6:end]))
                else
                    getproperty(__self__, k)
                end
            end
            # No id on this wrapper — the outer `#brm-macro-output` div in
            # the form is the persistent target (see buttons'
            # `hx_swap="innerHTML"`); putting the id here too would
            # duplicate ids after a button swap.
            h.div((render_step(k) for k in keys(result))...)
        end

        # Each per-step renderer reads its value off `context!().run`
        # directly. `compute_steps` populates the underlying IP cache under
        # the polling progress phases; bare reads here just hit the warm
        # cache. `render_step(k)` dispatches by step key (`:parse` →
        # `__self__.parse`, `:stan_code` → `stan.code`, etc.).
        data = let frame = __parent__.context!().run.df
            h.details(
                h.summary("Synthetic data ($(nrow(frame)) rows × $(ncol(frame)) cols: " *
                    join(names(frame), ", ") * ") — click to expand"),
                render_table(frame; sortable=false),
            )
        end
        parse = h.section(
            h.h3("1. Meta.parse — raw Julia AST"),
            h.pre(__parent__.context!().run.parse),
        )
        transform = let rewritten = __parent__.context!().run.transform
            h.section(
                h.h3("2. parse! — rewritten AST (= → @n/@x assign, ~ → @n/@x ~)"),
                h.pre(rewritten.transformed),
                h.h3("    locals classified by parse!"),
                h.pre(rewritten.alllocals),
            )
        end
        wrap = h.section(
            h.h3("3. _brm — full let-block (df spliced as a literal)"),
            h.pre(__parent__.context!().run.wrap),
        )
        brmi = h.section(
            h.h3("4. eval — BRMI value (parsed model)"),
            brmi_card(__parent__.context!().run.brmi),
        )
        vbrmi = let r = __parent__.context!().run
            fd_summary = r.n_dead == 0 ?
                h.span("logdensity + FD check: $(r.dim)/$(r.dim) active ✓"; data_status="success") :
                h.span("logdensity + FD check: $(r.n_dead) dead param(s)"; data_status="error")
            fd_body = h.div(
                h.p("dim = ", r.dim, ", logdensity = ", r.ldp),
                isempty(r.dead) ? "" :
                    h.p(; data_status="error")("dead param indices: ", r.dead),
                h.pre(r.grad),
            )
            h.section(
                h.h3("5. VBRMI — materialized action (blocks, dim, columns)"),
                vbrmi_card(r.vbrmi),
                h.details(h.summary(fd_summary), fd_body),
            )
        end
        bench = h.section(
            h.h3("6. Chairmarks @be — per-step"),
            # `h.pre(b)` would call 2-arg show -> `Benchmark([Sample(...)...])`.
            # Force the 3-arg `text/plain` show to get Chairmarks' pretty
            # multi-line summary (min/median/quantiles/etc.).
            [h.article(h.header(label),
                       h.pre(sprint(show, MIME("text/plain"), b)))
             for (label, b) in __parent__.context!().run.benches]...,
        )
        slic_model = let slic = __parent__.context!().run.sbbrmi
            h.section(
                h.h3("5a. SlicModel — SBBRMI @slic body"),
                h.pre(slic.model.model),
                h.p("data keys: ", h.code(sort(collect(keys(slic.data))))),
            )
        end
        # Stan-step renderers bundled. Dispatched from `index`'s
        # `render_step` by stripping the `stan_` prefix from the step key
        # and looking the suffix up on `stan`.
        @include stan = begin
            code = h.section(
                h.h3("5b. StanCode — transpiled Stan source"),
                h.pre(__parent__.context!().run.stan.src),
            )
            compile = let s = __parent__.context!().run.stan
                h.section(
                    h.h3("5c. StanCompile — BridgeStan shared library"),
                    h.p("stan file: ", h.code(s.file)),
                    h.p("compiled .so: ", h.code(s.lib)),
                )
            end
            instantiate = let s = __parent__.context!().run.stan
                h.section(
                    h.h3("6a. StanInstantiate — model bound to data"),
                    h.p("param_unc_num = ", s.dim),
                    h.p("init (narrow normal, rng=Xoshiro(42)):"),
                    h.pre(s.init),
                )
            end
            eval = h.section(
                h.h3("6b. StanEval — log density at init"),
                h.p("log_density = ", __parent__.context!().run.stan.log_density),
            )
            shapes = let frame = __parent__.context!().run.stan.param_shapes_df
                h.section(
                    h.h3("6b'. StanShapes — index count per base parameter (p + tp + gq)"),
                    h.p("total indexed entries: ", sum(frame.n_indices),
                        " across ", nrow(frame), " base params"),
                    render_table(frame; sortable=true),
                )
            end
        end
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
        # Build one h.section per detected `PPCKind` (so distributional
        # likelihoods like `Normal(loc, err)` get one panel per LP -- one
        # for `loc`, one for `log(err)` -- each with its own independent
        # picker). Returns `nothing` if no kind matched, otherwise a
        # `h.div` wrapping the per-kind sections.
        build_ppc_section(long, which::Symbol; id_prefix) = begin
            kinds = _ppc_kinds(__parent__.context!().run.brmi)
            isempty(kinds) && return nothing
            frame = __parent__.context!().run.df
            sections = Any[]
            for (i, kind) in enumerate(kinds)
                sec = _build_one_ppc(kind, long, frame, which;
                                     id_prefix="$id_prefix-$i")
                isnothing(sec) || push!(sections, sec)
            end
            isempty(sections) ? nothing : h.div(sections...)
        end

        # One section for one kind. `dispatch_spec` picks prior vs
        # posterior; `obs_y` is materialised lazily (only when needed +
        # only after the Binomial proportion conversion). No explicit
        # type annotation on `p` since this lives in a @include body and
        # the actual per-kind dispatch happens inside prior_spec /
        # posterior_spec / picker_dims / predictor_label / kind_tag.
        _build_one_ppc(kind, long, frame, which; id_prefix) = begin
            link_lbl  = kind.link_fn === identity ? string(kind.loc) :
                        "$(kind.link_fn)($(kind.loc))"
            pred_lbl  = predictor_label(kind)   # `label` is a @param; avoid shadowing
            kind_disp = display_name(kind)

            # Non-primary LPs (e.g. `err` in `Normal(loc, err)`) are not
            # likelihood locations -- framing them as a "predictive check
            # of <response>" is wrong, and there's nothing observed to
            # overlay. Switch to a posterior-of-<link(loc)> framing.
            heading = if which === :prior
                kind.is_primary ? "Prior predictive" : "Prior of $link_lbl"
            else
                kind.is_primary ? "Posterior predictive check" :
                               "Posterior of $link_lbl"
            end
            subject = kind.is_primary ? string(kind.response) : link_lbl
            title   = "$subject$pred_lbl -- $heading"
            spec = if which === :prior
                prior_spec(kind, long, frame; title)
            else
                # Binomial outcomes are counts; predicted `link(loc)` is
                # a probability. Convert observed counts to proportions
                # so both layers share the same response scale. Only
                # primary LPs need the obs vector at all.
                obs_y = if kind.is_primary
                    obs_y_raw = __parent__.context!().run.stan.fit.data_dict[Symbol(kind.response)]
                    is_binomial = kind.family === Binomial || kind.family === BinomialLogit
                    is_binomial && !isnothing(kind.n_trials) ?
                        obs_y_raw ./ __parent__.context!().run.stan.fit.data_dict[Symbol(kind.n_trials)] :
                        obs_y_raw
                else
                    nothing
                end
                posterior_spec(kind, long, frame, obs_y; title)
            end
            isnothing(spec) && return nothing

            cap = if which === :prior
                "Prior-predictive draws of $link_lbl$pred_lbl ($(kind.family) family, $kind_disp)"
            elseif kind.is_primary
                "Posterior draws of $link_lbl$pred_lbl, with observed $(kind.response) overlaid ($kind_disp)"
            else
                "Posterior draws of $link_lbl$pred_lbl ($kind_disp)"
            end

            dims = picker_dims(kind)
            plot_block = isnothing(dims) ?
                with_plot_caption(spec; plot_id="$id_prefix-ppc", title=cap) :
                with_plot_caption(spec; plot_id="$id_prefix-ppc", title=cap,
                                  auto_remap=(; dims))
            h.section(
                h.h4(heading, ": ", h.code(subject), pred_lbl,
                     " (", kind_disp, ")"),
                plot_block,
            )
        end

        # Stan-step renderers that read NamedTuple bundles directly off
        # `__parent__.context!().run.stan` — `:stan_generate` from `.generated`,
        # `:stan_fit_pathfinder` from `.posterior`, `:stan_fit_warmup` from
        # `.posterior.warmup`. Each composes `posterior_plots` (tabset +
        # wide-table details) with `build_ppc_section` (per-PPCKind sections).
        @include stan_section = begin
            generate = let g = __parent__.context!().run.stan.generated,
                           truth = __parent__.context!().run.stan.truth.df,
                           long = g.df, wide = g.wide_df, summary = g.summary_df
                plots = posterior_plots(long, wide, summary;
                                    id_prefix="brm-plot-generated",
                                    kind="Generated data (prior predictive)",
                                    truth)
                ppc_sec = build_ppc_section(long, :prior; id_prefix="brm-plot-generated")
                parts = Any[
                    h.h3("6c. StanGenerate — synthetic data from narrow-normal prior + param_constrain"),
                    h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                        "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
                ]
                isnothing(ppc_sec) || push!(parts, ppc_sec)
                push!(parts, plots.tabs, plots.wide_details)
                h.section(parts...)
            end
            @include fit = begin
                pathfinder = let p = __parent__.context!().run.stan.posterior,
                                truth = __parent__.context!().run.stan.truth.unc_df,
                                long = p.long_df, wide = p.wide_df,
                                summary = p.summary_df, full_long = p.full_long_df
                    plots = posterior_plots(long, wide, summary;
                                        id_prefix="brm-plot-pf",
                                        kind="Pathfinder posterior",
                                        truth)
                    ppc_sec = build_ppc_section(full_long, :posterior; id_prefix="brm-plot-pf")
                    parts = Any[
                        h.h3("6d. StanFit (Pathfinder) — variational approximation draws"),
                        h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                            "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
                    ]
                    isnothing(ppc_sec) || push!(parts, ppc_sec)
                    push!(parts, plots.tabs, plots.wide_details)
                    h.section(parts...)
                end
                warmup = let w = __parent__.context!().run.stan.posterior.warmup,
                             truth = __parent__.context!().run.stan.truth.unc_df,
                             long = w.long_df, wide = w.wide_df,
                             summary = w.summary_df, full_long = w.full_long_df,
                             diagnostics = w.diagnostics
                    plots = posterior_plots(long, wide, summary;
                                        id_prefix="brm-plot-warmup",
                                        kind="Warmup+MCMC posterior",
                                        truth)
                    ppc_sec = build_ppc_section(full_long, :posterior; id_prefix="brm-plot-warmup")
                    parts = Any[
                        h.h3("6d'. StanFit (Warmup+MCMC) — full Stan fit"),
                        h.p("n_divergent_samples: ", diagnostics.n_divergent_samples,
                            " · min ESS: ", minimum(diagnostics.ess)),
                        h.p("long format: ", nrow(long), " rows · ", ncol(long), " cols · ",
                            "wide format: ", nrow(wide), " rows · ", ncol(wide), " cols"),
                    ]
                    isnothing(ppc_sec) || push!(parts, ppc_sec)
                    push!(parts, plots.tabs, plots.wide_details)
                    h.section(parts...)
                end
            end
        end
    end

    @get index() = begin
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
            h.aside(; class="htmxo-status-banner", data_status="muted")(
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
                             " -- penalized thin-plate regression spline (rank-10 default: two unpenalized ",
                             "null-space columns plus eight range-space columns sharing a half-normal smoothing SD). ",
                             "No alternate ", h.code("bs"), " / ", h.code("t2"), " bases or ",
                             h.code("k"), " / ", h.code("knots"), " options yet."),
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
                    [e.preset_button for e in gallery.items() if e.tier.n == 0]...,
                ),
                h.label("Formula")(
                    h.textarea(formula;
                        name="formula", rows=8,
                        class="brm-formula-textarea",
                        # Edited formula no longer matches any preset --
                        # de-highlight every preset button.
                        oninput="document.querySelectorAll('.brm-preset-btn').forEach(b => b.classList.add('outline'))"),
                ),
                # Lazy stage-picker tabs. Each tab fetches its stage on
                # click; the response replaces the inner HTML of
                # `#brm-macro-output`. Labels match the previous button
                # row so the numeric prefix carries the macro-frontend
                # / VBRMI / SB-branch / SB-fit grouping. `hx_push_url`
                # so the address bar reflects the active tab; the
                # server gates `force=true` on `is_htmx(__req__)` so a
                # reload of that pushed URL re-attaches to the IP cache
                # rather than recomputing.
                htmx_tabset([
                    "1. Parse"             => string(query_url(__self__/"stage/parse";            force=true)),
                    "2. Transform"         => string(query_url(__self__/"stage/transform";        force=true)),
                    "3. Wrap"              => string(query_url(__self__/"stage/wrap";             force=true)),
                    "4. BRMI"              => string(query_url(__self__/"stage/brmi";             force=true)),
                    "5. VBRMI"             => string(query_url(__self__/"stage/vbrmi";            force=true)),
                    "6. Benchmark"         => string(query_url(__self__/"stage/bench";            force=true)),
                    "5a. SlicModel"        => string(query_url(__self__/"stage/slic_model";       force=true)),
                    "5b. StanCode"         => string(query_url(__self__/"stage/stan_code";        force=true)),
                    "5c. StanCompile"      => string(query_url(__self__/"stage/stan_compile";     force=true)),
                    "6a. StanInstantiate"  => string(query_url(__self__/"stage/stan_instantiate"; force=true)),
                    "6b. StanEval"         => string(query_url(__self__/"stage/stan_eval";        force=true)),
                    "6b'. StanShapes"      => string(query_url(__self__/"stage/stan_shapes";      force=true)),
                    "6c. StanGenerate"     => string(query_url(__self__/"stage/stan_generate";    force=true)),
                    "6d. StanFit (PF)"     => string(query_url(__self__/"stage/stan_fit_pathfinder"; force=true)),
                    "6d'. StanFit (Warmup)" => string(query_url(__self__/"stage/stan_fit_warmup"; force=true)),
                ];
                    active="6d. StanFit (PF)",
                    target="#brm-macro-output",
                    tab_attrs=_label -> (;
                        hx_include="#brm-macro-form",
                        hx_swap="innerHTML",
                        hx_push_url="true",
                    ),
                ),
                h.small("Bug-report helper:"),
                h.button("SB repro (current formula)";
                    type="submit",
                    formaction=string(__self__/"sb_repro"),
                    class="secondary"),
            ),
            # Persistent wrapper — tabs swap `innerHTML` into here so the
            # id survives polling/error responses. Default lazy load is
            # the 6d Pathfinder fit (the default active tab).
            h.div(; id="brm-macro-output")(
                lazy(query_url(__self__/"stage/stan_fit_pathfinder"; formula)),
            ),
        )
    end

    # Focused per-model views of the sbimpl intermediate artifacts. Each
    # route runs the pipeline just far enough and returns the relevant
    # source in `h.pre` (plus markdown_only serves the bare source via
    # `?plain` / `Accept: text/plain`, for piping into agents or curl).
    @get slic() = h.pre(context!().run.sbbrmi.model.model)

    @get stan() = h.pre(context!().run.stan.src)

    # Serve AoV's vega-embed runtime as plain JS so the docs page can
    # `<script src="...">` it before a gallery card swaps in. The
    # gallery card response embeds inline `<script>AoV.embed(…)>` calls
    # for the auto-PPC plot; htmx evaluates those on swap, but they
    # need `window.AoV` already defined. The docs theme loads this URL
    # via setupHtmxoEmbed. (Mirrors AoV's own `aov_runtime_js` route.)
    @get aov_runtime_js() = let
        wrapped = sprint(show, MIME"text/html"(), vega_runtime())
        body = replace(wrapped, r"^\s*<script[^>]*>"i => "")
        body = replace(body, r"</script>\s*$"i => "")
        MIMEResponse("application/javascript; charset=utf-8", body)
    end

    # Gallery bundle, mounted at /pipeline/gallery. `index` is the shell
    # (card grid of placeholder articles); `card(slug)` returns the full
    # rendered content for one example (formula + SLIC + auto-PPC + Stan
    # source); `record` triggers the AppData IP that walks every gallery
    # URL through `compute_steps` and dumps the static HTML for deploy.
    @include gallery = begin
        # Showable subset of `examples.entries()` (drops markdown-only
        # notes and bruno-* entries when bruno-ext.jl isn't on disk).
        items() = [e for e in __parent__.__parent__.examples.entries() if e.shown]

        # Card grid: one placeholder article per item, lazy-loading its
        # body via `hx-trigger=load` against `card/<slug>`. Per-card
        # placeholder is owned by ExampleEntry (`e.gallery_placeholder`).
        @get index() = let xs = items()
            h.div(; class="htmxo-gallery")(
                h.section(
                    h.h3("Presets"),
                    [e.gallery_placeholder for e in xs if e.tier.n == 0]...,
                ),
                h.section(
                    h.h3("Examples"),
                    [e.gallery_placeholder for e in xs if e.tier.n > 0]...,
                ),
            )
        end

        # Card content for one example/preset, addressed by slug.
        # Path-based so static-deploy recording works -- query-string
        # URLs lose their differentiator when GitHub Pages strips the
        # query before file lookup. Formula is set up under the example's
        # namespace (so `bruno-*` items get `dataset_extras(::Val{:bruno},
        # df)` extras) and run through `compute_steps` to terminal state
        # via `polling_fetchindex` -- same machinery the stage buttons
        # use, so the user sees the progress treebar during compile + fit.
        #
        # The rendered card contains: input formula, SLIC body, Stan
        # source, auto-PPC section. One self-contained HTML response,
        # recorded as `live-brm/pipeline/gallery/card/<slug>.html` for
        # static deploy.
        #
        # `force=true` invalidates both caches; gated on `is_htmx(__req__)`
        # so direct reloads don't blow the cache.
        @get card(slug; force::Bool=false) = begin
            slug = String(slug)
            item = let lookup = filter(e -> e.slug == slug, items())
                isempty(lookup) && return h.article(
                    h.p("Unknown gallery slug: ", h.code(slug); class="htmxo-card-error"))
                only(lookup)
            end
            ctx       = context(item.label, item.formula)
            ns        = ctx.namespace
            cache_key = (hash(item.formula), ns)
            do_force  = force && is_htmx(__req__)
            do_force && delete!(__appdata__.ppc_html_cache, cache_key)
            polling_fetchindex(
                compute_steps, item.formula, ns, :stan_fit_pathfinder;
                poll_url=query_url(__self__/"card/$(item.slug)"),
                label="Gallery card - $(item.label)",
                force=do_force,
            ) do _result
                get!(__appdata__.ppc_html_cache, cache_key) do
                    # All four panels share the same finished `run`. Stan
                    # source materialisation is cheap once compute_steps
                    # has run; SLIC body is even cheaper.
                    run        = ctx.run
                    full_long  = run.stan.posterior.full_long_df
                    ppc_div    = __parent__.stage(:stan_fit_pathfinder).build_ppc_section(
                                     full_long, :posterior;
                                     id_prefix="brm-gallery-$(item.slug)")
                    pipeline_url = string(query_url(__parent__/""; formula=item.formula, label=item.label))
                    h.article(; id="brm-gallery-card-$(item.slug)")(
                        h.h4(
                            h.a(item.label; href=pipeline_url, target="_blank",
                                title="Open in pipeline"),
                        ),
                        h.h5("Formula"),
                        h.pre(h.code(item.formula; class="language-julia")),
                        h.h5("SLIC model"),
                        h.pre(h.code(string(run.sbbrmi.model.model); class="language-julia")),
                        h.h5("Auto PPC"),
                        isnothing(ppc_div) ? h.p("(no PPC kind detected)") : ppc_div,
                        h.h5("Stan model"),
                        h.pre(h.code(run.stan.src; class="language-stan")),
                    )
                end
            end
        end

        # Drive the AppData IP to dump every gallery URL (gallery shell +
        # library + per-card content x full + HX shapes) into
        # `docs/src/public/live-brm/`. Long-running (Stan compile + fit
        # per item, serialised), so it goes through `polling_fetchindex`:
        # first hit kicks off a Task and returns a polling progress
        # fragment; subsequent polls show the per-path
        # `prepare_progress!` markers; when finished, the do-block
        # renders the summary article.
        #
        # Override the deploy URL prefix via `RECORD_BASE_PREFIX` env var
        # (default `/BayesianRegressionModels.jl/dev/live-brm`); override
        # the output directory via the `record_dir` query param.
        @get record(; record_dir::String="", record_base::String="", force::Bool=false) = begin
            rd = isempty(record_dir) ?
                joinpath(dirname(dirname(@__DIR__)), "docs", "src", "public", "live-brm") :
                record_dir
            rb = isempty(record_base) ?
                get(ENV, "RECORD_BASE_PREFIX", "/BayesianRegressionModels.jl/dev/live-brm") :
                record_base
            polling_fetchindex(__appdata__.record_gallery, rd, rb;
                               poll_url=query_url(__self__/"record"; record_dir=rd, record_base=rb),
                               label="Recording BRM gallery",
                               force) do summary
                h.article(
                    h.header(h.h2("Gallery recorded")),
                    h.p("Wrote ", h.code(string(summary.n_paths)),
                        " routes (x full + HX shapes) into ",
                        h.code(summary.record_dir), "."),
                    h.ul(
                        h.li(h.code(string(summary.n_html)),  " .html"),
                        h.li(h.code(string(summary.n_other)), " other"),
                        h.li(h.code(string(summary.n_items)), " gallery items"),
                    ),
                    h.p(h.strong("Next: "),
                        h.code("git add docs/src/public/live-brm && git commit && git push"),
                        " -- CI deploys the rest."),
                    h.p("Re-record (overwrites cache): ",
                        h.a("/pipeline/gallery/record?force=true";
                            href=__self__/"record?force=true")),
                )
            end
        end
    end

    @get debug(; q::String="") = h.pre(
        try
            isempty(q) ? "Pass ?q=<julia expr>" :
                first(string(Base.eval(@__MODULE__, Meta.parse(q))), 8000)
        catch e
            sprint(showerror, e)
        end
    )

    # StanBlocks-bug-report routes for external agents. Both endpoints render
    # the SlicModel body, generated Stan source, and BridgeStan compile output
    # (success msg or full error). HTMXO's `_resolve_response` auto-converts
    # to markdown when `Accept: text/plain` is requested, so the same URL works
    # for humans (browser) and agents (curl).
    #   curl -H 'Accept: text/plain' 'http://localhost:<port>/pipeline/sb_repro?formula=<url-encoded>'
    #   curl -H 'Accept: text/plain' 'http://.../pipeline/sb_repro/example?name=<slug>'
    @include sb_repro = begin
        @get index() = __parent__.context!().sb_repro_html

        @get example(; name::AbstractString="") = begin
            entry = __parent__.__parent__.examples.find_by_slug(name)
            isnothing(entry) && return h.div(
                h.h1("Example not found"),
                h.p("No example with slug ", h.code(name)),
            )
            __parent__.context(entry.label, entry.formula).sb_repro_html
        end
    end
end
