@htmx struct AppContext
    __appdata__::AppData = APPDATA

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
        app_layout(
            nav_sidebar([
                "Pipeline"  => "/pipeline",
                "Gallery"   => "/pipeline/gallery",
                "Examples"  => "/examples",
                "Library"   => "/library",
                "Structure" => "/structure",
            ]),
            content,
        );
        pico_version="2",
        extra_head=(
            h.title("BRM macro action"),
            h.style(HTMX.Raw(css)),
            htmxo_gallery_styles(),
            htmxo_syntax_head()...,
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
    @get index() = __self__.pipeline.index(__verb__)

    # Utility route: clears all in-memory caches on AppData (compute_steps
    # results, nested IP dicts, etc.). Useful after code changes when you
    # want to re-run a stage that's currently returning a stale cached
    # value. Mirrors the downstream app's ops-reset route.
    @get reset() = begin
        clear_mem_caches!(__appdata__)
        h.p("In-memory caches cleared.")
    end

    # DO/HTMXO type-tree introspection. `DynamicObjects.structure` returns a
    # `Node{:pre, …}` that's `Base.showable("text/html", …)`, so Cobweb's
    # auto-render-HTML-showable-children kicks in and emits the colored tree
    # straight into the page.
    @get structure() = h.div(
        h.h2("AppContext type structure"),
        h.p(h.small("DynamicObjects.structure(AppContext) — colors mark identical worst-case dependency sets")),
        DynamicObjects.structure(AppContext),
    ) 

    @include pipeline = PipelineRoutes()

    # Library of SLIC submodels shipped by BRM (sbimpl.jl). Each entry is
    # a `_sb_*` SlicModel that the @brm macro can route to (priors,
    # predictor terms, missing-data response, ...). Sourced by reflection
    # over the BayesianRegressionModels module rather than a hand-curated
    # list -- new submodels added in sbimpl.jl show up automatically.
    @include library = begin
        # Reflection: every `_sb_*` binding in BRM that's a SlicModel.
        # Sorted alphabetically for stable display.
        slic_submodels() = begin
            entries = Pair{Symbol,Any}[]
            for n in names(BayesianRegressionModels; all=true)
                startswith(string(n), "_sb_") || continue
                v = getproperty(BayesianRegressionModels, n)
                v isa StanBlocks.SlicModel || continue
                push!(entries, n => v)
            end
            sort!(entries; by=first)
        end

        @get index() = h.div(
            h.h1("SLIC submodel library"),
            h.p(
                "Reusable ",
                h.code("StanBlocks.@slic"),
                " submodels exported by ",
                h.code("BayesianRegressionModels"),
                ". The ",
                h.code("@brm"),
                " macro routes formula constructs (priors, predictor ",
                "terms, missing-data responses, ...) to these. Each ",
                "card shows the submodel body verbatim plus the data ",
                "kwargs it expects from the caller.",
            ),
            h.div(; class="brm-library-grid")(
                [h.article(
                    h.header(h.strong(string(nm))),
                    h.details(; open=true)(
                        h.summary("Body"),
                        h.pre(string(sm.model)),
                    ),
                    isempty(sm.data) ? "" :
                        h.details(
                            h.summary("Pre-bound data keys"),
                            h.code(join(sort(collect(keys(sm.data))), ", "))),
                )
                for (nm, sm) in slic_submodels()]...,
            ),
        )
    end

    # The Examples section mounts at /examples. Both the list view and per-slug
    # detail view share `@get index(slug)` (HTMXO registers `/examples` AND
    # `/examples/{slug}` thanks to slug's default). `@get mark` lives here too
    # since it operates exclusively on ExampleEntry; pill URLs hit /examples/mark.
    @include examples = begin
        (; examples_dir) = __appdata__
        # `label` is auto-forwarded from AppContext's `@param`; no explicit
        # `@param (; label) = __parent__` needed (and declaring it explicitly
        # collides with the auto-forward → "method overwritten" error).

        # ExampleEntry is constructed with `__parent__=__self__` (this
        # examples include) so each entry's rendering methods can build URLs
        # via its parent chain (`__parent__.__parent__.pipeline/...`): the
        # entry's parent is this examples include, whose parent is AppContext
        # (which holds `.pipeline`). Passing bare `__parent__` here would set
        # the entry's parent to AppContext, so `__parent__.__parent__` would
        # be AppContext's (nothing) parent — a 500 on every gallery render.
        files() = begin
            isdir(examples_dir) || mkpath(examples_dir)
            sort(filter(endswith(".jl"), readdir(examples_dir; join=true));
                 by=mtime, rev=true)
        end
        entries() = ExampleEntry[ExampleEntry(f; __parent__=__self__) for f in files()]
        find(label)        = findfirstelement(e -> e.label == label, entries())
        find_by_slug(slug) = findfirstelement(e -> e.slug == slug,   entries())
        persist!(label, formula) = begin
            e = find(label)
            e === nothing || e.save!(; new_formula=formula)
        end

        @get mark(; state::Symbol=Symbol("")) =
            find(label).toggle_status!(state).card.default

        @get flag() = find(label).cycle_flag!().card.default

        # List view vs detail view as separate derived-property methods; DO
        # supports multi-method dispatch on a single property name (the route
        # layer doesn't — see TODO below).
        # Sort-bar pill. `key` matches a `data-<key>` attribute on each card.
        # Click cycles off → desc → asc → off; drag between pills reorders
        # sort priority (left = highest). A small inline script below wires
        # this up to the `#brm-examples-list` container's DOM order.
        _sort_pill(key, label) = h.span(label;
            class="brm-sort-pill", draggable=true,
            data_sort_key=key, aria_pressed="false")

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
                if (search) {
                    if (lit) search.setAttribute('data-mode', 'literal');
                    else     search.removeAttribute('data-mode');
                }
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
                p.setAttribute('aria-pressed', s !== 0 ? 'true' : 'false');
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
                card.removeAttribute('data-flash');
            });
            document.body.addEventListener('htmx:afterRequest', e => {
                const el = e.detail.elt;
                if (!el || !el.classList.contains('brm-branch-btn')) return;
                const card = cardOf(el);
                if (!card) return;
                const ok = e.detail.successful && e.detail.xhr.status < 400;
                if (ok) {
                    card.setAttribute('data-flash', 'success');
                    setTimeout(() => card.removeAttribute('data-flash'), 1200);
                } else {
                    card.setAttribute('data-flash', 'error');
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
                    placeholder="Search (case-insensitive regex; e.g. interact|poisson)...",
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
                [e.card.default for e in entries()]...,
            ),
            h.script(HTMX.Raw(_sort_script)),
        )

        _index(slug::AbstractString, stage::AbstractString) = h.div(
            h.p(h.a("<- Back to Examples"; href=__prefix__)),
            find_by_slug(slug).card.with_preload(stage; force_open=true),
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
