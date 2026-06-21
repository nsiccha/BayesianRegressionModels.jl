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

    _TIER_LABELS   = ("T1", "T2", "T3")

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
    @struct tier = begin
        n     = parse(Int, get(_parsed.header, "tier", "1"))
        label = get(_TIER_LABELS, n, "T$n")
        pill  = h.span(label; class="brm-tier-pill", data_tier=string(n))
    end
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
    @struct stages = begin
        pass = _parse_stage_set("stages_pass")
        fail = _parse_stage_set("stages_fail")
    end

    # DOM ids derived once — HTMX targets reference these (hx_target= / id=).
    # Hashing the label keeps ids stable across requests without needing to
    # URL-escape the label.
    _label_hash = hash(label)

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
        is_active = status.value == target_state
        h.button(is_active ? active_text : inactive_text;
            type="button",
            class="brm-state-pill",
            data_state=string(target_state),
            aria_pressed=string(is_active),
            hx_get=string(query_url(__parent__/"mark"; label, state=target_state)),
            hx_target="#$(card.id)",
            hx_swap="outerHTML",
            onclick="event.stopPropagation()",
        )
    end

    # Flag pill — cycles through none → SB → SBBRM → both → none.
    # `SB` targets the StanBlocks-proper agent; `SBBRM` targets the BRM
    # sbimpl.jl agent; `both` goes to both. Flagged cards sort to the top.
    _flag_label = flag === :sb    ? "⚑ SB" :
                  flag === :sbbrm ? "⚑ SBBRM" :
                  flag === :both  ? "⚑ SB+SBBRM" :
                                    "flag"
    flag_pill = h.button(_flag_label;
        type="button",
        class="brm-flag-pill",
        data_flag=string(flag),
        hx_get=string(query_url(__parent__/"flag"; label)),
        hx_target="#$(card.id)",
        hx_swap="outerHTML",
        onclick="event.stopPropagation()",
    )

    @struct status = begin
        value = Symbol(get(_parsed.header, "status", "open"))
        id    = "status-$_label_hash"
        pills = h.span(; id, class="brm-status-pills")(
            state_pill(:done,          "✓ done",          "mark done"),
            state_pill(:deprioritized, "✓ deprioritized", "deprioritize"),
            flag_pill,
        )
    end

    # Stage list pulled from AppContext.stage_labels so the order + names
    # stay in sync with the pipeline page.
    _stages = __parent__.__parent__.stage_labels


    # Whether this entry should appear in gallery / preset UIs. Markdown-only
    # notes (no formula body) are skipped. Confidential client-project examples
    # (`<prefix>-*`) depend on their gitignored `<prefix>-ext.jl` registering the
    # dataset + allowlist; without that file on disk they'd error at compile, so
    # they're hidden too. Add a new confidential prefix to `_CONFIDENTIAL_EXT` —
    # the gate generalizes over the tuple rather than special-casing each one.
    _CONFIDENTIAL_EXT = ("bruno", "bordet")
    shown = begin
        gated = findfirst(p -> startswith(slug, "$p-"), _CONFIDENTIAL_EXT)
        !isnothing(formula) &&
            (isnothing(gated) ||
             isfile(joinpath(@__DIR__, "$(_CONFIDENTIAL_EXT[gated])-ext.jl")))
    end

    # Placeholder card emitted by the gallery shell: auto-fetches its body
    # via `hx-trigger="load"` against `pipeline/gallery/card/<slug>`. The
    # target route is polling_fetchindex-backed -- the user sees the
    # progress tree while Stan compile + Pathfinder fit run, then the full
    # card body (formula + SLIC + auto-PPC + Stan source) replaces this.
    gallery_placeholder = h.article(;
            id="brm-gallery-card-$slug",
            hx_get=string(query_url(__parent__.__parent__.pipeline.gallery/"card/$slug")),
            hx_trigger="load",
            hx_swap="outerHTML",
        )(
            h.h4(label),
            h.p("Loading..."),
        )

    # Quick-fill preset button shown in the pipeline editor for tier-0
    # entries. Active when its formula matches the page's current formula
    # @param; inactive presets get Pico's `outline` ghost-button class.
    preset_button = h.button(label;
        type="button",
        class=("brm-preset-btn" *
            (formula == __parent__.__parent__.pipeline.formula ? "" : " outline")),
        data_formula=formula,
        onclick="""
            document.querySelector('textarea[name=formula]').value = this.dataset.formula;
            document.querySelectorAll('.brm-preset-btn').forEach(b => b.classList.add('outline'));
            this.classList.remove('outline');
            const tab = document.querySelector('.tab-row a.primary') || document.querySelector('.tab-row a');
            if (tab) tab.click();
        """,
    )

    save!(; new_status=status.value, new_formula=formula,
            new_stages_pass=stages.pass, new_stages_fail=stages.fail,
            new_flag=flag) = begin
        io = IOBuffer()
        println(io, "# label: ", label)
        println(io, "# tier: ", tier.n)
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
        save!(; new_status = status.value == target ? :open : target)

    # Cycle: none → sb → sbbrm → both → none
    _flag_next = Dict(:none => :sb, :sb => :sbbrm, :sbbrm => :both, :both => :none)
    cycle_flag!() = save!(; new_flag = _flag_next[flag])

    # Mark a set of stage names as pass / fail and persist to disk. Passed
    # stages remove from the fail set (and vice versa) so the most recent
    # outcome wins. Returns the reloaded entry.
    mark_stages!(; passed=Symbol[], failed=Symbol[]) = begin
        new_pass = union(setdiff(stages.pass, failed), passed)
        new_fail = union(setdiff(stages.fail, passed), failed)
        save!(; new_stages_pass=new_pass, new_stages_fail=new_fail)
    end

    # Per-stage indicator pills rendered in the card summary. Each links to
    # `/examples/<slug>?stage=<name>`, a shareable GET URL that loads the
    # card with that stage's result pre-populated (without force=true).
    @struct stage = begin
        # Aliases to ExampleEntry's `stages` inline-struct (DO doesn't
        # auto-forward inline-struct names into sibling inline-structs, so
        # we pull what `state` reads explicitly — this also satisfies the
        # no-self-access lint by giving its body sibling props to read).
        pass = __parent__.stages.pass
        fail = __parent__.stages.fail
        state(name) = name in fail ? :fail :
                      name in pass ? :pass : :unknown
        indicators = h.span(; class="brm-stage-indicators")(
            [h.a(;
                href="$(__parent__.__parent__/slug)?stage=$stage_id",
                title="$stage_label -- $(state(stage_id))",
                onclick="event.stopPropagation()",
                class="brm-stage-indicator",
                data_state=string(state(stage_id)))(stage_label)
             for (stage_id, stage_label) in __parent__.__parent__.__parent__.stage_labels]...
        )
    end

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
    _sort_brokenness = length(stages.fail) - length(stages.pass)
    # 0 = open (top), 1 = done / deprioritized (bottom, tied). Ascending
    # surfaces open work first; default-on below.
    _sort_status     = status.value === :open ? 0 : 1
    _sort_progress   = length(stages.pass)
    # 0=none, 1=sb-only or sbbrm-only, 2=both → "most in need" sorts highest.
    _sort_flagged    = flag === :none ? 0 : flag === :both ? 2 : 1

    @struct card = begin
        id        = "example-card-$_label_hash"
        result_id = "example-result-$_label_hash"

        # Formula form with stage buttons. `push_url=true` in the standalone
        # detail view makes each button push `/examples/<slug>?stage=<id>` into
        # browser history (shareable, back/forward works); `push_url=false` in
        # the list view leaves the URL at `/examples`.
        formula_form(push_url::Bool) = begin
            stage_btns = [
                h.button(stage_label;
                    type="button",
                    class="brm-branch-btn",
                    data_stage_id=string(stage_id),
                    hx_get=string(query_url(pipeline/"stage/$stage_id"; force=true)),
                    hx_include="closest form",
                    hx_target="#$result_id",
                    hx_swap="innerHTML",
                    (push_url ? (; hx_push_url="$(__parent__.__parent__/slug)?stage=$stage_id") : (;))...,
                ) for (stage_id, stage_label) in _stages]
            h.form(
                h.input(; type="hidden", name="label", value=label),
                h.textarea(formula;
                    name="formula",
                    rows=max(3, count('\n', formula) + 1)),
                stage_btns...,
                h.button("SB repro ▶";
                    type="submit",
                    formaction=string(pipeline/"sb_repro"),
                    class="secondary"),
            )
        end

        # Card renderer, parameterized by an optional `preload_stage`. When set,
        # the result div is seeded with a `lazy(...)` that fires the stage GET
        # on first view — so `/examples/<slug>?stage=<name>` shows the card with
        # that stage's output already running/rendered (no click needed).
        with_preload(preload_stage::AbstractString=""; force_open::Bool=false) = begin
            children = Any[HTMXObjects.md_to_node(body)]
            if formula !== nothing
                # Push URL on stage-button clicks only in the detail view
                # (`force_open=true`); list view keeps URL at `/examples`.
                push!(children, formula_form(force_open))
                result_children = Any[]
                isempty(preload_stage) || push!(result_children,
                    lazy(query_url(pipeline/"stage/$preload_stage"; formula, label)))
                push!(children, h.div(; id=result_id)(result_children...))
            end
            h.article(;
                id,
                class="brm-example-card",
                data_state=string(status.value),
                data_mtime=string(_sort_mtime),
                data_tier=string(_sort_tier),
                data_complexity=string(_sort_complexity),
                data_brokenness=string(_sort_brokenness),
                data_status=string(_sort_status),
                data_progress=string(_sort_progress),
                data_flagged=string(_sort_flagged),
                data_label=label,
            )(
                h.details(; open=(force_open || status.value == :open))(
                    h.summary(
                        tier.pill, " ",
                        h.strong(label), " ",
                        status.pills, " ",
                        stage.indicators, " ",
                        permalink,
                    ),
                    children...,
                ),
            )
        end

        # Default card render (no stage pre-loaded; clicks on stage buttons
        # populate the inline result div on demand). `with_preload` handles
        # both list-mode and (when called with a non-empty preload_stage)
        # detail-mode.
        default = with_preload()
    end
end
