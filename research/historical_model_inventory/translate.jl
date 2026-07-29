#!/usr/bin/env julia

const DEFAULT_INPUT = joinpath(@__DIR__, "historical_catalog.tsv")
const DEFAULT_OUTPUT = joinpath(@__DIR__, "translations.tsv")
const DEFAULT_FAMILY_AUDIT = joinpath(@__DIR__, "family_audit.tsv")
const FAMILY_AUDIT_CLASSES = Set([
    "explicit-family-recovery",
    "defensible-semantic-inference",
    "genuinely-indeterminate",
])
const SUPPORT_CLASSES = Set([
    "already-expressible-verbatim",
    "already-expressible-via-semantic-rewrite",
    "genuinely-missing-brm-surface",
    "genuinely-missing-stanblocks-substrate",
    "historical-semantics-unresolved",
])

tsv_unescape(value) = replace(value,
    "\\t" => "\t", "\\n" => "\n", "\\r" => "\r", "\\\\" => "\\")
tsv_escape(value) = replace(string(value),
    '\\' => "\\\\", '\t' => "\\t", '\n' => "\\n", '\r' => "\\r")

function read_tsv(path)
    lines = readlines(path)
    isempty(lines) && error("empty TSV: $path")
    columns = split(first(lines), '\t'; keepempty=true)
    [Dict(columns .=> tsv_unescape.(split(line, '\t'; keepempty=true)))
     for line in Iterators.drop(lines, 1)]
end

function write_tsv(path, columns, rows)
    open(path, "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            println(io, join((tsv_escape(get(row, column, "")) for column in columns), '\t'))
        end
    end
end

function normalize_family(family)
    family = lowercase(strip(family))
    aliases = Dict(
        "normal" => "gaussian",
        "negative_binomial" => "negativebinomial",
        "nb" => "negativebinomial",
        "zero-inflated-poisson" => "zero_inflated_poisson",
        "zip" => "zero_inflated_poisson",
        "ordered_logistic" => "ordered_logistic",
        "ordinal" => "ordered_logistic",
        "cumulative" => "ordered_logistic",
        "logistic" => "bernoulli",
        "t" => "student_t",
        "student" => "student_t",
        "student-t" => "student_t",
        "student_t" => "student_t",
        "asym_laplace" => "asymmetriclaplace",
    )
    get(aliases, family, family)
end

function load_family_audit(path)
    rows = read_tsv(path)
    required = Set([
        "row_key", "recovered_family", "evidence_class", "evidence_strength",
        "authoritative_surface", "authoritative_revision",
        "authoritative_path_anchor", "retrieved_at_utc", "http_status",
        "explicit_evidence", "semantic_evidence", "dataset_evidence",
        "negative_evidence", "decision_rationale",
    ])
    missing = setdiff(required, Set(keys(first(rows))))
    isempty(missing) || error("family audit is missing columns: $(join(sort(collect(missing)), ','))")
    by_key = Dict{String,Dict{String,String}}()
    for row in rows
        key = row["row_key"]
        haskey(by_key, key) && error("duplicate family-audit row: $key")
        row["evidence_class"] in FAMILY_AUDIT_CLASSES ||
            error("invalid family-audit class for $key: $(row["evidence_class"])")
        if row["evidence_class"] == "genuinely-indeterminate"
            isempty(row["recovered_family"]) ||
                error("indeterminate family-audit row must not select a family: $key")
            isempty(row["negative_evidence"]) &&
                error("indeterminate family-audit row lacks negative evidence: $key")
        else
            isempty(row["recovered_family"]) &&
                error("resolved family-audit row lacks a family: $key")
        end
        by_key[key] = row
    end
    by_key
end

function infer_family(row, family_audit)
    explicit = normalize_family(row["family_claim"])
    !isempty(explicit) && return (explicit, "explicit")
    description = normalize_family(row["family_text_claim"])
    !isempty(description) && return (description, "inferred-from-description")

    row_key = row["source"] * ":" * row["key"]
    if haskey(family_audit, row_key)
        audit = family_audit[row_key]
        class = audit["evidence_class"]
        class == "genuinely-indeterminate" &&
            return ("", "authoritative-audit-genuinely-indeterminate")
        return (normalize_family(audit["recovered_family"]),
                "authoritative-audit-" * class)
    end

    formula = lowercase(row["formula_claim"])
    name = lowercase(row["name_claim"])
    key = lowercase(row["key"])
    evidence = string(name, " ", key, " ", formula)

    occursin("beta-binomial", evidence) || occursin("vint(", formula) &&
        return ("beta_binomial", "inferred-from-formula/name")
    (occursin("trials(", formula) || startswith(strip(formula), "p(") ||
     occursin("cbind(", formula)) &&
        return ("binomial", "inferred-from-response-syntax")
    occursin(r"negative[ _-]?binomial|\bnb[12]?\b", evidence) &&
        return ("negativebinomial", "inferred-from-name")
    occursin("zero-inflated poisson", evidence) || occursin("zip", key) &&
        return ("zero_inflated_poisson", "inferred-from-name")
    occursin("hurdle", evidence) && return ("hurdle_poisson", "inferred-from-name")
    occursin("poisson", evidence) && return ("poisson", "inferred-from-name")
    occursin("bernoulli", evidence) && return ("bernoulli", "inferred-from-name")
    occursin("heteroscedastic gaussian comparison", evidence) &&
        return ("gaussian", "inferred-from-catalog-title/formula")
    occursin("ordinal probit", evidence) &&
        return ("ordinal_probit", "inferred-from-catalog-title")
    (occursin("ordinal logistic", evidence) || occursin("polr", evidence)) &&
        return ("ordered_logistic", "inferred-from-catalog-title")
    (occursin("ordinal", evidence) || occursin("cumulative", evidence) ||
     occursin("trolley", evidence)) &&
        return ("ordinal_unresolved", "historical-link-unresolved")
    (occursin("logistic", evidence) || occursin("binary", evidence)) &&
        return ("bernoulli", "inferred-from-name")
    occursin("weibull", evidence) && return ("weibull", "inferred-from-name")
    occursin("lognormal", evidence) && return ("lognormal", "inferred-from-name")
    occursin("gamma", evidence) && return ("gamma", "inferred-from-name")
    occursin("beta regression", evidence) && return ("beta", "inferred-from-name")
    occursin("categorical", evidence) && return ("categorical", "inferred-from-name")
    occursin("student", evidence) && return ("t", "inferred-from-name")

    gaussian_cues = r"gaussian|linear regression|\bols\b|\blm\b|\blmm\b|anova|heteroscedastic|reaction time|height|weight|continuous"
    occursin(gaussian_cues, evidence) && return ("gaussian", "inferred-from-name")
    ("", "unresolved")
end

function semantic_route(row)
    source = row["source"]
    formula = lowercase(row["formula_claim"])
    name = lowercase(row["name_claim"])

    if source == "bmm"
        return ("brm_kernel", "faithful-candidate: group-local measurement-model parameters; likelihood cell still bespoke")
    end
    if source == "action_models"
        return ("brm_kernel", "faithful-candidate: formula is a linked population model for latent cognitive parameters; the observed-data action likelihood and trial-state update must remain in one kernel")
    end
    if source in ("epinowcast", "flocker", "mvgam")
        return ("stanblocks_plate", "faithful-candidate: structured latent/time-series cells exceed an ordinary population formula")
    end
    if source == "inla" && occursin(r"\bf\s*\(", formula)
        return ("stanblocks_plate", "faithful-candidate: INLA latent/group term requires an explicit plate declaration")
    end
    if source == "mcmcglmm" && (occursin("animal", formula) || occursin("bivariate", name))
        return ("stanblocks_plate", "faithful-candidate: relationship/cross-response covariance")
    end
    if occursin(r"\bmm\s*\(", formula)
        return ("stanblocks_plate", "faithful-candidate: weighted multi-membership effects need an explicit plate adapter; the stock BRM term is missing")
    end
    if occursin(r"\b(?:cens|censored|trunc)\s*\(", formula)
        return ("brm_kernel", "faithful-candidate: censoring/truncation needs a row-specific density or survival contribution; no generic response modifier is shipped")
    end
    if occursin("set_rescor(TRUE)", row["formula_claim"]) ||
       (occursin(r"\bmvbind\s*\(", formula) && !occursin("set_rescor(FALSE)", row["formula_claim"]))
        return ("stanblocks_plate", "faithful-candidate: correlated multivariate outcome requires explicit covariance and packed likelihood")
    end
    if occursin("gr(phylo", formula) || occursin("fcor(", formula) ||
       (source == "glmmtmb" && occursin(r"ar1\(|exp\([^|]*\|", formula))
        return ("stanblocks_plate", "faithful-candidate: structured covariance is not an ordinary IID group effect")
    end
    if !occursin('~', formula) || startswith(formula, "baselinenowcast(")
        return ("unsupported", "not a BRM regression declaration")
    end
    ("ordinary_brm", "direct population/likelihood route when the family and terms are supported")
end

replace_uncorrelated(rhs) = rhs # `||` is the current BRM marker syntax.

function normalize_interactions(rhs)
    # Current BRM deliberately uses `&` rather than R/brms `:` because Julia's
    # `:` precedence is incompatible with formula term splitting.
    rhs = replace(rhs, r"\b([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)\b" => s"\1 & \2")

    nstars = count(==('*'), rhs)
    nstars == 0 && return (rhs, "")
    nstars == 1 || return (rhs, "three-way/chained `*` expansion cannot be represented by current binary raw-column `&`")
    m = match(r"\b([A-Za-z_]\w*)\s*\*\s*([A-Za-z_]\w*)\b", rhs)
    isnothing(m) && return (rhs, "current `&` interactions require raw columns; transformed/non-atomic `*` operands need a data-derivation step")
    expansion = "$(m.captures[1]) + $(m.captures[2]) + ($(m.captures[1]) & $(m.captures[2]))"
    before = m.offset == firstindex(rhs) ? "" : rhs[firstindex(rhs):prevind(rhs, m.offset)]
    after_start = nextind(rhs, m.offset, length(m.match))
    after = after_start > lastindex(rhs) ? "" : rhs[after_start:lastindex(rhs)]
    (string(before, expansion, after), "")
end

function normalize_rhs(rhs)
    rhs = replace(rhs, "**" => "^", "True" => "true", "False" => "false")
    rhs = replace(rhs, r"^\s*-1\s*\+" => "0 +")
    rhs = replace(rhs, r"\bI\(" => "protect(")
    rhs = replace(rhs, r"\bscale\(" => "zscale(")
    # `S(x)` / `C(x)` are source-language type annotations. Current BRM is
    # dtype-driven, so preserve the raw column and require integer/categorical
    # data instead of wrapping a term that interaction/random-effect emitters
    # cannot consume.
    rhs = replace(rhs, r"\b(?:C|S|factor)\(\s*([A-Za-z_]\w*)\s*\)" => s"\1")
    rhs = replace(rhs, r"\bzerocorr\(\s*([^()]*)\|\s*([^()]*)\)" => s"(\1 || \2)")
    rhs = replace(rhs, r"\boffset\((log\([^()]+\))\)" => s"\1")
    rhs = replace(rhs, r"\bar\(\s*time\s*=\s*([A-Za-z_]\w*)\s*,\s*p\s*=\s*1\s*\)" => s"ar(\1; p=1)")
    rhs = replace(rhs, r"\{([^{}]+)\}" => s"protect(\1)")
    rhs = replace_uncorrelated(rhs)
    rhs
end

function split_top_level_commas(value)
    parts = String[]
    depth = 0
    start = firstindex(value)
    for index in eachindex(value)
        char = value[index]
        char == '(' && (depth += 1)
        char == ')' && (depth -= 1)
        if char == ',' && depth == 0
            push!(parts, strip(value[start:prevind(value, index)]))
            start = nextind(value, index)
        end
    end
    push!(parts, strip(value[start:lastindex(value)]))
    parts
end

function unwrap_distributional_formula(formula)
    stripped = strip(formula)
    !startswith(stripped, "bf(") && return (stripped, Dict{String,String}(), "")
    length(collect(eachmatch(r"\bbf\(", stripped))) == 1 ||
        return (stripped, Dict{String,String}(), "joint/multivariate bf declarations require a structured design")
    endswith(stripped, ')') ||
        return (stripped, Dict{String,String}(), "bf declaration has trailing joint-model terms")
    inner = stripped[nextind(stripped, firstindex(stripped), 3):prevind(stripped, lastindex(stripped))]
    parts = split_top_level_commas(inner)
    isempty(parts) && return (stripped, Dict{String,String}(), "empty bf declaration")
    count(==('~'), first(parts)) == 1 ||
        return (stripped, Dict{String,String}(), "bf main response formula is not a single declaration")
    auxiliary = Dict{String,String}()
    for part in Iterators.drop(parts, 1)
        if (match_result = match(r"^(sigma|phi|shape|alpha|zi|hu)\s*~\s*(.*)$", part)) !== nothing
            auxiliary[match_result.captures[1]] = strip(match_result.captures[2])
        elseif occursin('~', part)
            return (stripped, Dict{String,String}(), "nonlinear parameter formulas need an explicit current linked-predictor design")
        else
            return (stripped, Dict{String,String}(), "bf attribute `$part` needs family-specific semantic review")
        end
    end
    (first(parts), auxiliary, "")
end

function unwrap_inline_distributional_formula(formula)
    match_result = match(
        r"^\s*([A-Za-z_]\w*(?:\s*\|\s*(?:trials|se|resp_se)\([^)]*\))?)\s*~\s*(.*?)\s+\+\s+(sigma|phi|shape|alpha|zi|hu)\s*~\s*(.*?)\s*$",
        formula,
    )
    isnothing(match_result) && return (formula, Dict{String,String}())
    main_lhs, main_rhs, parameter, parameter_rhs = match_result.captures
    ("$(strip(main_lhs)) ~ $(strip(main_rhs))",
     Dict(parameter => strip(parameter_rhs)))
end

function rewrite_hsgp(rhs)
    matches = collect(eachmatch(r"\bhsgp\(([^()]*)\)", rhs))
    isempty(matches) && return (rhs, "")
    length(matches) == 1 || return (rhs, "multiple HSGP terms require a reviewed current GP design")
    match_result = only(matches)
    parts = split_top_level_commas(match_result.captures[1])
    positional = filter(part -> !occursin('=', part), parts)
    length(positional) == 1 ||
        return (rhs, "multi-axis HSGP is absent from the current one-input gp(...) surface")
    options = Dict{String,String}()
    for part in filter(part -> occursin('=', part), parts)
        kv = split(part, '='; limit=2)
        length(kv) == 2 || return (rhs, "could not parse HSGP option `$part`")
        options[strip(kv[1])] = strip(kv[2])
    end
    unknown = setdiff(Set(keys(options)), Set(["m", "c", "by", "centered", "share_cov"]))
    isempty(unknown) ||
        return (rhs, "HSGP options need semantic review: $(join(sort(collect(unknown)), ','))")
    get(options, "centered", "false") == "true" &&
        return (rhs, "historical centered HSGP semantics are not established as current gp(...) semantics")
    get(options, "share_cov", "true") == "false" &&
        return (rhs, "historical share_cov=false HSGP semantics are not established as current gp(...) semantics")
    current_options = String[]
    haskey(options, "m") && push!(current_options, "k=$(options["m"])")
    haskey(options, "c") && push!(current_options, "c=$(options["c"])")
    haskey(options, "by") && push!(current_options, "by=$(options["by"])")
    replacement = "gp($(only(positional))" *
        (isempty(current_options) ? "" : "; " * join(current_options, ", ")) * ")"
    before = match_result.offset == firstindex(rhs) ? "" : rhs[firstindex(rhs):prevind(rhs, match_result.offset)]
    after_start = nextind(rhs, match_result.offset, length(match_result.match))
    after = after_start > lastindex(rhs) ? "" : rhs[after_start:lastindex(rhs)]
    (string(before, replacement, after), "")
end

function normalize_predictor(rhs)
    rhs = normalize_rhs(rhs)
    rhs, hsgp_error = rewrite_hsgp(rhs)
    !isempty(hsgp_error) && return (rhs, hsgp_error)
    rhs, interaction_error = normalize_interactions(rhs)
    !isempty(interaction_error) && return (rhs, interaction_error)
    occursin(':', rhs) &&
        return (rhs, "source `:` interaction has a transformed/non-atomic operand; current `&` accepts raw columns only")
    occursin(r"\b[A-Za-z_]\w*\s*/\s*[A-Za-z_]\w*\b", rhs) &&
        return (rhs, "source fixed-effect `/` expansion needs explicit main/interaction or prederived contrast columns")
    occursin(r"\bbs\([^)]*,", rhs) &&
        return (rhs, "bs(...) arguments do not map mechanically to current s(...) semantics")
    rhs = replace(rhs, r"\bbs\(([^(),]+)\)" => s"s(\1)")
    (rhs, "")
end

function modifier_lhs(lhs, family)
    lhs = strip(lhs)
    if (m = match(r"^([A-Za-z_]\w*)\s*\|\s*(?:se|resp_se)\(([^,()]+)(?:,\s*sigma\s*=\s*(TRUE|True|true|FALSE|False|false))?\)$", lhs)) !== nothing
        outcome, scale, residual = strip(m.captures[1]), strip(m.captures[2]), something(m.captures[3], "false")
        family in ("gaussian", "student_t") ||
            return (outcome, "", "", "known response SE is only translated here for Gaussian or Student-t likelihoods")
        return (outcome, scale, lowercase(residual), "")
    end
    if (m = match(r"^([A-Za-z_]\w*)\s*\|\s*mi\(\s*\)$", lhs)) !== nothing
        family == "gaussian" ||
            return (lhs, "", "", "non-Normal response imputation is absent from the stock BRM mi surface")
        return ("mi($(m.captures[1]))", "", "", "")
    end
    if occursin(r"\|\s*mi\(", lhs)
        return (lhs, "", "", "predictor/measurement-error or argument-bearing mi semantics are not supported by the stock response-mi adapter")
    end
    (lhs, "", "", "")
end

function support_class(status, note, formula, family, route)
    if status == "ready"
        rewritten = occursin(r"\b(?:bf|hsgp|scale|offset|trials|cbind|I|S|C)\s*\(", formula) ||
                    occursin(':', formula) || occursin('*', formula) ||
                    family == "negativebinomial"
        return (rewritten ? "already-expressible-via-semantic-rewrite" : "already-expressible-verbatim", "")
    elseif status == "semantic-rewrite"
        return ("already-expressible-via-semantic-rewrite", "")
    elseif status == "route-specific"
        if occursin("stock BRM term is missing", note)
            return ("genuinely-missing-brm-surface", "StanBlocks plate substrate exists")
        end
        return ("historical-semantics-unresolved", "kernel/plate substrate control is separately probed")
    elseif status == "unresolved-family"
        return ("historical-semantics-unresolved", "family evidence remains indeterminate")
    end

    if occursin("NegativeBinomial2", note)
        return ("genuinely-missing-brm-surface", "neg_binomial_2_log additionally has a StanBlocks lpxf spelling/trace defect; ordinary neg_binomial_2 substrate is stanc-green")
    elseif occursin("LocationScale Student-t", note)
        return ("genuinely-missing-brm-surface", "reviewed SBBRMI-only adapter candidate is tracked separately; historical degrees-of-freedom semantics remain row-specific")
    elseif occursin("pairing remains historically unresolved", note)
        return ("historical-semantics-unresolved", "ZeroInflatedPoisson surface is landed; historical paired declaration remains unresolved")
    elseif occursin("basis", note) || occursin("bs(...)", note) ||
           occursin("does not map mechanically", note) || occursin("historical", note) ||
           occursin("size is absent", note) || occursin("Wald", note) ||
           occursin("ordinal link", note)
        return ("historical-semantics-unresolved", "")
    elseif occursin("ragged masked", note)
        return ("genuinely-missing-stanblocks-substrate", "")
    else
        return ("genuinely-missing-brm-surface", "StanBlocks building blocks may exist; see surface audits")
    end
end

function formula_columns(formula)
    function_names = Set{String}()
    for m in eachmatch(r"\b([A-Za-z_]\w*)\s*\(", formula)
        push!(function_names, m.captures[1])
    end
    named_arguments = Set{String}()
    for m in eachmatch(r"\b([A-Za-z_]\w*)\s*=", formula)
        push!(named_arguments, m.captures[1])
    end
    reserved = union(function_names, named_arguments, Set([
        "true", "false", "missing", "nothing", "Inf", "pi", "nl", "family",
        "sigma", "phi", "shape", "alpha", "mu", "loc", "rate",
        "log_rate", "log_odds", "log_mu", "log_phi", "nu_minus_two",
        "zi", "disc", "cmc", "cor", "intercept", "centered",
    ]))
    columns = String[]
    for m in eachmatch(r"\b[A-Za-z_]\w*\b", formula)
        token = m.match
        token in reserved && continue
        token in columns || push!(columns, token)
    end
    # `(terms | id | group)`: `id` names a correlation bucket, not a data column.
    for m in eachmatch(r"\|\s*([A-Za-z_]\w*)\s*\|\s*([A-Za-z_]\w*)", formula)
        filter!(!=(m.captures[1]), columns)
    end
    columns
end

function group_columns(formula)
    groups = String[]
    for m in eachmatch(r"\|(?:\s*[A-Za-z_]\w*\s*\|)?\s*([A-Za-z_]\w*)", formula)
        group = m.captures[1]
        group in groups || push!(groups, group)
    end
    groups
end

function translate_formula(formula, family, route)
    isempty(family) && return ("unresolved-family", "", "family is not explicit or defensibly inferred")
    route != "ordinary_brm" && return ("route-specific", "", "requires a faithful $route implementation; no ordinary-formula substitute")

    formula, auxiliary, distributional_error = unwrap_distributional_formula(formula)
    if !isempty(distributional_error)
        if occursin("nonlinear parameter formulas", distributional_error) ||
           occursin("joint/multivariate", distributional_error)
            return ("semantic-rewrite", "", distributional_error * "; current linked-predictor/kernel machinery exists, but this historical declaration needs a row-specific rewrite")
        end
        return ("unsupported", "", distributional_error)
    end
    formula, inline_auxiliary = unwrap_inline_distributional_formula(formula)
    merge!(auxiliary, inline_auxiliary)

    lower = lowercase(formula)
    blocked = Pair[
        r"\b(?:weights|vreal)\(" => "likelihood/frequency weights or vreal payload have no stock BRM emitter; known response SE is a separate supported rewrite",
        r"\bfcor\(" => "fixed/phylogenetic covariance has no stock BRM term adapter",
        r"\bgr\([^)]*\bcov\s*=" => "gr(...; by/id) exists, but fixed cov= structured covariance has no stock BRM adapter",
        r"\b(?:cs|t2|te|dynamic)\(" => "term has no semantics-preserving mechanical alias",
        r"\bmm\(" => "multi-membership term has no current emitter",
        r"\b(?:enw_|rw\()" => "domain-specific latent process",
        r"\b(?:occ|det|col|ex|trend)\s*:" => "multi-component domain model",
    ]
    for (pattern, reason) in blocked
        occursin(pattern, lower) && return ("unsupported", "", reason)
    end
    occursin(r"\bpoly\s*\(", lower) &&
        return ("semantic-rewrite", "", "polynomial basis is expressible through explicitly prederived columns, but raw/orthogonal basis identity and degree must be recovered")
    occursin(r"\|[^)]*(?::|/)[^)]*\)", formula) &&
        return ("semantic-rewrite", "", "nested/composite grouping factors are expressible after deriving stable integer group ids or expanding nesting; exact source identity must be reviewed")
    occursin(';', formula) &&
        return ("semantic-rewrite", "", "multiple components can live in one current declaration/kernel, but this source DSL needs a row-specific semantic rewrite")
    count(==('~'), formula) == 1 ||
        return ("semantic-rewrite", "", "multiple native linked declarations are supported; this source spelling needs a row-specific split")

    m = match(r"^\s*(.*?)\s*~\s*(.*?)\s*$", formula)
    isnothing(m) && return ("unsupported", "", "could not split response and population formula")
    lhs = strip(m.captures[1])
    rhs, predictor_error = normalize_predictor(strip(m.captures[2]))
    if !isempty(predictor_error)
        if occursin("transformed/non-atomic", predictor_error) ||
           occursin("three-way/chained", predictor_error) ||
           occursin("fixed-effect `/`", predictor_error)
            return ("semantic-rewrite", "", predictor_error * "; prederive the exact interaction/design columns")
        elseif occursin("basis", predictor_error) || occursin("historical", predictor_error)
            return ("unsupported", "", predictor_error)
        end
        return ("unsupported", "", predictor_error)
    end

    occursin(r"\bmi\s*\(", rhs) &&
        return ("unsupported", "", "predictor-side mi is not implemented; response-side Normal mi is supported")
    for match_result in eachmatch(r"\bme\(\s*[^,()]+\s*,\s*([^()]+)\)", rhs)
        tryparse(Float64, strip(match_result.captures[1])) === nothing &&
            return ("unsupported", "", "me(...) supports a positive scalar SD only; this row uses a per-row or nonliteral uncertainty")
    end

    normalized_auxiliary = Dict{String,String}()
    for (parameter, predictor) in auxiliary
        normalized, auxiliary_error = normalize_predictor(predictor)
        !isempty(auxiliary_error) &&
            return ("unsupported", "", "$parameter predictor: $auxiliary_error")
        normalized_auxiliary[parameter] = normalized
    end

    lhs, known_se, residual_se, modifier_error = modifier_lhs(lhs, family)
    !isempty(modifier_error) && return ("unsupported", "", modifier_error)

    outcome = lhs
    trials = ""
    if (tm = match(r"^([A-Za-z_]\w*)\s*\|\s*trials\((.+)\)$", lhs)) !== nothing
        outcome, trials = tm.captures
    elseif (pm = match(r"^p\(([^,]+),\s*([^)]+)\)$", lhs)) !== nothing
        outcome, trials = strip.(pm.captures)
    elseif (cm = match(r"^cbind\(([^,]+),\s*([^)]+)\)$", lhs)) !== nothing
        outcome = strip(cm.captures[1])
        trials = "$(strip(cm.captures[1])) + $(strip(cm.captures[2]))"
    elseif !occursin(r"^[A-Za-z_]\w*$", lhs)
        if startswith(lhs, "mi(") && endswith(lhs, ')')
            outcome = lhs
        elseif family in (
            "gaussian", "lognormal", "bernoulli", "binomial", "poisson",
            "ordered_logistic", "beta", "gamma", "weibull",
            "negativebinomial", "zero_inflated_poisson", "student_t",
        )
            return ("semantic-rewrite", "", "response expression is supported after an explicit named data-derivation step; transformed-response Jacobian/conditioning semantics still require review")
        end
    end

    if family == "gaussian"
        unsupported_auxiliary = setdiff(Set(keys(normalized_auxiliary)), Set(["sigma"]))
        isempty(unsupported_auxiliary) ||
            return ("unsupported", "", "Gaussian translation does not implement auxiliary predictors: $(join(sort(collect(unsupported_auxiliary)), ','))")
        sigma_rhs = get(normalized_auxiliary, "sigma", "1")
        if !isempty(known_se)
            if residual_se == "true"
                return ("semantic-rewrite",
                        "loc ~ $rhs\nlog(sigma) ~ $sigma_rhs\n$outcome ~ Normal(loc, sqrt($known_se^2 + sigma^2))",
                        "known response SE plus residual sigma is semantically expressible, but this exact historical scale combination needs a fresh executable row probe")
            end
            return ("ready", "loc ~ $rhs\n$outcome ~ Normal(loc, $known_se)",
                    "known response SE moved into the explicit Normal scale; no fractional likelihood weighting implied")
        end
        return ("ready", "loc ~ $rhs\nlog(sigma) ~ $sigma_rhs\n$outcome ~ Normal(loc, sigma)",
                "explicit Normal likelihood; named log-scale sigma predictor uses native linked-predictor syntax")
    elseif family == "lognormal"
        return ("ready", "loc ~ $rhs\nlog(sigma) ~ 1\n$outcome ~ LogNormal(loc, sigma)",
                "explicit LogNormal likelihood")
    elseif family == "bernoulli"
        return ("ready", "log_odds ~ $rhs\n$outcome ~ BernoulliLogit(log_odds)",
                "explicit Bernoulli-logit likelihood")
    elseif family == "binomial"
        isempty(trials) && return ("unsupported", "", "binomial size is absent; refusing to guess Bernoulli versus grouped Binomial")
        return ("ready", "log_odds ~ $rhs\n$outcome ~ BinomialLogit($trials, log_odds)",
                "trials/cbind response lowered to explicit BinomialLogit")
    elseif family == "poisson"
        return ("ready", "log_rate ~ $rhs\n$outcome ~ Poisson(exp(log_rate))",
                "explicit log-link Poisson likelihood; offset(...) becomes ordinary log exposure")
    elseif family == "ordered_logistic"
        strip(rhs) == "0" && return (
            "semantic-rewrite", "loc = 0\n$outcome ~ OrderedLogistic(loc)",
            "zero linear predictor is semantically available, but current population lowering rejects an empty `loc ~ 0` predictor; use an explicit vector/scalar binding after a row probe",
        )
        return ("ready", "loc ~ $rhs\n$outcome ~ OrderedLogistic(loc)",
                "current proportional-odds likelihood; non-logit historical links need separate review")
    elseif family == "beta"
        unsupported_auxiliary = setdiff(Set(keys(normalized_auxiliary)), Set(["phi"]))
        isempty(unsupported_auxiliary) ||
            return ("unsupported", "", "Beta translation does not implement auxiliary predictors: $(join(sort(collect(unsupported_auxiliary)), ','))")
        # Keep the vector-valued shape expressions inline. Binding them to
        # intermediate names currently loses StanBlocks' lpxf expression
        # provenance even though the same inline Beta likelihood is valid.
        phi_rhs = get(normalized_auxiliary, "phi", "1")
        body = "logit(mu) ~ $rhs\nlog(phi) ~ $phi_rhs\n$outcome ~ Beta(mu * phi, (1 - mu) * phi)"
        return ("ready", body, "mean/precision beta parameterization written explicitly with inline shape expressions")
    elseif family == "gamma"
        unsupported_auxiliary = setdiff(Set(keys(normalized_auxiliary)), Set(["shape", "alpha"]))
        isempty(unsupported_auxiliary) ||
            return ("unsupported", "", "Gamma translation does not implement auxiliary predictors: $(join(sort(collect(unsupported_auxiliary)), ','))")
        shape_rhs = get(normalized_auxiliary, "shape", get(normalized_auxiliary, "alpha", "1"))
        body = "log(mu) ~ $rhs\nlog(shape) ~ $shape_rhs\nrate = shape / mu\n$outcome ~ Gamma(shape, rate)"
        return ("ready", body, "mean/shape gamma model converted to Distributions shape/rate")
    elseif family == "weibull"
        body = "log(scale) ~ $rhs\nlog(shape) ~ 1\n$outcome ~ Weibull(shape, scale)"
        return ("ready", body, "explicit Weibull shape/scale likelihood; censoring remains unsupported")
    elseif family == "negativebinomial"
        unsupported_auxiliary = setdiff(Set(keys(normalized_auxiliary)), Set(["phi"]))
        isempty(unsupported_auxiliary) ||
            return ("unsupported", "", "NegativeBinomial2 design does not implement auxiliary predictors: $(join(sort(collect(unsupported_auxiliary)), ','))")
        phi_rhs = get(normalized_auxiliary, "phi", "1")
        proposed = "log(mu) ~ $rhs\nlog(phi) ~ $phi_rhs\n$outcome ~ NegativeBinomial2(mu, phi)"
        return ("ready", proposed,
                "native linked predictors and the SBBRMI-only NegativeBinomial2(mu,phi) adapter landed at canonical 11031f2; this exact census uses the non-log neg_binomial_2 path on StanBlocks 329a178, while the formerly broken neg_binomial_2_log trace was subsequently fixed in StanBlocks 144188a")
    elseif family == "zero_inflated_poisson"
        return ("unsupported", "", "the SBBRMI-only ZeroInflatedPoisson likelihood/RNG surface landed at canonical 11031f2, but these historical rows split mean and zero-inflation components across separate catalogue cards; their pairing and joint declaration remain unresolved")
    elseif family == "student_t"
        if !isempty(known_se)
            scale = residual_se == "true" ?
                "sqrt($known_se * $known_se + sigma * sigma)" : known_se
            sigma_decl = residual_se == "true" ? "log(sigma) ~ 1\n" : ""
            return (
                "semantic-rewrite",
                "loc ~ $rhs\n$(sigma_decl)$outcome ~ LocationScale(loc, $scale, TDist(nu))",
                "row-varying known response SE composes with the fitted residual sigma inside the LocationScale Student-t scale; the surface is executable, but `nu` remains symbolic until the historical degrees-of-freedom value/prior is recovered",
            )
        end
        return ("semantic-rewrite", "loc ~ $rhs\nlog(scale) ~ 1\n$outcome ~ LocationScale(loc, scale, TDist(nu))",
                "the SBBRMI-only LocationScale Student-t dispatcher landed at canonical 11031f2 and is executable; each row still needs its historical degrees-of-freedom value/prior recovered before this symbolic `nu` body is faithful")
    elseif family == "beta_binomial"
        return ("unsupported", "", "beta-binomial likelihood has StanBlocks building blocks but no stock BRM family marker/lowering")
    elseif family == "asymmetriclaplace"
        return ("unsupported", "", "asymmetric-Laplace quantile likelihood has no stock BRM family adapter")
    elseif family in ("categorical", "multivariate_gaussian", "multivariate_lognormal", "multivariate_skew_normal_gaussian")
        return ("unsupported", "", "outcome family `$family` has no stock ordinary BRM likelihood adapter; independent responses may be split only when authoritative residual independence is established")
    elseif family in ("sratio", "ordinal_probit", "ordinal_unresolved")
        return ("unsupported", "", "historical ordinal link/threshold semantics are not equivalent to stock OrderedLogistic")
    elseif family == "wald"
        return ("unsupported", "", "historical Wald/inverse-Gaussian parameterization is unresolved and no stock BRM likelihood mapping is audited")
    end
    ("unsupported", "", "likelihood family `$family` has no semantics-preserving current translation")
end

function translate_catalog(; input=DEFAULT_INPUT, output=DEFAULT_OUTPUT,
                           family_audit=DEFAULT_FAMILY_AUDIT)
    historical = read_tsv(input)
    family_audit_by_key = load_family_audit(family_audit)
    historical_keys = Set(row["source"] * ":" * row["key"] for row in historical)
    unknown_audit_keys = setdiff(Set(keys(family_audit_by_key)), historical_keys)
    isempty(unknown_audit_keys) ||
        error("family audit contains unknown rows: $(join(sort(collect(unknown_audit_keys)), ','))")
    rows = Dict{String,String}[]
    for row in historical
        route, route_note = semantic_route(row)
        inferred_family, inferred_provenance = infer_family(row, family_audit_by_key)
        row_key = row["source"] * ":" * row["key"]
        audit = get(family_audit_by_key, row_key, Dict{String,String}())
        for variant in ("exact-metadata", "inferred-family")
            family = variant == "exact-metadata" ? normalize_family(row["family_claim"]) : inferred_family
            family_provenance = variant == "exact-metadata" ?
                (isempty(family) ? "unresolved" : "explicit") : inferred_provenance
            status, body, translation_note = translate_formula(row["formula_claim"], family, route)
            class, secondary_gap = support_class(
                status,
                string(translation_note, "; ", route_note),
                row["formula_claim"],
                family,
                route,
            )
            class in SUPPORT_CLASSES || error("invalid support class for $row_key: $class")
            columns = formula_columns(row["formula_claim"])
            groups = group_columns(row["formula_claim"])
            out = copy(row)
            merge!(out, Dict(
                "variant" => variant,
                "family_selected" => family,
                "family_selected_provenance" => family_provenance,
                "family_audit_evidence_class" => get(audit, "evidence_class", ""),
                "family_audit_evidence_strength" => get(audit, "evidence_strength", ""),
                "family_audit_authoritative_surface" => get(audit, "authoritative_surface", ""),
                "family_audit_authoritative_revision" => get(audit, "authoritative_revision", ""),
                "family_audit_authoritative_path_anchor" => get(audit, "authoritative_path_anchor", ""),
                "family_audit_retrieved_at_utc" => get(audit, "retrieved_at_utc", ""),
                "family_audit_http_status" => get(audit, "http_status", ""),
                "family_audit_explicit_evidence" => get(audit, "explicit_evidence", ""),
                "family_audit_semantic_evidence" => get(audit, "semantic_evidence", ""),
                "family_audit_dataset_evidence" => get(audit, "dataset_evidence", ""),
                "family_audit_negative_evidence" => get(audit, "negative_evidence", ""),
                "family_audit_decision_rationale" => get(audit, "decision_rationale", ""),
                "semantic_route" => route,
                "route_note" => route_note,
                "translation_status" => status,
                "surface_support_class" => class,
                "surface_secondary_gap" => secondary_gap,
                "current_brm_body" => body,
                "translation_note" => translation_note,
                "data_columns" => join(columns, ','),
                "group_columns" => join(groups, ','),
                "data_shape_assumptions" => "synthetic n=12; positive continuous predictors; integer groups; family-valid outcome",
            ))
            push!(rows, out)
        end
    end
    columns = vcat(collect(keys(first(historical))), [
        "variant", "family_selected", "family_selected_provenance",
        "family_audit_evidence_class", "family_audit_evidence_strength",
        "family_audit_authoritative_surface", "family_audit_authoritative_revision",
        "family_audit_authoritative_path_anchor", "family_audit_retrieved_at_utc",
        "family_audit_http_status", "family_audit_explicit_evidence",
        "family_audit_semantic_evidence", "family_audit_dataset_evidence",
        "family_audit_negative_evidence", "family_audit_decision_rationale",
        "semantic_route", "route_note", "translation_status",
        "surface_support_class", "surface_secondary_gap", "current_brm_body",
        "translation_note", "data_columns", "group_columns", "data_shape_assumptions",
    ])
    write_tsv(output, columns, rows)
    visible = filter(row -> row["deployed"] == "true", rows)
    println("translation_rows=$(length(rows))")
    println("deployed_translation_rows=$(length(visible))")
    for variant in ("exact-metadata", "inferred-family")
        vr = filter(row -> row["variant"] == variant && row["deployed"] == "true", rows)
        counts = Dict(status => count(row -> row["translation_status"] == status, vr)
                      for status in sort(unique(row["translation_status"] for row in vr)))
        summary = join((string(key, ':', value) for (key, value) in sort(collect(counts))), ',')
        println("$variant=$summary")
    end
    route_counts = Dict(route => count(row -> row["semantic_route"] == route &&
                                       row["variant"] == "inferred-family" &&
                                       row["deployed"] == "true", rows)
                        for route in sort(unique(row["semantic_route"] for row in rows)))
    route_summary = join((string(key, ':', value) for (key, value) in sort(collect(route_counts))), ',')
    println("routes=$route_summary")
    inferred_visible = filter(row -> row["variant"] == "inferred-family" &&
                                     row["deployed"] == "true", rows)
    support_counts = Dict(class => count(row -> row["surface_support_class"] == class,
                                         inferred_visible)
                          for class in sort(collect(SUPPORT_CLASSES)))
    println("surface_support=" * join((string(key, ':', value)
                                       for (key, value) in sort(collect(support_counts))), ','))
    audit_class_counts = Dict(class => count(row -> row["evidence_class"] == class,
                                             values(family_audit_by_key))
                              for class in sort(collect(FAMILY_AUDIT_CLASSES)))
    println("family_audit=" * join((string(key, ':', value)
                                    for (key, value) in sort(collect(audit_class_counts))), ','))
    println("output=$(abspath(output))")
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    input = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_INPUT
    output = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT
    family_audit = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_FAMILY_AUDIT
    translate_catalog(; input, output, family_audit)
end
