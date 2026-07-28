#!/usr/bin/env julia

const DEFAULT_INPUT = joinpath(@__DIR__, "historical_catalog.tsv")
const DEFAULT_OUTPUT = joinpath(@__DIR__, "translations.tsv")

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
        "logistic" => "bernoulli",
    )
    get(aliases, family, family)
end

function infer_family(row)
    explicit = normalize_family(row["family_claim"])
    !isempty(explicit) && return (explicit, "explicit")
    description = normalize_family(row["family_text_claim"])
    !isempty(description) && return (description, "inferred-from-description")

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
    (occursin("logistic", evidence) || occursin("binary", evidence)) &&
        return ("bernoulli", "inferred-from-name")
    (occursin("ordinal", evidence) || occursin("trolley", evidence) ||
     occursin("polr", evidence) || occursin("cumulative", evidence)) &&
        return ("ordered_logistic", "inferred-from-name")
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
    if source in ("epinowcast", "flocker", "mvgam")
        return ("stanblocks_plate", "faithful-candidate: structured latent/time-series cells exceed an ordinary population formula")
    end
    if source == "inla" && occursin(r"\bf\s*\(", formula)
        return ("stanblocks_plate", "faithful-candidate: INLA latent/group term requires an explicit plate declaration")
    end
    if source == "mcmcglmm" && (occursin("animal", formula) || occursin("bivariate", name))
        return ("stanblocks_plate", "faithful-candidate: relationship/cross-response covariance")
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
    rhs = replace(rhs, r"\bI\(" => "protect(")
    rhs = replace(rhs, r"\bscale\(" => "zscale(")
    # `S(x)` / `C(x)` are source-language type annotations. Current BRM is
    # dtype-driven, so preserve the raw column and require integer/categorical
    # data instead of wrapping a term that interaction/random-effect emitters
    # cannot consume.
    rhs = replace(rhs, r"\b(?:C|S)\(\s*([A-Za-z_]\w*)\s*\)" => s"\1")
    rhs = replace(rhs, r"\bzerocorr\(\s*([^()]*)\|\s*([^()]*)\)" => s"(\1 || \2)")
    rhs = replace(rhs, r"\boffset\((log\([^()]+\))\)" => s"\1")
    rhs = replace(rhs, r"\{([^{}]+)\}" => s"protect(\1)")
    rhs = replace_uncorrelated(rhs)
    rhs
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
        "sigma", "zi", "disc", "cmc", "cor", "intercept", "centered",
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

    lower = lowercase(formula)
    blocked = Pair[
        r"\bbf\(" => "multi-formula/nonlinear brms declaration",
        r"\bmvbind\(" => "multivariate response declaration",
        r"\bmi\(" => "missing-value submodel",
        r"\bme\(" => "measurement-error submodel",
        r"\b(?:se|resp_se|weights|vreal)\(" => "weighted/error-aware response",
        r"\b(?:cens|censored|trunc)\(" => "censored/truncated response",
        r"\b(?:fcor|gr)\(" => "structured covariance",
        r"\b(?:cs|poly|t2|te|dynamic)\(" => "term has no semantics-preserving mechanical alias",
        r"\bmm\(" => "multi-membership term has no current emitter",
        r"\b(?:enw_|rw\()" => "domain-specific latent process",
        r"\b(?:occ|det|col|ex|trend)\s*:" => "multi-component domain model",
    ]
    for (pattern, reason) in blocked
        occursin(pattern, lower) && return ("unsupported", "", reason)
    end
    occursin(r"\|[^)]*(?::|/)[^)]*\)", formula) &&
        return ("unsupported", "", "nested/composite grouping factors require an explicit derived group id or reviewed gr(...; by=...) design")
    occursin(';', formula) && return ("unsupported", "", "multiple component formulas require an explicit joint declaration")
    count(==('~'), formula) == 1 || return ("unsupported", "", "formula is not one top-level response formula")

    m = match(r"^\s*(.*?)\s*~\s*(.*?)\s*$", formula)
    isnothing(m) && return ("unsupported", "", "could not split response and population formula")
    lhs, rhs = strip(m.captures[1]), normalize_rhs(strip(m.captures[2]))

    rhs, interaction_error = normalize_interactions(rhs)
    !isempty(interaction_error) && return ("unsupported", "", interaction_error)
    occursin(':', rhs) &&
        return ("unsupported", "", "source `:` interaction has a transformed/non-atomic operand; current `&` accepts raw columns only")

    # Old spline/HSGP call signatures carried backend-specific arguments. Only
    # the argument-free aliases are mechanical; anything richer needs review.
    occursin(r"\bbs\([^)]*,", rhs) &&
        return ("unsupported", "", "bs(...) arguments do not map mechanically to current s(...) semantics")
    occursin(r"\bhsgp\([^)]*,", rhs) &&
        return ("unsupported", "", "multi-axis/multi-argument hsgp does not map to current one-input gp(...)")
    occursin(r"\bhsgp\([^)]*(?:by|share_cov|iso|centered)\s*=", rhs) &&
        return ("unsupported", "", "hsgp options require a current gp(...) semantic redesign")
    rhs = replace(rhs, r"\bbs\(([^(),]+)\)" => s"s(\1)")
    rhs = replace(rhs, r"\bhsgp\(" => "gp(", r"\bm\s*=" => "k=")

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
        return ("unsupported", "", "response expression needs an explicit data-derivation step")
    end

    if family == "gaussian"
        return ("ready", "loc ~ $rhs\nlog(sigma) ~ 1\n$outcome ~ Normal(loc, sigma)",
                "explicit Normal likelihood; residual scale estimated on log scale")
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
        return ("ready", "loc ~ $rhs\n$outcome ~ OrderedLogistic(loc)",
                "current proportional-odds likelihood; non-logit historical links need separate review")
    elseif family == "beta"
        # Keep the vector-valued shape expressions inline. Binding them to
        # intermediate names currently loses StanBlocks' lpxf expression
        # provenance even though the same inline Beta likelihood is valid.
        body = "logit(mu) ~ $rhs\nlog(phi) ~ 1\n$outcome ~ Beta(mu * phi, (1 - mu) * phi)"
        return ("ready", body, "mean/precision beta parameterization written explicitly with inline shape expressions")
    elseif family == "gamma"
        body = "log(mu) ~ $rhs\nlog(shape) ~ 1\nrate = shape / mu\n$outcome ~ Gamma(shape, rate)"
        return ("ready", body, "mean/shape gamma model converted to Distributions shape/rate")
    elseif family == "weibull"
        body = "log(scale) ~ $rhs\nlog(shape) ~ 1\n$outcome ~ Weibull(shape, scale)"
        return ("ready", body, "explicit Weibull shape/scale likelihood; censoring remains unsupported")
    elseif family == "negativebinomial"
        return ("unsupported", "", "current NegativeBinomial Julia/Stan parameterizations are documented as non-identical")
    elseif family == "zero_inflated_poisson"
        return ("unsupported", "", "row does not authoritatively pair the mean and zero-inflation submodels")
    end
    ("unsupported", "", "likelihood family `$family` has no semantics-preserving current translation")
end

function translate_catalog(; input=DEFAULT_INPUT, output=DEFAULT_OUTPUT)
    historical = read_tsv(input)
    rows = Dict{String,String}[]
    for row in historical
        route, route_note = semantic_route(row)
        inferred_family, inferred_provenance = infer_family(row)
        for variant in ("exact-metadata", "inferred-family")
            family = variant == "exact-metadata" ? normalize_family(row["family_claim"]) : inferred_family
            family_provenance = variant == "exact-metadata" ?
                (isempty(family) ? "unresolved" : "explicit") : inferred_provenance
            status, body, translation_note = translate_formula(row["formula_claim"], family, route)
            columns = formula_columns(row["formula_claim"])
            groups = group_columns(row["formula_claim"])
            out = copy(row)
            merge!(out, Dict(
                "variant" => variant,
                "family_selected" => family,
                "family_selected_provenance" => family_provenance,
                "semantic_route" => route,
                "route_note" => route_note,
                "translation_status" => status,
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
        "semantic_route", "route_note", "translation_status", "current_brm_body",
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
    println("output=$(abspath(output))")
    rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    input = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_INPUT
    output = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT
    translate_catalog(; input, output)
end
