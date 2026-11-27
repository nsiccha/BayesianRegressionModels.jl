# Exact replay of an independent Gaussian intercept observed additively through
# a Normal response. Work in the standardized coordinate used by draw transport.
StanBlocks.@deffun begin
    brm_intercept_posterior(residual::vector[n], noise::vector[n], index::int[n],
                            scale::real, groups::int)::matrix[groups, 2] = begin
        moments = rep_matrix(0., groups, 2)
        for g in 1:groups
            moments[g, 2] = 1.
        end
        for i in 1:n
            if !(noise[i] > 0.)
                reject("condition_groups requires a positive Normal measurement SD")
            end
            g = index[i]
            sd = moments[g, 2]
            marginal_sd = hypot(scale * sd, noise[i])
            gain = sd * (scale * sd / marginal_sd)
            moments[g, 1] += gain * ((residual[i] - scale * moments[g, 1]) / marginal_sd)
            moments[g, 2] = sd * (noise[i] / marginal_sd)
        end
        moments
    end
end

_condition_error(group, response, message) =
    throw(ArgumentError("condition_groups ($group=:$response): $message"))

function _sb_condition_site(sb, body, producers, declarations, group, response)
    fail(msg) = _condition_error(group, response, msg)
    obs = filter(d -> d.role === :observation && d.data_source === response &&
        isempty(d.context), declarations)
    length(obs) == 1 && !(response in sb.held_out) ||
        fail("the replay must supply one observed model-scope Normal response named `$response`")
    measurement = only(obs)
    measurement.family === :normal && length(measurement.arguments) == 2 &&
        isempty(measurement.keywords) || fail("expected an ordinary Normal response")
    predictor, noise = measurement.arguments
    blocks = filter(declarations) do d
        entry = get(sb.preproc, get(d.keywords, :group_idx, nothing), nothing)
        d.role === :prior && entry isa PreprocEntry && entry.raw_ref === group
    end
    length(blocks) == 1 || fail("expected one independent Gaussian intercept grouped by `$group`")
    block = only(blocks)
    family = _ranef_declared_family(block, sb.bindings)
    simple = family === :ranef_intercept
    simple || (family in (:ranef_correlated_draws, :ranef_correlated_draws_generic) &&
        sb.data[block.keywords.n_terms] == 1) ||
        fail("the selected group must be a single independent Gaussian intercept")
    index = block.keywords.group_idx
    term = block.target
    if !simple
        # The single margin must enter the row mean with unit loading.
        gather = Expr(:ref, term, index, 1)
        rows = [s.args[1] for s in body if Meta.isexpr(s, :(=), 2) && s.args[2] == gather]
        length(rows) == 1 || fail("the selected margin must be an intercept with unit loading")
        term = only(rows)
    end
    haskey(producers, predictor) || fail("the Normal mean must be a row predictor")
    mean_stmt = body[producers[predictor]]
    Meta.isexpr(mean_stmt, :(=), 2) && mean_stmt.args[1] === predictor ||
        fail("the Normal mean must be an identity predictor containing the selected intercept")
    # Remove precisely one additive summand. A slope, link or interaction
    # involving the block cannot be treated as a unit-loading observation.
    terms = Any[]
    function summands(x)
        if Meta.isexpr(x, :call) && x.args[1] === :+
            foreach(summands, x.args[2:end])
        else
            push!(terms, x)
        end
    end
    summands(mean_stmt.args[2])
    count(==(term), terms) == 1 || fail("the Normal mean must include the intercept exactly once")
    filter!(!=(term), terms)
    profile = isempty(terms) ? 0. : length(terms) == 1 ? only(terms) : Expr(:call, :+, terms...)
    (; block, family, simple, noise, profile)
end

function _sb_condition_statement(stmt, simple, response, profile, noise)
    # Use the actual configured family, preserving its fitted scale prior,
    # name, constraint and any inputs. Only its standardized draw changes.
    call = stmt.args[3]
    base = simple ? ranef_intercept : call.args[1]
    z, scale = simple ? (:xi, :(exp(log_scale))) : (:z_flat, :(tau[1]))
    stmts = Any[]
    drew = false
    for s in base.model.args
        if Meta.isexpr(s, :call, 3) && s.args[1] === :~ && s.args[2] === z
            drew = true
            push!(stmts, :(conditional_moments = $brm_intercept_posterior(
                measurement_residual, measurement_noise, group_idx, $scale, n_groups)))
            push!(stmts, :($z ~ normal(conditional_moments[:, 1], conditional_moments[:, 2]; n=n_groups)))
        else
            push!(stmts, s)
        end
    end
    drew || error("conditional replay: expected standardized draw `$z` in the selected intercept")
    model = StanBlocks.SlicModel(Expr(:block, stmts...), base.data, base.mod, base.observations)
    kwargs = only(call.args[2:end])
    args = Expr(:parameters, kwargs.args...,
                Expr(:kw, :measurement_residual, :($response .- $profile)),
                Expr(:kw, :measurement_noise, :($noise .+ rep_vector(0., num_elements($response)))))
    Expr(:call, :~, stmt.args[2], Expr(:call, model, args))
end

# Move just the required profile and noise producers ahead of their intercept.
# Other statements, including scoped submodels, keep their original ordering.
function _sb_condition_order(body, producers, changed)
    prerequisites(i) = sort!([producers[k] for k in
        _sb_read_symbols!(Set{Symbol}(), body[i].args[end]) if haskey(producers, k)])
    ordered = Any[]
    emitted = BitSet()
    # A shared upstream profile (e.g. a fitted area effect) only needs to be
    # visited once. An input depending on any selected intercept would require
    # joint conditioning or a non-Gaussian update.
    independent = BitSet()
    function append_once(i)
        i in emitted && return
        push!(ordered, body[i])
        push!(emitted, i)
    end
    function prior_input(i)
        i in changed && throw(ArgumentError("condition_groups: a measurement profile or SD " *
            "depends on a conditioned intercept; the selected updates must be independent"))
        i in independent && return
        push!(independent, i)
        foreach(prior_input, prerequisites(i))
        append_once(i)
    end
    for i in eachindex(body)
        i in changed && foreach(prior_input, prerequisites(i))
        append_once(i)
    end
    Expr(:block, ordered...)
end

function _sb_condition_groups(sb, conditions, redraw)
    body = copy(sb.model.model.args)
    bindings = copy(sb.bindings)
    producers = Dict{Symbol,Int}()
    for (i, stmt) in enumerate(body)
        lhs = Meta.isexpr(stmt, :(=), 2) ? stmt.args[1] :
              Meta.isexpr(stmt, :call, 3) && stmt.args[1] === :~ ? stmt.args[2] : nothing
        target = _sb_bound_name(lhs)
        isnothing(target) || haskey(sb.data, target) || (producers[target] = i)
    end
    declarations = _sb_plan_declarations(sb.model.model, sb.data, sb.parent, sb.model.mod)
    changed = BitSet()
    for (group, response) in pairs(conditions)
        site = _sb_condition_site(sb, body, producers, declarations, group, response)
        key = site.block.target
        i = producers[key]
        body[i] = _sb_condition_statement(body[i], site.simple, response, site.profile, site.noise)
        push!(changed, i)
        bindings[key] = merge(bindings[key], (; family=site.family, condition_response=response, redraw_groups=Tuple(redraw)))
    end
    ordered = _sb_condition_order(body, producers, changed)
    model = StanBlocks.SlicModel(ordered, sb.data, sb.model.mod, sb.model.observations)
    SBBRMI(sb.parent, model, sb.data, sb.preproc, copy(sb.held_out), bindings, sb.source)
end
