# ---- Julia-first kernel cells on StanBlocks ---------------------------------
#
# A kernel cell (`@plate for`, or the deprecated `kernel(...) do`) is ordinary
# Julia source. RKBRMI evaluates it as written. StanBlocks traces most of the
# same source unchanged: operators, dotted broadcasts, indexing, and every
# function whose Stan builtin shares its Julia name (`exp`, `log`, `sum`,
# `abs2.`, ...). This pass supplies the SB backend's remaining Stan spellings:
#
# - a deterministic call whose Julia callable is a differently named Stan
#   builtin (`logistic` -> `inv_logit`, `normcdf` -> `Phi`, `cumsum` ->
#   `cumulative_sum`, `fill`/`zeros`/`ones` -> `rep_vector`/`rep_matrix`,
#   `vcat` -> `append_row`, ...);
# - distribution statistics over an explicit constructor (`mean`/`var` of
#   lower-truncated LogNormal, and its separate `logcdf`/`logccdf`), using the numerical
#   primitives in StanBlocks (`sb_distribution_statistics.jl`);
# - an observation constructor from Distributions (`y ~ Normal(mu, s)`),
#   through the `_sb_stan_distribution_call` table the top-level likelihoods
#   use, also as the family of `censored`/`truncated`/`interval_censored`/
#   `weighted`.
#
# A call is matched by the callable the cell's own module binds, never by its
# spelling, so a caller function that happens to be named `logistic` is left
# to StanBlocks. Everything else passes through unchanged: StanBlocks
# spellings keep their current Stan source, and a caller helper resolves to the
# `@deffun` of the same name, whose Julia methods are what RKBRMI calls.
#
# A model-scope expression (`u = vcat(a, b)`, a likelihood argument) holds the
# callable `@brm` evaluated, and `_sb_scalar_expr` spells it through the same
# `_sb_julia_stan_call` table.

# The value `head` names in `scope`, or `nothing` when nothing binds it there.
_sb_cell_binding(scope, head) = nothing
_sb_cell_binding(scope::Module, head::Symbol) =
    isdefined(scope, head) ? getfield(scope, head) : nothing
_sb_cell_binding(scope, head::GlobalRef) =
    isdefined(head.mod, head.name) ? getfield(head.mod, head.name) : nothing
function _sb_cell_binding(scope::Module, head::Expr)
    Meta.isexpr(head, :., 2) && head.args[2] isa QuoteNode || return nothing
    owner = _sb_cell_binding(scope, head.args[1])
    field = head.args[2].value
    owner isa Module && field isa Symbol && isdefined(owner, field) ?
        getfield(owner, field) : nothing
end

# The Stan builtin a Julia callable is spelled as, for calls and dotted calls.
_sb_julia_stan_name(callable) = nothing
_sb_julia_stan_name(::typeof(LogExpFunctions.logistic)) = :inv_logit
_sb_julia_stan_name(::typeof(LogExpFunctions.log1pexp)) = :log1p_exp
_sb_julia_stan_name(::typeof(LogExpFunctions.log1mexp)) = :log1m_exp
_sb_julia_stan_name(::typeof(LogExpFunctions.logsumexp)) = :log_sum_exp
# The standard normal CDF and its inverse, one argument each.
_sb_julia_stan_name(::typeof(StatsFuns.normcdf)) = :Phi
_sb_julia_stan_name(::typeof(StatsFuns.norminvcdf)) = :inv_Phi
_sb_julia_stan_name(::typeof(cumsum)) = :cumulative_sum

# The Stan call for a Julia call, or `nothing` to keep the authored call.
function _sb_julia_stan_call(callable, args)
    name = _sb_julia_stan_name(callable)
    name === nothing ? nothing : Expr(:call, name, args...)
end
# Constructors whose Stan builtin depends on the result's rank.
_sb_julia_stan_call(::typeof(fill), args) =
    length(args) == 2 ? Expr(:call, :rep_vector, args...) :
    length(args) == 3 ? Expr(:call, :rep_matrix, args...) : nothing
_sb_julia_stan_call(::typeof(zeros), args) = _sb_julia_stan_call(fill, Any[0.0; args])
_sb_julia_stan_call(::typeof(ones), args) = _sb_julia_stan_call(fill, Any[1.0; args])
# Stan concatenates two values at a time.
_sb_julia_stan_call(::typeof(vcat), args) = length(args) < 2 ? nothing :
    foldl((a, b) -> Expr(:call, :append_row, a, b), args)
# The square matrix with a vector on its diagonal.
_sb_julia_stan_call(::Type{Diagonal}, args) =
    length(args) == 1 ? Expr(:call, :diag_matrix, args...) : nothing
_sb_julia_stan_call(::typeof(diagm), args) =
    length(args) == 1 ? Expr(:call, :diag_matrix, args...) : nothing

_sb_cell_combinator(head) =
    head in (:censored, :truncated, :interval_censored, :weighted)

_sb_cell_keyword_argument(arg) = Meta.isexpr(arg, (:parameters, :kw))

# One call argument in StanBlocks spelling; keyword names stay as written.
_sb_cell_argument(arg, scope) = _sb_cell_value(arg, scope)
function _sb_cell_argument(arg::Expr, scope)
    Meta.isexpr(arg, :kw, 2) &&
        return Expr(:kw, arg.args[1], _sb_cell_value(arg.args[2], scope))
    Meta.isexpr(arg, :parameters) &&
        return Expr(:parameters, (_sb_cell_argument(a, scope) for a in arg.args)...)
    _sb_cell_value(arg, scope)
end

_sb_cell_keywords!(keywords, arg) = keywords
_sb_cell_keywords!(keywords, arg::Symbol) = push!(keywords, arg => arg)
function _sb_cell_keywords!(keywords, arg::Expr)
    if Meta.isexpr(arg, :kw, 2)
        push!(keywords, arg.args[1] => arg.args[2])
    elseif Meta.isexpr(arg, :parameters)
        foreach(a -> _sb_cell_keywords!(keywords, a), arg.args)
    end
    keywords
end

# A deterministic cell expression in StanBlocks spelling.
_sb_cell_value(x, scope) = x
function _sb_cell_value(x::Expr, scope)
    if x.head === :call
        head = first(x.args)
        args = Any[_sb_cell_argument(a, scope) for a in x.args[2:end]]
        if !any(_sb_cell_keyword_argument, args)
            callable = _sb_cell_binding(scope, head)
            statistic = _sb_distribution_statistic(callable, args, scope)
            statistic === nothing || return statistic
            rewritten = callable === nothing ? nothing : _sb_julia_stan_call(callable, args)
            rewritten === nothing || return rewritten
        end
        return Expr(:call, head, args...)
    elseif _brm_is_broadcast_call(x)
        head = first(x.args)
        callable = _sb_cell_binding(scope, head)
        name = callable === nothing ? nothing : _sb_julia_stan_name(callable)
        return Expr(:., something(name, head),
            Expr(:tuple, (_sb_cell_argument(a, scope) for a in x.args[2].args)...))
    elseif x.head === :. && length(x.args) == 2 && x.args[2] isa QuoteNode
        return Expr(:., _sb_cell_value(x.args[1], scope), x.args[2])
    end
    Expr(x.head, (_sb_cell_value(a, scope) for a in x.args)...)
end

# An observation's law in StanBlocks spelling: a Distributions constructor
# becomes its Stan family call, and a combinator spells its family argument.
# Any other family head is a StanBlocks family token and stays as written
# (Stan's `logistic` family is not the Julia `logistic` function).
function _sb_cell_family(x, scope)
    Meta.isexpr(x, :call) || return _sb_cell_value(x, scope)
    head = first(x.args)
    callable = _sb_cell_binding(scope, head)
    callable isa BRMComponent && return _sb_component_cell_call(callable, x, scope)
    args = Any[_sb_cell_argument(a, scope) for a in x.args[2:end]]
    if callable isa Type && callable <: Distribution
        positional = Tuple(filter(!_sb_cell_keyword_argument, args))
        keywords = foldl(_sb_cell_keywords!, filter(_sb_cell_keyword_argument, args);
            init=Pair{Symbol,Any}[])
        return _sb_cell_distribution_call(callable, positional, NamedTuple(keywords))
    elseif head isa Symbol && _sb_cell_combinator(head)
        family = findfirst(!_sb_cell_keyword_argument, x.args[2:end])
        family === nothing || (args[family] = _sb_cell_family(x.args[1 + family], scope))
    end
    Expr(:call, head, args...)
end

# A Distributions constructor in a cell observation: the top-level table, except
# for laws that only a cell observes.
_sb_cell_distribution_call(callable, positional, keywords) =
    _sb_stan_distribution_call(callable, positional, keywords)
# `y[rows[i]] ~ MvNormal(location, covariance)` is one joint density of the
# cell's slice. Julia gives an empty slice log density 0, which Stan's
# `multi_normal` refuses, so the cell observes `brm_mvnormal` instead.
function _sb_cell_distribution_call(::Type{<:MvNormal}, positional, keywords)
    isempty(keywords) || error(
        "sbimpl: an in-cell `MvNormal` observation takes no keywords, got ",
        "$(collect(keys(keywords)))")
    length(positional) == 2 || error(
        "sbimpl: an in-cell observation `y[rows[i]] ~ MvNormal(location, covariance)` ",
        "takes a location vector and a covariance matrix; got $(length(positional)) arguments")
    Expr(:call, brm_mvnormal, positional...)
end

StanBlocks.@deffun begin
    brm_mvnormal_rng(vector[n], location::vector[n], covariance::matrix[n, n])::vector[n] = begin
        result::vector[n]
        if n > 0
            result = multi_normal_rng(location, covariance)
        end
        return result
    end
    brm_mvnormal_lpdfs(value::vector[n], location::vector[n], covariance::matrix[n, n])::real = begin
        result::real = 0.0
        if n > 0
            result = multi_normal_lpdf(value, location, covariance)
        end
        return result
    end
    @lhs @lpxf brm_mvnormal_lpdf(value::vector[n], location::vector[n], covariance::matrix[n, n])::real =
        brm_mvnormal_lpdfs(value, location, covariance)
end

# An unbound family token (`normal`) names the StanBlocks family of that name.
function _sb_cell_family_binding(scope, head)
    callable = _sb_cell_binding(scope, head)
    callable === nothing && head isa Symbol && isdefined(StanBlocks, head) ?
        getfield(StanBlocks, head) : callable
end

_sb_cell_statement(x, scope) = _sb_cell_value(x, scope)
function _sb_cell_statement(x::Expr, scope)
    Meta.isexpr(x, :local, 1) &&
        return Expr(:local, _sb_cell_statement(only(x.args), scope))
    # A cell's `@scan` block and its step loop hold statements of their own.
    _is_brm_scan(x) &&
        return Expr(:macrocall, x.args[1:end-1]..., _sb_cell_block(x.args[end], scope))
    Meta.isexpr(x, :for) &&
        return Expr(:for, _sb_cell_value(x.args[1], scope), _sb_cell_block(x.args[2], scope))
    Meta.isexpr(x, :call, 3) && x.args[1] === :~ || return _sb_cell_value(x, scope)
    law = x.args[3]
    if Meta.isexpr(law, :call) && first(law.args) === :weighted
        law = _brm_cell_typed_weight(head -> _sb_cell_family_binding(scope, head),
            law, "sbimpl"; normal=c -> c === Normal || c === StanBlocks.normal)
    end
    Expr(:call, :~, x.args[2], _sb_cell_family(law, scope))
end

function _sb_cell_statements(statements, scope)
    out = Any[]
    for statement in statements
        # A hidden Markov observation is its emission locals and the
        # forward-algorithm family (sb_hidden_markov.jl).
        if _sb_is_cell_hidden_markov(statement, scope)
            append!(out, _sb_cell_hidden_markov(statement, scope))
            continue
        end
        # A latent in total coordinates is two statements (cell_total_coordinates.jl).
        total = _sb_cell_total_coordinates(statement, scope)
        total === nothing ? push!(out, _sb_cell_statement(statement, scope)) :
            append!(out, total)
    end
    out
end
_sb_cell_block(x, scope) = Meta.isexpr(x, :block) ?
    Expr(:block, _sb_cell_statements(x.args, scope)...) : _sb_cell_statement(x, scope)

# StanBlocks traces a cell in statement order. A bare local prior can be
# authored after its readers, just like a model-scope prior. Schedule those
# declarations through the shared dependency walk, retaining the relative
# order of every other statement (including assignments and entire scans).
# This operates on one cell only; it never moves a draw across a plate or
# changes the sequential steps inside a scan.
_sb_cell_unwrap_local(x) = Meta.isexpr(x, :local, 1) ? only(x.args) : x
function _sb_cell_prior_order(statements, inputs)
    assignments = Dict{Symbol,Vector{Int}}()
    for (i, statement) in enumerate(statements)
        s = _sb_cell_unwrap_local(statement)
        Meta.isexpr(s, :(=), 2) || continue
        name = _sb_bound_name(s.args[1])
        name === nothing || push!(get!(assignments, name, Int[]), i)
    end
    declarations = Dict{Symbol,Int}()
    for (i, statement) in enumerate(statements)
        s = _sb_cell_unwrap_local(statement)
        isxcall(s, :~) || continue
        name = _sb_bound_name(s.args[2])
        name === nothing && continue
        # An input or an assigned value is observed, not a fresh draw. An
        # explicit `local` declaration instead shadows that outer binding.
        _sb_local_name(statement) === nothing &&
            (name in inputs || haskey(assignments, name)) && continue
        get!(declarations, name, i)
    end
    isempty(declarations) && return statements
    priors = Set(values(declarations))
    keys = [Symbol(:cell_statement_, i) for i in eachindex(statements)]
    dependencies = [Set{Int}() for _ in statements]
    previous = nothing
    for (i, statement) in enumerate(statements)
        prior = i in priors
        s = _sb_cell_unwrap_local(statement)
        refs = _brm_cell_value_refs!(Set{Symbol}(), prior ? s.args[3] : s)
        union!(dependencies[i], (j for (name, j) in declarations if name in refs && j != i))
        if prior
            # Preserve the version of an ordinary cell value read by a prior:
            # its preceding assignment stays before it, and the next write
            # stays after it. A forward dependency is retained as well, so a
            # real cycle fails instead of changing the law.
            for name in refs
                writers = get(assignments, name, Int[])
                isempty(writers) && continue
                before = findlast(<(i), writers)
                producer = before === nothing ? first(writers) : writers[before]
                push!(dependencies[i], producer)
                after = findfirst(>(max(i, producer)), writers)
                after === nothing || push!(dependencies[writers[after]], i)
            end
        else
            previous === nothing || push!(dependencies[i], previous)
            previous = i
        end
    end
    operations = [_BRMPreparedOperation(keys[i], :cell, statement,
        Tuple(keys[j] for j in sort!(collect(dependencies[i]))))
        for (i, statement) in enumerate(statements)]
    byname = Dict(op.name => op.expression for op in operations)
    Any[byname[key] for key in _brm_operation_order(operations)]
end

# ---- `AR1Normal` in a cell: the BRM-owned AR(1) submodel ---------------------
#
# The `@slic` submodels `y[p] ~ AR1Normal(...)` calls (`ar1_normal.jl`): the
# exact AR(1) density over one series, at the recorded integer times or one
# step apart. `obs` is the caller's observed series; without it the submodel
# draws the series. A mean, scale or persistence that varies along the series
# is read per step (`mu[t]`); a shared one is read as its one value (`sigma`).
# StanBlocks substitutes each argument where the body reads it, so a shared
# value filled to the series' length would be rebuilt at every step, and the
# gradient would grow with the square of the series' length.
function _sb_ar1_normal_body(along::NTuple{3,Bool}, timed::Bool)
    mu, sigma, phi = map((:mu, :sigma, :phi), along) do name, varies
        t -> varies ? :($name[$t]) : name
    end
    step = timed ? quote
        k = time[t] - time[t - 1]
        obs[t] ~ normal($(mu(:t)) + $(phi(:t)) ^ k * (obs[t - 1] - $(mu(:(t - 1)))),
            $(sigma(:t)) * sqrt((1 - $(phi(:t)) ^ (2 * k)) / (1 - $(phi(:t)) ^ 2)))
    end : quote
        obs[t] ~ normal($(mu(:t)) + $(phi(:t)) * (obs[t - 1] - $(mu(:(t - 1)))), $(sigma(:t)))
    end
    quote
        @scan begin
            obs[1] ~ normal($(mu(1)), $(sigma(1)) / sqrt(1 - $(phi(1)) ^ 2))
            for t in 2:n
                $(step.args...)
            end
        end
    end
end

# One submodel per pattern of which of `mu`, `sigma`, `phi` vary along the
# series, with a time column (`timed`) or one step apart.
# Anonymous submodels are values, so construct them without top-level eval.
# Keep the same scope and empty docstring metadata as an anonymous `@slic`.
const _sb_ar1_normal_models = Dict{Tuple{NTuple{3,Bool},Bool},StanBlocks.SlicModel}()
for along in Iterators.product((false, true), (false, true), (false, true)), timed in (false, true)
    _sb_ar1_normal_models[(along, timed)] =
        StanBlocks.SlicModel(_sb_ar1_normal_body(along, timed),
            Dict{Symbol,Any}(:docstring => ""), @__MODULE__)
end

# A cell's statements with each `y ~ AR1Normal(...)` calling the AR(1)
# submodel: on the observed series (`obs = y`), or drawing it when the
# response is omitted and `y` is no cell parameter.
function _sb_cell_ar1_normal_calls(statements, params)
    any(_is_brm_ar1_normal, statements) || return statements
    used = _brm_cell_value_refs!(Set{Symbol}(params), Expr(:block, statements...))
    map(statements) do statement
        _is_brm_ar1_normal(statement) || return statement
        spec = _brm_kernel_ar1_normal(statement, params, statements)
        observed = spec.response in params
        observed || spec.length != :(length($(spec.response))) || error(
            "sbimpl: `$(spec.response) ~ AR1Normal(...)` with `$(spec.response)` omitted " *
            "draws a series whose length nothing in the cell gives; pass " *
            "`time = t[p]`, or an argument read along the series")
        mu, sigma, phi = spec.arguments
        keywords = Any[Expr(:kw, :mu, mu), Expr(:kw, :sigma, sigma),
            Expr(:kw, :phi, phi), Expr(:kw, :n, spec.length)]
        spec.time === nothing || push!(keywords, Expr(:kw, :time, spec.time))
        node = spec.response
        if observed
            pushfirst!(keywords, Expr(:kw, :obs, spec.response))
            node = Symbol(spec.response, :_ar1)
            while node in used
                node = Symbol(node, :_)
            end
            push!(used, node)
        end
        submodel = _sb_ar1_normal_models[(spec.along, spec.time !== nothing)]
        Expr(:call, :~, node, Expr(:call, submodel, Expr(:parameters, keywords...)))
    end
end

# Whether an emitted statement is a cell's AR(1) submodel call.
# The plan reads a copy of the emitted body, so the submodel is matched by
# its model, not by identity.
_sb_plan_ar1_normal_call(x) = Meta.isexpr(x, :call, 3) && x.args[1] === :~ &&
    Meta.isexpr(x.args[3], :call) && (head = first(x.args[3].args);
        head isa StanBlocks.SlicModel && any(submodel -> isequal(head.model, submodel.model),
            values(_sb_ar1_normal_models)))
