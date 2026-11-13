# Statistical components as self-contained RKPPL submodels. Each component
# allocates its own parameters and returns its post-processed value, the
# shape SBBRMI emits with `popefs`, `ranef_*_draws`, `_sb_mo` and `_sb_hsgp`.
# Data preparation and the combination of component values stay in the main
# block. Single-expression algebra is written inline rather than as a submodel.

const _RK_KERNEL_MACRO = Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel")))

# Install one definition per distinct signature and body. Components with
# identical configuration share a definition; a different configuration (or
# a collision with an authored name) receives a fresh name. `doc`
# (`@_rk_definition_doc`) documents a new definition.
function _rk_ast_shared_definition!(definitions, taken, base, arguments, body;
        kernel=false, head=:(=), doc=nothing)
    arguments = collect(Any, arguments)
    for definition in definitions
        definition = _brm_undoc(definition)
        kernel == Meta.isexpr(definition, :macrocall) || continue
        inner = kernel ? last(definition.args) : definition
        Meta.isexpr(inner, head, 2) || continue
        call = first(inner.args)
        Meta.isexpr(call, :call) || continue
        isequal(call.args[2:end], arguments) || continue
        isequal(last(inner.args), body) || continue
        return first(call.args)
    end
    callee = _rk_ast_fresh_name(string(base), taken)
    definition = Expr(head, Expr(:call, callee, arguments...), body)
    kernel && (definition = Expr(:macrocall, _RK_KERNEL_MACRO, LineNumberNode(0), definition))
    push!(definitions, _rk_documented_definition(definition, doc))
    callee
end

_rk_ast_component_body(statements...) = Expr(:block, statements...)

# A component call samples its submodel under `name`; its locals are scoped
# there (`name.beta`, `name.sd`, ...).
_rk_ast_component_call(name, callee, arguments...) =
    Expr(:call, :~, name, Expr(:call, callee, arguments...))

# Caller values in authored priors are explicit component inputs. Otherwise a
# caller named `tau` or `sigma` would be captured by the component's own draw.
# Call heads retain their complete emitted module bindings.
function _rk_ast_component_prior_inputs(priors, occupied)
    reads = Set{Symbol}()
    function collect_reads(value)
        value isa Symbol && push!(reads, value)
        value isa Expr || return
        foreach(collect_reads, last(_rk_ast_value_operands(value)))
    end
    foreach(collect_reads, priors)
    values = sort!(collect(reads); by=string)
    used = union(Set{Symbol}(occupied), reads)
    inputs = Symbol[]
    for j in eachindex(values)
        push!(inputs, _rk_ast_fresh_name("prior_input_$j", used))
    end
    replacements = Dict(zip(values, inputs))
    function replace_reads(value)
        value isa Symbol && return get(replacements, value, value)
        value isa Expr ? _rk_ast_map_operands(replace_reads, value) : value
    end
    map(replace_reads, priors), inputs, values
end

# The sentence a component's docstring adds for those prior inputs.
_rk_prior_inputs_doc(inputs) = isempty(inputs) ? "" : string(" ",
    join(("`$input`" for input in inputs), ", "),
    length(inputs) == 1 ? " is a value" : " are values",
    " the authored priors read, passed in by the caller.")

# Scalar population coefficient families admitted on RKPPL's design-matrix
# route, with their positional argument roles.
const _RK_POPULATION_FAMILIES = (
    Normal=(:loc, :scale), Cauchy=(:loc, :scale), Laplace=(:loc, :scale),
    Logistic=(:loc, :scale), StudentT=(:nu, :loc, :scale), Flat=(),
    Uniform=(:lower, :upper))

_rk_ast_population_family(prior) = prior !== nothing &&
    haskey(_RK_POPULATION_FAMILIES, first(prior)) &&
    length(last(prior)) == length(getproperty(_RK_POPULATION_FAMILIES, first(prior)))

# One argument shared by every coefficient stays scalar; otherwise each
# column states its own value.
function _rk_ast_population_argument(values)
    all(isequal(first(values)), values) ? first(values) : Expr(:vect, values...)
end

"""Population effects over design columns: coefficients allocated inside, `X * beta_pop` returned.

Predictors with identical design columns share one design matrix; `design` is
the base name of a new one. `designs` records each matrix with the component
calls that read it, their priors and their coefficients' coordinate records
(`coefficients`, one `(; predictor, coefficient, paired)` per column), for
`_rk_ast_join_shared_designs!`."""
function _rk_ast_population_component!(definitions, statements, taken, name, design,
        columns, priors, designs, coefficients)
    matrix = Expr(:call, :hcat, columns...)
    shared = findfirst(record -> isequal(last(record.assignment.args), matrix), designs)
    if shared === nothing
        design = _rk_ast_fresh_name(design, taken)
        assignment = Expr(:(=), design, matrix)
        push!(statements, assignment)
        designs[design] = (; columns, assignment, entries=Any[])
    else
        design = shared
    end
    entry(call, mixed) = push!(designs[design].entries,
        (; name, call, priors, coefficients, mixed))
    product = Expr(:call, :*, :X, :beta_pop)
    families = unique(first.(priors))
    if length(families) > 1
        # Coefficients with different prior families keep one statement each,
        # as SBBRMI's configured generic population block does.
        declarations = [Symbol(:beta_pop_, j) for j in eachindex(priors)]
        body = Expr(:block)
        for (declaration, (family, args)) in zip(declarations, priors)
            push!(body.args, Expr(:call, :~, declaration, Expr(:call, family, args...)))
        end
        push!(body.args, Expr(:(=), :beta_pop, Expr(:vect, declarations...)))
        push!(body.args, Expr(:return, product))
        doc = @_rk_definition_doc """
            Population-level effects whose coefficients have different prior laws.

            Draws one coefficient per column of the design matrix `X`, each with its
            own prior (`beta_pop_1`, `beta_pop_2`, …), and returns the linear
            predictor `X * beta_pop`."""
        callee = _rk_ast_shared_definition!(definitions, taken,
            "brm_mixed_population_effects", (:X,), body; doc)
        call = _rk_ast_component_call(name, callee, design)
        entry(call, true)
        push!(statements, call)
        return name
    end
    family = only(families)
    roles = getproperty(_RK_POPULATION_FAMILIES, family)
    body = _rk_ast_component_body(
        Expr(:call, :.~, Expr(:ref, :beta_pop, Expr(:call, :(:), 1, :ncoef)),
            _rk_ast_dotted(family, roles...)),
        Expr(:return, product))
    base = family === :Normal ? "brm_population_effects" :
        string("brm_population_effects_", lowercase(string(family)))
    law = string(family, "(", join(roles, ", "), ")")
    per_coefficient = isempty(roles) ? "" : string(" Each prior argument (",
        join(("`$role`" for role in roles), ", "),
        ") is one value for all coefficients or a vector with one per coefficient.")
    doc = @_rk_definition_doc """
        Population-level effects: one coefficient per column of the design matrix `X`.

        Draws `beta_pop[1:ncoef] ~ $law` and returns the linear predictor
        `X * beta_pop`. `ncoef` is the number of columns of `X`.$per_coefficient"""
    callee = _rk_ast_shared_definition!(definitions, taken, base, (:X, :ncoef, roles...), body;
        doc)
    arguments = [_rk_ast_population_argument([last(prior)[j] for prior in priors])
        for j in eachindex(roles)]
    call = _rk_ast_component_call(name, callee, design, length(columns), arguments...)
    entry(call, false)
    push!(statements, call)
    name
end

# A population component's coefficient records: `<name>.beta_pop` indexed by
# column, or one `<name>.beta_pop_<j>` declaration per column for mixed families.
# Each carries the law its column was given.
_rk_ast_population_prior(entry, j) =
    Expr(:call, first(entry.priors[j]), last(entry.priors[j])...)

function _rk_ast_population_records!(coordinates, entry)
    for (j, c) in enumerate(entry.coefficients)
        c.paired || continue
        _rk_coordinate_record!(coordinates, (; kind=:population,
            declaration=Symbol(entry.name, entry.mixed ? ".beta_pop_$j" : ".beta_pop"),
            index=entry.mixed ? () : (j,), predictor=c.predictor, coefficient=c.coefficient,
            prior=_rk_ast_population_prior(entry, j)))
    end
end

# A prior argument of a joint population block: one value shared by every
# coefficient of every predictor, or a coefficients x predictors literal matrix
# (`hcat` of one column per predictor). `nothing` when a value is not a literal.
function _rk_ast_joint_population_argument(values::AbstractMatrix)
    all(isequal(first(values)), values) && return first(values)
    all(v -> v isa Real, values) || return nothing
    Expr(:call, :hcat, (Expr(:vect, values[:, t]...) for t in axes(values, 2))...)
end

# A design column's part of a shared matrix name: the intercept's column of
# ones is `Intercept`, as its coefficient is.
_rk_ast_design_label(column::Symbol) = string(column)
function _rk_ast_design_label(column::Expr)
    Meta.isexpr(column, :call) && first(column.args) === :ones && return "Intercept"
    error("RK backend: internal: design column `$column` has no name")
end

# A design matrix read by one predictor keeps that predictor's name and its
# own population component. One read by several belongs to none of them: it is
# named for its columns, and the predictors' coefficients become one joint
# block over it (user decision `1pj2p5j`): `pop_<columns>` owns a coefficients
# x predictors matrix `beta_pop`, computes `X * beta_pop` once, and each
# predictor reads its column, `pop_<target> = pop_<columns>[:, t]`. Its
# coordinates are `pop_<columns>.beta_pop.<j>.<t>`. The recorded statements are
# this emission's own, not yet published.
function _rk_ast_join_shared_designs!(definitions, statements, designs, taken, coordinates)
    for design in sort!(collect(keys(designs)); by=string)
        record = designs[design]
        entries = record.entries
        if length(entries) == 1
            _rk_ast_population_records!(coordinates, only(entries))
            continue
        end
        labels = map(_rk_ast_design_label, record.columns)
        shared = _rk_ast_fresh_name(join(("X", labels...), "_"), taken)
        record.assignment.args[1] = shared
        foreach(entry -> entry.call.args[3].args[2] = shared, entries)
        if !_rk_ast_joint_population!(definitions, statements, taken, coordinates,
                shared, labels, entries)
            foreach(entry -> _rk_ast_population_records!(coordinates, entry), entries)
        end
    end
    designs
end

# The joint block of `_rk_ast_join_shared_designs!`. Its coefficients share one
# prior family, whose arguments may differ per coefficient and predictor. When
# they need families or argument values a two-axis declaration cannot state
# (mixed families, non-literal varying arguments, per-element `Uniform`
# bounds), the predictors keep their own components and return `false`.
function _rk_ast_joint_population!(definitions, statements, taken, coordinates,
        design, labels, entries)
    any(entry -> entry.mixed, entries) && return false
    families = unique(first(prior) for entry in entries for prior in entry.priors)
    length(families) == 1 || return false
    family = only(families)
    roles = getproperty(_RK_POPULATION_FAMILIES, family)
    ncoef, ntargets = length(first(entries).priors), length(entries)
    arguments = map(eachindex(roles)) do r
        _rk_ast_joint_population_argument(
            [last(entry.priors[j])[r] for j in 1:ncoef, entry in entries])
    end
    any(isnothing, arguments) && return false
    family === :Uniform && any(a -> a isa Expr, arguments) && return false
    body = _rk_ast_component_body(
        Expr(:call, :.~, Expr(:ref, :beta_pop, Expr(:call, :(:), 1, :ncoef),
                Expr(:call, :(:), 1, :ntargets)),
            _rk_ast_dotted(family, roles...)),
        Expr(:return, Expr(:call, :*, :X, :beta_pop)))
    base = family === :Normal ? "brm_joint_population_effects" :
        string("brm_joint_population_effects_", lowercase(string(family)))
    doc = @_rk_definition_doc """
        Population-level effects of several predictors that share one design matrix `X`.

        Draws one coefficient per column of `X` and predictor,
        `beta_pop[1:ncoef, 1:ntargets] ~ $(family)($(join(roles, ", ")))`, and returns the
        linear predictors `X * beta_pop`, one column per predictor."""
    callee = _rk_ast_shared_definition!(definitions, taken, base,
        (:X, :ncoef, :ntargets, roles...), body; doc)
    joint = _rk_ast_fresh_name(join(("pop", labels...), "_"), taken)
    first_call = findfirst(s -> s === first(entries).call, statements)
    insert!(statements, first_call,
        _rk_ast_component_call(joint, callee, design, ncoef, ntargets, arguments...))
    for (t, entry) in enumerate(entries)
        entry.call.head = :(=)
        entry.call.args = Any[entry.name, Expr(:ref, joint, :(:), t)]
        for (j, c) in enumerate(entry.coefficients)
            c.paired && _rk_coordinate_record!(coordinates, (; kind=:population,
                declaration=Symbol(joint, ".beta_pop"), index=(j, t),
                predictor=c.predictor, coefficient=c.coefficient,
                prior=_rk_ast_population_prior(entry, j)))
        end
    end
    true
end

# A monotonic effect owns its increment simplex and, when it has one, its
# coefficient. A coefficient-free term returns only its contrast.
function _rk_ast_monotonic_component!(definitions, statements, taken, name, index,
        alpha, prior)
    contrast = :(cumsum(vcat(0.0, simplex_incr))[c])
    simplex = Expr(:call, :~, :simplex_incr, Expr(:call, :Dirichlet, :alpha))
    if prior === nothing
        body = _rk_ast_component_body(simplex, Expr(:return, contrast))
        doc = @_rk_definition_doc """
            A monotonic effect's contrast of the ordinal codes `c` (`mo(x)` without a coefficient).

            Draws the increments `simplex_incr ~ Dirichlet(alpha)` and returns
            `cumsum(vcat(0.0, simplex_incr))[c]`, which rises from 0 at the lowest level to 1
            at the highest."""
        callee = _rk_ast_shared_definition!(definitions, taken,
            "brm_monotonic_value", (:c, :alpha), body; doc)
        push!(statements, _rk_ast_component_call(name, callee, index, _rk_ast_alpha(alpha)))
        return name
    end
    family, args = prior
    roles = haskey(_RK_POPULATION_FAMILIES, family) &&
        length(getproperty(_RK_POPULATION_FAMILIES, family)) == length(args) ?
        getproperty(_RK_POPULATION_FAMILIES, family) :
        Tuple(Symbol(:arg_, j) for j in eachindex(args))
    body = _rk_ast_component_body(simplex,
        Expr(:call, :~, :beta, Expr(:call, family, roles...)),
        Expr(:return, Expr(:call, :.*, :beta, contrast)))
    base = family === :Normal ? "brm_monotonic_effect" :
        string("brm_monotonic_effect_", lowercase(string(family)))
    doc = @_rk_definition_doc """
        A monotonic effect of the ordinal codes `c` (`mo(x)`).

        Draws the increments `simplex_incr ~ Dirichlet(alpha)` and the coefficient
        `beta ~ $(family)($(join(roles, ", ")))`, and returns
        `beta .* cumsum(vcat(0.0, simplex_incr))[c]`: the effect rises monotonically from 0
        at the lowest level to `beta` at the highest."""
    callee = _rk_ast_shared_definition!(definitions, taken, base, (:c, :alpha, roles...), body;
        doc)
    push!(statements, _rk_ast_component_call(name, callee, index,
        _rk_ast_alpha(alpha), args...))
    name
end
# A concentration as the program reads it: a generated input's name, or a literal.
_rk_ast_alpha(alpha::AbstractVector) = Expr(:vect, alpha...)
_rk_ast_alpha(alpha) = alpha

# Several predictors' monotonic effects of one column, as one component: one
# increment simplex per predictor (the columns of `simplex_incr`), one
# cumulative table, and the coefficients `beta` when the effects have them.
# The predictors gather their columns directly by the codes. Returns the call
# statement and, per predictor,
# its coefficient's `(declaration, index)` or `nothing`. One shared prior
# family broadcasts over `beta[1:ntargets]`; mixed families give one
# statement per predictor. A shared Dirichlet concentration is one vector;
# per-predictor concentrations are the columns of an increments x predictors
# matrix (`Dirichlet.(eachcol(alpha))`).
function _rk_ast_joint_monotonic_component!(definitions, taken, name, alphas, priors;
        spelled=nothing, nincr=length(first(alphas)))
    ntargets = length(priors)
    shared = all(isequal(first(alphas)), alphas)
    law = shared ? Expr(:call, :Dirichlet, :alpha) :
        Expr(:., :Dirichlet, Expr(:tuple, Expr(:call, :eachcol, :alpha)))
    simplex = Expr(:call, :.~, Expr(:call, :eachcol, Expr(:ref, :simplex_incr,
            Expr(:call, :(:), 1, :nincr), Expr(:call, :(:), 1, :ntargets))), law)
    table = :(cumsum(vcat(zeros(1, ntargets), simplex_incr); dims=1))
    # `spelled` names each predictor's concentration input; the columns of their
    # `hcat` are the per-predictor concentrations.
    alpha = spelled !== nothing ? (shared ? first(spelled) : Expr(:call, :hcat, spelled...)) :
        shared ? Expr(:vect, first(alphas)...) :
        Expr(:vcat, (Expr(:row, (alphas[t][i] for t in 1:ntargets)...)
            for i in eachindex(first(alphas)))...)
    suffix = shared ? "" : "_by_column"
    inputs = (alpha, nincr, ntargets)
    increments = string("Draws each predictor's increments, the columns of `simplex_incr[1:nincr, 1:ntargets]`, ",
        shared ? "from `Dirichlet(alpha)`" : "from the Dirichlet of its own concentration, a column of `alpha`")
    if all(isnothing, priors)
        body = _rk_ast_component_body(simplex, Expr(:return, table))
        doc = @_rk_definition_doc """
            The monotonic contrasts of one ordinal column in `ntargets` predictors, as one table.

            $increments, and returns the levels-by-predictors table
            `cumsum(vcat(zeros(1, ntargets), simplex_incr); dims = 1)`, which each predictor
            reads at the codes."""
        callee = _rk_ast_shared_definition!(definitions, taken,
            "brm_joint_monotonic_values" * suffix, (:alpha, :nincr, :ntargets), body; doc)
        return _rk_ast_component_call(name, callee, inputs...), fill(nothing, ntargets)
    end
    any(isnothing, priors) && error("RK backend: internal: joint monotonic block " *
        "`$name` mixes effects with and without a coefficient")
    families = unique(first.(priors))
    # Return the tiny levels × predictors table. Each predictor gathers only
    # its own column, avoiding an observations × predictors materialization.
    scaled_table = :((cumsum(vcat(zeros(1, ntargets), simplex_incr); dims=1)) .* beta')
    effect = Expr(:return, scaled_table)
    if length(families) == 1
        family = only(families)
        roles = haskey(_RK_POPULATION_FAMILIES, family) &&
            length(getproperty(_RK_POPULATION_FAMILIES, family)) == length(last(first(priors))) ?
            getproperty(_RK_POPULATION_FAMILIES, family) :
            Tuple(Symbol(:arg_, j) for j in eachindex(last(first(priors))))
        body = _rk_ast_component_body(simplex,
            Expr(:call, :.~, Expr(:ref, :beta, Expr(:call, :(:), 1, :ntargets)),
                _rk_ast_dotted(family, roles...)),
            effect)
        base = (family === :Normal ? "brm_joint_monotonic_effects" :
            string("brm_joint_monotonic_effects_", lowercase(string(family)))) * suffix
        doc = @_rk_definition_doc """
            The monotonic effects of one ordinal column in `ntargets` predictors, as one table.

            $increments, and one coefficient per predictor,
            `beta[1:ntargets] ~ $(family)($(join(roles, ", ")))`. Returns the levels-by-predictors
            table of cumulative increments scaled by `beta`, which each predictor reads at the codes."""
        callee = _rk_ast_shared_definition!(definitions, taken, base,
            (:alpha, :nincr, :ntargets, roles...), body; doc)
        arguments = [_rk_ast_population_argument([last(prior)[j] for prior in priors])
            for j in eachindex(roles)]
        betas = [(Symbol(name, ".beta"), (t,)) for t in 1:ntargets]
        return _rk_ast_component_call(name, callee, inputs..., arguments...), betas
    end
    # Coefficients with different prior families keep one statement each.
    coefficients = [Symbol(:beta_, t) for t in 1:ntargets]
    body = _rk_ast_component_body(simplex,
        (Expr(:call, :~, coefficient, Expr(:call, family, args...))
            for (coefficient, (family, args)) in zip(coefficients, priors))...,
        Expr(:(=), :beta, Expr(:vect, coefficients...)),
        effect)
    doc = @_rk_definition_doc """
        The monotonic effects of one ordinal column in `ntargets` predictors whose coefficients
        have different prior laws, as one table.

        $increments, and each predictor's coefficient from its own prior
        (`beta_1`, `beta_2`, …). Returns the levels-by-predictors table of cumulative
        increments scaled by `beta`, which each predictor reads at the codes."""
    callee = _rk_ast_shared_definition!(definitions, taken,
        "brm_joint_monotonic_mixed_effects" * suffix, (:alpha, :nincr, :ntargets), body; doc)
    betas = [(Symbol(name, ".", coefficient), ()) for coefficient in coefficients]
    _rk_ast_component_call(name, callee, inputs...), betas
end

# One shared positive prior broadcasts over the margins; distinct priors are
# stated per margin and collected into the same `tau` vector. A known scale
# (`sd(...) = value`) arrives as the name of its input and enters `tau` as that
# value; only a law is sampled.
function _rk_ast_group_scales(priors, index)
    held(prior) = prior isa Union{Real,Symbol}
    if all(isequal(first(priors)), priors) && !held(first(priors))
        prior = first(priors)
        return Expr[Expr(:call, :.~, Expr(:ref, :tau, index),
            _rk_ast_dotted(prior.args[1], prior.args[2:end]...))]
    end
    scales = Any[held(prior) ? prior : Symbol(:tau_, j) for (j, prior) in pairs(priors)]
    statements = Expr[Expr(:call, :~, scale, prior)
                      for (scale, prior) in zip(scales, priors) if !held(prior)]
    push!(statements, Expr(:(=), :tau, Expr(:vect, scales...)))
    statements
end

# Group-level effects allocate their scales, correlation factor and
# standardized draws, and return the non-centered levels-by-margins matrix.
# `priors` holds one positive prior per margin; with `scale_value` the scales
# are instead a supplied value (shared R2D2M2 budgets). A Student-t block
# (`nu`, a constant or graph value) also draws one mixing weight per level,
# `w ~ InverseGamma(nu/2, nu/2)`, and scales that level's row by `sqrt(w)`:
# the same scale mixture as StanBlocks' `_sb_student_t_ranef`.
# Rows are level positions, one per level of `g`: callers gather them with
# positions computed from the data (`b[g_level, k]`), which a `levels(g)` axis
# would instead read as labels.
function _rk_ast_group_component!(definitions, statements, taken, name, group, K, eta,
        priors; scale_value=nothing, nu=nothing, L_value=nothing)
    priors, prior_inputs, prior_values = _rk_ast_component_prior_inputs(priors,
        (:g, :K, :eta, :tau, :L, :z, :w, :nu, (Symbol(:tau_, j) for j in 1:K)...))
    index = K > 1 ? Expr(:call, :(:), 1, :K) : Expr(:call, :(:), 1, 1)
    rows = Expr(:call, :(:), 1, Expr(:call, :length, Expr(:call, :levels, :g)))
    body = Expr(:block)
    scale_value === nothing && append!(body.args, _rk_ast_group_scales(priors, index))
    # `L_value`: a known correlation's constant Cholesky factor (`cor(:, ID) = R`).
    K > 1 && L_value === nothing &&
        push!(body.args, Expr(:call, :~, :L, Expr(:call, :LKJCholesky, :K, :eta)))
    push!(body.args, Expr(:call, :.~, Expr(:ref, :z, rows, index),
        _rk_ast_dotted(:Normal, 0, 1)))
    value = K > 1 ?
        Expr(:call, :*, :z, Expr(:call, :transpose, Expr(:call, :.*, :tau, :L))) :
        Expr(:call, :.*, :z, Expr(:ref, :tau, 1))
    if nu !== nothing
        half = Expr(:call, :/, :nu, 2)
        push!(body.args, Expr(:call, :.~, Expr(:ref, :w, rows),
            _rk_ast_dotted(:InverseGamma, half, half)))
        value = Expr(:call, :.*, _rk_ast_dotted(:sqrt, :w), value)
    end
    push!(body.args, Expr(:return, value))
    arguments = Any[:g]
    K > 1 && push!(arguments, :K, :eta)
    scale_value === nothing || push!(arguments, :tau)
    L_value === nothing || push!(arguments, :L)
    nu === nothing || push!(arguments, :nu)
    append!(arguments, prior_inputs)
    base = string(scale_value === nothing ? "brm_" : "brm_scaled_",
        nu === nothing ? "" : "student_t_",
        L_value === nothing ? "" : "known_cor_",
        K > 1 ? "correlated_group_effects" : "group_effects")
    scales = scale_value === nothing ?
        (K > 1 ? "the scales `tau[1:K]` (drawn from their priors)" :
            "the scale `tau[1]` (drawn from its prior)") :
        "the scales `tau` (an input)"
    correlation = K == 1 ? "" : L_value === nothing ?
        ", the Cholesky factor of the margins' correlation `L ~ LKJCholesky(K, eta)`" :
        ", the Cholesky factor `L` of a known correlation (an input)"
    levels = K > 1 ? "`z * transpose(tau .* L)`" : "`z .* tau[1]`"
    mixing = nu === nothing ? "" : string(" Each level also draws a Student-t mixing ",
        "weight `w ~ InverseGamma(nu / 2, nu / 2)` and its row is scaled by `sqrt(w)`.")
    doc = @_rk_definition_doc """
        Group-level (varying) effects of grouping `g`$(K > 1 ? ", `K` correlated margins per level" : "").

        Non-centered, from $scales$correlation and standardized effects
        `z[1:length(levels(g)), 1:$(K > 1 ? "K" : "1")] ~ Normal(0, 1)`. Returns the
        level-by-margin matrix $levels, one row per level of `g` in order.$mixing$(_rk_prior_inputs_doc(prior_inputs))"""
    callee = _rk_ast_shared_definition!(definitions, taken, base, arguments, body; doc)
    values = Any[group]
    K > 1 && push!(values, K, eta)
    scale_value === nothing || push!(values, scale_value)
    L_value === nothing || push!(values, L_value)
    nu === nothing || push!(values, nu)
    append!(values, prior_values)
    push!(statements, _rk_ast_component_call(name, callee, values...))
    name
end

# The `levels(s) x K` scales of a stratified block: every stratum's scale of
# margin `j` takes `priors[j]`. One shared law broadcasts over the whole matrix
# `tau`; distinct laws are stated per margin, one `tau_<j>[levels(s)]` each.
function _rk_ast_stratum_scales(priors, index)
    strata = Expr(:call, :levels, :s)
    if all(isequal(first(priors)), priors)
        prior = first(priors)
        return Expr[Expr(:call, :.~, Expr(:ref, :tau, strata, index),
            _rk_ast_dotted(prior.args[1], prior.args[2:end]...))]
    end
    Expr[Expr(:call, :.~, Expr(:ref, Symbol(:tau_, j), strata),
            _rk_ast_dotted(prior.args[1], prior.args[2:end]...))
        for (j, prior) in enumerate(priors)]
end

# Row `i`'s scales, read at its stratum's label `s[i]`: a row of the shared-law
# matrix, or each margin's own vector. Only a declaration over `levels(s)`
# reads by label; a matrix collected from the per-margin vectors (`hcat`) has
# positional rows `1:S`, which a String label cannot index.
function _rk_ast_stratum_row_scales(priors)
    row = Expr(:ref, :s, :i)
    all(isequal(first(priors)), priors) && return Expr(:ref, :tau, row, :(:))
    Expr(:vect, (Expr(:ref, Symbol(:tau_, j), row) for j in eachindex(priors))...)
end

# Stratified group-level effects: one scale vector (and factor) per stratum.
# `priors` holds one positive prior per margin, shared by every stratum.
function _rk_ast_stratified_group_component!(definitions, statements, taken, name,
        group, stratum, K, eta, priors)
    priors, prior_inputs, prior_values = _rk_ast_component_prior_inputs(priors,
        (:g, :s, :eta, :tau, :L, :k, :z, :b, :i, (Symbol(:tau_, j) for j in 1:K)...))
    index = Expr(:call, :(:), 1, K)
    body = Expr(:block, _rk_ast_stratum_scales(priors, index)...)
    if K > 1
        cell = Expr(:call, :~, Expr(:ref, :L, :k), Expr(:call, :LKJCholesky, K, :eta))
        push!(body.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
            Expr(:for, Expr(:(=), :k, Expr(:call, :levels, :s)), Expr(:block, cell))))
    end
    push!(body.args, Expr(:call, :.~, Expr(:ref, :z, Expr(:call, :levels, :g), index),
        _rk_ast_dotted(:Normal, 0, 1)))
    scale = _rk_ast_stratum_row_scales(priors)
    raw = Expr(:ref, :z, Expr(:ref, :g, :i), :(:))
    value = K == 1 ? Expr(:call, :.*, scale, raw) :
        Expr(:call, :*, Expr(:call, :.*, scale, Expr(:ref, :L, Expr(:ref, :s, :i))), raw)
    cell = Expr(:(=), Expr(:ref, :b, :i, index), value)
    push!(body.args, Expr(:macrocall, Symbol("@plate"), LineNumberNode(0),
        Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, :g)), Expr(:block, cell))))
    push!(body.args, Expr(:return, :b))
    correlation = K > 1 ? (:eta,) : ()
    doc = @_rk_definition_doc """
        Group-level effects of grouping `g` whose scales$(K > 1 ? " and correlation" : "") differ by stratum `s`.

        Draws one scale per stratum and margin$(K > 1 ? ", a Cholesky factor `L[k] ~ LKJCholesky($K, eta)` per stratum" : "")
        and standardized effects `z[levels(g), 1:$K] ~ Normal(0, 1)`. Returns one row per
        observation `i`, its level's effects scaled by its stratum's scales$(K > 1 ? " and factor" : "").$(_rk_prior_inputs_doc(prior_inputs))"""
    callee = _rk_ast_shared_definition!(definitions, taken,
        "brm_stratified_group_effects", (:g, :s, correlation..., prior_inputs...), body; doc)
    push!(statements, _rk_ast_component_call(name, callee, group, stratum,
        (K > 1 ? (eta,) : ())..., prior_values...))
    name
end

# An HSGP effect allocates its length scale(s), marginal scale and basis
# weights and returns the spectral-weighted basis product. `inputs` pairs its
# data arguments with their main-block values: the axes of a
# squared-exponential term, or a periodic term's prepared basis and
# frequencies. The validity floor is passed in where a prior reads it.
function _rk_ast_hsgp_component!(definitions, statements, taken, name, inputs,
        floors, rho_priors, sigma_prior, value, extent; scales, truncated=true,
        base="brm_hsgp_effect")
    formals = first.(inputs)
    priors, prior_inputs, prior_values = _rk_ast_component_prior_inputs(
        [rho_priors; sigma_prior],
        (formals..., :rho_floor, :rho_iso, :rho, :sigma, :beta_raw, :spectral_scale,
            (Symbol(:rho_, j) for j in eachindex(rho_priors))...))
    rho_priors, sigma_prior = priors[1:end-1], last(priors)
    body = Expr(:block)
    floor_argument = truncated ? (:rho_floor,) : ()
    if length(rho_priors) == 1
        prior = only(rho_priors)
        truncated && (prior = Expr(:call, :restricted, prior, :rho_floor, Inf))
        push!(body.args, _rk_ast_hyper_statement(:rho_iso, prior))
    else
        rhos = Symbol[]
        for (j, prior) in enumerate(rho_priors)
            rho = Symbol(:rho_, j)
            push!(rhos, rho)
            truncated && (prior = Expr(:call, :restricted, prior,
                Expr(:ref, :rho_floor, j), Inf))
            push!(body.args, _rk_ast_hyper_statement(rho, prior))
        end
        push!(body.args, Expr(:(=), :rho, Expr(:vect, rhos...)))
    end
    push!(body.args, _rk_ast_hyper_statement(:sigma, sigma_prior))
    push!(body.args, Expr(:call, :.~, Expr(:ref, :beta_raw, extent),
        _rk_ast_dotted(:Normal, 0, 1)))
    # The weights' spectral scales, a value of the component that learned SVI
    # centering reads (src/nlme_svi_centering.jl).
    push!(body.args, Expr(:(=), :spectral_scale, scales))
    push!(body.args, Expr(:return, value))
    periodic = occursin("periodic", base)
    lengths = length(rho_priors) == 1 ? "the length scale `rho_iso`" :
        "one length scale per axis, `rho = [$(join(("rho_$j" for j in eachindex(rho_priors)), ", "))]`"
    floor = truncated ? ", each truncated below at the floor `rho_floor` the basis resolves" : ""
    doc = @_rk_definition_doc """
        A $(periodic ? "periodic " : "")Hilbert-space Gaussian-process (HSGP) effect.

        Draws $lengths$floor, the marginal scale `sigma` and standardized basis
        coefficients `beta_raw ~ Normal(0, 1)`. Each basis function is weighted by the
        square root of the covariance's spectral density (`spectral_scale`), and the
        effect at the rows is the basis times the weighted coefficients.$(_rk_prior_inputs_doc(prior_inputs))"""
    callee = _rk_ast_shared_definition!(definitions, taken, base,
        (formals..., floor_argument..., prior_inputs...), body; doc)
    push!(statements, _rk_ast_component_call(name, callee, last.(inputs)...,
        (truncated ? (floors,) : ())..., prior_values...))
    name
end
