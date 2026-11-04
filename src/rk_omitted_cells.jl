# ---- What an omitted response leaves of the plate cells ---------------------
#
# Omitting a response means not conditioning on it: what no remaining likelihood
# reaches leaves the sampler. SBBRMI moves it to generated quantities and samples
# only the active margins of a shared `|ID|` block (`sb_marginal_ranef.jl`). RK
# has no generated quantities, so such a value is simply not in the program, and
# a shared block is planned over the predictors a likelihood reaches.
#
# Dropping the omitted observation from its cell is not enough for that. The cell
# usually computes the observation's location in statements of its own, and
# reports them (`loc = base[i] + ...; z[i] = loc; y[i] ~ Normal(loc, s)`). Those
# statements stay in the cell, and they read the predictors, margins and scalars
# that only `y` needed, which would stay as prior-only coordinates that the
# Stan program of the same BRMI does not sample.
#
# So once a response is omitted, a cell keeps
#
# - what a kept likelihood reaches: the statements its observations read, and
#   those of any cell output a kept likelihood reads, through the whole program;
# - any other statement that adds no coordinate: one reading only those values
#   and data, so a reported `auc[i] = sum(conc)` over the kept concentrations
#   stays readable.
#
# A statement reading anything else goes with every value computed from it. A
# cell output (`z[i]`) that goes is no longer in the program, as on the top-level
# route. A fully observed program is unchanged, and a response absent from a
# replay's design (`_BRMAbsentResponse`) is not omitted: the fit observed it.

# Whether the program omits a response: an in-cell observation with no data, or a
# top-level law over model values that nothing reads.
function _rk_program_omits_response(program, kernels)
    any(kernel -> any(input -> input.kind === :omitted, kernel.inputs), kernels) && return true
    read = Set{Symbol}()
    foreach(operation -> union!(read, operation.dependencies), program.operations)
    any(operation -> operation.role === :parameter && !isempty(operation.dependencies) &&
        !(operation.name in read), program.operations)
end

# The cells of a program that omits a response, restricted as above, and the
# program with each cell depending only on what it still reads. Native submodels
# stay whole: their source can observe data this pass cannot see.
function _rk_cells_under_omission(program, kernels, observations, submodels)
    _rk_program_omits_response(program, kernels) || return program, kernels
    reach = _rk_likelihood_reach(program, kernels, observations, submodels)
    operations = Dict(operation.name => operation for operation in program.operations)
    kept = Dict{Symbol,_RKPreparedKernelAssignment}()
    retained = Set{Symbol}()
    function free(name)
        (name in reach.names || name in retained) && return true
        operation = get(operations, name, nothing)
        operation === nothing && return !haskey(reach.owners, name)
        # A bound column is an operation too, and adds no coordinate.
        operation.role === :assignment ? all(free, operation.dependencies) :
            operation.role === :extension && haskey(program.context.data, name)
    end
    for name in program.order, (k, kernel) in enumerate(kernels)
        kernel.name === name || continue
        restricted = _rk_kernel_restricted(kernel, reach.needed[k], free)
        restricted.collected === nothing || push!(retained, restricted.name)
        union!(retained, restricted.outputs)
        kept[name] = restricted
    end
    remaining = Tuple(kept[kernel.name] for kernel in kernels if _rk_kernel_has_work(kept[kernel.name]))
    _rk_with_kernel_reads(program, remaining, kernels, reach.owners), remaining
end

# A cell nothing is left of: no value, no observation, no statement that must run.
_rk_kernel_has_work(kernel) = kernel.collected !== nothing || !isempty(kernel.outputs) ||
    !isempty(kernel.observations) || !isempty(kernel.scan_observed) ||
    any(statement -> isempty(_rk_cell_statement_writes(statement)), kernel.body)

# `names`: every model name a kept likelihood reaches. `needed[k]`: the names cell
# `k` must compute for them. `owners`: the cell writing each cell output.
function _rk_likelihood_reach(program, kernels, observations, submodels)
    operations = Dict(operation.name => operation for operation in program.operations)
    native = Dict(submodel.name => submodel for submodel in submodels)
    owners = Dict{Symbol,Int}()
    for (k, kernel) in enumerate(kernels)
        owners[kernel.name] = k
        for output in kernel.outputs
            # Two cells writing one name leave it a cell value (`_rk_kernel_output_names`).
            haskey(owners, output) ? (owners[output] = 0) : (owners[output] = k)
        end
    end
    names = Set{Symbol}()
    needed = [Set{Symbol}() for _ in kernels]
    function visit(name)
        name in names && return
        push!(names, name)
        k = get(owners, name, 0)
        if k != 0
            kernel = kernels[k]
            if name !== kernel.name
                push!(needed[k], name)
            elseif kernel.collected !== nothing
                _brm_cell_value_refs!(needed[k], kernel.collected)
            end
            walk(k)
        elseif haskey(native, name)
            foreach(visit, native[name].globals)
        else
            operation = get(operations, name, nothing)
            operation === nothing || foreach(visit, operation.dependencies)
        end
    end
    function walk(k)
        kernel = kernels[k]
        reads = needed[k]
        locals = Set{Symbol}(kernel.params)
        for statement in Iterators.reverse(kernel.body)
            writes = _rk_cell_statement_writes(statement)
            union!(locals, writes)
            !isempty(writes) && isdisjoint(writes, reads) && continue
            _brm_cell_value_refs!(reads, statement)
        end
        for (param, input) in zip(kernel.params, kernel.inputs)
            param in reads && input.kind !== :omitted && visit(input.source)
        end
        for name in collect(reads)
            name in locals && continue
            (name in kernel.globals || haskey(owners, name)) && visit(name)
        end
    end
    foreach(observation -> visit(observation.key), observations)
    foreach(submodel -> visit(submodel.name), submodels)
    for (k, kernel) in enumerate(kernels)
        _rk_kernel_observation_reads!(needed[k], kernel)
        foreach(entry -> push!(needed[k], entry.param), kernel.scan_observed)
        walk(k)
    end
    (; names, needed, owners)
end

# The cell without the statements and outputs that read what `free` refuses.
function _rk_kernel_restricted(kernel, needed, free)
    blocked = Set{Symbol}(param for (param, input) in zip(kernel.params, kernel.inputs)
        if input.kind === :omitted || !free(input.source))
    locals = Set{Symbol}(kernel.params)
    foreach(statement -> union!(locals, _rk_cell_statement_writes(statement)), kernel.body)
    available(reads) = isdisjoint(reads, blocked) && all(free, setdiff(reads, locals))
    body = Any[]
    for statement in kernel.body
        writes = _rk_cell_statement_writes(statement)
        if isempty(writes) || !isdisjoint(writes, needed) || !(
                _rk_is_cell_latent(statement) || _rk_is_cell_gmrf(statement)) &&
                available(_brm_cell_value_refs!(Set{Symbol}(), statement))
            push!(body, statement)
        else
            union!(blocked, writes)
        end
    end
    collected = kernel.collected === nothing ||
        available(_brm_cell_value_refs!(Set{Symbol}(), kernel.collected)) ?
        kernel.collected : nothing
    outputs = Tuple(output for output in kernel.outputs if !(output in blocked))
    restricted = _RKPreparedKernelAssignment(kernel.name, kernel.params, body, collected,
        kernel.scope, kernel.inputs, kernel.globals, kernel.columns, kernel.group_values,
        kernel.observations, kernel.cells, kernel.scan_observed, kernel.cell_column, outputs)
    reads = _rk_kernel_reads(restricted)
    _RKPreparedKernelAssignment(kernel.name, kernel.params, body, collected,
        kernel.scope, kernel.inputs, filter(in(reads), kernel.globals), kernel.columns,
        kernel.group_values, kernel.observations, kernel.cells, kernel.scan_observed,
        kernel.cell_column, outputs)
end

# `_brm_reachable_operations` walks a cell's operation through its dependencies,
# which list everything its authored body read: restrict those to what each cell
# still reads. A cell nothing is left of depends on nothing.
function _rk_with_kernel_reads(program, kernels, authored, owners)
    remaining = Dict(kernel.name => kernel for kernel in kernels)
    gone = Set{Symbol}(kernel.name for kernel in authored if !haskey(remaining, kernel.name))
    operations = map(program.operations) do operation
        operation.name in gone && return _BRMPreparedOperation(operation.name, operation.role,
            operation.expression, ())
        kernel = get(remaining, operation.name, nothing)
        kernel === nothing && return operation
        reads = _rk_kernel_reads(kernel)
        sources = Set{Symbol}(input.source for (param, input) in zip(kernel.params, kernel.inputs)
            if param in reads)
        union!(sources, kernel.globals)
        for name in reads
            k = get(owners, name, 0)
            k != 0 && push!(sources, authored[k].name)
        end
        _BRMPreparedOperation(operation.name, operation.role, operation.expression,
            Tuple(name for name in operation.dependencies if name in sources))
    end
    _BRMPreparedProgram(program.context, operations, _brm_operation_order(operations))
end

# ---- Shared blocks that stay whole ------------------------------------------
#
# The exact split of a shared `|ID|` block needs a plain Gaussian block with
# sampled scales (`_sb_marginal_eligible`). Any other block that a likelihood
# partly reaches stays sampled whole on SBBRMI: Student-t mixing, R2D2-derived or
# known scales, stratified `gr(g, by=b)` and predictors joining several grouping
# columns. RK then plans the same whole block, so both backends sample one law on
# one layout; its unread margins are read by no predictor. A block no likelihood
# reaches is not planned at all, on either backend.

# Whether SBBRMI samples block `key` whole when a likelihood reaches part of it.
function _rk_block_stays_whole(@nospecialize(brmi::BRMI), context, key, bucket, margins)
    _sb_bucket_joins_columns(bucket) && return true
    specs = ranef_effect_priors(brmi)
    r2d2_scales = any(s -> s.class === :sd && s.family === r2d2 && s.id === first(key), specs) ||
        any(p -> isnothing(p.predictor), r2d2_priors(brmi)) ||
        let scoped = Set(p.predictor for p in r2d2_priors(brmi))
            !isempty(scoped) && all(m -> m.predictor in scoped, margins)
        end
    direct = [s for s in specs if s.id === first(key) && !(s.class === :sd && s.family === r2d2)]
    overrides = isempty(direct) ? Dict() : _brm_resolve_ranef_effect_overrides(direct,
        Dict(key => margins); prefix="RK backend")
    !_sb_marginal_eligible(bucket.group_desc, get(overrides, key, nothing),
        r2d2_scales ? true : nothing)
end

# The group declarations of `predictors` and, for a block that stays whole, those
# of its other predictors.
Base.@nospecializeinfer function _rk_block_declarations(@nospecialize(brmi::BRMI), context, predictors)
    declarations = context.group_declarations
    all(d -> d.predictor in predictors, declarations) && return collect(declarations)
    planned = Set{Symbol}(d.id for d in declarations if d.predictor in predictors &&
        d.id !== nothing && !d.uncorrelated)
    whole = Set{Symbol}()
    if !isempty(planned)
        for (key, bucket) in _sb_collect_id_buckets(context)
            first(key) in planned || continue
            _rk_block_stays_whole(brmi, context, key, bucket, _sb_id_bucket_margins(bucket)) &&
                push!(whole, first(key))
        end
    end
    [d for d in declarations if d.predictor in predictors || (d.id !== nothing && d.id in whole)]
end
