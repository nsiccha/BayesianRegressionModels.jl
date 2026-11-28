# Cell observations identify their missing data explicitly. For ambiguous
# top-level leaves, use the same generative declaration roles as prior replay.
Base.@nospecializeinfer function _rk_omitted_response_columns(@nospecialize(brmi::BRMI))
    missing = Dict{Symbol,Any}()
    outputs = keys(_brm_kernel_output_reads(brmi))
    for (_, raw) in _brm_operation_entries(brmi)
        cell = _brm_prune_cell(raw)
        isnothing(cell) && continue
        scanned = _rk_scan_observed_params(cell.body, cell.params)
        for (param, arg) in zip(cell.params, cell.positionals)
            _rk_kernel_omitted_response(arg, outputs) || continue
            param in scanned || _sb_cell_param_observed(cell.body, param) || continue
            gathered = arg isa ExprColumn
            column = gathered ? first(getargs(arg)) : arg
            key = name(column)
            missing[key] = gathered ? _BRMAbsentResponse{Float64}(key) :
                _BRMAbsentResponse{Vector{Float64}}(key)
        end
    end
    program = _brm_prepare_program(brmi)
    read = Set(dep for op in program.operations for dep in op.dependencies)
    if any(op -> op.role === :parameter && !(op.name in read) &&
            (!isempty(op.dependencies) || _sb_plan_observation_only(brmi, op.name)), program.operations)
        for key in _brm_stan_unbound_observations(brmi; _rk_cell_module(brmi)...)
            get!(missing, key, _BRMAbsentResponse{Float64}(key))
        end
    end
    missing
end

# Conditional generation works in the full model's declaration order. A
# generated prior can therefore read a fitted value or a completed predecessor.
# Coordinates that get a different name or matrix column when predictors are
# pooled are identified by their BRM coefficient addresses.
_rk_draw_path(name::Symbol) = Tuple(Symbol.(split(string(name), '.')))
_rk_at_path(value, path) = foldl(getproperty, path; init=value)

function _rk_population_slots(rk)
    Dict((r.predictor, r.coefficient) =>
        (; path=_rk_draw_path(r.declaration), index=get(r, :index, ()))
        for r in _rk_coordinate_records(rk) if r.kind === :population)
end

function _rk_widened_groups(fit, target)
    groups(rk) = filter(r -> r.kind === :ranef, _rk_coordinate_records(rk))
    fitted = groups(fit)
    widened = Dict{Symbol,Any}()
    for full in groups(target)
        matches = filter(fitted) do r
            r.group == full.group && r.id == full.id && r.bucket_kind == full.bucket_kind &&
                !isdisjoint(r.margins, full.margins)
        end
        isempty(matches) && continue
        active = only(matches)
        active.margins == full.margins && continue
        positions = [only(findall(==(m), full.margins)) for m in active.margins]
        root = _rk_record_root(full.z)
        bucket = only(b for b in _rk_replay_buckets(target.plan)
            if b.id == full.id && b.group == full.group && b.kind == full.bucket_kind &&
                Tuple((m.predictor, m.coefficient) for m in b.margins) == full.margins)
        remaining = setdiff(eachindex(full.margins), positions)
        widened[root] = (; source=_rk_record_root(active.z), positions, remaining,
            eta=_rk_bucket_eta(bucket), priors=full.scale_priors, split=full.scales !== nothing)
    end
    widened
end

# LKJ onion rows are independent in active-first order. At row j of the
# full K-dimensional factor the squared radius is Beta((j-1)/2, η+(K-j)/2)
# and the direction is uniform on the (j-2)-sphere. This is the conditional
# used by the Stan completion in sb_marginal_ranef.jl.
function _rk_generate_group!(generated, taken, root, group, levels)
    fresh(suffix) = _rk_ast_fresh_name(string("generated_", root, "_", suffix), taken)
    a, m = length(group.positions), length(group.remaining)
    K = a + m
    scales = map(group.remaining) do i
        name = fresh("scale_$i")
        push!(generated, Expr(:call, :~, name, group.priors[i]))
        name
    end
    radii = map(a+1:K) do j
        name = fresh("radius_$j")
        beta = :($(group.eta) + $((K-j)/2))
        push!(generated, :($name ~ Beta($((j-1)/2), $beta)))
        name
    end
    direction, innovation = fresh("direction"), fresh("innovation")
    n = sum(a:K-1)
    push!(generated, :($direction[1:$n] .~ Normal.(0, 1)))
    push!(generated, :($innovation[1:$levels, 1:$m] .~ Normal.(0, 1)))
    (; scales, radii, direction, innovation)
end

function _rk_complete_group(active, group, names, sampled)
    (; positions, remaining) = group
    a = length(positions)
    order = vcat(positions, remaining)
    K = length(order)
    L = zeros(K, K)
    L[1:a, 1:a] = a == 1 ? ones(1, 1) : active.L
    offset = 0
    for (r, name) in enumerate(names.radii)
        j = a + r
        u = view(sampled[names.direction], offset+1:offset+j-1)
        y = sampled[name]
        L[j, 1:j-1] = (sqrt(y) / norm(u)) .* u
        L[j, j] = sqrt(1-y)
        offset += j-1
    end
    z = hcat(active.z, sampled[names.innovation])
    if !issorted(order)
        # Permuting the rows permutes the effects. Restore the positive
        # Cholesky diagonal by an orthogonal change of basis, without forming
        # a covariance or solving through a nearly singular fitted factor.
        factor = lq(L[invperm(order), :])
        signs = transpose(sign.(diag(factor.L)))
        L = Matrix(factor.L) .* signs
        z = (z * factor.Q') .* signs
    end
    tau = zeros(K)
    tau[positions] = haskey(active, :tau) ? active.tau : [active[Symbol(:tau_, i)] for i in 1:a]
    tau[remaining] = [sampled[name] for name in names.scales]
    scales = group.split ? (; (Symbol(:tau_, i) => tau[i] for i in 1:K)...) : (; tau)
    merge(scales, (; L, z))
end

# Lift one sampling declaration under a fresh name, reading its parents from
# the target model. Project all its sampled locals, including those its return
# value does not expose on the prediction design.
function _rk_generated_site!(generated, taken, statement, root, paths)
    fresh = _rk_ast_fresh_name("generated_" * string(root), taken)
    lifted = _rk_rename_symbols(_brm_undoc(statement), Dict(root => fresh))
    push!(generated, lifted)
    map(paths) do path
        length(path) == 1 && return fresh
        field = foldl((base, key) -> Expr(:., base, QuoteNode(key)), Base.tail(path); init=fresh)
        alias = _rk_ast_fresh_name(join(("generated", path...), "_"), taken)
        push!(generated, Expr(:(=), alias, field))
        alias
    end
end

# A restricted law with constant mass has the same conditional draw as its
# normalized truncation. Leave other restrictions to RKPPL's generator check.
function _rk_fixed_law_expression(x)
    x isa Number || x in (:Inf, :(-Inf)) ||
        (Meta.isexpr(x, :call) && all(_rk_fixed_law_expression, x.args[2:end]))
end
function _rk_constant_restriction_mass(law)
    args = law.head === :. ? last(law.args).args : law.args[2:end]
    all(_rk_fixed_law_expression, args) && return true
    length(args) == 3 || return false
    base, lo, hi = args
    # A zero-centred Gaussian has half its mass on each side of zero even
    # when another sampled parameter supplies its scale.
    Meta.isexpr(base, :call, 3) && base.args[1] === :Normal && base.args[2] == 0 &&
        lo == 0 && hi in (Inf, :Inf)
end

# A prior may depend on another sampled declaration through arbitrary model
# assignments. Group independent declarations into one native @generate query.
# Only a conditional dependency needs a second query on the completed draw.
function _rk_generation_layers(emitted, steps)
    free = _rk_block_free_reads(emitted.defs)
    reads = Dict{Symbol,Set{Symbol}}()
    for statement in emitted.main.args
        outputs = _rk_source_outputs!(Set{Symbol}(), statement)
        refs = setdiff(_rk_source_references(statement, free), outputs)
        for name in outputs
            reads[name] = refs
        end
    end
    roots = Set(s.root for s in steps)
    function parents(name)
        pending = collect(reads[name])
        visited = Set{Symbol}()
        found = Set{Symbol}()
        while !isempty(pending)
            dep = pop!(pending)
            dep in visited && continue
            push!(visited, dep)
            if dep in roots
                push!(found, dep)
            else
                append!(pending, get(reads, dep, ()))
            end
        end
        found
    end
    dependencies = Dict(s.root => (s.generate ? parents(s.root) : Set{Symbol}()) for s in steps)
    remaining = copy(steps)
    layers = Vector{typeof(steps)}()
    while !isempty(remaining)
        layer = filter(s -> isempty(dependencies[s.root]), remaining)
        isempty(layer) && error("rk_generated_draws: cyclic dependencies between generated priors")
        push!(layers, layer)
        done = Set(s.root for s in layer)
        filter!(s -> !(s.root in done), remaining)
        foreach(deps -> setdiff!(deps, done), values(dependencies))
    end
    layers
end
