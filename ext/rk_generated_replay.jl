function _rk_ordinary_generator(readout)
    (; fit, target) = readout
    emitted = BRM._rk_retained_program(target)
    fitted = Dict(p.path => p for p in parameter_metadata(_rk_bound_layout(fit)))
    params = parameter_metadata(target.model.layout)
    source_slots = BRM._rk_population_slots(fit)
    target_slots = BRM._rk_population_slots(target)
    widened = BRM._rk_widened_groups(fit, target)
    unmapped = setdiff(keys(source_slots), keys(target_slots))
    isempty(unmapped) || throw(ArgumentError("rk_generated_draws: the replay has no " *
        "population coefficient for $(collect(unmapped))"))
    source_pop = Set(s.path for s in values(source_slots))
    target_pop = Set(s.path for s in values(target_slots))
    transfers = Dict{Tuple,Vector{NamedTuple}}()
    target_params = Dict(p.path => p for p in params)
    for path in copy(source_pop)
        at(slots) = Dict(address => s.index for (address, s) in slots if s.path == path)
        # Unpaired coefficients can share this declaration. If the block is
        # unchanged, keep every entry, including those without an address.
        if get(target_params, path, nothing) == fitted[path] && at(source_slots) == at(target_slots)
            delete!(source_pop, path)
            delete!(target_pop, path)
        elseif length(at(source_slots)) != prod(fitted[path].shape)
            throw(ArgumentError("rk_generated_draws: some fitted coefficients at $path " *
                "have no address in the regrouped population block"))
        end
    end
    for (address, source) in source_slots
        source.path in source_pop || continue
        dest = target_slots[address]
        push!(get!(Vector{NamedTuple}, transfers, dest.path), (; dest.index, source))
    end
    copied = Set{Tuple}(source_pop)
    seen = Set{Symbol}()
    steps = NamedTuple[]
    roots = unique(first(p.path) for p in params)
    for statement in emitted.main.args
        outputs = BRM._rk_source_outputs!(Set{Symbol}(), statement)
        for root in filter(in(outputs), roots)
            root in seen && continue
            push!(seen, root)
            entries = filter(p -> first(p.path) === root, params)
            group = get(widened, root, nothing)
            generate = if isnothing(group)
                needed = map(entries) do p
                    if p.path in target_pop
                        return length(get(transfers, p.path, ())) < prod(p.shape)
                    end
                    haskey(fitted, p.path) && p.path ∉ source_pop || return true
                    fitted[p.path] == p || throw(DimensionMismatch(
                        "rk_generated_draws: incompatible fitted and target shapes at $(p.path)"))
                    push!(copied, p.path)
                    false
                end
                # Each pooled population coefficient has an independent law.
                # A different partially retained submodel would need its own
                # conditional law before any of its locals could be redrawn.
                any(needed) && !all(needed) && !all(p -> p.path in target_pop, entries) &&
                    throw(ArgumentError("rk_generated_draws: no conditional law for the " *
                        "partially retained parameters of `$root`"))
                any(needed)
            else
                union!(copied, (path for path in keys(fitted) if first(path) === group.source))
                true
            end
            push!(steps, (; root, entries, group, statement, generate))
        end
    end
    isempty(setdiff(roots, seen)) || error("rk_generated_draws: missing declarations for $(setdiff(roots, seen))")
    lost = setdiff(keys(fitted), copied)
    isempty(lost) || throw(ArgumentError("rk_generated_draws: the replay has no " *
        "conditional transport for fitted parameters $(collect(lost))"))
    layers = map(BRM._rk_generation_layers(emitted, steps)) do layer
        taken = BRM._rk_source_symbols!(Set{Symbol}(), emitted.main)
        generated = Any[]
        sites = map(layer) do step
            aliases = if !isnothing(step.group)
                levels = first(only(p.shape for p in step.entries if last(p.path) === :z))
                BRM._rk_generate_group!(generated, taken, step.root, step.group, levels)
            elseif step.generate
                BRM._rk_generated_site!(generated, taken, step.statement, step.root,
                    [p.path for p in step.entries])
            else
                Symbol[]
            end
            (; step.root, step.entries, step.group, aliases)
        end
        query = if !isempty(generated)
            block = Expr(:macrocall, Symbol("@generate"), LineNumberNode(0), Expr(:block, generated...))
            source = BRM._rk_with_main(emitted, Expr(:block, emitted.main.args..., block))
            program = _RKProgram(BRM._rk_prior_rng_source(source, BRM._rk_constant_restriction_mass))
            bound = _rk_bind_program(program, target.plan; frozen=_rk_frozen(target))
            ReactiveKernelsPPL.prepare_generated(bound)
        end
        (; sites, query)
    end
    (; layers, transfers, target_pop)
end

function BRM._rk_generate_replay_points(readout, U, rng)
    fit_layout = _rk_bound_layout(readout.fit)
    readout.target === readout.fit && return U
    target_layout = readout.target.model.layout
    points = Matrix{Float64}(undef, target_layout.total, size(U, 2))
    generator = BRM._rk_once!(() -> _rk_ordinary_generator(readout), readout.generator)
    for (k, u) in enumerate(eachcol(U))
        fitted = constrain(fit_layout, u)
        physical = (;)
        for layer in generator.layers
            sampled = isnothing(layer.query) ? nothing :
                layer.query(physical, Random.Xoshiro(rand(rng, UInt64)))
            for step in layer.sites
                if !isnothing(step.group)
                    group = BRM._rk_complete_group(fitted[step.group.source],
                        step.group, step.aliases, sampled)
                    physical = _rk_with_generated(physical, (step.root,), group)
                    continue
                end
                physical = _rk_transfer_site(physical, fitted, step, generator, sampled)
            end
        end
        points[:, k] = unconstrain(target_layout, physical)
    end
    points
end

function _rk_transfer_site(physical, fitted, step, generator, sampled)
    for (i, p) in pairs(step.entries)
        value = if !isempty(step.aliases)
            sampled[step.aliases[i]]
        elseif p.path in generator.target_pop
            isempty(p.shape) ? nothing : Array{Float64}(undef, p.shape)
        else
            BRM._rk_at_path(fitted, p.path)
        end
        for slot in get(generator.transfers, p.path, ())
            source = BRM._rk_at_path(fitted, slot.source.path)
            coefficient = isempty(slot.source.index) ? source : source[slot.source.index...]
            if isempty(slot.index)
                value = coefficient
            else
                value[slot.index...] = coefficient
            end
        end
        physical = _rk_with_generated(physical, p.path, value)
    end
    physical
end
