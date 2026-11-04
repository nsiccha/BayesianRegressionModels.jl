# Structural width screening. This runs before preparing any Laplace AD.
function _brm_spl_width_limit(value)
    value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) ||
        throw(ArgumentError("max_cell_width must be a finite positive integer"))
    Int(value)
end

# Greedy in observation/cell order. A removed term updates every cell through
# reverse incidence; correlated margins and all levels always move together.
# Ties use declaration order, independent of dictionary/hash iteration order.
function _brm_spl_select_width(constraints, declarations, forced, bound)
    active = trues(length(declarations))
    incident = [Tuple{Int,Int}[] for _ in declarations]
    widths = Int[]
    for (i, c) in enumerate(constraints)
        push!(widths, c.extra + sum(last, c.counts; init=0))
        for (t, contribution) in c.counts
            push!(incident[t], (i, contribution))
        end
    end
    events = Dict{Symbol,NamedTuple}()
    for (i, c) in enumerate(constraints)
        widths[i] <= bound && continue
        # Contributions do not change when another whole term is removed.
        # Sort once, instead of rescanning all terms after every removal.
        choices = sort!([(t, k) for (t, k) in c.counts
            if active[t] && declarations[t] ∉ forced]; by=x -> (-x[2], x[1]))
        for (choice, contribution) in choices
            widths[i] <= bound && break
            events[declarations[choice]] = (; c.observation, c.cell, c.part,
                width=widths[i], bound, contribution)
            active[choice] = false
            for (j, k) in incident[choice]
                widths[j] -= k
            end
        end
        widths[i] <= bound || _brm_nlme_error("brm_laplace_problem: observation " *
            "`$(c.observation)`, cell $(c.cell), $(c.part) has integrated width " *
            "$(widths[i]) above max_cell_width=$bound; forced terms " *
            join((string(declarations[t]) for (t, _) in c.counts if active[t]), ", ") *
            " cannot be sampled automatically; increase the bound or relax `integrate`")
    end
    (; active, events, max_width=maximum(widths; init=0))
end

function _brm_spl_screen(backend, candidates, forced, max_cell_width)
    bound = _brm_spl_width_limit(max_cell_width)
    declarations = [t.declaration for t in candidates]
    layout = _brm_spl_layout(backend, unique(t.grouping for t in candidates);
        declarations=Set(declarations))
    owner = zeros(Int, length(layout.names))
    for b in layout.blocks
        t = findfirst(d -> _brm_nlme_owned(b.declaration, d), declarations)
        owner[layout.integrated[vec(b.z)]] .= t
    end
    constraints = NamedTuple[]
    # Retain structural reads for the automatic planner. A cell's current
    # numerical Hessian cannot certify its dependence at other outer points.
    other = setdiff(Set(_brm_slap_other_observations(backend.plan)),
        keys(_brm_plan_certified_observations(backend)))
    uncertified = Dict{Symbol,Set{Symbol}}()
    function record!(observation, cell, part, reads; extra=0)
        counts = Dict{Int,Int}()
        for i in reads
            t = owner[i]
            t == 0 && continue
            counts[t] = get(counts, t, 0) + 1
            observation in other && push!(get!(Set{Symbol}, uncertified,
                declarations[t]), observation)
        end
        push!(constraints, (; observation, cell, part, extra, counts=sort!(collect(counts))))
    end
    u0 = zeros(length(layout.names))
    pointwise = _rk_pointwise_values(_rk_pointwise_query(backend), u0)
    for name in sort!(collect(keys(pointwise)))
        ncells = length(pointwise[name])
        ncells == 0 && continue
        q = _rk_spl_factor_queries(backend, name; layout.partitioned, u0)
        # The same choice of route as `_brm_spl_observation_cells!`.
        query = _rk_nlme_cell_query(backend, (name,); layout.partitioned, u0, locality=:ignore)
        prepared = _brm_spl_prepare_factors(backend, name, u0, ncells,
            layout.integrated, layout.population; query=q, whole=_rk_spl_cell_whole(query))
        if prepared === nothing
            footprint = _rk_spl_cell_footprint(query; shared=false)
            _rk_spl_cell_batch(query, u0) do batch
                for i in 1:ncells
                    record!(name, i, :cell,
                        _rk_spl_cell_reads(footprint, u0, i; context=batch))
                end
            end
        else
            for c in prepared.cells
                record!(name, c.i, :outer, c.direct.sel)
                for (j, reduction) in enumerate(c.terms), t in reduction
                    record!(name, c.i, (:summand, j, t.t), t.sel)
                end
            end
        end
    end
    selected = _brm_spl_select_width(constraints, declarations, forced, bound)
    integrated_dimension = count(i -> owner[i] != 0 && selected.active[owner[i]], eachindex(owner))
    (; selected..., uncertified, integrated_dimension,
        sampled_dimension=length(owner) - integrated_dimension)
end
