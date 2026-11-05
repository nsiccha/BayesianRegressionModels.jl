# Stan implementation of the same bounded spatial recurrence and exact OU law.
# Geometry and compressed time maps are data; all sampled-length arithmetic
# stays in the generated model and receives Stan's native reverse derivative.
StanBlocks.@deffun @juliacompat begin
    brm_gmrf_multiresolution_basis(spatial_length::real, east::vector[P], north::vector[P],
            origin::int[P], item_ptr::int[PP], item_position::int[NI], item_lag::vector[NI],
            neighbor_ptr::int[PP], neighbor::int[NN], state_pattern::int[P],
            basis_ptr::int[PP], basis_node::int[B], M::int, minimum_distance::real)::vector[B] = begin
        basis = rep_vector(0.0, B)
        if spatial_length < minimum_distance / 1024.0
            for p in 1:P
                basis[basis_ptr[p + 1] - 1] = 1.0
            end
            return basis
        end
        for p in 1:P
            weights = brm_gmrf_vecchia_weights(state_pattern[p], sqrt(2.0) / spatial_length,
                1.0, east, north, origin, item_ptr, item_position, item_lag, M)
            first = basis_ptr[p]
            last = basis_ptr[p + 1] - 1
            basis[last] = sqrt(weights[M + 1])
            for k in neighbor_ptr[p]:(neighbor_ptr[p + 1] - 1)
                parent = neighbor[k]
                a = first
                b = basis_ptr[parent]
                while a <= last && b < basis_ptr[parent + 1]
                    if basis_node[a] == basis_node[b]
                        basis[a] += weights[k - neighbor_ptr[p] + 1] * basis[b]
                        a += 1
                        b += 1
                    elseif basis_node[a] < basis_node[b]
                        a += 1
                    else
                        b += 1
                    end
                end
            end
        end
        for p in 1:P
            variance = 0.0
            for j in basis_ptr[p]:(basis_ptr[p + 1] - 1)
                variance += square(basis[j])
            end
            for j in basis_ptr[p]:(basis_ptr[p + 1] - 1)
                basis[j] /= sqrt(variance)
            end
        end
        basis
    end

    brm_gmrf_multiresolution_logdensity(field::vector[D], temporal_length::real,
            L::matrix[C, C], times::vector[T], path_ptr::int[PP], path_time::int[N])::real = begin
        lp = -N * C * log(2.0 * pi) / 2.0
        for c in 1:C
            lp -= N * log(L[c, c])
        end
        white = rep_vector(0.0, C)
        for p in 1:(PP - 1)
            first = path_ptr[p]
            for s in first:(path_ptr[p + 1] - 1)
                gap = s == first ? 0.0 : times[path_time[s]] - times[path_time[s - 1]]
                independent = s == first || temporal_length < gap / 1024.0
                decay = independent ? 0.0 : exp(-gap / temporal_length)
                variance = independent ? 1.0 : -expm1(-2.0 * gap / temporal_length)
                q = 0.0
                for c in 1:C
                    r = field[c + C * (s - 1)] -
                        (s == first ? 0.0 : decay * field[c + C * (s - 2)])
                    for k in 1:(c - 1)
                        r -= L[c, k] * white[k]
                    end
                    white[c] = r / L[c, c]
                    q += square(white[c])
                end
                lp -= (C * log(variance) + q / variance) / 2.0
            end
        end
        lp
    end

    @lpxf brm_gmrf_multiresolution_lpdf(field::vector[D], temporal_length::real,
            L::matrix[C, C], times::vector[T], path_ptr::int[PP], path_time::int[N])::real =
        brm_gmrf_multiresolution_logdensity(field, temporal_length, L, times, path_ptr, path_time)

    @stanonly brm_gmrf_multiresolution_rng(vector[D], temporal_length::real,
            L::matrix[C, C], times::vector[T], path_ptr::int[PP], path_time::int[N])::vector[D] = begin
        field = rep_vector(0.0, D)
        white = rep_vector(0.0, C)
        for p in 1:(PP - 1)
            first = path_ptr[p]
            for s in first:(path_ptr[p + 1] - 1)
                gap = s == first ? 0.0 : times[path_time[s]] - times[path_time[s - 1]]
                independent = s == first || temporal_length < gap / 1024.0
                decay = independent ? 0.0 : exp(-gap / temporal_length)
                scale = independent ? 1.0 : sqrt(-expm1(-2.0 * gap / temporal_length))
                for c in 1:C
                    white[c] = normal_rng(0.0, 1.0)
                    r = 0.0
                    for k in 1:c
                        r += L[c, k] * white[k]
                    end
                    field[c + C * (s - 1)] = scale * r +
                        (s == first ? 0.0 : decay * field[c + C * (s - 2)])
                end
            end
        end
        field
    end

    brm_gmrf_multiresolution_lpdfs(field::vector[D], temporal_length::real,
            L::matrix[C, C], times::vector[T], path_ptr::int[PP], path_time::int[N])::vector[D] = begin
        out = rep_vector(0.0, D)
        out[1] = brm_gmrf_multiresolution_lpdf(field, temporal_length, L, times, path_ptr, path_time)
        out
    end

    brm_gmrf_multiresolution_read(field::vector[D], weights::vector[B], C::int, R::int,
            basis_ptr::int[PP], basis_node::int[B], read_ptr::int[PP], path_ptr::int[PP],
            mapid::int[B], run_ptr::int[MP], run_start::int[NR], run_stop::int[NR],
            run_delta::int[NR])::vector[R] = begin
        out = rep_vector(0.0, R)
        for p in 1:(PP - 1)
            for j in basis_ptr[p]:(basis_ptr[p + 1] - 1)
                m = mapid[j]
                for r in run_ptr[m]:(run_ptr[m + 1] - 1)
                    for i in run_start[r]:run_stop[r]
                        for c in 1:C
                            out[c + C * (read_ptr[p] + i - 2)] += weights[j] *
                                field[c + C * (path_ptr[basis_node[j]] + i + run_delta[r] - 2)]
                        end
                    end
                end
            end
        end
        out
    end
end

_sb_gmrf_multiresolution = StanBlocks.@slic begin
    sigma ~ lognormal(0.0, 1.0; lower=0.0)
    spatial_length ~ lognormal(0.0, 1.0; lower=0.0)
    temporal_length ~ lognormal(0.0, 1.0; lower=0.0)
    field::vector[n_field] ~ brm_gmrf_multiresolution(temporal_length, L, times, path_ptr, path_time)
    weights = brm_gmrf_multiresolution_basis(spatial_length, east, north, origin, item_ptr,
        item_position, item_lag, neighbor_ptr, neighbor, state_pattern, basis_ptr, basis_node, M, minimum_distance)
    values = brm_gmrf_multiresolution_read(field, weights, C, n_read, basis_ptr, basis_node,
        read_ptr, path_ptr, mapid, run_ptr, run_start, run_stop, run_delta)
    return sigma * values[index]
end

function _sb_gmrf_multiresolution_data(prepared)
    s, g = prepared.state, prepared.state.support
    spatial = _brm_gmrf_footprint_data(g.spatial.conditionals)
    plan = _brm_multiresolution_read_plan(g)
    (; n_field=g.ncoordinates, C=g.nchannels, n_read=g.nchannels * length(g.reads.state_time),
       times=g.times, path_ptr=g.paths.state_ptr, path_time=Int.(g.paths.state_time),
       spatial.east, spatial.north, spatial.origin, spatial.item_ptr, spatial.item_position,
       spatial.item_lag, spatial.neighbor_ptr, spatial.neighbor, spatial.state_pattern, spatial.M,
       minimum_distance=isfinite(g.spatial.minimum_distance) ? g.spatial.minimum_distance : floatmax(Float64),
       basis_ptr=g.spatial.basis_ptr, basis_node=Int.(g.spatial.basis_node),
       read_ptr=g.reads.state_ptr, mapid=Int.(plan.mapid), run_ptr=plan.ptr,
       run_start=Int.(plan.starts), run_stop=Int.(plan.stops), run_delta=Int.(plan.delta), index=s.index)
end
