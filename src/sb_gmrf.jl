# Stan and native Julia evaluate the same scalar sparse LDL and normalized
# field law. The emitted model retains original vertex/channel/time ordering;
# only the spatial factor uses the data-only fill-reducing permutation.
StanBlocks.@deffun @juliacompat begin
    brm_gmrf_factor(mass_scale::real, stiffness_scale::real, mass::vector[N], gptr::int[NP],
            grow::int[NZ], gval::vector[NZ], lptr::int[NP], lrow::int[NL],
            previous_ptr::int[NP], previous_col::int[ND], previous_pos::int[ND])::vector[N + NL] = begin
        ldl_values = rep_vector(0.0, NL)
        diagonal = rep_vector(0.0, N)
        work = rep_vector(0.0, N)
        for j in 1:N
            for p in lptr[j]:(lptr[j + 1] - 1)
                work[lrow[p]] = 0.0
            end
            for p in gptr[j]:(gptr[j + 1] - 1)
                if grow[p] >= j
                    work[grow[p]] = stiffness_scale * gval[p]
                end
            end
            work[j] += mass_scale * mass[j]
            for q in previous_ptr[j]:(previous_ptr[j + 1] - 1)
                k = previous_col[q]
                start = previous_pos[q]
                v = ldl_values[start] * diagonal[k]
                for p in start:(lptr[k + 1] - 1)
                    work[lrow[p]] -= ldl_values[p] * v
                end
            end
            diagonal[j] = work[j]
            ldl_values[lptr[j]] = 1.0
            for p in (lptr[j] + 1):(lptr[j + 1] - 1)
                ldl_values[p] = work[lrow[p]] / diagonal[j]
            end
        end
        append_row(diagonal, ldl_values)
    end

    brm_gmrf_solve(b::vector[N], factor::vector[NF], lptr::int[NP],
            lrow::int[NL])::vector[N] = begin
        x = rep_vector(0.0, N)
        for i in 1:N
            x[i] = b[i]
        end
        for j in 1:N
            for p in (lptr[j] + 1):(lptr[j + 1] - 1)
                x[lrow[p]] -= factor[N + p] * x[j]
            end
        end
        for j in 1:N
            x[j] /= factor[j]
        end
        for step in 1:N
            j = N + 1 - step
            for p in (lptr[j] + 1):(lptr[j + 1] - 1)
                x[j] -= factor[N + p] * x[lrow[p]]
            end
        end
        x
    end

    brm_gmrf_logdensity(field::vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], times::vector[T], permutation::int[N], mass::vector[N],
            gptr::int[NP], grow::int[NZ], gval::vector[NZ], lptr::int[NP], lrow::int[NL],
            previous_ptr::int[NP], previous_col::int[ND], previous_pos::int[ND])::real = begin
        # B = C/ell + (ell/2) G gives Q = B C^-1 B/(2pi), without
        # overflowing an inverse-squared range before it cancels.
        factor = brm_gmrf_factor(1.0 / spatial_length, spatial_length / 2.0,
            mass, gptr, grow, gval, lptr, lrow,
            previous_ptr, previous_col, previous_pos)
        logdet_space = 2.0 * sum(log(segment(factor, 1, N))) - sum(log(mass)) - N * log(2.0 * pi)
        lp = T * C * logdet_space / 2.0 - D * log(2.0 * pi) / 2.0
        for c in 1:C
            lp -= N * T * log(L[c, c])
        end
        residual::matrix[N, C]
        for t in 1:T
            a = t == 1 ? 0.0 : exp(-(times[t] - times[t - 1]) / temporal_length)
            variance = t == 1 ? 1.0 : -expm1(-2.0 * (times[t] - times[t - 1]) / temporal_length)
            for c in 1:C
                for i in 1:N
                    index = permutation[i] + N * ((c - 1) + C * (t - 1))
                    z = field[index]
                    if t > 1
                        z -= a * field[index - N * C]
                    end
                    for k in 1:(c - 1)
                        z -= residual[i, k] * L[c, k]
                    end
                    residual[i, c] = z / L[c, c]
                end
            end
            quadratic = 0.0
            for c in 1:C
                for i in 1:N
                    stiffness = 0.0
                    for p in gptr[i]:(gptr[i + 1] - 1)
                        stiffness += gval[p] * residual[grow[p], c]
                    end
                    z = mass[i] * (residual[i, c] / spatial_length) +
                        (spatial_length / 2.0) * stiffness
                    quadratic += square(z / (sqrt(4.0 * pi * mass[i]) * sqrt(variance)))
                end
            end
            lp -= quadratic + N * C * log(variance) / 2.0
        end
        lp
    end

    @lpxf brm_gmrf_lpdf(field::vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], times::vector[T], permutation::int[N], mass::vector[N],
            gptr::int[NP], grow::int[NZ], gval::vector[NZ], lptr::int[NP], lrow::int[NL],
            previous_ptr::int[NP], previous_col::int[ND], previous_pos::int[ND])::real =
        brm_gmrf_logdensity(field, spatial_length, temporal_length, L, times,
            permutation, mass, gptr, grow, gval, lptr, lrow, previous_ptr, previous_col, previous_pos)

    @stanonly brm_gmrf_rng(vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], times::vector[T], permutation::int[N], mass::vector[N],
            gptr::int[NP], grow::int[NZ], gval::vector[NZ], lptr::int[NP], lrow::int[NL],
            previous_ptr::int[NP], previous_col::int[ND], previous_pos::int[ND])::vector[D] = begin
        factor = brm_gmrf_factor(1.0 / spatial_length, spatial_length / 2.0,
            mass, gptr, grow, gval, lptr, lrow,
            previous_ptr, previous_col, previous_pos)
        field = rep_vector(0.0, D)
        residual::matrix[N, C]
        white::vector[N]
        for t in 1:T
            a = t == 1 ? 0.0 : exp(-(times[t] - times[t - 1]) / temporal_length)
            variance = t == 1 ? 1.0 : -expm1(-2.0 * (times[t] - times[t - 1]) / temporal_length)
            for c in 1:C
                for i in 1:N
                    white[i] = normal_rng(0.0, sqrt(2.0 * pi * mass[i]))
                end
                residual[:, c] = brm_gmrf_solve(white, factor, lptr, lrow)
            end
            for c in 1:C
                for i in 1:N
                    z = 0.0
                    for k in 1:c
                        z += L[c, k] * residual[i, k]
                    end
                    index = permutation[i] + N * ((c - 1) + C * (t - 1))
                    field[index] = sqrt(variance) * z
                    if t > 1
                        field[index] += a * field[index - N * C]
                    end
                end
            end
        end
        field
    end

    brm_gmrf_lpdfs(field::vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], times::vector[T], permutation::int[N], mass::vector[N],
            gptr::int[NP], grow::int[NZ], gval::vector[NZ], lptr::int[NP], lrow::int[NL],
            previous_ptr::int[NP], previous_col::int[ND], previous_pos::int[ND])::vector[D] = begin
        out = rep_vector(0.0, D)
        out[1] = brm_gmrf_lpdf(field, spatial_length, temporal_length, L, times,
            permutation, mass, gptr, grow, gval, lptr, lrow, previous_ptr, previous_col, previous_pos)
        out
    end

    brm_gmrf_rows(sigma::real, field::vector[D], index1::int[R], index2::int[R],
            index3::int[R], weight1::vector[R], weight2::vector[R], weight3::vector[R])::vector[R] =
        sigma * (weight1 .* field[index1] + weight2 .* field[index2] + weight3 .* field[index3])
end

# The Julia method a shared helper's Stan builtin call resolves to.
modified_bessel_second_kind(order::Integer, x::Real) = besselk(order, x)

# The footprint form of the same covariance: one conditional per distinct
# geometry, in the scalar correlation, with the channel factor whitening each
# state (`gmrf_vecchia.jl` is its native counterpart and holds the derivation).
StanBlocks.@deffun @juliacompat begin
    brm_gmrf_matern1(x::real)::real = x > 0.0 ? x * modified_bessel_second_kind(1, x) : 1.0

    # Conditional weights of pattern `p` in the first entries, its variance last.
    brm_gmrf_vecchia_weights(p::int, kappa::real, temporal_length::real, east::vector[P],
            north::vector[P], origin::int[NT], item_ptr::int[NIP], item_position::int[NI],
            item_lag::vector[NI], M::int)::vector[M + 1] = begin
        first_item = item_ptr[p]
        m = item_ptr[p + 1] - first_item
        out = rep_vector(0.0, M + 1)
        A = rep_matrix(0.0, M, M)
        center = origin[p]
        for j in 1:m
            pj = item_position[first_item + j - 1]
            lj = item_lag[first_item + j - 1]
            for i in j:m
                pk = item_position[first_item + i - 1]
                lk = item_lag[first_item + i - 1]
                A[i, j] = brm_gmrf_matern1(kappa * hypot(east[pk] - east[pj], north[pk] - north[pj])) *
                    exp(-abs(lk - lj) / temporal_length)
            end
            out[j] = brm_gmrf_matern1(kappa * hypot(east[center] - east[pj], north[center] - north[pj])) *
                exp(-abs(lj) / temporal_length)
        end
        for j in 1:m
            d = A[j, j]
            for k in 1:(j - 1)
                d -= square(A[j, k])
            end
            d = sqrt(d)
            A[j, j] = d
            for i in (j + 1):m
                value = A[i, j]
                for k in 1:(j - 1)
                    value -= A[i, k] * A[j, k]
                end
                A[i, j] = value / d
            end
        end
        variance = 1.0
        for i in 1:m
            value = out[i]
            for k in 1:(i - 1)
                value -= A[i, k] * out[k]
            end
            out[i] = value / A[i, i]
            variance -= square(out[i])
        end
        for step in 1:m
            i = m + 1 - step
            value = out[i]
            for k in (i + 1):m
                value -= A[k, i] * out[k]
            end
            out[i] = value / A[i, i]
        end
        out[M + 1] = variance
        out
    end

    brm_gmrf_vecchia_logdensity(field::vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], east::vector[P], north::vector[P], origin::int[NT], item_ptr::int[NIP],
            item_position::int[NI], item_lag::vector[NI], pattern_ptr::int[NPP], pattern_states::int[N],
            neighbor_ptr::int[NNP], neighbor::int[NN], state_pattern::int[N], M::int)::real = begin
        kappa = sqrt(2.0) / spatial_length
        z = rep_matrix(0.0, C, N)
        for s in 1:N
            for c in 1:C
                value = field[c + C * (s - 1)]
                for k in 1:(c - 1)
                    value -= z[k, s] * L[c, k]
                end
                z[c, s] = value / L[c, c]
            end
        end
        lp = -N * C * log(2.0 * pi) / 2.0
        for c in 1:C
            lp -= N * log(L[c, c])
        end
        for p in 1:NT
            weights = brm_gmrf_vecchia_weights(p, kappa, temporal_length, east, north, origin,
                item_ptr, item_position, item_lag, M)
            m = item_ptr[p + 1] - item_ptr[p]
            q = 0.0
            for slot in pattern_ptr[p]:(pattern_ptr[p + 1] - 1)
                s = pattern_states[slot]
                for c in 1:C
                    r = z[c, s]
                    for a in 1:m
                        r -= weights[a] * z[c, neighbor[neighbor_ptr[s] + a - 1]]
                    end
                    q += square(r)
                end
            end
            count = (pattern_ptr[p + 1] - pattern_ptr[p]) * C
            lp -= (count * log(weights[M + 1]) + q / weights[M + 1]) / 2.0
        end
        lp
    end

    @lpxf brm_gmrf_vecchia_lpdf(field::vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], east::vector[P], north::vector[P], origin::int[NT], item_ptr::int[NIP],
            item_position::int[NI], item_lag::vector[NI], pattern_ptr::int[NPP], pattern_states::int[N],
            neighbor_ptr::int[NNP], neighbor::int[NN], state_pattern::int[N], M::int)::real =
        brm_gmrf_vecchia_logdensity(field, spatial_length, temporal_length, L, east, north, origin,
            item_ptr, item_position, item_lag, pattern_ptr, pattern_states, neighbor_ptr, neighbor,
            state_pattern, M)

    @stanonly brm_gmrf_vecchia_rng(vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], east::vector[P], north::vector[P], origin::int[NT], item_ptr::int[NIP],
            item_position::int[NI], item_lag::vector[NI], pattern_ptr::int[NPP], pattern_states::int[N],
            neighbor_ptr::int[NNP], neighbor::int[NN], state_pattern::int[N], M::int)::vector[D] = begin
        kappa = sqrt(2.0) / spatial_length
        z = rep_matrix(0.0, C, N)
        field = rep_vector(0.0, D)
        for s in 1:N
            p = state_pattern[s]
            weights = brm_gmrf_vecchia_weights(p, kappa, temporal_length, east, north, origin,
                item_ptr, item_position, item_lag, M)
            m = item_ptr[p + 1] - item_ptr[p]
            for c in 1:C
                value = sqrt(weights[M + 1]) * normal_rng(0.0, 1.0)
                for a in 1:m
                    value += weights[a] * z[c, neighbor[neighbor_ptr[s] + a - 1]]
                end
                z[c, s] = value
            end
            for c in 1:C
                value = 0.0
                for k in 1:c
                    value += L[c, k] * z[k, s]
                end
                field[c + C * (s - 1)] = value
            end
        end
        field
    end

    brm_gmrf_vecchia_lpdfs(field::vector[D], spatial_length::real, temporal_length::real,
            L::matrix[C, C], east::vector[P], north::vector[P], origin::int[NT], item_ptr::int[NIP],
            item_position::int[NI], item_lag::vector[NI], pattern_ptr::int[NPP], pattern_states::int[N],
            neighbor_ptr::int[NNP], neighbor::int[NN], state_pattern::int[N], M::int)::vector[D] = begin
        out = rep_vector(0.0, D)
        out[1] = brm_gmrf_vecchia_lpdf(field, spatial_length, temporal_length, L, east, north, origin,
            item_ptr, item_position, item_lag, pattern_ptr, pattern_states, neighbor_ptr, neighbor,
            state_pattern, M)
        out
    end
end

_sb_gmrf_vecchia = StanBlocks.@slic begin
    sigma ~ lognormal(0.0, 1.0; lower=0.0)
    spatial_length ~ lognormal(0.0, 1.0; lower=0.0)
    temporal_length ~ lognormal(0.0, 1.0; lower=0.0)
    field::vector[n_field] ~ brm_gmrf_vecchia(spatial_length, temporal_length, L, east, north, origin,
        item_ptr, item_position, item_lag, pattern_ptr, pattern_states, neighbor_ptr, neighbor,
        state_pattern, M)
    return sigma * field[index]
end

_sb_gmrf = StanBlocks.@slic begin
    sigma ~ lognormal(0.0, 1.0; lower=0.0)
    spatial_length ~ lognormal(0.0, 1.0; lower=0.0)
    temporal_length ~ lognormal(0.0, 1.0; lower=0.0)
    field::vector[n_field] ~ brm_gmrf(spatial_length, temporal_length, L, times,
        permutation, mass, gptr, grow, gval, lptr, lrow, previous_ptr, previous_col, previous_pos)
    return brm_gmrf_rows(sigma, field, index1, index2, index3, weight1, weight2, weight3)
end

_sb_is_direct_term(::typeof(gmrf)) = true
function _sb_emit_direct_expr!(stmts, data, target::Symbol, ::typeof(gmrf), term,
        summands; term_overrides=Dict{Symbol,Any}(), kwargs...)
    push!(summands, _sb_predictor_term!(stmts, data, gmrf, term; target, term_overrides))
end
_sb_term_slot_config(::Val{:spatial_length}, entry) =
    _sb_gp_scale_prior(entry.spec, _sb_term_prior_spelling(entry))
_sb_term_slot_config(::Val{:temporal_length}, entry) =
    _sb_gp_scale_prior(entry.spec, _sb_term_prior_spelling(entry))

function _sb_gmrf_submodel_expr(overrides, term, footprint::Bool; multiresolution::Bool=false)
    slots = (:sigma, :spatial_length, :temporal_length)
    cfgs = map(slot -> _sb_term_cfg(overrides, term, slot), slots)
    # The submodel itself, not its name: a caller's `mod` need not see this module.
    base = multiresolution ? _sb_gmrf_multiresolution : footprint ? _sb_gmrf_vecchia : _sb_gmrf
    all(isnothing, cfgs) && return base
    _sb_configured_term_submodel(base, zip(slots, cfgs)...)
end

function _brm_gmrf_factor_data(mesh)
    p, G = mesh.factor_pattern, mesh.factor_stiffness
    (; permutation=p.permutation, mass=mesh.factor_mass,
       gptr=G.colptr, grow=rowvals(G), gval=nonzeros(G),
       lptr=p.colptr, lrow=p.rowval, previous_ptr=p.previous_ptr,
       previous_col=p.previous_col, previous_pos=p.previous_pos)
end

# A footprint's pattern tables with absolute positions and lags, for the
# Stan helpers above: they evaluate each conditional without the native tables.
function _brm_gmrf_footprint_data(f::_BRMGMRFFootprint)
    npatterns = length(f.pattern_ptr) - 1
    item_position = Vector{Int}(undef, length(f.item_position))
    item_lag = Vector{Float64}(undef, length(f.item_lag))
    origin = Vector{Int}(undef, npatterns)
    state_pattern = Vector{Int}(undef, length(f.position))
    for p in 1:npatterns
        origin[p] = f.geometry_position[f.geometry_ptr[p]]
        for q in f.item_ptr[p]:(f.item_ptr[p + 1] - 1)
            item_position[q] = f.geometry_position[f.geometry_ptr[p] + f.item_position[q] - 1]
            item_lag[q] = f.lag[f.lag_ptr[p] + f.item_lag[q] - 1]
        end
        for slot in f.pattern_ptr[p]:(f.pattern_ptr[p + 1] - 1)
            state_pattern[f.pattern_states[slot]] = p
        end
    end
    (; east=f.east, north=f.north, origin, item_ptr=f.item_ptr, item_position, item_lag,
       pattern_ptr=f.pattern_ptr, pattern_states=Vector{Int}(f.pattern_states),
       neighbor_ptr=f.neighbor_ptr, neighbor=Vector{Int}(f.neighbor), state_pattern, M=f.max_items)
end

_sb_gmrf_data(prepared) = get(prepared.state, :multiresolution, nothing) !== nothing ?
    _sb_gmrf_multiresolution_data(prepared) : prepared.state.neighbors === nothing ?
    _sb_gmrf_mesh_data(prepared) : _sb_gmrf_footprint_data(prepared)

function _sb_gmrf_footprint_data(prepared)
    s = prepared.state
    (; n_field=s.footprint.ncoordinates, _brm_gmrf_footprint_data(s.footprint)..., index=s.index)
end

function _sb_gmrf_mesh_data(prepared)
    s = prepared.state
    (; n_field=s.layout.ncoordinates,
       times=s.times, _brm_gmrf_factor_data(s.mesh)...,
       index1=s.indices[:, 1], index2=s.indices[:, 2], index3=s.indices[:, 3],
       weight1=s.weights[:, 1], weight2=s.weights[:, 2], weight3=s.weights[:, 3])
end

# A fresh replay's read-support count: the new data's value of the field that
# held it, else the fitted count (a literal, or a field the data omits).
function _sb_gmrf_fresh_count(df, state, option::Symbol)
    source = get(state, Symbol(option, :_source), nothing)
    (source === nothing || !_brm_df_has_column(df, source)) && return getproperty(state, option)
    _brm_gmrf_count(option, _sb_df_column(df, source))
end

function _sb_replay_gmrf!(new_data, new_preproc, handled, key, e, df, freeze)
    training = e.const_.prepared
    rows = Dict{Symbol,Any}(s => _sb_df_column(df, s) for s in training.source)
    if !freeze && get(training.state, :multiresolution, nothing) !== nothing
        state = training.state
        times = _brm_gmrf_times(rows[training.source[3]])
        channels = length(training.source) == 3 ? [1] : _brm_fit_levels(rows[training.source[4]])
        if state.correlation !== nothing
            all(c -> c in state.channels, channels) || throw(ArgumentError(
                "gmrf replay with channel correlation cannot introduce new channel labels"))
            channels = state.channels
        end
        multiresolution = _sb_gmrf_fresh_count(df, state, :multiresolution)
        support = _brm_multiresolution_support(rows[training.source[1]], rows[training.source[2]],
            rows[training.source[3]], length(channels); rank=multiresolution)
        training = _BRMPreparedTerm(gmrf, training.source,
            merge(state, (; times, channels, support, multiresolution)), training.dependencies)
    elseif !freeze && training.state.neighbors !== nothing
        # A fresh fit's footprint is its own rows' read pairs.
        state = training.state
        times = _brm_gmrf_times(rows[training.source[3]])
        channels = length(training.source) == 3 ? [1] : _brm_fit_levels(rows[training.source[4]])
        if state.correlation !== nothing
            all(c -> c in state.channels, channels) || throw(ArgumentError(
                "gmrf replay with channel correlation cannot introduce new channel labels"))
            channels = state.channels
        end
        neighbors = _sb_gmrf_fresh_count(df, state, :neighbors)
        footprint = _brm_vecchia_footprint(rows[training.source[1]], rows[training.source[2]],
            rows[training.source[3]], length(channels); neighbors)
        training = _BRMPreparedTerm(gmrf, training.source,
            merge(state, (; times, channels, footprint, neighbors)), training.dependencies)
    elseif !freeze
        state = training.state
        vertices = _sb_df_column(df, state.vertices_source)
        cells = _sb_df_column(df, state.cells_source)
        mesh = _brm_gmrf_mesh(vertices, cells)
        times = _brm_gmrf_times(rows[training.source[3]])
        channels = length(training.source) == 3 ? [1] :
            _brm_fit_levels(rows[training.source[4]])
        # A declared correlation retains its channel labels, including when
        # some channels have no replay rows. Changing that joint law needs a
        # new formula, as for the other correlated structured terms.
        if state.correlation !== nothing
            all(c -> c in state.channels, channels) || throw(ArgumentError(
                "gmrf replay with channel correlation cannot introduce new channel labels"))
            channels = state.channels
        end
        layout = _BRMGMRFStateLayout(size(vertices, 1), channels, times)
        training = _BRMPreparedTerm(gmrf, training.source,
            merge(state, (; vertices, cells, mesh, times=layout.times, channels=layout.channels, layout,
                          nvertices=size(vertices, 1))), training.dependencies)
    end
    # A frozen replay keeps the fitted support, which a data-valued count sized.
    freeze && _sb_check_replay_settings(df, :gmrf, Tuple(
        source => getproperty(training.state, option)
        for option in (:neighbors, :multiresolution)
        for source in (get(training.state, Symbol(option, :_source), nothing),)
        if source !== nothing))
    prepared = _brm_replay_gmrf(training, (; data=rows))
    for (name, value) in pairs(_sb_gmrf_data(prepared))
        owned = e.const_.keys[name]
        new_data[owned] = value
        push!(handled, owned)
    end
    s = prepared.state
    if s.correlation === nothing
        new_data[e.const_.L_key] = Matrix{Float64}(I, length(s.channels), length(s.channels))
        push!(handled, e.const_.L_key)
    end
    s.neighbors === nothing && get(s, :multiresolution, nothing) === nothing &&
        push!(handled, s.vertices_source, s.cells_source)
    new_preproc[key] = PreprocEntry(:gmrf, merge(e.const_, (; prepared)), e.raw_ref, true)
end

function _sb_predictor_term!(stmts, data, ::typeof(gmrf), term;
        target::Symbol, term_overrides=Dict{Symbol,Any}(),
        term_scope::Symbol=:population, kwargs...)
    arguments = _brm_gmrf_arguments(term)
    footprint = arguments.neighbors !== nothing
    multiresolution = arguments.multiresolution !== nothing
    raw = (arguments.axes..., arguments.time,
        (arguments.channel === nothing ? () : (arguments.channel,))...,
        (footprint || multiresolution ? () : (arguments.mesh.vertices, arguments.mesh.cells))...)
    inner = Dict{Symbol,Any}(_sb_inner_data(:gmrf, value) for value in raw)
    prepared = _brm_prepare_term(term, target, (; data=inner))
    # Preparation here receives raw support data, while the resolved prior
    # configurations arrive separately. Preserve those same configurations in
    # fitted metadata, so replay and descriptions agree with the emitted law.
    prior_keys = (:sigma_prior, :spatial_prior, :temporal_prior)
    priors = map(prior_keys, (:sigma, :spatial_length, :temporal_length)) do key, slot
        cfg = _sb_term_cfg(term_overrides, term, slot)
        _sb_refuse_derived(cfg, "a `gmrf` parameter")
        isnothing(cfg) ? getproperty(prepared.state, key) :
            _sb_known_cfg(cfg) ? _BRMKnownTermValue(cfg.known) : cfg.prior
    end
    prepared = _BRMPreparedTerm(gmrf, prepared.source,
        merge(prepared.state, NamedTuple{prior_keys}(priors)), prepared.dependencies)
    state = prepared.state
    stem = Symbol("gmrf_", target, "_", join(prepared.source, "_"))
    _sb_record_term_binding!(data, stem, target, term; scope=term_scope,
        family=:gmrf, term_label=stem, term_parameters=_BRM_GMRF_PARAMETER_BINDINGS)
    field_data = _sb_gmrf_data(prepared)
    keys_ = NamedTuple{keys(field_data)}(map(key -> Symbol(stem, :_, key), keys(field_data)))
    for (name, value) in pairs(field_data)
        key = keys_[name]
        haskey(data, key) && error("sbimpl: gmrf reserves data key `$key`")
        data[key] = value
    end
    L_key = if state.correlation === nothing
        key = Symbol(stem, :_L)
        data[key] = Matrix{Float64}(I, length(state.channels), length(state.channels))
        key
    else
        if state.correlation.kind === :data
            # The named matrix can also be present as a raw model input.
            # It is a fitted constant, just like its Cholesky factor.
            cor_name, cor_value = _sb_inner_data(:gmrf, arguments.cor)
            data[cor_name] = cor_value
            _sb_record_static!(data, cor_name)
        end
        _sb_level_correlation_factor!(data, state.correlation)
    end
    _sb_record_preproc!(data, footprint || multiresolution ? keys_.index : keys_.index1, PreprocEntry(:gmrf,
        (; prepared, keys=keys_, L_key), prepared.source, true))
    submodel = _sb_gmrf_submodel_expr(term_overrides, term, footprint; multiresolution)
    push!(stmts, Expr(:call, :~, stem, _sb_term_model_call(submodel, term_overrides, term;
        data, stem, L=L_key, keys_...)))
    stem
end
