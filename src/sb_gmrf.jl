# Stan and native Julia evaluate the same scalar sparse LDL and normalized
# field law. The emitted model retains original vertex/channel/time ordering;
# only the spatial factor uses the data-only fill-reducing permutation.
StanBlocks.@deffun @juliacompat begin
    brm_gmrf_factor(kappa2::real, mass::vector[N], gptr::int[NP],
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
                    work[grow[p]] = gval[p]
                end
            end
            work[j] += kappa2 * mass[j]
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
        kappa2 = 2.0 / square(spatial_length)
        factor = brm_gmrf_factor(kappa2, mass, gptr, grow, gval, lptr, lrow,
            previous_ptr, previous_col, previous_pos)
        logdet_space = 2.0 * sum(log(segment(factor, 1, N))) - sum(log(mass)) - N * log(4.0 * pi * kappa2)
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
                    z = kappa2 * mass[i] * residual[i, c]
                    for p in gptr[i]:(gptr[i + 1] - 1)
                        z += gval[p] * residual[grow[p], c]
                    end
                    quadratic += square(z) / mass[i]
                end
            end
            lp -= (quadratic / (4.0 * pi * kappa2 * variance) + N * C * log(variance)) / 2.0
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
        kappa2 = 2.0 / square(spatial_length)
        factor = brm_gmrf_factor(kappa2, mass, gptr, grow, gval, lptr, lrow,
            previous_ptr, previous_col, previous_pos)
        field = rep_vector(0.0, D)
        residual::matrix[N, C]
        white::vector[N]
        for t in 1:T
            a = t == 1 ? 0.0 : exp(-(times[t] - times[t - 1]) / temporal_length)
            variance = t == 1 ? 1.0 : -expm1(-2.0 * (times[t] - times[t - 1]) / temporal_length)
            for c in 1:C
                for i in 1:N
                    white[i] = normal_rng(0.0, sqrt(4.0 * pi * kappa2 * mass[i]))
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

function _sb_gmrf_submodel_expr(overrides, term)
    slots = (:sigma, :spatial_length, :temporal_length)
    cfgs = map(slot -> _sb_term_cfg(overrides, term, slot), slots)
    all(isnothing, cfgs) && return :_sb_gmrf
    _sb_configured_term_submodel(_sb_gmrf, zip(slots, cfgs)...)
end

function _brm_gmrf_factor_data(mesh)
    p, G = mesh.factor_pattern, mesh.factor_stiffness
    (; permutation=p.permutation, mass=mesh.factor_mass,
       gptr=G.colptr, grow=rowvals(G), gval=nonzeros(G),
       lptr=p.colptr, lrow=p.rowval, previous_ptr=p.previous_ptr,
       previous_col=p.previous_col, previous_pos=p.previous_pos)
end

function _sb_gmrf_data(prepared)
    s = prepared.state
    (; n_field=s.layout.ncoordinates,
       times=s.times, _brm_gmrf_factor_data(s.mesh)...,
       index1=s.indices[:, 1], index2=s.indices[:, 2], index3=s.indices[:, 3],
       weight1=s.weights[:, 1], weight2=s.weights[:, 2], weight3=s.weights[:, 3])
end

function _sb_replay_gmrf!(new_data, new_preproc, handled, key, e, df, freeze)
    training = e.const_.prepared
    rows = Dict{Symbol,Any}(s => _sb_df_column(df, s) for s in training.source)
    if !freeze
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
    push!(handled, s.vertices_source, s.cells_source)
    new_preproc[key] = PreprocEntry(:gmrf, merge(e.const_, (; prepared)), e.raw_ref, true)
end

function _sb_predictor_term!(stmts, data, ::typeof(gmrf), term;
        target::Symbol, term_overrides=Dict{Symbol,Any}(), kwargs...)
    arguments = _brm_gmrf_arguments(term)
    raw = (arguments.axes..., arguments.time,
        (arguments.channel === nothing ? () : (arguments.channel,))...,
        arguments.mesh.vertices, arguments.mesh.cells)
    inner = Dict{Symbol,Any}(_sb_inner_data(:gmrf, value) for value in raw)
    prepared = _brm_prepare_term(term, target, (; data=inner))
    state = prepared.state
    stem = Symbol("gmrf_", target, "_", join(prepared.source, "_"))
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
    _sb_record_preproc!(data, keys_.index1, PreprocEntry(:gmrf,
        (; prepared, keys=keys_, L_key), prepared.source, true))
    submodel = _sb_gmrf_submodel_expr(term_overrides, term)
    push!(stmts, Expr(:call, :~, stem, _sb_term_model_call(submodel, term_overrides, term;
        data, stem, L=L_key, keys_...)))
    stem
end
