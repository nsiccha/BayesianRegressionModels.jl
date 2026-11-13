# Numerical HSGP source is authored graph data. Its sizes and settings are not:
# each axis's basis count (`modes`), boundary factor and fixed domain
# (`settings`) are bound inputs every graph reads, so one program serves every
# value (user principle `0ikeidi`); `B = prod(modes)` is the basis size. Automatic
# fits are computed from the current bound raw axes. A model-derived axis always
# has fixed fits and is read from its graph value. Basis and spectral arithmetic
# never cross an ordinary BRM helper boundary.
function _rk_ast_graph_plate(arguments, parameters, body)
    body = Meta.isexpr(body, :block) ? deepcopy(body) : Expr(:block, body)
    last = findlast(x -> !(x isa LineNumberNode), body.args)
    Meta.isexpr(body.args[last], :return) ||
        (body.args[last] = Expr(:return, body.args[last]))
    Expr(:do, Expr(:call, Expr(:., :ReactiveKernels, QuoteNode(:plate)), arguments...),
        Expr(:->, Expr(:tuple, parameters...), body))
end

# Axis inputs of a squared-exponential HSGP graph: a one-dimensional basis
# reads `axis`; per-axis suffixes, tuples and one-column frequency matrices
# appear only for a tensor basis.
_rk_hsgp_per_axis(stem, D) = D == 1 ? [Symbol(stem)] : [Symbol(stem, :_, j) for j in 1:D]
_rk_hsgp_modes(options) = options.k isa Tuple ? options.k : (options.k,)
_rk_hsgp_axis_inputs(term) = _rk_hsgp_per_axis(:axis, length(term.columns))

# The term's size and settings inputs (`_rk_plan_hsgp_term!`), as each graph
# reads them: `modes` where it sizes the basis or bounds the length scale,
# `settings` where it fits or fixes the interval (an automatic interval already
# fitted is read from `fit`). Their positions: axis `j`'s boundary factor, then
# each fixed axis's center and half-width.
_rk_hsgp_setting_actuals(options, formals) =
    map(f -> f === :modes ? options.modes : options.settings, formals)
_rk_hsgp_fit_setting_formals(options) =
    _rk_hsgp_orthogonal(options) === nothing ? (:settings,) : (:modes, :settings)
_rk_hsgp_basis_setting_formals(term) =
    _rk_hsgp_fitted(term) && get(term.options, :anchor, nothing) === nothing ?
        (:modes,) : (:modes, :settings)
_rk_hsgp_boundary(j; settings=:settings) = :($settings[$j])
_rk_hsgp_fixed_center(D, j; settings=:settings) = :($settings[$(D + 2j - 1)])
_rk_hsgp_fixed_width(D, j; settings=:settings) = :($settings[$(D + 2j)])
# An anchored term's anchor follows: axis `j`'s anchored value.
_rk_hsgp_anchor_value(options, D, j; settings=:settings) =
    :($settings[$(D + (options.fixed_fits === nothing ? 0 : 2D) + j)])
# Axis `j`'s mode in tensor basis column `b`: the column-major position of `b`
# in the `modes[1] x ... x modes[D]` grid, as `CartesianIndices` orders it.
function _rk_hsgp_tensor_mode(j)
    stride = j == 1 ? 1 : j == 2 ? :(modes[1]) :
        Expr(:call, :*, (:(modes[$i]) for i in 1:j-1)...)
    :(div(b - 1, $stride) % modes[$j] + 1)
end

# A `rows=` term also reads its bound 0/1 mask (decision 1fk9b29): the formal
# follows the axes in every graph that fits or evaluates the basis.
_rk_hsgp_mask(term) = get(term.options, :mask, nothing)
_rk_hsgp_mask_formals(term) = _rk_hsgp_mask(term) === nothing ? () : (:mask,)
_rk_hsgp_mask_actuals(term) = _rk_hsgp_mask(term) === nothing ? () : (_rk_hsgp_mask(term),)

# An automatic interval is fitted to fit-role copies of the axes (and mask)
# planned by `_rk_plan_hsgp_term!`: the axes themselves when the model is
# built, the training axes on a frozen `reprocess`. It is a fitted value of the
# program (`_rk_ast_hsgp_fit!`), and the basis and floor graphs read it as
# their trailing `fit` argument.
_rk_hsgp_fit_axes(term) = get(term.options, :fit_axes, nothing)
_rk_hsgp_fitted(term) = _rk_hsgp_fit_axes(term) !== nothing
_rk_hsgp_fit_formals(term) = _rk_hsgp_fitted(term) ? (:fit,) : ()
_rk_hsgp_fit_graph_formals(term) = (_rk_hsgp_per_axis(:axis_fit, length(term.columns))...,
    (_rk_hsgp_mask(term) === nothing ? () : (:mask_fit,))...,
    (_rk_hsgp_level_projected(term.options) ? (:group_fit,) : ())...,
    _rk_hsgp_fit_setting_formals(term.options)...)
_rk_hsgp_fit_actuals(term) = (_rk_hsgp_fit_axes(term)...,
    (_rk_hsgp_mask(term) === nothing ? () : (term.options.fit_mask,))...,
    (_rk_hsgp_level_projected(term.options) ? (term.options.fit_group_index,) : ())...,
    _rk_hsgp_setting_actuals(term.options, _rk_hsgp_fit_setting_formals(term.options))...)
_rk_hsgp_clamped(options) = get(options, :extrapolate, :error) === :clamp
_rk_hsgp_orthogonal(options) = get(options, :orthogonal, nothing)
# A `by=` term's `orthogonal_to=` projection is fitted per level (decision
# 15hn91t) on the fit-role rows' level codes (`_rk_plan_projection_levels!`).
_rk_hsgp_level_projected(options) = haskey(options, :fit_group_index)

# Where each fitted quantity sits in a term's `fit` vector: every axis's center
# and half-width, then every axis's training range when the basis clamps, then
# for `orthogonal_to=` each basis column's mean — after the axis mean and its
# sum of squares, and followed by each column's projection coefficient, when
# the basis is orthogonal to the linear term. A per-level projection repeats
# that projection block once per level, from offset `projection`.
function _rk_hsgp_fit_index(options, D)
    ranges = _rk_hsgp_clamped(options) ? 2D : 0
    axis = _rk_hsgp_orthogonal(options) === :linear ? 2 : 0
    (; center=j -> 2j - 1, width=j -> 2j,
        lower=j -> 2D + 2j - 1, upper=j -> 2D + 2j,
        axis_mean=2D + ranges + 1, axis_ss=2D + ranges + 2, columns=2D + ranges + axis,
        projection=2D + ranges)
end

# The length of one level's projection block: the column means, after the
# axis mean and sum of squares and followed by the slopes for `:linear`. `B` is
# the graph's basis size value.
_rk_hsgp_projection_stride(orthogonal::Symbol, B) = orthogonal === :linear ? :(2 + 2 * $B) : B

# Each axis's half-width, and its center when the basis reads it: the `settings`
# input's center and half-width for a fixed domain, otherwise computed from the
# current axis -- over the masked rows only, widened to contain an anchor -- and
# scaled by the axis's boundary factor: the law of `_brm_fit_hsgp`.
function _rk_ast_hsgp_fits!(body, options, inputs; centered=true, mask=:mask)
    D = length(inputs)
    centers, widths = _rk_hsgp_per_axis(:center, D), _rk_hsgp_per_axis(:width, D)
    fixed = options.fixed_fits !== nothing
    masked = get(options, :mask, nothing) !== nothing
    anchor = get(options, :anchor, nothing)
    for j in 1:D
        x, center, width = inputs[j], centers[j], widths[j]
        (centered || !fixed) && push!(body.args, Expr(:(=), center,
            fixed ? _rk_hsgp_fixed_center(D, j) :
            masked ? :(sum($x .* $mask) / sum($mask)) : :(sum($x) / length($x))))
        if !fixed
            deviation = masked ? :(maximum(abs.($x .- $center) .* $mask)) :
                :(maximum(abs.($x .- $center)))
            anchor === nothing || (deviation = :(max($deviation,
                abs($(_rk_hsgp_anchor_value(options, D, j)) - $center))))
            push!(body.args, Expr(:(=), width, :($(_rk_hsgp_boundary(j)) * $deviation)))
        else
            push!(body.args, Expr(:(=), width, _rk_hsgp_fixed_width(D, j)))
        end
    end
    centers, widths
end

# The centers and half-widths a graph reads from its `fit` argument.
function _rk_ast_hsgp_read_fit!(body, options, D; centered=true)
    index = _rk_hsgp_fit_index(options, D)
    centers, widths = _rk_hsgp_per_axis(:center, D), _rk_hsgp_per_axis(:width, D)
    for j in 1:D
        centered && push!(body.args, :($(centers[j]) = fit[$(index.center(j))]))
        push!(body.args, :($(widths[j]) = fit[$(index.width(j))]))
    end
    centers, widths
end

# `extrapolate=:clamp` evaluates the basis at each value clamped into the
# training data range of its axis (a no-op on the training rows).
function _rk_ast_hsgp_clamp(cell, row_values, D)
    clamped = Dict{Symbol,Any}(row_values[j] => :(clamp($(row_values[j]),
        $(Symbol(:support_lower_, j)), $(Symbol(:support_upper_, j)))) for j in 1:D)
    _rk_ast_hsgp_substitute(cell, clamped)
end

# The squared frequencies and the basis cell of one row (`b` indexes the mode),
# for the given centers and half-widths, in a graph that defines `B` and reads
# `modes`. Returns the cell, the names of a row's axis values and the graph
# inputs a row zips.
function _rk_ast_hsgp_cell!(body, D, centers, widths, inputs)
    frequencies = D == 1 ? [:omega2] : _rk_hsgp_per_axis(:frequency, D)
    for j in 1:D
        # These are the tensor basis coordinates in the same Julia/Stan
        # column-major order; no prepared numerical basis is shipped. A
        # tensor axis's modes are a graph value of their own, which its
        # frequency plate then reads.
        mode_values = if D == 1
            :(1:B)
        else
            name = Symbol(:modes_, j)
            push!(body.args, Expr(:(=), name,
                _rk_ast_graph_plate([:(1:B)], [:b], _rk_hsgp_tensor_mode(j))))
            name
        end
        frequency = _rk_ast_graph_plate([mode_values], [:mode],
            :((mode * pi / (2 * $(widths[j])))^2))
        push!(body.args, Expr(:(=), frequencies[j], frequency))
    end
    if D == 1
        cell = :(sin(sqrt(omega2[b]) * (x - center + width)) / sqrt(width))
        return cell, [:x], Any[only(inputs)]
    end
    push!(body.args, :(widths = $(Expr(:tuple, widths...))))
    push!(body.args, :(centers = $(Expr(:tuple, centers...))))
    push!(body.args, :(omega2 = hcat($(frequencies...))))
    row_values = [Symbol(:x_, j) for j in 1:D]
    cell = Expr(:call, :*, [:(sin(sqrt(omega2[b, $j]) *
        ($(row_values[j]) - centers[$j] + widths[$j])) / sqrt(widths[$j]))
        for j in 1:D]...)
    cell, row_values, Any[inputs...]
end

# A term's automatic interval, fitted once over its fit-role axes and mask:
# `<id>_fit = brm_hsgp_fit_graph(x_fit, ...)`, the `fit` vector the basis and
# floor graphs read (`_rk_hsgp_fit_index`). It reads only fit-role inputs, so
# it is a fitted value of the program (`brm_fitted_state`). Returns its name.
function _rk_ast_hsgp_fit!(definitions, statements, term, taken)
    options = term.options
    D, B = length(_rk_hsgp_modes(options)), :B
    formals = _rk_hsgp_fit_graph_formals(term)
    inputs = collect(formals[1:D])
    masked = _rk_hsgp_mask(term) !== nothing
    body = Expr(:block)
    centers, widths = _rk_ast_hsgp_fits!(body, options, inputs; mask=:mask_fit)
    entries = Any[]
    for j in 1:D
        push!(entries, centers[j], widths[j])
    end
    if _rk_hsgp_clamped(options)
        for j in 1:D
            x = inputs[j]
            lower, upper = Symbol(:support_lower_, j), Symbol(:support_upper_, j)
            push!(body.args, masked ?
                :($lower = minimum(ifelse.(mask_fit .> 0, $x, Inf))) :
                :($lower = minimum($x)))
            push!(body.args, masked ?
                :($upper = maximum(ifelse.(mask_fit .> 0, $x, -Inf))) :
                :($upper = maximum($x)))
            push!(entries, lower, upper)
        end
    end
    value = Expr(:vect, entries...)
    orthogonal = _rk_hsgp_orthogonal(options)
    if orthogonal !== nothing
        # The column means and projections are fitted on the basis at the
        # fit-role axes (their masked rows), as the basis graph evaluates it at
        # the rows.
        push!(body.args, :(B = prod(modes)))
        cell, row_values, row_inputs = _rk_ast_hsgp_cell!(body, D, centers, widths, inputs)
        _rk_hsgp_clamped(options) && (cell = _rk_ast_hsgp_clamp(cell, row_values, D))
        inner = _rk_ast_graph_plate([:(1:$B)], [:b], cell)
        row_value = :values
        if masked
            push!(row_inputs, :mask_fit)
            push!(row_values, :m)
            row_value = :(m .* values)
        end
        fit_rows = _rk_ast_graph_plate(row_inputs, row_values,
            Expr(:block, :(values = $inner), row_value))
        push!(body.args, :(fit_basis = stack($fit_rows; dims=1)))
        value = if _rk_hsgp_level_projected(options)
            _rk_ast_hsgp_level_projection_fit!(Val(orthogonal), body, value, inputs,
                                               B, options.n_groups, masked)
        else
            _rk_ast_hsgp_projection_fit!(Val(orthogonal), body, value, inputs, B,
                                         masked ? :(sum(mask_fit)) : nothing)
        end
    end
    push!(body.args, :(fit = $value), :(return fit))
    doc = @_rk_definition_doc """
        The fitted interval of an HSGP term's basis, from its training (fit-role) axes.

        Returns one vector `fit`: each axis's center and half-width$(_rk_hsgp_clamped(options) ?
            ", its observed lower and upper bounds" : "")$(orthogonal === nothing ? "" :
            ", and the basis columns' means and projections for `orthogonal_to`"). The basis and
        length-scale floor graphs read it, so a prediction uses the training interval."""
    entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_fit_graph",
        collect(formals), body; kernel=true, doc)
    name = _rk_ast_fresh_name(string(options.id, "_fit"), taken)
    push!(statements, Expr(:(=), name, Expr(:call, entry, _rk_hsgp_fit_actuals(term)...)))
    name
end

# `orthogonal_to=`'s projection fitted on `fit_basis`, the basis at the fit-role
# rows (rows outside a mask are zero, and `rows` counts the rest), appended to
# the `fit` vector `value`: the law of `_brm_hsgp_projection_fit`.
function _rk_ast_hsgp_projection_fit!(::Val{:intercept}, body, value, _inputs, B, rows)
    count = rows === nothing ? :(length(fit_phi)) : rows
    columns = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
            fit_phi = fit_basis[:, b]
            sum(fit_phi) / $count
        end))
    push!(body.args, :(column_means = $columns))
    :(vcat($value, column_means))
end

function _rk_ast_hsgp_projection_fit!(::Val{:linear}, body, value, inputs, B, rows)
    rows === nothing || error(
        "BRM backend lowering: internal error — `orthogonal_to=:linear` with `rows=`")
    x = only(inputs)
    push!(body.args, :(axis_mean = sum($x) / length($x)))
    push!(body.args, :(fit_axis_centered = $x .- axis_mean))
    push!(body.args, :(axis_ss = sum(fit_axis_centered .^ 2)))
    columns = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
            fit_phi = fit_basis[:, b]
            phi_mean = sum(fit_phi) / length(fit_phi)
            fit_centered = fit_phi .- phi_mean
            [phi_mean, axis_ss > 1e-12 ?
                sum(fit_axis_centered .* fit_centered) / axis_ss : 0.0]
        end))
    push!(body.args, :(column_fits = stack($columns; dims=2)))
    :(vcat($value, [axis_mean, axis_ss], column_fits[1, :], column_fits[2, :]))
end

# A per-level projection: the block of `_rk_ast_hsgp_projection_fit!` fitted on
# each level's fit-role rows (`group_fit .== g`, within the mask), one block per
# level appended to `value` in level order (`_rk_hsgp_projection_stride`).
function _rk_ast_hsgp_level_projection_fit!(::Val{:intercept}, body, value, _inputs,
                                            B, G, masked)
    weight = masked ? :(ifelse.(group_fit .== g, 1.0, 0.0) .* mask_fit) :
        :(ifelse.(group_fit .== g, 1.0, 0.0))
    columns = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
            sum(fit_basis[:, b] .* in_level) / level_rows
        end))
    levels = _rk_ast_graph_plate([:(1:$G)], [:g], Base.remove_linenums!(quote
            in_level = $weight
            level_rows = sum(in_level)
            $columns
        end))
    push!(body.args, :(level_column_means = stack($levels; dims=2)))
    :(vcat($value, vec(level_column_means)))
end

function _rk_ast_hsgp_level_projection_fit!(::Val{:linear}, body, value, inputs,
                                            B, G, masked)
    masked && error(
        "BRM backend lowering: internal error — `orthogonal_to=:linear` with `rows=`")
    x = only(inputs)
    columns = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
            fit_phi = fit_basis[:, b]
            phi_mean = sum(fit_phi .* in_level) / level_rows
            fit_centered = (fit_phi .- phi_mean) .* in_level
            [phi_mean, axis_ss > 1e-12 ?
                sum(fit_axis_centered .* fit_centered) / axis_ss : 0.0]
        end))
    levels = _rk_ast_graph_plate([:(1:$G)], [:g], Base.remove_linenums!(quote
            in_level = ifelse.(group_fit .== g, 1.0, 0.0)
            level_rows = sum(in_level)
            axis_mean = sum($x .* in_level) / level_rows
            fit_axis_centered = ($x .- axis_mean) .* in_level
            axis_ss = sum(fit_axis_centered .^ 2)
            column_fits = stack($columns; dims=2)
            vcat([axis_mean, axis_ss], column_fits[1, :], column_fits[2, :])
        end))
    push!(body.args, :(level_fits = stack($levels; dims=2)))
    :(vcat($value, vec(level_fits)))
end

# The basis matrix and squared frequencies of one basis structure (modes,
# boundary factor, fits, orthogonalization); terms with the same structure
# share the definition. Its caller destructures both results.
function _rk_ast_hsgp_basis_graph!(definitions, term, taken)
    options = term.options
    D, B = length(_rk_hsgp_modes(options)), :B
    inputs = _rk_hsgp_axis_inputs(term)
    fitted = _rk_hsgp_fitted(term)
    body = Expr(:block, :(B = prod(modes)))
    centers, widths = fitted ? _rk_ast_hsgp_read_fit!(body, options, D) :
        _rk_ast_hsgp_fits!(body, options, inputs; mask=:mask)
    # Each row cell zips the axes; its mode cells close over the row's axis
    # values and the graph's whole `omega2`, widths and centers. An anchored
    # term subtracts the basis row at its anchor; a masked one zeroes the rows
    # outside its mask (decision 1fk9b29).
    cell, row_values, row_inputs = _rk_ast_hsgp_cell!(body, D, centers, widths, inputs)
    anchor = get(options, :anchor, nothing)
    masked = _rk_hsgp_mask(term) !== nothing
    if _rk_hsgp_clamped(options) && fitted
        index = _rk_hsgp_fit_index(options, D)
        for j in 1:D
            push!(body.args, :($(Symbol(:support_lower_, j)) = fit[$(index.lower(j))]),
                :($(Symbol(:support_upper_, j)) = fit[$(index.upper(j))]))
        end
        cell = _rk_ast_hsgp_clamp(cell, row_values, D)
    end
    inner = _rk_ast_graph_plate([:(1:$B)], [:b], cell)
    row_value = :values
    if anchor !== nothing
        anchored = _rk_ast_hsgp_substitute(cell, Dict(zip(row_values,
            Any[_rk_hsgp_anchor_value(options, D, j) for j in 1:D])))
        push!(body.args, :(anchor_values = $(_rk_ast_graph_plate([:(1:$B)], [:b], anchored))))
        row_value = :(values .- anchor_values)
    end
    if masked
        push!(row_inputs, :mask)
        push!(row_values, :m)
        row_value = :(m .* $row_value)
    end
    outer = _rk_ast_graph_plate(row_inputs, row_values,
        Expr(:block, :(values = $inner), row_value))
    push!(body.args, :(basis_rows = $outer))
    orthogonal = _rk_hsgp_orthogonal(options)
    level_projected = _rk_hsgp_level_projected(options)
    if level_projected
        # A per-level projection (`by=`): every row projected with its own
        # level's fitted block (`_rk_ast_hsgp_level_projection_fit!`), read at
        # the row's level code.
        index = _rk_hsgp_fit_index(options, D)
        stride = _rk_hsgp_projection_stride(orthogonal, B)
        push!(body.args, :(raw_basis = stack(basis_rows; dims=1)))
        push!(body.args, :(level_offset = $(index.projection) .+ (group_index .- 1) .* $stride))
        if orthogonal === :intercept
            column = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
                    centered = raw_basis[:, b] .- fit[level_offset .+ b]
                    $(masked ? :(mask .* centered) : :centered)
                end))
        else
            push!(body.args, :(axis_centered = $(only(inputs)) .- fit[level_offset .+ 1]))
            column = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
                    centered = raw_basis[:, b] .- fit[level_offset .+ (2 + b)]
                    centered .- axis_centered .* fit[level_offset .+ (2 + B + b)]
                end))
        end
        push!(body.args, :(basis_columns = $column))
        push!(body.args, :(PHI = stack(basis_columns; dims=2)))
    elseif orthogonal === :intercept
        # `orthogonal_to=:intercept`: center every basis column over the rows
        # (the masked ones; the others stay zero) — the fit-role column means
        # when fitted, otherwise the current rows' own (a model-derived axis
        # at every draw, as SB's `brm_hsgp_center_columns`).
        push!(body.args, :(raw_basis = stack(basis_rows; dims=1)))
        if fitted
            index = _rk_hsgp_fit_index(options, D)
            mean = :(fit[$(index.columns) + b])
        else
            mean = :(sum(phi) / $(masked ? :(sum(mask)) : :(length(phi))))
        end
        centered = :(phi .- $mean)
        column = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
                phi = raw_basis[:, b]
                $(masked ? :(mask .* $centered) : centered)
            end))
        push!(body.args, :(basis_columns = $column))
        push!(body.args, :(PHI = stack(basis_columns; dims=2)))
    elseif orthogonal === :linear
        # `orthogonal_to=:linear` (one axis, by preparation): center every
        # basis column and project out the centered axis, column by column,
        # from the current axis values — the law of SB's in-graph
        # `brm_hsgp_orthogonalize_linear`, including its degenerate-axis guard.
        x = only(inputs)
        push!(body.args, :(raw_basis = stack(basis_rows; dims=1)))
        if !fitted
            push!(body.args, :(axis_centered = $x .- sum($x) / length($x)))
            push!(body.args, :(axis_ss = sum(axis_centered .^ 2)))
            column = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
                    phi = raw_basis[:, b]
                    centered = phi .- sum(phi) / length(phi)
                    axis_ss > 1e-12 ?
                        centered .- axis_centered .*
                            (sum(axis_centered .* centered) / axis_ss) : centered
                end))
        else
            # The column means and projections fitted on the basis at the
            # fit-role axis (`_rk_ast_hsgp_fit!`), applied to the rows.
            index = _rk_hsgp_fit_index(options, D)
            push!(body.args, :(axis_mean = fit[$(index.axis_mean)]))
            push!(body.args, :(axis_ss = fit[$(index.axis_ss)]))
            push!(body.args, :(axis_centered = $x .- axis_mean))
            column = _rk_ast_graph_plate([:(1:$B)], [:b], Base.remove_linenums!(quote
                    phi_mean = fit[$(index.columns) + b]
                    centered = raw_basis[:, b] .- phi_mean
                    axis_ss > 1e-12 ?
                        centered .- axis_centered .* fit[$(index.columns) + B + b] :
                        centered
                end))
        end
        push!(body.args, :(basis_columns = $column))
        push!(body.args, :(PHI = stack(basis_columns; dims=2)))
    elseif options.fixed_fits !== nothing && !options.latent
        # Only an explicit domain admits an axis with no rows (decision
        # 1fk9b29): its basis is then `0 x B`, which `stack` cannot size. The
        # basis reads data and its settings only, so both arms are inactive
        # under AD.
        push!(body.args, :(PHI = isempty(basis_rows) ? zeros(0, $B) :
            stack(basis_rows; dims=1)))
    else
        push!(body.args, :(PHI = stack(basis_rows; dims=1)))
    end
    push!(body.args, Expr(:return, Expr(:tuple, :PHI, :omega2)))
    doc = @_rk_definition_doc """
        The Hilbert-space basis of an HSGP term at the rows.

        Returns `(PHI, omega2)`: `PHI` holds one column per basis function, the Laplacian
        eigenfunctions on the term's interval evaluated at each row's axes, and `omega2`
        the matching squared frequencies, which the spectral density reads."""
    _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_basis_graph",
        [inputs; _rk_hsgp_mask_formals(term)...; _rk_hsgp_fit_formals(term)...;
         _rk_hsgp_basis_setting_formals(term)...;
         (level_projected ? (:group_index,) : ())...],
        body; kernel=true, doc)
end

# The length-scale validity floor `(4L/pi) * sqrt(log(100) / (k^2 - 1))` per
# axis (the largest for an isotropic length scale, `0.0` for a single mode),
# emitted only where a prior or hyper-predictor reads it. Automatic fits read
# the current axes through a shared graph; a fixed domain's floor reads its
# `settings` input in the main block. Both read the basis counts from `modes`.
function _rk_ast_hsgp_floor!(definitions, statements, term, taken; fit=nothing)
    options = term.options
    D = length(_rk_hsgp_modes(options))
    body = Expr(:block)
    fitted = fit !== nothing
    inputs = _rk_hsgp_axis_inputs(term)
    _, widths = fitted ? _rk_ast_hsgp_read_fit!(body, options, D; centered=false) :
        _rk_ast_hsgp_fits!(body, options, inputs; centered=false, mask=:mask)
    floors = [:(modes[$j] == 1 ? 0.0 : (4 * $(widths[j]) / pi) *
        sqrt(log($(_BRM_HSGP_WEIGHT_THRESHOLD)) / (modes[$j]^2 - 1))) for j in 1:D]
    value = D == 1 ? only(floors) : options.iso ? Expr(:call, :max, floors...) :
        Expr(:vect, floors...)
    name = _rk_ast_fresh_name(string(options.id, "_rho_floor"), taken)
    if options.fixed_fits === nothing
        push!(body.args, :(rho_floor = $value), :(return rho_floor))
        # The floor reads only the fitted interval: the term's `fit` when the
        # interval is fitted to fit-role inputs.
        settings = fitted ? (:modes,) : (:modes, :settings)
        formals = fitted ? [:fit; settings...] :
            [inputs; _rk_hsgp_mask_formals(term)...; settings...]
        actuals = fitted ? (fit, _rk_hsgp_setting_actuals(options, settings)...) :
            (term.columns..., _rk_hsgp_mask_actuals(term)...,
             _rk_hsgp_setting_actuals(options, settings)...)
        doc = @_rk_definition_doc """
            The smallest length scale an HSGP term's basis resolves, per axis.

            For `m` basis functions on an axis of half-width `L` the floor is
            `(4L / pi) * sqrt(log(threshold) / (m^2 - 1))` (`0.0` for one function), below
            which the omitted frequencies would carry too much weight."""
        entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_rho_floor_graph",
            formals, body; kernel=true, doc)
        push!(statements, Expr(:(=), name, Expr(:call, entry, actuals...)))
    else
        fits = Dict{Symbol,Any}(widths[j] =>
            _rk_hsgp_fixed_width(D, j; settings=options.settings) for j in 1:D)
        fits[:modes] = options.modes
        push!(statements, Expr(:(=), name, _rk_ast_hsgp_substitute(value, fits)))
    end
    name
end

_rk_ast_hsgp_substitute(value::Symbol, values) = get(values, value, value)
_rk_ast_hsgp_substitute(value::Expr, values) =
    Expr(value.head, (_rk_ast_hsgp_substitute(a, values) for a in value.args)...)
_rk_ast_hsgp_substitute(value, values) = value

# Periodic bases have no fitted interval. Keep the Fourier columns in the
# same order as `_brm_apply_hsgp_periodic`: cosines, then sines, without the
# constant harmonic. All sizes, periods and anchors remain graph inputs.
function _rk_ast_hsgp_periodic_basis_graph!(definitions, term, taken)
    body = quote
        harmonics = Float64.(1:modes[1])
        omega2 = vcat(harmonics, harmonics)
        w0 = 2pi / settings[1]
        frequencies = w0 .* harmonics
        angles = Float64.(axis) .* transpose(frequencies)
        basis = hcat(cos.(angles), sin.(angles))
    end
    value = :basis
    if get(term.options, :anchor, nothing) !== nothing
        push!(body.args, :(anchor_angles = frequencies .* settings[2]),
            :(anchor_basis = vcat(cos.(anchor_angles), sin.(anchor_angles))),
            :(anchored = basis .- transpose(anchor_basis)))
        value = :anchored
    end
    if _rk_hsgp_mask(term) !== nothing
        push!(body.args, :(PHI = ifelse.(mask .!= 0, $value, 0.0)))
    else
        push!(body.args, :(PHI = $value))
    end
    push!(body.args, :(return (PHI, omega2)))
    doc = @_rk_definition_doc """
        The periodic HSGP Fourier basis: cosine columns followed by sine columns,
        with no constant harmonic. Returns the basis and each column's harmonic
        index. The period and optional anchor are supplied in `settings`."""
    _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_periodic_basis_graph",
        [:axis, _rk_hsgp_mask_formals(term)..., :modes, :settings], body; kernel=true, doc)
end

# The existing periodic validity-floor bisection as a data-only scan. Keep
# its arithmetic and iteration count identical to `_brm_hsgp_periodic_rho_lower`.
function _rk_ast_hsgp_periodic_floor!(definitions, statements, term, taken, bindings)
    bessel = _rk_value_callee!(bindings, SpecialFunctions.besselix, taken)
    body = quote
        K = modes[1]
        target = $(_BRM_HSGP_WEIGHT_THRESHOLD^-2)
        low_ratio = $bessel(K, 1e-12) / $bessel(1, 1e-12) - target
        high_ratio = $bessel(K, 1e7) / $bessel(1, 1e7) - target
        valid = K <= 1 || (low_ratio < 0 && high_ratio > 0)
        checked_K = valid ? K : error("hsgp: internal periodic validity-floor bracket failed")
        bounds = ReactiveKernels.scan(1:200; init=(log(1e-12), log(1e7))) do bracket, step
            lo = bracket[1]
            hi = bracket[2]
            mid = (lo + hi) / 2
            a = exp(mid)
            below = $bessel(checked_K, a) / $bessel(1, a) - target < 0
            next_lo = below ? mid : lo
            next_hi = below ? hi : mid
            ((next_lo, next_hi), (next_lo + next_hi) / 2)
        end
        floor = checked_K > 1 ? 1 / sqrt(exp(bounds[end])) : 0.0
        return floor
    end
    doc = @_rk_definition_doc """
        The periodic length-scale floor: the scale where the last harmonic's
        spectral amplitude is 1/100 of the first. A single harmonic has floor zero."""
    entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_periodic_floor_graph",
        [:modes], body; kernel=true, doc)
    name = _rk_ast_fresh_name(string(term.options.id, "_rho_floor"), taken)
    push!(statements, :($name = $entry($(term.options.modes))))
    name
end

function _rk_ast_hsgp_weight_cell(::Val{:periodic}, options, D, bindings, taken)
    bessel = _rk_value_callee!(bindings, SpecialFunctions.besselix, taken)
    :(exp(log(sigma) + (log(2) + log($bessel(omega2[b], inv(rho^2)))) / 2))
end

function _rk_ast_hsgp_weight_cell(::Val{:exp_quad}, options, D, bindings, taken)
    frequency(j) = D == 1 ? :(omega2[b]) : :(omega2[b, $j])
    exponent_parts = options.iso ? [:(rho^2 * $(frequency(j))) for j in 1:D] :
        [:(rho[$j]^2 * $(frequency(j))) for j in 1:D]
    exponent = D == 1 ? only(exponent_parts) : Expr(:call, :+, exponent_parts...)
    scale = options.iso ? :(sigma * (rho * sqrt(2pi))^($D / 2)) :
        Expr(:call, :*, :sigma, [:(sqrt(rho[$j] * sqrt(2pi))) for j in 1:D]...)
    :($scale * exp(-0.25 * $exponent))
end

# The spectral-weighted basis product, in two graphs: the per-weight spectral
# scales, which the caller binds to the local `scales` names so that they are
# a value of the program (learned SVI centering reads an HSGP weight's scale
# there, src/nlme_svi_centering.jl), and the basis product of those scales and
# the standardized weights `z`. Returns both calls. `inputs` pairs each graph
# argument with its value: a squared-exponential term passes its axes and the
# graphs compose its basis; a periodic term passes its prepared `PHI` and
# `omega2`. The scales are a vector over the modes, or levels × modes when a
# hyper-predictor scales `sd` or `length_scale` by level.
function _rk_ast_hsgp_value_graph!(definitions, term, taken, inputs, sigma, rho, z, scales, bindings;
        group_index=nothing, multigroup=nothing)
    options = term.options
    D = length(term.columns)
    rho_by_group = any(p -> p.hyper === :length_scale, options.hyper_plans)
    sigma_by_group = any(p -> p.hyper === :sd, options.hyper_plans)
    by_group = group_index !== nothing && (sigma_by_group || rho_by_group)
    # A one-dimensional basis has one frequency per mode, not a one-column matrix.
    weight_cell = _rk_ast_hsgp_weight_cell(Val(get(options, :cov, :exp_quad)),
        options, D, bindings, taken)
    # Mode cells close over `omega2` and the scales, read under their names.
    weight_plate(sigma_value, rho_value) = _rk_ast_graph_plate([:(axes(omega2, 1))], [:b],
        _rk_ast_hsgp_substitute(weight_cell, Dict(:sigma => sigma_value, :rho => rho_value)))
    formals = first.(inputs)
    composed = get(options, :cov, :exp_quad) !== :periodic
    # A per-level projection reads each row's level (the `group_index` below).
    projected = composed && _rk_hsgp_level_projected(options)
    basis = composed ? _rk_ast_hsgp_basis_graph!(definitions, term, taken) : nothing
    basis_statements() = composed ? [Expr(:(=), Expr(:tuple, :PHI, :omega2),
        Expr(:call, basis, formals..., (projected ? (:group_index,) : ())...))] : Expr[]

    # The scales.
    scale_arguments = [formals..., :sigma, :rho]
    projected && push!(scale_arguments, :group_index)
    scale_body = Expr(:block, basis_statements()...)
    if group_index === nothing
        # Hyper-predicted scales arrive as one-element vectors.
        sigma_value, rho_value = :sigma, :rho
        sigma_by_group && (push!(scale_body.args, :(sigma_value = sigma[1])); sigma_value = :sigma_value)
        rho_by_group && (push!(scale_body.args, :(rho_value = rho[1])); rho_value = :rho_value)
        push!(scale_body.args, :(weights = $(weight_plate(sigma_value, rho_value))),
            :(return weights))
    elseif !by_group
        push!(scale_body.args, :(weights = $(weight_plate(:sigma, :rho))), :(return weights))
    else
        frequencies = D == 1 ? :(transpose(omega2)) : :(transpose(vec(sum(omega2; dims=2))))
        push!(scale_body.args, :(frequencies = $frequencies),
            :(scale = sigma .* (rho .* sqrt(2pi)).^($D / 2)),
            :(exponent = (rho .^ 2) .* frequencies),
            :(spectra = scale .* exp.(-0.25 .* exponent)), :(return spectra))
    end
    doc = @_rk_definition_doc """
        The weight of each HSGP basis function: the square root of the covariance's
        spectral density at its frequency, for the marginal scale `sigma` and length
        scale `rho`$(by_group ? " (one column per group)" : "")."""
    scale_entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_spectral_scales",
        scale_arguments, scale_body; kernel=true, doc)
    scale_values = Any[last.(inputs)..., sigma, rho]
    projected && push!(scale_values, group_index)

    # The basis product.
    arguments = [formals..., :scales, :z]
    body = Expr(:block, basis_statements()...)
    # `cor=multigroup(...)`: the group weights of each basis frequency are
    # correlated by its multigroup coherence (src/level_correlation.jl), which
    # reads `rho`; the level-correlation emitter supplies the statement
    # (src/rk_structured.jl).
    zw = :z
    if multigroup !== nothing
        push!(arguments, :rho)
        weights = multigroup.weights
        if Meta.isexpr(weights, :block)
            statements = filter(x -> !(x isa LineNumberNode), weights.args)
            append!(body.args, statements[1:end-1])
            weights = last(statements)
        end
        push!(body.args, :(z_multigroup = $weights))
        zw = :z_multigroup
    end
    if group_index === nothing
        push!(body.args, :(value = PHI * (scales .* z)), :(return value))
    else
        push!(arguments, :group_index)
        push!(body.args, by_group ? :(scaled_z = $zw .* scales) :
            :(scaled_z = $zw .* transpose(scales)))
        # The same row-wise contraction as the scalar sum: gather the
        # original group's mode weights, multiply by the row's basis,
        # then reduce only the mode axis.
        push!(body.args, :(rows = scaled_z[group_index,:]),
            :(value = vec(sum(PHI .* rows; dims=2))), Expr(:return,:value))
    end
    # The call below passes these last, after `group_index`.
    multigroup === nothing || append!(arguments, multigroup.formals)
    # Terms with the same spectral shape share one explicit numerical graph.
    doc = @_rk_definition_doc """
        An HSGP term's value at each row: the basis `PHI` times the basis weights `scales`
        times the standardized coefficients `z`$(group_index === nothing ? ", `PHI * (scales .* z)`" :
            ", each row reading its group's weights")."""
    entry = _rk_ast_shared_definition!(definitions, taken, "brm_hsgp_spectral_graph",
        arguments, body; kernel=true, doc)
    values = Any[last.(inputs)..., scales, z]
    multigroup === nothing || push!(values, rho)
    group_index === nothing || push!(values, group_index)
    multigroup === nothing || append!(values, multigroup.values)
    Expr(:call, scale_entry, scale_values...), Expr(:call, entry, values...)
end
