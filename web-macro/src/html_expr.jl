# Styled HTML rendering for BRMI / VBRMI cards.
#
# Each symbol gets a deterministic color, data columns are bold, parameters
# are italic, and likelihood statements are underlined. The tree walker
# (_html_expr) converts an ExprColumn AST into a nest of <span>s.
#
# Eventually this belongs in an ext of the main package (rendering depends on
# HTMX.jl which is a web-only concern); keeping it in its own file makes that
# future move a single-file relocation.
#
# TODO: HTMX.jl should grow a generic `htmx_node(x)::Node` extension point
# so downstream packages can overload once and have every consumer dispatch
# automatically. For now these card functions are wired in by hand.

# Deterministic HSL color per symbol (golden-ratio spread for visual variety).
_symbol_color(name::Symbol) = "hsl($(mod(hash(name) * 137, 360)), 60%, 40%)"

# A colored <span> with role-based font styling. Data columns are normal
# weight; parameters (latent/sampled) are bold. Per-symbol color is the only
# inline style -- it is data-derived, not presentational.
_styled_name(name::Symbol, role::Symbol) = h.span(string(name);
    class=role == :parameter ? "brm-sym brm-sym-param" : "brm-sym",
    style="color:$(_symbol_color(name))")

_html_expr(x::NamedColumn{<:Any, <:DataColumn})  = _styled_name(name(x), :data)
_html_expr(x::NamedColumn{<:Any, MissingColumn}) = _styled_name(name(x), :parameter)
_html_expr(x::NamedColumn)                       = _styled_name(name(x), :derived)
_html_expr(x::Int)                               = h.span(string(x); class="brm-num")
_html_expr(x::Float64)                           = h.span(string(x); class="brm-num")
_html_expr(x::Number)                            = h.span(string(x); class="brm-num")
_html_expr(x::DataColumn) = h.span("data($(eltype(parent(x))))"; class="brm-muted")
_html_expr(x::MaterializedColumn) = _html_expr(getbroadcast(x))
_html_expr(x::LikelihoodColumn)   = h.span(
    _html_expr(parent(x)), h.span(" .~ "; class="brm-op-dark"), _html_expr(rhs(x)))

# Infix operators: always parenthesized so inner expressions like (1 + b | g1)
# keep their grouping. Top-level callers (_html_brmi_row) use _html_infix
# directly to skip the outermost parens.
_html_expr(x::ExprColumn{<:Union{typeof.((~,*,+,|,doublepipe,assign))...}}) = begin
    h.span("(", _html_infix(x), ")")
end

_html_infix(x::ExprColumn) = begin
    op_str = " $(getop(x)) "
    args = getargs(x)
    parts = Any[]
    for (i, arg) in enumerate(args)
        i > 1 && push!(parts, h.span(op_str; class="brm-op"))
        push!(parts, _html_expr(arg))
    end
    h.span(parts...)
end

# Function-call style: fname(args...; kwargs...)
_html_expr(x::ExprColumn) = begin
    fname = getf(x) isa Function ? nameof(getf(x)) :
            getf(x) isa Type     ? nameof(getf(x)) : string(getf(x))
    args = getargs(x)
    kw = getkwargs(x)
    parts = Any[h.span(string(fname); class="brm-fname"), "("]
    for (i, arg) in enumerate(args)
        i > 1 && push!(parts, ", ")
        push!(parts, _html_expr(arg))
    end
    if length(kw) > 0
        push!(parts, "; ")
        for (i, (k, v)) in enumerate(pairs(kw))
            i > 1 && push!(parts, ", ")
            push!(parts, "$k=", _html_expr(v))
        end
    end
    push!(parts, ")")
    h.span(parts...)
end

# Broadcasted objects (from VBRMI materialization): walk their inner structure
_html_expr(x::Base.Broadcast.Broadcasted) = begin
    fname = x.f isa Function ? nameof(x.f) :
            x.f isa Type     ? nameof(x.f) : string(x.f)
    args = x.args
    if x.f in (+, -, *, /)
        parts = Any[]
        for (i, arg) in enumerate(args)
            i > 1 && push!(parts, h.span(" $(x.f) "; class="brm-op"))
            push!(parts, _html_expr(arg))
        end
        return h.span(parts...)
    end
    parts = Any[h.span(string(fname); class="brm-fname"), "("]
    for (i, arg) in enumerate(args)
        i > 1 && push!(parts, ", ")
        push!(parts, _html_expr(arg))
    end
    push!(parts, ")")
    h.span(parts...)
end

# Arrays / views from block parameter slots: show as a compact shape description
_html_expr(x::SubArray) = h.span("param[$(join(size(x), "×"))]"; class="brm-arr")
_html_expr(x::AbstractVector{<:Number}) = h.span("vec[$(length(x))]"; class="brm-vec")
_html_expr(x::Base.RefValue) = _html_expr(x[])
_html_expr(x::AbstractMatrix) = h.span("mat[$(join(size(x), "×"))]"; class="brm-arr")

_html_expr(x) = h.span(sprint(show, x; context=:compact=>true); class="brm-muted")

function brmi_card(brmi::BRMI)
    rows = [_html_brmi_row(key, parent(value)) for (key, value) in pairs(brmi.operations)]
    h.article(; class="brm-card")(
        h.header(h.strong("BRMI"),
            h.small(" -- $(length(brmi.operations)) operations")),
        h.div(; class="brm-expr-list")(rows...),
    )
end

_leaf_column(x::NamedColumn) = x
_leaf_column(x::ExprColumn) = _leaf_column(getargs(x)[1])
_leaf_column(x) = x

function _html_brmi_row(key, op::ExprColumn{typeof(~)})
    lhs_leaf = _leaf_column(getargs(op)[1])
    is_likelihood = lhs_leaf isa NamedColumn && parent(lhs_leaf) isa DataColumn
    content = _html_infix(op)
    h.div(content; class=is_likelihood ? "brm-likelihood" : "")
end
_html_brmi_row(key, op::ExprColumn{typeof(assign)}) = h.div(_html_infix(op))
_html_brmi_row(key, op) = h.div(
    _styled_name(key, :data),
    h.span(": "; class="brm-num"),
    h.span(sprint(show, op); class="brm-muted"),
)

function vbrmi_card(vbrmi::VBRMI)
    brmi = getfield(vbrmi, :parent)
    (; meta) = vbrmi
    n_dim = LogDensityProblems.dimension(vbrmi)
    n_mat = length(meta.materialized)
    n_blocks = length(meta.blocks)

    mat_rows = [begin
        nc = get(brmi.operations, key, nothing)
        inner = nc !== nothing ? parent(nc) : nothing
        if inner isa DataColumn
            h.div(_styled_name(key, :data),
                h.span(": data($(eltype(parent(inner))))"; class="brm-muted"))
        elseif value isa LikelihoodColumn
            expr = inner !== nothing ? _html_infix(inner) : _styled_name(key, :data)
            h.div(expr; class="brm-likelihood")
        elseif inner !== nothing
            expr = inner isa ExprColumn{<:Union{typeof(~),typeof(assign)}} ?
                _html_infix(inner) : _html_expr(inner)
            shape = "$(eltype(parent(value)))[$(length(parent(value)))]"
            h.div(expr, h.span(" -> $shape"; class="brm-shape"))
        else
            h.div(_styled_name(key, :derived),
                h.span(": $(sprint(show, value))"; class="brm-muted"))
        end
    end for (key, value) in pairs(meta.materialized)]

    blocks_rows = [begin
        role = key === :__population__ ? :derived : :data
        h.div(
            _styled_name(key, role),
            [h.div(; class="brm-block-part")(sprint(show, part))
                for part in parts]...,
        )
    end for (key, parts) in pairs(meta.blocks)]

    h.article(; class="brm-card")(
        h.header(h.strong("VBRMI"),
            h.small(" -- dim $n_dim, $n_mat materialized, $n_blocks blocks")),
        h.h6(; class="brm-subhead")("materialized"),
        h.div(; class="brm-expr-list brm-indent")(mat_rows...),
        h.h6(; class="brm-subhead")("blocks"),
        h.div(; class="brm-expr-list brm-indent")(blocks_rows...),
    )
end
