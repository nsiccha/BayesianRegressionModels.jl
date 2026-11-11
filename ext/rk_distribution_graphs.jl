# BRM owns the completion geometry and its statistical correction. Base-law
# arithmetic comes only from the public RKPPL distribution graph hook.
_rk_density_argument(argument, index) = :($argument isa Real ? $argument : $argument[$index])

function _rk_density_score(F, arguments::Tuple, value)
    if BRM._brm_lower_family(F)
        isempty(arguments) && return nothing
        lower, base_arguments = first(arguments), Base.tail(arguments)
        base = BRM._brm_lower_base(F)
        object = ReactiveKernelsPPL.sampling_distribution_graph(base, base_arguments)
        object isa Expr || return nothing
        normalizer = BRM._brm_lower_normalized(F) ?
            :($lower > $(BRM._brm_lower_floor(base)) ? $object.logccdf($lower) : 0.0) : 0.0
        return :(isfinite($lower) && $value >= $lower ?
            $object.logpdf($value) - $normalizer : -Inf)
    end
    object = ReactiveKernelsPPL.sampling_distribution_graph(F, arguments)
    object isa Expr || return nothing
    :($object.logpdf($value))
end

function _rk_density_sum(F, arguments::Tuple, value)
    at = Tuple(_rk_density_argument(arg, :j) for arg in arguments)
    score = _rk_density_score(F, at, :($value[j]))
    score === nothing && return nothing
    # Only extents enter the validation helper, so it does not obscure a
    # law argument's elementwise footprint or numerical arithmetic.
    lengths = [:($arg isa Real ? length($value) : length($arg)) for arg in arguments]
    indices = Expr(:call, GlobalRef(BRM, :_brm_density_indices), :(length($value)), lengths...)
    quote
        density_indices = $indices
        density_terms = ReactiveKernels.plate(density_indices) do j
            $score
        end
        sum(density_terms)
    end
end

function _rk_density_definition(entry, inputs, body)
    statements = Meta.isexpr(body, :block) ?
        filter(x -> !(x isa LineNumberNode), body.args) : Any[body]
    statements[end] = Expr(:return, last(statements))
    Expr(:macrocall, Symbol("@kernel"), LineNumberNode(0),
        Expr(:(=), Expr(:call, entry, inputs...), Expr(:block, statements...)))
end

function _rk_density_spec(inputs, body; valid=nothing)
    body === nothing && return nothing
    # The public hook reconstructs this local spec on printed-source replay.
    # Nothing is registered globally, and no argument AST is evaluated here.
    mod = _rk_new_module()
    Core.eval(mod, :(const ReactiveKernels = $ReactiveKernels))
    Core.eval(mod, :(using .ReactiveKernels: @kernel))
    if valid === nothing
        return Core.eval(mod, _rk_density_definition(:brm_owned_density, inputs, body))
    end
    Core.eval(mod, _rk_density_definition(:brm_owned_valid_density, inputs, body))
    call = Expr(:call, :brm_owned_valid_density, inputs...)
    Core.eval(mod, _rk_density_definition(:brm_owned_density, inputs, :($valid ? $call : -Inf)))
end

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{<:BRM._BRMShares{F}}, args::Tuple) where {F}
    length(args) >= 2 || return nothing
    law_args = ntuple(j -> Symbol(:argument_, j), length(args)-2)
    _rk_density_spec((:value, :n, :total, law_args...), _rk_density_sum(F, law_args, :value))
end

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{<:BRM._BRMGroupedShares{F}}, args::Tuple) where {F}
    length(args) >= 2 || return nothing
    law_args = ntuple(j -> Symbol(:argument_, j), length(args)-2)
    _rk_density_spec((:value, :ends, :residuals, law_args...), _rk_density_sum(F, law_args, :value))
end

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{<:BRM._BRMLowerLaw{F,N}}, args::Tuple) where {F,N}
    isempty(args) && return nothing
    law_args = ntuple(j -> Symbol(:argument_, j), length(args)-1)
    _rk_density_spec((:value, :lower, law_args...),
        _rk_density_sum(BRM._BRMLowerLaw{F,N}, (:lower, law_args...), :value))
end

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{<:BRM._BRMLowerShares{F}}, args::Tuple) where {F}
    length(args) >= 3 || return nothing
    law_args = ntuple(j -> Symbol(:argument_, j), length(args)-3)
    density = _rk_density_sum(F, (:lower, law_args...), :value)
    density === nothing && return nothing
    valid = Expr(:call, GlobalRef(BRM, :_brm_lower_shares_valid), :total, :lower, :(length(value)))
    _rk_density_spec((:value, :n, :total, :lower, law_args...), density; valid)
end

function ReactiveKernelsPPL.sampling_distribution_graph(::Type{<:BRM._BRMLowerGroupedShares{F}}, args::Tuple) where {F}
    length(args) >= 3 || return nothing
    law_args = ntuple(j -> Symbol(:argument_, j), length(args)-3)
    density = _rk_density_sum(F, (:lower, law_args...), :value)
    density === nothing && return nothing
    valid = Expr(:call, GlobalRef(BRM, :_brm_lower_grouped_valid), :ends, :residuals, :lower)
    _rk_density_spec((:value, :ends, :residuals, :lower, law_args...), density; valid)
end

function _rk_entry_weight_definition(entry, F, n)
    n < 0 && return nothing
    args = ntuple(j -> Symbol(:argument_, j), n)
    density = _rk_density_score(F, args, :value)
    density === nothing && return nothing
    normal = _rk_density_score(BRM.Normal, (0.0, 1.0), :z)
    _rk_density_definition(entry, (:z, :value, args...), :($density + z - $normal))
end

function _rk_group_weight_definition(entry, F, n)
    n < 0 && return nothing
    args = ntuple(j -> Symbol(:argument_, j), n)
    density = _rk_density_sum(F, args, :value)
    density === nothing && return nothing
    normal = _rk_density_score(BRM.Normal, (0.0, 1.0), :(z[k]))
    jacobian = BRM._brm_lower_family(F) ?
        Expr(:call, GlobalRef(BRM, :_brm_lower_shares_logjac), :z, :total, first(args)) :
        :(sum(log.(value)) - log(total))
    body = quote
        $(density.args[1:end-1]...)
        density = $(last(density.args))
        reference_indices = eachindex(z)
        reference_terms = ReactiveKernels.plate(reference_indices) do k
            $normal
        end
        density + $jacobian - sum(reference_terms)
    end
    inputs = (:z, :value, :total, args...)
    if BRM._brm_lower_family(F)
        valid = Expr(:call, GlobalRef(BRM, :_brm_lower_shares_valid), :total, first(args), :(length(value)))
        helper = Symbol(entry, :_valid_density)
        call = Expr(:call, helper, inputs...)
        return [_rk_density_definition(helper, inputs, body),
            _rk_density_definition(entry, inputs, :($valid ? $call : -Inf))]
    end
    [_rk_density_definition(entry, inputs, body)]
end

# An emitted program may run in an application which does not import the
# distribution package. Bind the public mapping's global leaves explicitly,
# so printing and reparsing the definition preserves the same identities.
function _rk_density_source_bindings!(bindings, entry, definition)
    names = Dict{GlobalRef,Symbol}()
    function bind(node)
        if node isa GlobalRef
            return get!(names, node) do
                name = Symbol(entry, :_graph_leaf_, length(names)+1)
                push!(bindings, name => getfield(node.mod, node.name))
                name
            end
        end
        node isa Expr || return node
        Expr(node.head, map(bind, node.args)...)
    end
    bind(definition)
end

function BRM._rk_callable_source!(definitions, bindings, entry,
        ::BRM._BRMLowerEntryWeight{F,N}) where {F,N}
    definition = _rk_entry_weight_definition(entry, F, N)
    definition === nothing && return nothing
    push!(definitions, _rk_density_source_bindings!(bindings, entry, definition))
    true
end

function BRM._rk_callable_source!(definitions, bindings, entry,
        ::BRM._BRMGroupShareWeight{F,N}) where {F,N}
    source = _rk_group_weight_definition(entry, F, N)
    source === nothing && return nothing
    block = _rk_density_source_bindings!(bindings, entry, Expr(:block, source...))
    append!(definitions, block.args)
    true
end
