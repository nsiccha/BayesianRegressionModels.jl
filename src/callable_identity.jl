# src/callable_identity.jl — what a callable contributes to a model's identity.
#
# Serialization is a transport format, not a fingerprint. For a generated
# function (anonymous, a closure, a local or keyword-body function) it writes
# what names that function in this process. In `Main` that is the whole type:
# the generated name, the file and line of its methods and process-wide object
# numbers. Elsewhere it is the module and generated name, without the body. So
# the same model has another digest after one more anonymous definition, on
# another line or once another anonymous function has been serialized, while a
# changed body in a package keeps its digest.
#
# Here a value holding a generated function enters an identity through what the
# function computes: its methods' signatures and lowered code, and its captured
# values through ordinary serialization. Everything else serializes unchanged,
# so an identity of a model without generated functions keeps its bytes.

mutable struct _BRMCallableSerializer{I<:IO} <: Serialization.AbstractSerializer
    io::I
    counter::Int
    table::IdDict{Any,Any}
    pending_refs::Vector{Int}
    # The generated function types met so far, each by its first-visit position,
    # so a repeated or recursive reference ends in that position.
    generated::IdDict{Core.TypeName,Int}
end
_BRMCallableSerializer(io) = _BRMCallableSerializer(io, 0, IdDict{Any,Any}(), Int[],
    IdDict{Core.TypeName,Int}())

# A named function's type is `#name`, its instance bound to `name`. Every other
# hidden function type has a generated name; a keyword body such as `##f#3` is
# bound too, under a name that is itself generated.
function _brm_generated_function_type(@nospecialize(t::DataType))
    t <: Function || return false
    name = String(t.name.name)
    startswith(name, '#') || return false
    bound = chop(name; head=1, tail=0)
    '#' in bound && return true
    !(isdefined(t.name.module, Symbol(bound)) && getglobal(t.name.module, Symbol(bound)) isa t)
end

# Without line information and local names, which lowered code keeps beside `code`.
_brm_method_form(m::Method) = (m.sig, m.nargs, m.isva,
    m.source === nothing ? nothing : Base.uncompressed_ast(m).code)

function Serialization.serialize_type_data(s::_BRMCallableSerializer, @nospecialize(t::DataType))
    _brm_generated_function_type(t) ||
        return invoke(Serialization.serialize_type_data, Tuple{Any,DataType}, s, t)
    name = t.name
    if haskey(s.generated, name)
        serialize(s, s.generated[name])
    else
        s.generated[name] = length(s.generated)
        serialize(s, map(_brm_method_form, collect(Base.MethodList(name.mt))))
        serialize(s, map(_brm_method_form, collect(methods(Core.kwcall, (Any, name.wrapper, Vararg)))))
    end
    serialize(s, t.parameters)
end

# Lowered code reaches a nested generated function through its generated global.
function Serialization.serialize(s::_BRMCallableSerializer, g::GlobalRef)
    if startswith(String(g.name), '#') && isdefined(g.mod, g.name)
        value = getglobal(g.mod, g.name)
        t = value isa Type ? Base.unwrap_unionall(value) : typeof(value)
        t isa DataType && _brm_generated_function_type(t) && return serialize(s, value)
    end
    invoke(serialize, Tuple{Serialization.AbstractSerializer,GlobalRef}, s, g)
end

_brm_value_digest(x) = (io = IOBuffer(); serialize(io, x); bytes2hex(sha256(take!(io))))

# `x` itself, or a digest of what it computes when it holds a generated function.
function _brm_stable_value(x)
    io = IOBuffer()
    s = _BRMCallableSerializer(io)
    serialize(s, x)
    isempty(s.generated) ? x : "generated-callable:" * bytes2hex(sha256(take!(io)))
end

_brm_stable_bindings(bindings::Vector{Pair{Symbol,Any}}) =
    Pair{Symbol,Any}[name => _brm_stable_value(value) for (name, value) in bindings]

# A model's declaration as source text. Outside a plate cell a callable prints as
# what names it in this process (`Normal(#3(mu), 0.8)`), which a rebuilt model
# does not reproduce. What it computes is the bindings' part of an identity.
_brm_stable_declaration(text::AbstractString) =
    replace(text, r"#[^\s()\[\],;.\"]*\d" => "#generated")
_brm_declaration_text(x) =
    _brm_stable_declaration(sprint(show, x; context=:brm_documentation => false))
