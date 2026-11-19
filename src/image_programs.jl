# A downstream workload can build models from another package's source.
# Only the package currently being written can own new incremental-image data.
function _brm_image_target()
    (ccall(:jl_generating_output, Cint, ()) == 0 ||
        Base.JLOptions().incremental == 0) && return nothing
    @static if isdefined(Base, :precompilation_target)
        Base.precompilation_target
    elseif isdefined(Base, :precompilation_stack)
        isempty(Base.precompilation_stack) ? nothing : last(Base.precompilation_stack)
    else
        error("BRM: cannot identify the package whose image is being generated")
    end
end

# A short list per image. Keys can include module or function identities;
# comparing those at lookup also works across process hash seeds. Runtime
# misses never add entries. The owner is rooted by Julia's package loader.
struct _BRMImageBuilder
    source::Expr
    code::NamedTuple{(:body, :columns, :source),Tuple{Expr,Expr,Expr}}
end
struct _RKImageDefinition
    digest::String
    bindings::Vector{Pair{Symbol,Any}}
    mod::Module
    inputs::_RKSourceInputs
end
struct _BRMImagePrograms
    builders::Vector{_BRMImageBuilder}
    definitions::Vector{_RKImageDefinition}
end
_BRMImagePrograms() = _BRMImagePrograms(_BRMImageBuilder[], _RKImageDefinition[])

_brm_image_matches(entry::_BRMImageBuilder, source) = _brm_same_syntax(entry.source, source)
_brm_image_matches(entry::_RKImageDefinition, (digest, bindings)) =
    entry.digest == digest && _rk_same_bindings(entry.bindings, bindings)

const _BRM_IMAGE_PROGRAMS_NAME = Symbol("#BayesianRegressionModels_image_programs")
const _BRM_IMAGE_PROGRAMS_LOCK = ReentrantLock()
const _BRM_LOADED_PROGRAMS = _BRMImagePrograms[]

function _brm_image_table(owner)
    isdefined(owner, _BRM_IMAGE_PROGRAMS_NAME) || return nothing
    table = getglobal(owner, _BRM_IMAGE_PROGRAMS_NAME)
    table isa _BRMImagePrograms ? table : nothing
end

function _brm_register_image(owner::Module)
    table = _brm_image_table(owner)
    isnothing(table) && return
    lock(_BRM_IMAGE_PROGRAMS_LOCK) do
        any(t -> t === table, _BRM_LOADED_PROGRAMS) || push!(_BRM_LOADED_PROGRAMS, table)
    end
    nothing
end
_brm_register_image(id::Base.PkgId) = _brm_register_image(Base.root_module(id))

function __init__()
    push!(Base.package_callbacks, _brm_register_image)
    foreach(_brm_register_image, Base.loaded_modules_array())
end

# Runtime opens search only images with entries. A workload writes solely to
# the root module Julia is currently compiling.
function _brm_image_find(field, key)
    target = _brm_image_target()
    owner = isnothing(target) ? nothing : Base.root_module(target)
    lock(_BRM_IMAGE_PROGRAMS_LOCK) do
        tables = isnothing(owner) ? _BRM_LOADED_PROGRAMS : (_brm_image_table(owner),)
        for table in tables
            isnothing(table) && continue
            for entry in getfield(table, field)
                _brm_image_matches(entry, key) && return entry
            end
        end
        nothing
    end
end

function _brm_image_record(field, entry)
    target = _brm_image_target()
    isnothing(target) && return
    owner = Base.root_module(target)
    lock(_BRM_IMAGE_PROGRAMS_LOCK) do
        table = _brm_image_table(owner)
        if isnothing(table)
            isdefined(owner, _BRM_IMAGE_PROGRAMS_NAME) && return
            table = _BRMImagePrograms()
            Core.eval(owner, Expr(:const, Expr(:(=), _BRM_IMAGE_PROGRAMS_NAME, QuoteNode(table))))
        end
        push!(getfield(table, field), entry)
    end
    nothing
end

# A source block may interpolate arbitrary Julia objects, including data arrays
# and closures with captures. A model supplied as plain Julia syntax needs none
# of those. Use the ordinary construction path for other objects.
_brm_image_syntax(x) = false
_brm_image_syntax(::Union{Symbol,String,Char,Bool,Nothing,LineNumberNode,
    Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,
    Float16,Float32,Float64}) = true
_brm_image_syntax(x::GlobalRef) = x.mod === Core || x.mod === Base || x.mod === (@__MODULE__)
_brm_image_syntax(x::QuoteNode) = _brm_image_syntax(x.value)
_brm_image_syntax(x::Expr) = all(_brm_image_syntax, x.args)

# Expr equality considers a literal 1 and 1.0 equal. A called Julia method can
# dispatch on that difference. The image key preserves exact leaf identities.
_brm_same_syntax(x, y) = x === y
_brm_same_syntax(x::QuoteNode, y::QuoteNode) = _brm_same_syntax(x.value, y.value)
_brm_same_syntax(x::Expr, y::Expr) = x.head === y.head && _brm_same_arguments(x.args, y.args)
_brm_same_arguments(xs, ys) = length(xs) == length(ys) &&
    all(i -> _brm_same_syntax(xs[i], ys[i]), eachindex(xs))
