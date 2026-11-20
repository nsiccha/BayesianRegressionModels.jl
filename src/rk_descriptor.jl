# src/rk_descriptor.jl — the descriptor of the program an RKBRMI runs.
#
# `brm_descriptor(::SBBRMI)` describes the Stan program StanBlocks generates. An
# `RKBRMI` runs another program, BRM's emitted RK source lowered by RKPPL, and a
# model whose law has no Stan form (an `_rk_observation_source!` family without
# an `@lpxf` twin) has no Stan program at all. This descriptor reads the RK
# program from the backend itself, so its `held_out`, `completions` and
# `car_units` are those of the program that runs, and builds no Stan program.
#
# Nothing here is a second source. The identity hashes the program text and
# callables that the checkpoint identities already hash (`_rk_program_statements`,
# `_brm_stable_bindings`; `_brm_rk_binding_identity` adds the bound data), and
# the inputs are the backend's bound inputs (`rk_bound_data`), classified by the
# plan's own records of what it observes, generates and sizes.

"""
    BRMRKDescriptor

The content identity and input inventory of the program an [`RKBRMI`](@ref)
runs. Construct with [`brm_descriptor`](@ref)`(rk::RKBRMI; name)`.

| field | what it is |
| --- | --- |
| `id` | `"rk:<sha256>"`: the hash of the backend's emitted RK program, its definitions, main statements and binding names (without macro-call places or documentation) and its callables (a generated function enters by what it computes). Independent of `name` and of the data's values: the same program and callables give the same id. It changes with the formula, `held_out`, `completions`, `car_units` or a callable, and when a BRM release emits the model differently |
| `name` | informational label: `name=` at construction, else `:model` |
| `formula` | the canonical rendering of the declaration (`show` of [`brm_source`](@ref)`(rk)`), as on a [`BRMDescriptor`](@ref) |
| `inputs` | `Tuple` of [`BRMInput`](@ref), one per input the RK graph binds ([`rk_bound_data`](@ref)), in that order |
| `columns` | the dataframe columns the declaration reads ([`brm_columns`](@ref)) |
| `held_out` | the responses whose likelihoods the backend withholds. The program does not read them, so they are not among `inputs` |
| `program` | `(; defs, main, bindings)`: a copy of the emitted program `id` hashes |

The fields of each input mean what they mean on a Stan descriptor, read for an
RK program:

- `type`: the element type of the bound value (`:Float64`, `:Int64`, …), or
  the type of a scalar.
- `size`: the bound value's size, `()` for a scalar. An RK program takes its
  sizes from its inputs' values; an extent computed from the data is an input
  of its own, with `derived = true`.
- `constraints` is `(;)` and `inlined` is `false`.
- `observed`: the program conditions on the input. `held_out` is `false`.
- `column`: the dataframe column the input is or copies, or `nothing` for a
  value BRM generates or the program computes.
- `transform`: `nothing` for a dataframe column: the RK graph does the
  preprocessing (`zscale`, group indices, …) itself, so the program reads the
  raw column. A plate cell's port of a response's nested values (`<cell>_input_y`)
  is that response's column too. A fit-role input (`<column>_fit`) is `:fit`:
  a copy of `column` that the program fits its preprocessing from (a grouping's
  levels, a `center`/`zscale` mean and SD), so a replay keeps the fit's
  preprocessing. It holds the column's own values on a first build and the
  training column on a frozen `reprocess` replay. An input BRM generates names
  what it holds: `:levels` (the level table of a factor or `mo()` term, fitted
  when BRM plans the model), `:level_count`, `:prior_constant` or
  `:known_correlation`. A value the program computes from its data (a design
  matrix, a grouping's level table `<g>_fit_levels`, each row's level
  `<g>_level`), which RK preparation evaluates once and binds in place of its
  definition, is `:computed`; on a frozen replay that is the fit's value.

No Stan program is built, so the Stan-only readers (`brm_output`,
`brm_operation`, the coordinate resolvers) have no method for it. The RK
parameters are the backend's ([`rk_restore_draws`](@ref)).
"""
struct BRMRKDescriptor{P}
    id::String
    name::Symbol
    formula::String
    inputs::Tuple{Vararg{BRMInput}}
    columns::Tuple{Vararg{Symbol}}
    held_out::Tuple{Vararg{Symbol}}
    program::P
end

"""
    brm_descriptor(rk::RKBRMI; name=nothing) -> BRMRKDescriptor

The descriptor of the RK program `rk` runs: its content identity `id`, its
`inputs` with their dataframe provenance, and the `columns` and `formula` of
its declaration (see [`BRMRKDescriptor`](@ref)). It reads the backend's own
program, so `held_out`, `completions` and `car_units` are the backend's, and it
builds no Stan program: it also describes a model whose law exists only on RK,
such as a whole-vector `_rk_observation_source!` family without an `@lpxf` twin.
Requires the ReactiveKernels extension, as every `RKBRMI` does.
"""
function brm_descriptor(rk::RKBRMI; name::Union{Nothing,Symbol}=nothing)
    emitted = _rk_retained_program(rk)
    brmi = brm_source(rk)
    inputs = _brm_rk_descriptor_inputs(rk, emitted)
    held_out = _brm_rk_held_out(rk.plan)
    columns = Set{Symbol}(data_columns(brmi))
    raw = _brm_rk_raw_columns(brmi)
    for response in (_rk_observed_names(rk.plan)..., held_out...)
        response in raw && push!(columns, response)
    end
    foreach(i -> isnothing(i.column) || push!(columns, i.column), inputs)
    program = (; defs=deepcopy(emitted.defs), main=deepcopy(emitted.main),
        bindings=copy(emitted.bindings))
    BRMRKDescriptor(_brm_rk_program_id(emitted), something(name, :model),
        _brm_descriptor_formula(brmi), inputs, Tuple(sort!(collect(columns))), held_out,
        program)
end

function _brm_rk_program_id(emitted::_RKEmittedProgram)
    io = IOBuffer()
    serialize(io, (format=1, program=_rk_program_statements(emitted),
        callables=_brm_stable_bindings(emitted.bindings)))
    "rk:" * bytes2hex(sha256(take!(io)))
end

_brm_rk_held_out(_plan) = ()
_brm_rk_held_out(plan::_RKHeldOutPlan) = Tuple(sort!(collect(plan.held_out)))

# The dataframe columns a declaration's operations hold, responses included.
function _brm_rk_raw_columns(brmi::BRMI)
    columns = Dict{Symbol,Any}()
    foreach(op -> _rk_raw_data_columns!(columns, op), _brm_operation_values(brmi))
    union!(Set{Symbol}(keys(columns)), data_columns(brmi))
end

function _brm_rk_descriptor_inputs(rk, emitted)
    plan = rk.plan
    bound = rk_bound_data(rk)
    observed = Set{Symbol}(_rk_observed_names(plan))
    generated = _rk_generated_input_entries(plan)
    extents = Set{Symbol}(first(extent) for extent in _rk_extent_bindings(plan))
    data = union!(_brm_rk_raw_columns(brm_source(rk)), keys(_rk_input_bindings(plan)))
    ports = _brm_rk_port_columns(plan)
    # RK preparation evaluates the program's data-only definitions (a design
    # matrix, group indices) and binds their values in their place; the
    # program's own inputs do not name them.
    source = Set{Symbol}(keys(_rk_source_data_columns(rk, emitted)))
    # A fit-role input (`_rk_fit_twin!`) is a bound copy of the column the
    # program fits its preprocessing from; it names the column it copies.
    fits = _rk_plan_fit_inputs(plan)
    function column(name)
        (haskey(generated, name) || name in extents) && return nothing
        copied = _brm_rk_fit_column(fits, name)
        copied in data ? copied : get(ports, copied, nothing)
    end
    Tuple(_brm_rk_input(name, value; observed=name in observed,
            generated=get(generated, name, nothing), derived=name in extents,
            column=column(name), fit=haskey(fits, name), computed=!(name in source))
        for (name, value) in pairs(bound))
end

# The value a fit-role input copies, followed through a copy of a copy (a
# component's input of its caller's fit-role input); `name` itself otherwise.
function _brm_rk_fit_column(fits, name::Symbol)
    seen = Set{Symbol}()
    while haskey(fits, name) && !(name in seen)
        push!(seen, name)
        name = fits[name]
    end
    name
end

# A plate cell reads a response's nested values through a port of its own
# (`<cell>_input_<column>`); the cell's input records name the column.
_brm_rk_port_columns(_plan) = Dict{Symbol,Symbol}()
_brm_rk_port_columns(plan::_RKHeldOutPlan) = _brm_rk_port_columns(plan.parent)
_brm_rk_port_columns(plan::_RKValuePlan) = Dict{Symbol,Symbol}(
    input.source => input.column
    for assignment in plan.assignments if assignment isa _RKPreparedKernelAssignment
    for input in assignment.inputs
    if input.kind === :element && input.column isa Symbol && input.source !== input.column)

function _brm_rk_input(name, value; observed, generated, derived, column, fit, computed)
    transform = !isnothing(generated) ? generated.kind : fit ? :fit :
        computed ? :computed : nothing
    BRMInput(name, _brm_rk_input_type(value), _brm_rk_input_size(value), (;),
        observed, false, derived, false, column, transform)
end

_brm_rk_input_type(value::AbstractArray) = Symbol(string(eltype(value)))
_brm_rk_input_type(value) = Symbol(string(typeof(value)))
_brm_rk_input_size(value::AbstractArray) = size(value)
_brm_rk_input_size(_value) = ()

brm_columns(d::BRMRKDescriptor) = d.columns

"""
    required_brm_inputs(d::BRMRKDescriptor) -> Vector{Symbol}

The inputs of the RK program that are not extents the data compute: the
neither-`derived`-nor-`inlined` subset of `d.inputs`, as for a
[`BRMDescriptor`](@ref).
"""
required_brm_inputs(d::BRMRKDescriptor) =
    Symbol[i.name for i in d.inputs if !i.derived && !i.inlined]

Base.show(io::IO, d::BRMRKDescriptor) = begin
    print(io, "BRMRKDescriptor `", d.name, "` (id ", d.id, ")\n")
    print(io, "  formula:  ", replace(strip(d.formula), "\n" => "\n            "), "\n")
    print(io, "  columns:  ", join(d.columns, ", "), "\n")
    print(io, "  inputs:   ", join(required_brm_inputs(d), ", "), "\n")
    isempty(d.held_out) || print(io, "  held out: ", join(d.held_out, ", "), "\n")
end
