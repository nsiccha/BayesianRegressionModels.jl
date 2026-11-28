# The emitted ReactiveKernelsPPL program as source text (`rkppl_source`).
#
# Julia's own `Expr` printer writes a documented definition as
# `Core.@doc "…" definition`, keeps a `#= file:line =#` marker in every macro
# call and indents the body of `name(args...) = begin … end` and of a `do`
# block by eight columns. This printer writes the program the way one types it:
# a docstring above its definition, four columns per level, a conditional
# value as `a ? b : c`. An expression without statements inside it is written
# by `_brm_print_source`, the printer of `@brm` source, so operators, calls and
# keywords read the same in a declaration and in its emitted program.
#
# The text parses back to the emitted program (`test/rk_source_text.jl`), up to
# how a chain of sums or products is grouped. The emitter writes the sum of a
# predictor's terms as one call, `.+(a, b, c)`, and so does Julia's printer; the
# text reads `a .+ b .+ c`, which parses nested to the left. A scalar chain the
# emitter built nested, `(a * b) * c`, reads `a * b * c`, which parses as one
# call. Both have the value of the emitted form. Only the text changes: the
# emitted program stays as it is, and with it the program identity that fitted
# states and checkpoints store (`_rk_program_statements`).

"""
    rkppl_source(rk::RKBRMI; part = :all) -> String

The ReactiveKernelsPPL program `rk` runs, as source text.

`RKBRMI` emits one program per model: definitions (the submodels and numerical
graphs its statements call) and a main block with one statement per prior,
value and observation. `rkppl_source` returns that program as it would be
typed:

- each definition under its docstring, a submodel as `@rkppl name(args...) =
  begin … end`, a numerical graph as `ReactiveKernels.@kernel name(args...) =
  begin … end`;
- then the model, `@rkppl begin … end`, with the authored docstring of a
  declaration above its statement.

A sum or product of several terms reads as it is typed, `a .+ b .+ c`.

`part = :main` returns the statements of the main block alone, one per line
and without the enclosing `@rkppl begin … end`; `part = :definitions` returns
the definitions alone.

The text names the program's inputs and the `brm_*` preparation functions; it
does not carry their values. [`rk_bound_data`](@ref) returns the bound inputs
and [`rkppl_model`](@ref) the captured model in the module that binds those
functions. Load ReactiveKernels and ReactiveKernelsPPL first.

```julia
rk = RKBRMI(model)
print(rkppl_source(rk; part = :main))
write("model.jl", rkppl_source(rk))
```
"""
function rkppl_source(backend::RKBRMI; part::Symbol=:all)
    part in (:all, :main, :definitions) || throw(ArgumentError(
        "rkppl_source: `part` is :all, :main or :definitions, not $(repr(part))"))
    program = _rk_retained_program(backend)
    io = IOBuffer()
    if part !== :main
        _rk_print_definitions(io, program.defs)
        part === :all && !isempty(program.defs) && print(io, "\n")
    end
    if part === :all
        println(io, "@rkppl begin")
        _rk_print_statements(io, program.main.args, 4)
        println(io, "end")
    elseif part === :main
        _rk_print_statements(io, program.main.args, 0)
    end
    String(take!(io))
end

function _rk_print_definitions(io::IO, definitions)
    for (i, definition) in enumerate(definitions)
        i == 1 || println(io)
        _rk_print_definition(io, definition)
        println(io)
    end
end

# One definition in its typed spelling: an `@rkppl` submodel (emitted bare, as
# an assignment definition), a `@kernel` graph under the macro it was written
# with, or an ordinary function.
function _rk_print_definition(io::IO, definition)
    while _brm_is_doc(definition)
        _rk_print_docstring(io, definition.args[3], 0; block=true)
        definition = definition.args[4]
    end
    kind = _rk_source_definition(definition).kind
    if Meta.isexpr(definition, :macrocall)
        body = only(filter(arg -> !(arg isa LineNumberNode), definition.args[2:end]))
        kind === :kernel ? (_rk_print_macro_name(io, first(definition.args)); print(io, " ")) :
            print(io, "@rkppl ")
        _rk_print_code(io, body, 0)
    else
        kind === :rkppl && print(io, "@rkppl ")
        _rk_print_code(io, definition, 0)
    end
end

# A macro's name as it is typed, `@plate` or `ReactiveKernels.@kernel`. Julia
# prints the qualified name on its own as `ReactiveKernels.var"@kernel"`.
_rk_print_macro_name(io::IO, name::Symbol) = print(io, name)
_rk_print_macro_name(io::IO, name::GlobalRef) = print(io, nameof(name.mod), ".", name.name)
function _rk_print_macro_name(io::IO, name)
    if Meta.isexpr(name, :., 2) && name.args[2] isa QuoteNode
        Base.show_unquoted(io, name.args[1])
        print(io, ".", name.args[2].value)
    else
        Base.show_unquoted(io, name)
    end
end

# A docstring above its statement. A definition's docstring, and any that
# spans lines, is a triple-quoted block; a one-line docstring of a declaration
# is a string literal on its own line, as `@brm` source prints it.
function _rk_print_docstring(io::IO, markdown, indent::Int; block::Bool=false)
    pad = " "^indent
    if !(markdown isa AbstractString)
        print(io, pad)
        _brm_print_source(io, markdown)
        println(io)
    elseif block || occursin('\n', markdown)
        text = replace(rstrip(markdown), "\\" => "\\\\", "\$" => "\\\$",
            "\"\"\"" => "\\\"\\\"\\\"")
        println(io, pad, "\"\"\"")
        for line in split(text, '\n')
            println(io, isempty(line) ? "" : pad * line)
        end
        println(io, pad, "\"\"\"")
    else
        println(io, pad, repr(markdown))
    end
end

function _rk_print_statements(io::IO, statements, indent::Int)
    for statement in statements
        statement isa LineNumberNode && continue
        _rk_print_statement(io, statement, indent)
        println(io)
    end
end

function _rk_print_statement(io::IO, statement, indent::Int)
    while _brm_is_doc(statement)
        _rk_print_docstring(io, statement.args[3], indent)
        statement = statement.args[4]
    end
    # A bare string before a declaration is its docstring inside a block.
    statement isa AbstractString && return print(io, " "^indent, repr(statement))
    print(io, " "^indent)
    _rk_print_code(io, statement, indent)
end

# Heads that hold statements, or that Julia's printer writes over several
# lines. An expression with none of them inside is written on one line.
const _RK_SOURCE_STRUCTURE = (:block, :do, :if, :elseif, :for, :while, :function, :macrocall,
    :return, :let, :->)
_rk_source_structured(x) = false
_rk_source_structured(x::Expr) = x.head in _RK_SOURCE_STRUCTURE ||
    (x.head !== :quote && any(_rk_source_structured, x.args))

_rk_source_statements(x) = Any[x]
_rk_source_statements(x::Expr) = x.head === :block ?
    Any[a for a in x.args if !(a isa LineNumberNode)] : Any[x]

# The body of a block-shaped construct, one level deeper, then its `end`.
function _rk_print_body(io::IO, body, indent::Int)
    println(io)
    _rk_print_statements(io, _rk_source_statements(body), indent + 4)
    print(io, " "^indent, "end")
end

# `x` as code whose continuation lines are indented relative to `indent`; the
# caller has written the indentation of the first line.
_rk_print_code(io::IO, x, indent::Int, outer::Int=0; left::Bool=false) =
    _brm_print_source(io, x, outer; left)
function _rk_print_code(io::IO, x::Expr, indent::Int, outer::Int=0; left::Bool=false)
    _rk_source_structured(x) || return _brm_print_source(io, x, outer; left)
    head = x.head
    if _brm_is_doc(x)
        # A documented expression in value position keeps its text as written.
        _rk_print_docstring(io, x.args[3], indent)
        print(io, " "^indent)
        _rk_print_code(io, x.args[4], indent)
    elseif head === :block
        print(io, "begin")
        _rk_print_body(io, x, indent)
    elseif head === :do
        call, lambda = x.args
        _rk_print_code(io, call, indent, typemax(Int))
        parameters = lambda.args[1]
        names = Meta.isexpr(parameters, :tuple) ? parameters.args : Any[parameters]
        print(io, " do")
        isempty(names) || print(io, " ")
        for (i, name) in enumerate(names)
            i == 1 || print(io, ", ")
            _brm_print_source(io, name)
        end
        _rk_print_body(io, lambda.args[2], indent)
    elseif head === :if && !any(a -> Meta.isexpr(a, :block), x.args[2:end]) &&
            length(x.args) == 3
        # `a ? b : c`: Julia parses the ternary without block wrappers.
        parens = outer > 0
        parens && print(io, "(")
        _rk_print_code(io, x.args[1], indent, 1)
        print(io, " ? ")
        _rk_print_code(io, x.args[2], indent, 1)
        print(io, " : ")
        _rk_print_code(io, x.args[3], indent, Meta.isexpr(x.args[3], :if) ? 0 : 1)
        parens && print(io, ")")
    elseif head === :if
        _rk_print_conditional(io, x, indent, "if")
        print(io, " "^indent, "end")
    elseif head === :for || head === :while
        print(io, head, " ")
        condition = x.args[1]
        if head === :for && Meta.isexpr(condition, :(=), 2)
            _brm_print_source(io, condition.args[1])
            print(io, " in ")
            _rk_print_code(io, condition.args[2], indent)
        else
            _rk_print_code(io, condition, indent)
        end
        _rk_print_body(io, x.args[2], indent)
    elseif head === :function
        print(io, "function ")
        _rk_print_code(io, x.args[1], indent)
        length(x.args) == 1 ? print(io, " end") : _rk_print_body(io, x.args[2], indent)
    elseif head === :return
        print(io, "return")
        value = only(x.args)
        value === nothing || (print(io, " "); _rk_print_code(io, value, indent))
    elseif head === :macrocall
        _rk_print_macro_name(io, x.args[1])
        for argument in x.args[2:end]
            argument isa LineNumberNode && continue
            argument === nothing && continue
            print(io, " ")
            _rk_print_code(io, argument, indent, typemax(Int))
        end
    elseif head === :(=) && length(x.args) == 2
        _rk_print_code(io, x.args[1], indent)
        print(io, " = ")
        _rk_print_code(io, x.args[2], indent)
    elseif (op = _brm_source_infix(x)) !== nothing
        precedence = Base.operator_precedence(op)
        parens = precedence < outer || (precedence == outer && !left)
        parens && print(io, "(")
        operands = head === :call ? x.args[2:end] : x.args
        for (i, operand) in enumerate(operands)
            i == 1 || print(io, op === :(:) ? ":" : " $op ")
            _rk_print_code(io, operand, indent, precedence; left=i == 1)
        end
        parens && print(io, ")")
    elseif head === :call
        _rk_print_code(io, x.args[1], indent, typemax(Int))
        _rk_print_arguments(io, x.args[2:end], "(", ")", indent)
    elseif _brm_is_broadcast_call(x)
        _rk_print_code(io, x.args[1], indent, typemax(Int))
        _rk_print_arguments(io, x.args[2].args, ".(", ")", indent)
    elseif head === :ref
        _rk_print_code(io, x.args[1], indent, typemax(Int))
        _rk_print_arguments(io, x.args[2:end], "[", "]", indent)
    elseif head === :vect
        _rk_print_arguments(io, x.args, "[", "]", indent)
    elseif head === :tuple
        _rk_print_arguments(io, x.args, "(", length(x.args) == 1 ? ",)" : ")", indent)
    else
        # Any other syntax keeps Julia's own spelling at this indentation.
        Base.show_unquoted(io, _brm_display_source(x, true), indent, -1)
    end
end

# `if … elseif … else` with statement bodies, without the closing `end`.
function _rk_print_conditional(io::IO, x::Expr, indent::Int, keyword)
    print(io, keyword, " ")
    condition = x.args[1]
    # An `elseif` condition is a block holding its line number.
    Meta.isexpr(condition, :block) && (condition = only(_rk_source_statements(condition)))
    _rk_print_code(io, condition, indent)
    println(io)
    _rk_print_statements(io, _rk_source_statements(x.args[2]), indent + 4)
    length(x.args) == 3 || return
    otherwise = x.args[3]
    if Meta.isexpr(otherwise, :elseif)
        print(io, " "^indent)
        _rk_print_conditional(io, otherwise, indent, "elseif")
    else
        println(io, " "^indent, "else")
        _rk_print_statements(io, _rk_source_statements(otherwise), indent + 4)
    end
end

# Call, index and literal arguments. Keywords are written `name=value`, as
# `_brm_print_arguments` writes them, each where the call has it: after the
# semicolon, or among the positional arguments.
function _rk_print_arguments(io::IO, arguments, open, close, indent::Int)
    print(io, open)
    positional = Any[a for a in arguments if !Meta.isexpr(a, :parameters)]
    keywords = Any[]
    for a in arguments
        Meta.isexpr(a, :parameters) && append!(keywords, a.args)
    end
    for (i, argument) in enumerate(positional)
        i == 1 || print(io, ", ")
        _rk_print_argument(io, argument, indent)
    end
    if !isempty(keywords)
        print(io, isempty(positional) ? ";" : "; ")
        for (i, keyword) in enumerate(keywords)
            i == 1 || print(io, ", ")
            _rk_print_argument(io, keyword, indent)
        end
    end
    print(io, close)
end
function _rk_print_argument(io::IO, argument, indent::Int)
    if Meta.isexpr(argument, :kw, 2)
        _brm_print_source(io, argument.args[1])
        print(io, "=")
        _rk_print_code(io, argument.args[2], indent)
    else
        _rk_print_code(io, argument, indent)
    end
end
