# The binding reader itself depends only on Julia's Docs. Include the exact
# producer source to test macro expansion without loading numerical backends.
module AuthoredBindingDocumentation
using Test
include(joinpath(@__DIR__, "..", "src", "authored_documentation.jl"))

macro documented_value(name, value)
    previous = gensym(:previous_documentation)
    source = QuoteNode(Expr(:(=), name, value))
    esc(quote
        local $previous = $_brm_binding_docstring($__module__, $(QuoteNode(name)))
        Core.@__doc__ $name = (; value=$value, documentation=())
        $name = merge($name, (; documentation=$_brm_binding_documentation(
            $__module__, $(QuoteNode(name)), $source; previous=$previous)))
    end)
end

"""**Original header.**

Keep α, `x`, and a literal \$ unchanged.
"""
@documented_value documented 1
const original = documented

# A new undocumented definition does not borrow the preceding binding's docs.
@documented_value documented 2
const replacement = documented

module UndocumentedScope end

@testset "Julia outer-macro docs retain raw text and definition identity" begin
    doc = only(original.documentation)
    @test doc.path == (:documented,)
    @test doc.owner === :documented
    @test doc.markdown == "**Original header.**\n\nKeep α, `x`, and a literal \$ unchanged.\n"
    @test doc.file === Symbol(@__FILE__)
    @test doc.line > 0
    @test doc.statement == :(documented = 1)
    @test isempty(replacement.documentation)
    @test original.value == 1
    @test replacement.value == 2
    @test _brm_binding_docstring(UndocumentedScope, :absent) === nothing
    @test Base.Docs.meta(UndocumentedScope; autoinit=false) === nothing
    emitted = :(result = value + 1)
    wrapped = _brm_documented_statement(emitted, original.documentation)
    @test _brm_strip_normalized_docs(wrapped) == emitted
    @test wrapped.args[3] == doc.markdown
    @test wrapped.args[2] == LineNumberNode(doc.line, doc.file)
    @test wrapped.args[4] !== emitted
end

@testset "A component's root documentation has its definition name" begin
    doc = BRMDocumentation((), :piece, "Component description.",
        :(function piece(x); x; end), nothing, 0)
    artifact = (; meta=(; documentation=(doc,)))
    @test brm_documentation_markdown(artifact) == "### `piece`\n\nComponent description.\n\n"
end
end
