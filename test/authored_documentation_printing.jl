using Test, BayesianRegressionModels
const DOC_PRINT_BRM = BayesianRegressionModels

@testset "Documented function source is readable and parses without evaluation" begin
    sources = [
        quote
            "Reusable component."
            function component(x)
                "A local slope."
                theta ~ Normal(0, 1)
                @plate for i in eachindex(x)
                    "The cell result."
                    expected[i] = theta * x[i]
                end
                return expected
            end
        end,
        quote
            "A short component."
            component(x) = begin
                "The only declaration."
                result = error("printing must not execute this")
            end
        end,
        quote
            function outer(x)
                "A nested definition."
                function inner(y)
                    "The result with α and a literal \$ sign."
                    result = y + 1
                    return result
                end
                return inner(x)
            end
        end,
    ]
    for source in sources
        source = DOC_PRINT_BRM._brm_normalize_docs(source)
        original = deepcopy(source)
        statement = only(filter(x -> !(x isa LineNumberNode), source.args))
        rendered = sprint(DOC_PRINT_BRM._brm_print_source, statement)
        parsed = Meta.parse("begin\n" * rendered * "\nend")
        @test !occursin("Core.@doc", rendered)
        @test !occursin("#=", rendered)
        @test count(==('\n'), rendered) >= 3
        @test DOC_PRINT_BRM._brm_display_source(
            DOC_PRINT_BRM._brm_normalize_docs(parsed), true) ==
            DOC_PRINT_BRM._brm_display_source(source, true)
        @test source == original
        hidden = sprint(DOC_PRINT_BRM._brm_print_source, statement;
            context=:brm_documentation => false)
        @test DOC_PRINT_BRM._brm_display_source(
            DOC_PRINT_BRM._brm_strip_docs(Meta.parse("begin\n" * hidden * "\nend")), true) ==
            DOC_PRINT_BRM._brm_display_source(DOC_PRINT_BRM._brm_strip_docs(source), true)
        @test !occursin("A local slope.", hidden)
        @test !occursin("The only declaration.", hidden)
        @test !occursin("A nested definition.", hidden)
    end
end
