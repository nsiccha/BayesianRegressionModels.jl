using Test, BayesianRegressionModels, Distributions, Serialization
const BRM = BayesianRegressionModels
const DOC_DATA = (; x=[-0.4, 0.2, 0.8], y=[0.1, 0.3, 0.7])
const DOC_SOURCE = quote
    """**Expected response** with a slope and intercept.

    The author supplies this Markdown: α, `x`, and a literal \$ sign.
    """
    mu ~ 1 + x
    "Observation scale."
    sigma ~ Exponential(1)
    "Measured response."
    y ~ Normal(mu, sigma)
    "Intercept prior."
    effect(mu, Intercept) ~ Normal(0, 2)
end
const DOC_CELL_SOURCE = quote
    "Global slope."
    theta ~ Normal(0, 1)
    @plate for i in eachindex(x)
        "Cell mean, not a global parameter."
        mean_i = theta * x[i]
        "Predicted value."
        expected[i] = mean_i
        "Cell observation."
        y[i] ~ Normal(mean_i, 1)
    end
end
nodocs(x) = sprint(show, x; context=:brm_documentation => false)
doc_signature(x) = [(d.path, d.owner, d.markdown) for d in brm_documentation(x)]

@testset "Authored documentation is metadata, never documentation registration" begin
    b = brm(DOC_SOURCE; mod=@__MODULE__)
    plain = brm(BRM._brm_strip_docs(DOC_SOURCE); mod=@__MODULE__)
    m, p = b(DOC_DATA), plain(DOC_DATA)
    @test length(brm_documentation(b)) == 4
    @test doc_signature(b) == doc_signature(m)
    @test brm_columns(b) == brm_columns(plain)
    @test nodocs(m) == nodocs(p)
    @test all(d -> d.line > 0, brm_documentation(m))
    @test all(d -> d.file !== nothing, brm_documentation(m))
    @test !isdefined(@__MODULE__, :mu)
    @test !isdefined(@__MODULE__, :sigma)
    @test occursin("**Expected response**", string(m))
    @test occursin("α", brm_documentation_markdown(m))
    for source in (string(b), string(m))
        replay = brm(source; mod=@__MODULE__)(DOC_DATA)
        @test doc_signature(replay) == doc_signature(m)
        @test nodocs(replay) == nodocs(m)
    end
    docs = brm_documentation(m)
    empty!(docs[1].statement.args)
    @test !isempty(first(brm_documentation(m)).statement.args)
    io = IOBuffer(); serialize(io, m); seekstart(io)
    @test doc_signature(deserialize(io)) == doc_signature(m)
    @test doc_signature(deepcopy(m)) == doc_signature(m)

    macro_model = @brm DOC_DATA begin
        "Mean value."
        mu ~ 1 + x
        y ~ Normal(mu, 1)
    end
    @test only(brm_documentation(macro_model; path=:mu)).markdown == "Mean value."
    @test nodocs(macro_model) == nodocs(brm(BRM._brm_strip_docs(quote
        mu ~ 1 + x
        y ~ Normal(mu, 1)
    end); mod=@__MODULE__)(DOC_DATA))
end

@testset "Tuple expansion, prior partition, merge and replay" begin
    b = brm(quote
        "Shared predictor explanation."
        (log(q), log(r)) ~ 1 + x
        "Slope prior."
        effect(q, x) ~ Normal(0, 1)
    end; mod=@__MODULE__)
    m = b(DOC_DATA)
    @test [d.path for d in brm_documentation(m)][1:2] == [(:q,), (:r,)]
    @test length(brm_documentation(priors_of(b))) == 1
    @test length(brm_documentation(structure_of(b))) == 2
    @test doc_signature(merge(structure_of(m), priors_of(m))) == doc_signature(m)
    @test doc_signature(merge(structure_of(b), priors_of(b))(DOC_DATA)) == doc_signature(m)
    @test doc_signature(BRM._brm_rebind_brmi(m, (; x=[1.0, 2.0]))) == doc_signature(m)
    replacement = brm(quote
        "Replacement predictor."
        log(q) ~ 1
    end; mod=@__MODULE__)(DOC_DATA)
    @test only(brm_documentation(merge(m, replacement); path=:q)).markdown == "Replacement predictor."
    bare = brm(:(log(q) ~ 1); mod=@__MODULE__)(DOC_DATA)
    @test isempty(brm_documentation(merge(m, bare); path=:q))
    read = brm(:(y ~ Normal(q, 1)); mod=@__MODULE__)(DOC_DATA)
    @test only(brm_documentation(merge(m, read); path=:q)).markdown == "Shared predictor explanation."
    @test only(brm_documentation(brm(string(m); mod=@__MODULE__)(DOC_DATA); path=:q)).markdown == "Shared predictor explanation."
end

@testset "Plate-local declarations retain their scope and source" begin
    b = brm(DOC_CELL_SOURCE; mod=@__MODULE__)
    m = b(DOC_DATA)
    @test [d.path for d in brm_documentation(m)] ==
        [(:theta,), (:expected, :mean_i), (:expected, :expected), (:expected, :y)]
    @test doc_signature(brm(string(m); mod=@__MODULE__)(DOC_DATA)) == doc_signature(m)
    @test doc_signature(brm(string(b); mod=@__MODULE__)(DOC_DATA)) == doc_signature(m)
    @test !occursin("@doc", nodocs(m))

    grouped_source = quote
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            "Grouped expected values."
            expected[i] = sum(obs_time[obs[i]])
        end
    end
    grouped_data = (; subject=[1, 2], obs_subject=[1, 1, 2], obs_time=[1.0, 2.0, 3.0])
    grouped = brm(grouped_source; mod=@__MODULE__)(grouped_data)
    @test occursin("groups(", string(grouped))
    @test doc_signature(brm(string(grouped); mod=@__MODULE__)(grouped_data)) == doc_signature(grouped)

    scan_source = quote
        @plate for i in eachindex(y)
            @scan begin
                "Initial latent state."
                h[1] ~ Normal(0, 1)
                for t in 2:T
                    "Later latent states."
                    h[t] ~ Normal(h[t - 1], 1)
                end
            end
            y[i] ~ Normal(h, 1)
        end
    end
    scan_data = (; y=[[0.1, 0.2], [0.3, 0.4]], T=2)
    scanned = brm(scan_source; mod=@__MODULE__)(scan_data)
    @test length(brm_documentation(scanned)) == 2
    @test doc_signature(brm(string(scanned); mod=@__MODULE__)(scan_data)) == doc_signature(scanned)
end

# The producer artifact is available without loading the RK package extension.
# Documentation is retained beside the identical executable source and bindings.
@testset "RK artifacts retain authored records without numerical source changes" begin
    model = brm(DOC_SOURCE; mod=@__MODULE__)(DOC_DATA)
    plain = brm(BRM._brm_strip_docs(DOC_SOURCE); mod=@__MODULE__)(DOC_DATA)
    artifact = BRM.emit_rk_artifact(model; case_id="authored-documentation")
    control = BRM.emit_rk_artifact(plain; case_id="authored-documentation-control")
    @test BRM._brm_strip_docs(artifact.ast) == control.ast
    @test artifact.defs == control.defs
    @test artifact.bindings == control.bindings
    @test isequal(rk_artifact_inputs(artifact), rk_artifact_inputs(control))
    @test doc_signature(artifact) == doc_signature(model)
    @test isempty(brm_documentation(control))
    mktemp() do path, io
        close(io)
        BRM.write_rk_artifact(path, artifact)
        @test doc_signature(BRM.read_rk_artifact(path)) == doc_signature(model)
    end
end

# Optional export acceptance, matching the existing description_tex.jl mode.
# Quarto and Poppler are required when requested; renderer failures propagate.
if "--render" in ARGS
    @testset "Authored source and Markdown survive HTML/PDF export" begin
        output = get(() -> mktempdir(), ENV, "BRM_AUTHORED_DOC_OUTPUT")
        mkpath(output)
        model = brm(DOC_SOURCE; mod=@__MODULE__)(DOC_DATA)
        cell = brm(DOC_CELL_SOURCE; mod=@__MODULE__)(DOC_DATA)
        report = brm_description_markdown(brm_description(brm_descriptor(SBBRMI(model))))
        header = raw"""
        ---
        title: Authored declaration documentation
        format:
          html:
            embed-resources: true
            code-overflow: wrap
          pdf:
            pdf-engine: lualatex
            mainfont: DejaVu Serif
            monofont: DejaVu Sans Mono
            include-in-header:
              text: |
                \usepackage{fvextra}
                \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\}}
        ---

        """
        document = header * report *
            "\n# Cell documentation\n\n" * brm_documentation_markdown(cell) *
            "\n# Authored model source\n\n```julia\n" * string(model) *
            "\n```\n\n# Authored cell source\n\n```julia\n" * string(cell) * "\n```\n"
        input = joinpath(output, "authored-documentation.qmd")
        write(input, document)
        run(`quarto render $input --to html`)
        run(`quarto render $input --to pdf`)
        html = read(joinpath(output, "authored-documentation.html"), String)
        pdf = joinpath(output, "authored-documentation.pdf")
        pdftext = read(`pdftotext $pdf -`, String)
        @test filesize(pdf) > 0
        for text in ("Expected response", "Cell observation.", "Predicted value.")
            @test occursin(text, html)
            @test occursin(text, pdftext)
        end
        @test occursin("α", pdftext)
        @test !occursin('\ufffd', pdftext)
    end
end

@testset "Executable Stan, descriptor identity and report rendering" begin
    m = brm(DOC_SOURCE; mod=@__MODULE__)(DOC_DATA)
    p = brm(BRM._brm_strip_docs(DOC_SOURCE); mod=@__MODULE__)(DOC_DATA)
    sb, control = SBBRMI(m; mod=@__MODULE__), SBBRMI(p; mod=@__MODULE__)
    @test stan_code(sb) == stan_code(control)
    @test isequal(stan_data(sb), stan_data(control))
    d, control_d = brm_descriptor(sb), brm_descriptor(control)
    @test d.id == control_d.id
    @test occursin("**Expected response**", d.formula)
    @test nodocs(brm(d.formula; mod=@__MODULE__)(DOC_DATA)) == control_d.formula
    @test doc_signature(sb) == doc_signature(m)
    @test doc_signature(d) == doc_signature(m)
    report = brm_description(d)
    @test doc_signature(report) == doc_signature(m)
    @test report.complete == brm_description(control_d).complete
    @test report.model_id == control_d.id
    @test occursin("**Expected response**", brm_description_markdown(report))
    german = brm_description_markdown(brm_description(d; texts=description_texts(:de)))
    @test occursin("Dokumentation der Deklarationen", german)
    @test occursin("**Expected response**", german)
end
