# test/vector_parameters.jl — declared vector parameters, `x[1:n] .~ Family.(args...)`,
# and the classification and RK target of `x ~ MvNormal(...)`.
#
# Run: julia --project=test test/vector_parameters.jl
#
# Decision `19ssmud` (2026-10-25): a sized vector parameter with an elementwise prior is
# declared with RKPPL's broadcast spelling, `phi[1:n] .~ Normal.(0.0, 0.5; lower=0.0,
# upper=2.0)`: `n` elements, each drawn from the scalar law, which is what Julia's
# broadcasting means. Before it the only spelling was a factor regression
# (`phi ~ 0 + factor(g)` + `effect(phi, g) ~ ...`), `x ~ MvNormal(...)` (unbounded Normal
# only) was classified as an omitted response on SBBRMI, and RKBRMI refused every vector
# parameter. Each case checks the Stan declaration, the plan role, the printed source,
# an independent density and gradient on RKBRMI, and RKBRMI against compiled Stan.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Logging: Warn

# The one Stan declaration of `name` (a whole-program substring could match another one).
function stan_declaration(code, name)
    lines = filter(l -> occursin(Regex("^\\s*vector[<\\[].*\\]\\s+$name;"), l), split(code, '\n'))
    only(strip.(lines))
end
roles(sb) = [(d.target, d.role, d.dimension) for d in generative_plan(sb).declarations]
reparsed(brmi, data) = Base.invokelatest(
    Base.invokelatest(Core.eval, @__MODULE__, BRM._brm(string(brmi))), data)

function check_vector_case(builder, data, label, oracle; declaration, statement, rk_source)
    before = deepcopy(data)
    brmi = builder(data)
    sb = @test_logs min_level=Warn SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test stan_declaration(code, declaration.name) == declaration.text
    @test occursin(statement, code)
    @test (declaration.name, :prior, (Symbol(declaration.name, :_n),)) in roles(sb)
    @test !occursin("$(declaration.name)_gen", code)
    # The model prints as written and re-parses to the same program.
    @test BRM.stan_code(SBBRMI(reparsed(brmi, data); mod=@__MODULE__)) == code
    backend, problem = consumer_problem(brmi)
    @test occursin(rk_source, sprint(Base.show_unquoted, BRM._rk_emit_ast(backend.plan).main))
    stan = consumer_stan(brmi, "vector-parameters-" * label)
    names = coordinate_names(backend.model.layout)
    mapping = [name => string(name) for name in names]
    n = length(names)
    for u in (zeros(n), fill(0.3, n), collect(range(-0.5, 0.4; length=n)))
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    @test isequal(data, before)
    brmi
end

const vp_data = (; y=[0.3, -0.2, 0.5, 0.1], n=4, ws=[0.5, 1.0, 1.5, 2.0])
const vp_cov = [1.0 0.5 0.0 0.0; 0.5 1.0 0.0 0.0; 0.0 0.0 1.0 0.2; 0.0 0.0 0.2 1.0]

const vp_bounded = @brm begin
    phi[1:n] .~ Normal.(0.0, 0.5; lower = 0.0, upper = 2.0)
    y ~ Normal(phi, 1.0)
end
const vp_elementwise = @brm begin
    w[1:length(ws)] .~ Normal.(0.0, ws; lower = 0.0)
    y ~ Normal(w, 1.0)
end
const vp_hierarchical = @brm begin
    tau ~ Exponential(1.0)
    z[1:4] .~ Normal.(-0.5, tau)
    y ~ Normal(z, 0.5)
end
const vp_truncated = @brm begin
    s[1:n] .~ truncated.(Normal.(0.0, 1.0), 0.0, Inf)
    y ~ Normal(0.0, s)
end
const vp_mvnormal = @brm begin
    rho ~ MvNormal(fill(-0.5, 4), 1.0)
    y ~ Normal(rho, 1.0)
end
const vp_mvnormal_cov = @brm begin
    rho ~ MvNormal(zeros(4), [1.0 0.5 0.0 0.0; 0.5 1.0 0.0 0.0; 0.0 0.0 1.0 0.2; 0.0 0.0 0.2 1.0])
    y ~ Normal(rho, 1.0)
end

likelihood(mu, sigma) = sum(logpdf.(Normal.(mu, sigma), vp_data.y))

@stestset "bounded declared vector" begin
    oracle(u) = begin
        phi = uniform_value.(u, 0.0, 2.0)
        sum(logpdf.(Normal(0.0, 0.5), phi)) + likelihood(phi, 1.0) +
            sum(uniform_logjac.(u, 0.0, 2.0))
    end
    brmi = check_vector_case(vp_bounded, vp_data, "bounded", oracle;
        declaration=(; name=:phi, text="vector<lower=0.0, upper=2.0>[phi_n] phi;"),
        statement="phi ~ normal(0.0, 0.5);",
        rk_source="phi[1:4] .~ restricted.(Normal.(0.0, 0.5), 0.0, 2.0)")
    @test string(brmi) == "phi[1:n] .~ Normal.(0.0, 0.5; lower=0.0, upper=2.0)\ny ~ Normal(phi, 1.0)"
    # A declaration is model structure, like a scalar parameter's.
    @test occursin("phi[1:n] .~", string(structure_of(brmi)))
    sb = SBBRMI(brmi; mod=@__MODULE__)
    ids = Set(c.id for c in brm_description(sb).components)
    @test (:parameter, :phi) in ids && (:observation, :phi) ∉ ids
    @test brm_output(brm_descriptor(sb), :phi).dims == (4,)
    # The declaration is described: one element law shared by every coordinate.
    description = brm_description(sb)
    @test description.complete
    @test occursin("Each coordinate of this vector independently follows the same " *
                   "separately described prior", BRM.brm_description_markdown(description))
    @test occursin("Jede Koordinate dieses Vektors folgt unabhängig", BRM.brm_description_markdown(
        brm_description(sb; texts=BRM.description_texts(:de))))
end

@stestset "data-vector argument applies elementwise" begin
    oracle(u) = begin
        w = exp.(u)
        sum(logpdf.(Normal.(0.0, vp_data.ws), w)) + likelihood(w, 1.0) + sum(u)
    end
    check_vector_case(vp_elementwise, vp_data, "elementwise", oracle;
        declaration=(; name=:w, text="vector<lower=0.0>[w_n] w;"),
        statement="w ~ normal(0.0, ws);",
        rk_source="w[1:4] .~ restricted.(Normal.(0.0, ws), 0.0, Inf)")
    # A data-only expression is evaluated at build time and shipped as data.
    computed = @brm begin
        w[1:n] .~ Normal.(fill(0.5, n), ws ./ 2)
        y ~ Normal(w, 1.0)
    end
    code = BRM.stan_code(SBBRMI(computed(vp_data); mod=@__MODULE__))
    @test occursin("w ~ normal(w_arg1, w_arg2);", code)
end

@stestset "sampled hyperparameter and literal length" begin
    oracle(u) = begin
        tau, z = exp(u[1]), u[2:end]
        logpdf(Exponential(1.0), tau) + sum(logpdf.(Normal(-0.5, tau), z)) +
            likelihood(z, 0.5) + u[1]
    end
    check_vector_case(vp_hierarchical, vp_data, "hierarchical", oracle;
        declaration=(; name=:z, text="vector[z_n] z;"),
        statement="z ~ normal(-0.5, tau);",
        rk_source="z[1:4] .~ Normal.(-0.5, tau)")
end

@stestset "normalized truncation as the element law" begin
    oracle(u) = begin
        s = exp.(u)
        sum(logpdf.(truncated(Normal(0.0, 1.0), 0.0, Inf), s)) + likelihood(0.0, s) + sum(u)
    end
    check_vector_case(vp_truncated, vp_data, "truncated", oracle;
        declaration=(; name=:s, text="vector<lower=0.0>[s_n] s;"),
        statement="s ~ lower_conditioning_normal(0.0, 0.0, 1.0);",
        rk_source="s[1:4] .~ HalfNormal.(1.0)")
end

@stestset "MvNormal vector parameters are priors and build on RKBRMI" begin
    oracle_iso(u) = logpdf(MvNormal(fill(-0.5, 4), 1.0), u) + likelihood(u, 1.0)
    check_vector_case(vp_mvnormal, vp_data, "mvnormal", oracle_iso;
        declaration=(; name=:rho, text="vector[rho_n] rho;"),
        statement="rho ~ normal(rho_mu, 1.0);",
        rk_source="rho[1:4] .~ Normal.([-0.5, -0.5, -0.5, -0.5], 1.0)")
    oracle_cov(u) = logpdf(MvNormal(zeros(4), vp_cov), u) + likelihood(u, 1.0)
    check_vector_case(vp_mvnormal_cov, vp_data, "mvnormal-cov", oracle_cov;
        declaration=(; name=:rho, text="vector[rho_n] rho;"),
        statement="rho ~ multi_normal(rho_mu, rho_scale);",
        rk_source="rho[1:4] ~ MvNormal([0.0, 0.0, 0.0, 0.0], [1.0 0.5 0.0 0.0;")
end

# A per-section vector read by position in a plate cell: the shape the factor
# regression `phi ~ 0 + factor(section)` + `effect(phi, section) ~ ...` used to stand in for.
@stestset "plate cells read a declared vector by position" begin
    sections = (; total=[9.0, 11.5, 10.2], g=[1.0, 1.2, 0.9])
    model = @brm begin
        phi[1:length(total)] .~ Normal.(0.0, 0.5; lower = 0.0, upper = 2.0)
        @plate for s in eachindex(total)
            total[s] ~ Normal(10.0 * g[s], phi[s])
        end
    end
    oracle(u) = begin
        phi = uniform_value.(u, 0.0, 2.0)
        sum(logpdf.(Normal(0.0, 0.5), phi)) +
            sum(logpdf.(Normal.(10.0 .* sections.g, phi), sections.total)) +
            sum(uniform_logjac.(u, 0.0, 2.0))
    end
    brmi = model(sections)
    sb = @test_logs min_level=Warn SBBRMI(brmi; mod=@__MODULE__)
    @test (:phi, :prior, (:phi_n,)) in roles(sb)
    backend, problem = consumer_problem(brmi)
    stan = consumer_stan(brmi, "vector-parameters-plate")
    names = coordinate_names(backend.model.layout)
    mapping = [name => string(name) for name in names]
    for u in (zeros(3), fill(0.3, 3), [-0.5, 0.0, 0.4])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
end

@stestset "declaration errors" begin
    # refused: a data-vector argument must have the declared length; the vector is
    # `n` elements, each with one element of the argument.
    short = @brm begin
        w[1:3] .~ Normal.(0.0, ws)
        y ~ Normal(sum(w), 1.0)
    end
    @test_throws r"argument `ws` has 4 elements, not 3" SBBRMI(short(vp_data); mod=@__MODULE__)
    @test_throws r"argument `ws` has 4 elements, not 3" RKBRMI(short(vp_data))
    # refused: a typed left-hand side is Stan syntax; the user chose Distributions-style
    # spellings over it (decisions `1c18vfn`, `187g4va`, `19ssmud`).
    @test_throws r"typed left-hand side `rho::vector\[3\]` is not @brm syntax.*rho\[1:n\] \.~" Core.eval(
        @__MODULE__, BRM._brm(:(begin; rho::vector[3] ~ Normal(0.0, 1.0); y ~ Normal(rho, 1.0); end)))
end

# The documented omitted-response rule is unchanged for an undeclared name: a
# statement reading a data vector stays an observation (held-out-and-prior-draws.md,
# "Which unbound statements are observations").
@stestset "an undeclared name over a data vector stays an omitted response" begin
    undeclared = @brm begin
        a ~ Normal(0.0, ws; lower = 0.0)
        y ~ Normal(sum(a), 1.0)
    end
    sb = @test_logs (:warn, r"^sbimpl: `a` bind\(s\) no data column") SBBRMI(
        undeclared(vp_data); mod=@__MODULE__)
    @test (:a, :observation, ()) in roles(sb)
end
