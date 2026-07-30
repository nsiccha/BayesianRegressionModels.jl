# test/simplex_priors.jl — simplex-valued parameter priors (`s ~ Dirichlet(...)`).
#
# Run: julia --project=. test/simplex_priors.jl
# Set BRM_SIMPLEX_RUNTIME=0 to skip the BridgeStan density/gradient probes.
#
# The inciting case is a monotonic per-EVENT effect inside a `kernel(...)` cell:
# diet multiplies dose bioavailability per dose event, so what the cell needs is a
# K-simplex PARAMETER it can index by an ordinal level — not a per-observation
# contrast column. Before this path the two near-miss spellings were:
#
#   * `mo(c)` / `mo1(c)`, which are population linear-predictor terms. `mo1(c)`
#     as a bare non-data LHS is accepted and hands back a per-ROW contrast vector
#     of length `n_rows`; its Dirichlet increments live inside the `_sb_mo`
#     submodel and are unreachable from the formula.
#   * the scalar-parameter prior path, which is scalar-only: `Dirichlet` had no
#     emission at all, so the statement fell through to the linear-predictor
#     walker and died in a raw `_sb_predictor_term!` MethodError.
#
# `_sb_emit_vector_prior!` closes that: a non-data LHS with a `Dirichlet` RHS
# declares `simplex[K]` and is addressable anywhere later in the formula, cells
# included. It is a separate seam from the scalar `_sb_emit_prior!` because a
# multivariate family's declared SIZE comes from its hyperparameters, so the
# concentration has to reach `data` — mirroring the R2D2 share prepass.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Dirichlet, Exponential, Normal, logpdf
import BayesianRegressionModels as BRM

const SIMPLEX_CACHE = joinpath(tempdir(), "brm-simplex-priors")
const SIMPLEX_RUNTIME = get(ENV, "BRM_SIMPLEX_RUNTIME", "1") != "0"

const SIMPLEX_N = 6
simplex_df() = (;
    t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:SIMPLEX_N],
    dose    = fill(100.0, SIMPLEX_N),
    dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:SIMPLEX_N],
    diet    = [1, 2, 3, 1, 2, 3],
    weight  = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    subject = collect(1:SIMPLEX_N),
)

code_of(sb) = StanBlocks.stan_code(sb.model)

function stanc_accepts(model)
    result = StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic=false)
    result.ok || @error "stanc rejected simplex-prior model" output=result.output
    result.ok
end

function instantiate(model)
    isdir(SIMPLEX_CACHE) || mkpath(SIMPLEX_CACHE)
    code = StanBlocks.stan_code(model)
    StanBlocks.stan_instantiate(model; path=joinpath(SIMPLEX_CACHE, string(hash(code)) * ".stan"))
end

fixed_q(dim) = [0.1 * ((i % 5) - 2) for i in 1:dim]

# ---------------------------------------------------------------------- models
# The inciting shape: a 3-simplex indexed by an ordinal DIET level per dose event
# inside the kernel cell.
#
# Concentrations are spelled out as literals in each variant deliberately. `@brm`
# is a macro over the formula block, so a bare Julia symbol on the RHS is parsed
# as a formula LOCAL, not interpolated -- `Dirichlet(3, alpha)` with a captured
# `alpha` reaches the emitter as a `NamedColumn`, which the validator rejects.
simplex_kernel_flat(df) = @brm df begin
    sigma ~ Exponential(1)
    diet_share ~ Dirichlet(3, 1.0)
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, diet, dv, log_CL) do ts, d, di, yy, lCL
        mu = d * diet_share[di] * exp(-exp(lCL) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# Same program, asymmetric concentration -- only the DATA differs, which is what
# lets the semantic-parity check below compare two instantiations of one binary.
const SIMPLEX_ALPHA2 = [2.0, 1.5, 3.0]
simplex_kernel_asym(df) = @brm df begin
    sigma ~ Exponential(1)
    diet_share ~ Dirichlet([2.0, 1.5, 3.0])
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, diet, dv, log_CL) do ts, d, di, yy, lCL
        mu = d * diet_share[di] * exp(-exp(lCL) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# Symmetric spelled as an explicit flat vector: must be the same model as
# `simplex_kernel_flat`, data included.
simplex_kernel_flat_vec(df) = @brm df begin
    sigma ~ Exponential(1)
    diet_share ~ Dirichlet([1.0, 1.0, 1.0])
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, diet, dv, log_CL) do ts, d, di, yy, lCL
        mu = d * diet_share[di] * exp(-exp(lCL) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# No simplex statement: the negative control for dimension accounting and for
# "the new path never fires unasked".
plain_kernel_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + (1 | p | subject)
    pred ~ kernel(t, dose, dv, log_CL) do ts, d, yy, lCL
        mu = d * exp(-exp(lCL) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

@testset "declaration and emission" begin
    sb = SBBRMI(simplex_kernel_flat(simplex_df()))
    code = code_of(sb)

    # Concentration is DATA, under the derived `<target>_alpha` name -- the same
    # shape `_sb_emit_r2d2_params!` uses for its explained-variance shares.
    @test sb.data[:diet_share_alpha] == [1.0, 1.0, 1.0]
    @test occursin("vector[diet_share_alpha_n] diet_share_alpha;", code)

    # StanBlocks types the LHS of `~ dirichlet(alpha)` as `simplex[dims(alpha)[1]]`,
    # so the declaration needs no size annotation from BRM.
    @test occursin("simplex[diet_share_alpha_n] diet_share;", code)
    @test occursin("diet_share ~ dirichlet(diet_share_alpha);", code)

    # It is a real sampled parameter, not a generated-quantities prior draw.
    params_block = split(split(code, "parameters {")[2], "}")[1]
    @test occursin("simplex[diet_share_alpha_n] diet_share;", params_block)
    @test !occursin("dirichlet_simplex_rng", code)

    # Both Distributions.jl constructor forms are the same model.
    sym = SBBRMI(simplex_kernel_flat_vec(simplex_df()))
    @test code_of(sym) == code
    @test sym.data[:diet_share_alpha] == [1.0, 1.0, 1.0]

    # Asymmetric concentrations ride the same path and change only the data.
    asym = SBBRMI(simplex_kernel_asym(simplex_df()))
    @test code_of(asym) == code
    @test asym.data[:diet_share_alpha] == SIMPLEX_ALPHA2

    # A prior declaration is not a population formula. `linear_predictors` reports
    # every non-data `~` binding, so `_sb_is_prior_declaration` is what keeps the
    # `effect(...)` prepass from asking `popcoefnames` about it -- exactly the
    # invariant the scalar `log_F_bottle ~ Normal(0, 0.5)` case already relies on.
    # A Dirichlet declaration behaves identically to a scalar one here, including
    # `popcoefnames` throwing rather than naming a column.
    brmi = sb.parent
    @test :diet_share in Symbol[x.name for x in linear_predictors(brmi)]
    @test BRM._sb_is_prior_declaration(brmi, :diet_share)
    @test BRM._sb_is_prior_declaration(brmi, :sigma)
    @test_throws ErrorException popcoefnames(brmi, :diet_share)
    @test_throws ErrorException popcoefnames(brmi, :sigma)
end

@testset "addressable inside a kernel(...) cell" begin
    sb = SBBRMI(simplex_kernel_flat(simplex_df()))
    code = code_of(sb)
    # The cell indexes the SIMPLEX by the per-event ordinal level -- the thing the
    # `mo1(c)` per-row contrast could not express.
    @test occursin("diet_share[diet[plate_i__pl_1]]", code)
    @test stanc_accepts(sb.model)
end

@testset "the new path never fires unasked" begin
    # A formula with no vector-valued prior must not gain a simplex, a Dirichlet,
    # or a `_alpha` data entry. (Byte-identity of non-Dirichlet emission against
    # the pre-change tree is a landing receipt, not something one process can
    # assert; this is the in-process half of it.)
    sb = SBBRMI(plain_kernel_model(simplex_df()))
    code = code_of(sb)
    @test !occursin("simplex", code)
    @test !occursin("dirichlet", code)
    @test !any(k -> endswith(string(k), "_alpha"), keys(sb.data))
    # The seam's default declines every family it does not know.
    @test BRM._sb_emit_vector_prior!(Any[], Dict{Symbol,Any}(), :s, Normal, nothing) == false
    @test BRM._sb_emit_vector_prior!(Any[], Dict{Symbol,Any}(), :s, Exponential, nothing) == false
end

@testset "coexists with scalar and population effect priors" begin
    sb = SBBRMI(@brm simplex_df() begin
        sigma ~ Exponential(1)
        log_F ~ Normal(0.0, 0.5)
        diet_share ~ Dirichlet(3, 1.0)
        effect(log_CL, weight) ~ Normal(0.0, 0.1)
        log_CL ~ 1 + weight + (1 | p | subject)
        pred ~ kernel(t, dose, diet, dv, log_CL) do ts, d, di, yy, lCL
            mu = d * exp(log_F) * diet_share[di] * exp(-exp(lCL) * ts)
            yy ~ normal(mu, sigma)
            mu
        end
    end)
    code = code_of(sb)
    @test occursin("simplex[diet_share_alpha_n] diet_share;", code)
    @test occursin("real log_F;", code)
    @test sb.data[:diet_share_alpha] == [1.0, 1.0, 1.0]
    @test stanc_accepts(sb.model)
end

@testset "validation is loud" begin
    df = simplex_df()
    bad(body) = @test_throws ErrorException SBBRMI(body)

    # A one-element simplex is deterministically [1.0] -- no parameter to sample.
    bad(@brm df begin
        s ~ Dirichlet(1, 1.0)
        log_CL ~ 1 + (1 | p | subject)
        pred ~ kernel(t, dose, dv, log_CL) do ts, d, yy, lCL
            mu = d * s[1] * exp(-exp(lCL) * ts); yy ~ normal(mu, 1.0); mu
        end
    end)

    # Non-positive / non-finite concentrations.
    bad(@brm df begin
        s ~ Dirichlet([1.0, 0.0, 1.0])
        log_CL ~ 1 + (1 | p | subject)
        pred ~ kernel(t, dose, dv, log_CL) do ts, d, yy, lCL
            mu = d * s[1] * exp(-exp(lCL) * ts); yy ~ normal(mu, 1.0); mu
        end
    end)

    # Wrong arity.
    bad(@brm df begin
        s ~ Dirichlet(3, 1.0, 2.0)
        log_CL ~ 1 + (1 | p | subject)
        pred ~ kernel(t, dose, dv, log_CL) do ts, d, yy, lCL
            mu = d * s[1] * exp(-exp(lCL) * ts); yy ~ normal(mu, 1.0); mu
        end
    end)

    # A data-backed concentration is a different model, not a hyperparameter.
    bad(@brm df begin
        s ~ Dirichlet(weight)
        log_CL ~ 1 + (1 | p | subject)
        pred ~ kernel(t, dose, dv, log_CL) do ts, d, yy, lCL
            mu = d * s[1] * exp(-exp(lCL) * ts); yy ~ normal(mu, 1.0); mu
        end
    end)

    # `Dirichlet` is deliberately absent from the shared likelihood name table,
    # so a simplex-valued RESPONSE still fails loudly instead of half-working.
    @test isnothing(BRM._sb_stan_dist_name(Dirichlet))

    # The validator itself, addressed directly.
    @test BRM._sb_dirichlet_alpha(:s, (3, 2.0), (;)) == [2.0, 2.0, 2.0]
    @test BRM._sb_dirichlet_alpha(:s, ([1, 2, 3],), (;)) == [1.0, 2.0, 3.0]
    @test_throws ErrorException BRM._sb_dirichlet_alpha(:s, (3, 1.0), (; foo=1))
    @test_throws ErrorException BRM._sb_dirichlet_alpha(:s, (0, 1.0), (;))
    @test_throws ErrorException BRM._sb_dirichlet_alpha(:s, (3, -1.0), (;))
    @test_throws ErrorException BRM._sb_dirichlet_alpha(:s, ("nope",), (;))

    # The `@brm`-is-a-macro trap gets NAMED, not just typed at. A captured Julia
    # variable arrives as a formula-local `NamedColumn`, and the bare type name
    # alone reads like a BRM bug rather than a spelling mistake.
    local_alpha = BRM.NamedColumn(:my_alpha, BRM.MissingColumn())
    @test_throws "formula-local column" BRM._sb_dirichlet_alpha(:s, (3, local_alpha), (;))
    @test_throws "my_alpha" BRM._sb_dirichlet_alpha(:s, (3, local_alpha), (;))
    @test_throws "formula-local column" BRM._sb_dirichlet_alpha(:s, (local_alpha,), (;))
    # ...and a plain wrong type still gets the plain message, no spurious hint.
    @test !occursin("formula-local column",
                    try BRM._sb_dirichlet_alpha(:s, (3, "nope"), (;)) catch e; sprint(showerror, e) end)
end

if SIMPLEX_RUNTIME
    @testset "BridgeStan runtime" begin
        sb_with    = SBBRMI(simplex_kernel_flat(simplex_df()))
        sb_without = SBBRMI(plain_kernel_model(simplex_df()))

        p_with    = instantiate(sb_with.model)
        p_without = instantiate(sb_without.model)

        dim = LogDensityProblems.dimension(p_with)
        # A `simplex[K]` costs K-1 unconstrained coordinates.
        @test dim == LogDensityProblems.dimension(p_without) + 2

        q = fixed_q(dim)
        lp = LogDensityProblems.logdensity(p_with, q)
        lp_grad, grad = LogDensityProblems.logdensity_and_gradient(p_with, q)
        @test isfinite(lp)
        @test isfinite(lp_grad)
        @test length(grad) == dim
        @test all(isfinite, grad)

        # The simplex is a named constrained parameter, K coordinates wide.
        names = StanBlocks.BridgeStan.param_names(p_with.model)
        shares = findall(n -> startswith(n, "diet_share"), names)
        @test length(shares) == 3

        # -------------------------------------------------- semantic parity
        # `Dirichlet(alpha)`'s Julia parameterization IS Stan's, so changing ONLY
        # the concentration must move the target by exactly the Dirichlet's own
        # contribution at the same point. Both builds emit byte-identical code
        # (only the `data` value differs), so the jacobian and every other term
        # cancels in the difference.
        alpha1, alpha2 = fill(1.0, 3), SIMPLEX_ALPHA2
        m1 = sb_with
        m2 = SBBRMI(simplex_kernel_asym(simplex_df()))
        @test code_of(m1) == code_of(m2)
        p1, p2 = instantiate(m1.model), instantiate(m2.model)
        s = StanBlocks.BridgeStan.param_constrain(p1.model, q)[shares]
        @test isapprox(sum(s), 1.0; atol=1e-12)

        # `~` in Stan drops terms that are constant in the PARAMETERS, and the
        # concentration is registered as `data` -- so the Dirichlet
        # log-normalizer, a function of `alpha` alone, never reaches `target`.
        # What does reach it is the kernel below. (This is why the parity claim
        # is about the kernel and not about `logpdf` directly: the full logpdf
        # difference here is off by `lmnB(alpha2) - lmnB(alpha1)` ~= 4.4 nats.)
        dirichlet_kernel(a, x) = sum((a .- 1) .* log.(x))
        # Pin that kernel against Distributions.jl rather than trusting the
        # algebra: for one `alpha`, `logpdf` and the kernel differ by the same
        # constant at every point, so their DIFFERENCES across two points agree.
        s_ref = [0.2, 0.3, 0.5]
        for a in (alpha1, alpha2)
            @test isapprox(
                logpdf(Dirichlet(a), s) - logpdf(Dirichlet(a), s_ref),
                dirichlet_kernel(a, s) - dirichlet_kernel(a, s_ref); atol=1e-10)
        end

        expected = dirichlet_kernel(alpha2, s) - dirichlet_kernel(alpha1, s)
        actual = LogDensityProblems.logdensity(p2, q) - LogDensityProblems.logdensity(p1, q)
        @test isapprox(actual, expected; atol=1e-8)
    end
end
