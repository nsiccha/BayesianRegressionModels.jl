# Run: julia --startup-file=no --project=test test/description_known_values.jl
#
# Descriptions of random-effect blocks held at known values and of
# `NegativeBinomial2` observations (snag brm-description-256b3b35):
# - `cor(:, ID) = I` leaves the margins independent; it is not described as
#   correlated, and its covariance is D², not DΩD;
# - `cor(:, ID) = R` is described as held at a known matrix;
# - a known `sd(...) = v` or `cor(:, ID) = R` is the point mass `Dirac`, a
#   described law rather than a coverage gap, a matrix shown as a matrix;
# - a known-value block's sampled scales are its `(…, :sd, i)` priors only:
#   the vector its emission declares for them is not listed again as a
#   separate `(:parameter, :<binding>_tau)` prior;
# - `NegativeBinomial2(mu, phi)` is described by its mean and dispersion, at
#   top level and inside a `@plate` cell.
using Test, BayesianRegressionModels, Distributions

const DATA = let n = 24
    (; g=repeat(["a", "b", "c"], 8), x=collect(range(-1, 1; length=n)),
       z=sin.(1:n), y=cos.(1:n), count=rem.(1:n, 7),
       unit_cell=repeat(["k1", "k2", "k3", "k4"], inner=6), cell=["k1", "k2", "k3"],
       total=[3, 5, 4])
end
priors_at(r, id) = filter(p -> p.id == id, r.priors)
block(r, id) = only(c for c in r.components if c.kind === :random_effect && c.id[2] === id)
fragment_prose(r, needle) = filter(p -> occursin(needle, p), r.prose)

@testset "a correlation held at the identity leaves the margins independent" begin
    model = @brm DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + z + (1 + x + z | gs | g)
        sd(:, gs) ~ truncated(Normal(0, 0.5); lower = 0)
        cor(:, gs) = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(model)
    r = brm_description(sb)
    @test r.complete
    @test isempty(r.diagnostics)
    c = block(r, :gs)
    @test c.keywords.correlation === :identity
    @test only(fragment_prose(r, "Group deviations for `g`")) ==
        "Group deviations for `g` have 3 margin(s) across 3 fitted levels. This block is " *
        "independent of separately declared blocks. Its correlation is held at the " *
        "identity: its margins are independent."
    @test "\\mathbf b_{1,i}\\sim\\mathcal N_{3}(0,D_{1}^2)" in r.equations
    @test !any(e -> occursin("\\Omega_{1}", e), r.equations)
    # The known correlation is still inventoried, as the point mass at I.
    cor = only(priors_at(r, (:random_effect, :gs, :g, :correlation)))
    @test cor.source.kind === :known
    md = brm_description_markdown(r)
    @test occursin("\\operatorname{Dirac}\\left(\\begin{bmatrix}1.0&0.0&0.0\\\\" *
        "0.0&1.0&0.0\\\\0.0&0.0&1.0\\end{bmatrix}\\right)", md)
    # The sampled scales are the block's three `sd` priors, listed once.
    @test length([p for p in r.priors if p.id[1:min(4, end)] == (:random_effect, :gs, :g, :sd)]) == 3
    @test isempty(priors_at(r, (:parameter, :b_gs_g_tau)))
    @test !occursin("Logical parameter: `parameter` / `b_gs_g_tau`", md)
    german = brm_description(sb; texts=description_texts(:de))
    @test german.complete
    @test any(p -> endswith(p, "Seine Korrelation ist auf die Einheitsmatrix festgelegt: " *
        "Seine Koeffizienten sind unabhängig."), german.prose)
end

@testset "a known correlation and known scales are described, not gaps" begin
    model = @brm DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        sd(mu, p, Intercept) = 0.3
        sd(mu, p, x) ~ Exponential(2)
        cor(:, p) = [1.0 0.4; 0.4 1.0]
        y ~ Normal(mu, sigma)
    end
    r = brm_description(SBBRMI(model))
    @test r.complete
    @test block(r, :p).keywords.correlation === :known
    @test any(p -> endswith(p, "Its correlation is not estimated: it is held at a known matrix."), r.prose)
    # A known correlation keeps the full covariance DΩD.
    @test "\\mathbf b_{1,i}\\sim\\mathcal N_{2}(0,D_{1}\\Omega_{1}D_{1})" in r.equations
    md = brm_description_markdown(r)
    @test occursin("\\operatorname{Dirac}\\left(0.3\\right)", md)
    @test occursin("\\operatorname{Dirac}\\left(\\begin{bmatrix}1.0&0.4\\\\0.4&1.0\\end{bmatrix}\\right)", md)
    @test only(priors_at(r, (:random_effect, :p, :g, :sd, 1))).source.kind === :known
    @test only(priors_at(r, (:random_effect, :p, :g, :sd, 2))).source.kind === :selector
    # Some scale known: the sampled ones are `b_p_g_tau_free`, not a second prior.
    @test isempty(priors_at(r, (:parameter, :b_p_g_tau_free)))
    @test isempty(priors_at(r, (:parameter, :b_p_g_tau)))
end

@testset "an estimated correlation is unchanged" begin
    model = @brm DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | g)
        y ~ Normal(mu, sigma)
    end
    r = brm_description(SBBRMI(model))
    @test r.complete
    @test block(r, :p).keywords.correlation === :estimated
    @test any(p -> endswith(p, "Its margins are correlated."), r.prose)
    @test "\\mathbf b_{1,i}\\sim\\mathcal N_{2}(0,D_{1}\\Omega_{1}D_{1})" in r.equations
end

@testset "NegativeBinomial2 is described by its mean and dispersion" begin
    model = @brm DATA begin
        phi ~ Exponential(1)
        log(mu) ~ 1 + x
        count ~ NegativeBinomial2(mu, phi)
    end
    r = brm_description(SBBRMI(model))
    @test r.complete
    @test any(p -> endswith(p, "Negative binomial distribution with mean μ and dispersion φ " *
        "(first and second argument): its variance is μ + μ²/φ."), r.prose)
    @test any(e -> startswith(e, "\\mathrm{count}\\sim{}\\operatorname{NegBinomial}_{\\mathrm{mean,dispersion}}("),
        r.equations)

    # The same family observing a total in a `@plate` cell.
    cell = @brm DATA begin
        cv ~ LogNormal(-2, 1)
        log_a ~ 1 + x
        rows = groups(unit_cell => cell)
        @plate for k in eachindex(total)
            total[k] ~ NegativeBinomial2(sum(exp.(log_a[rows[k]])), 1.0 / cv^2)
        end
    end
    r = brm_description(SBBRMI(cell); texts=description_texts(:de))
    @test r.complete
    @test any(e -> occursin("\\operatorname{NegBinomial}_{\\mathrm{Mittelwert,Dispersion}}(", e), r.equations)
end
