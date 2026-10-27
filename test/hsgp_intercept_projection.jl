# `hsgp(x...; orthogonal_to=:intercept)`: every basis column centered over the
# term's rows, for any number of raw axes (and a model-derived axis), so the
# term sums to zero over its rows and a level elsewhere in the model is not
# shared with the field. Both projections are fitted on the training rows and
# frozen on replay; `freeze_constants=false` re-fits them on the new rows.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LinearAlgebra
using ReactiveKernels, ReactiveKernelsPPL, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const RUNTIME = get(ENV, "BRM_HSGP_PROJECTION_RUNTIME", "1") != "0"
const CACHE = joinpath(tempdir(), "brm-hsgp-projection")

function field_df(; n=24)
    i = 1:n
    x = 2.0 .* sin.(0.7 .* i) .+ 0.1 .* i
    z = cos.(1.3 .* i) .- 0.05 .* i
    keep = isodd.(i)
    g = repeat(["a", "b", "c"], outer=n ÷ 3)
    y = 0.5 .* x .- 0.2 .* z .+ 0.1 .* sin.(3 .* i)
    (; x, z, keep, g, y)
end
rows_of(df, rows) = map(column -> column[rows], df)

raw_2d = @brm begin
    mu ~ 1 + hsgp(x, z; k=(4, 3))
    y ~ Normal(mu, 1)
end
centered_2d = @brm begin
    mu ~ 1 + hsgp(x, z; k=(4, 3), orthogonal_to=:intercept)
    y ~ Normal(mu, 1)
end
centered_aniso = @brm begin
    mu ~ 1 + hsgp(x, z; k=(3, 3), iso=false, orthogonal_to=:intercept)
    y ~ Normal(mu, 1)
end
centered_domain = @brm begin
    mu ~ 1 + hsgp(x, z; k=(3, 3), domain=((-6.0, 6.0), (-4.0, 4.0)),
                  orthogonal_to=:intercept)
    y ~ Normal(mu, 1)
end
centered_rows = @brm begin
    mu ~ 1 + hsgp(x, z; k=(3, 3), rows=keep, orthogonal_to=:intercept)
    y ~ Normal(mu, 1)
end
linear_1d = @brm begin
    mu ~ 1 + x + hsgp(x; k=5, orthogonal_to=:linear)
    y ~ Normal(mu, 1)
end
# A multiplicative field: without the projection its level is a second copy
# of the weights' scale.
weighted_field = @brm begin
    u ~ 0 + hsgp(x, z; k=(3, 3), orthogonal_to=:intercept)
    w ~ Normal(0, 1; lower=0)
    rate = w * exp(u)
    y ~ Normal(rate, 1)
end

column_sums(PHI) = vec(sum(PHI; dims=1))
basis(sb) = sb.data[only(k for k in keys(sb.data) if startswith(string(k), "PHI_hsgp"))]

@testset "every basis column is centered over the term's rows" begin
    df = field_df()
    raw = basis(SBBRMI(raw_2d(df); mod=@__MODULE__))
    for model in (centered_2d, centered_aniso, centered_domain)
        sb = SBBRMI(model(df); mod=@__MODULE__)
        PHI = basis(sb)
        @test maximum(abs, column_sums(PHI)) < 1e-12
        @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model);
                                     warn_pedantic=false).ok
    end
    # The centered basis is the raw basis minus each column's mean, so any
    # weights give the raw term minus its mean over the rows.
    PHI = basis(SBBRMI(centered_2d(df); mod=@__MODULE__))
    @test PHI ≈ raw .- sum(raw; dims=1) ./ size(raw, 1) atol=1e-14
    w = [0.3sin(b) for b in 1:size(PHI, 2)]
    @test PHI * w ≈ raw * w .- sum(raw * w) / length(df.x) atol=1e-12

    # A `rows=` term is centered over the rows that carry it; the others stay 0.
    sb = SBBRMI(centered_rows(df); mod=@__MODULE__)
    PHI = basis(sb)
    @test all(iszero, PHI[.!df.keep, :])
    @test maximum(abs, column_sums(PHI[df.keep, :])) < 1e-12
    @test maximum(abs, column_sums(PHI)) < 1e-12

    # The fitted projection is recorded with the basis.
    projection = sb.preproc[:PHI_hsgp_x_z].const_.projection
    @test projection.orthogonal === :intercept
    @test length(projection.column_means) == 9
end

@testset "a frozen replay evaluates the training projection" begin
    df = field_df()
    rows = [2, 5, 6, 11, 17]
    for model in (centered_2d, centered_domain, centered_rows, linear_1d)
        sb = SBBRMI(model(df); mod=@__MODULE__)
        replay = reprocess(sb, rows_of(df, rows))
        @test basis(replay) ≈ basis(sb)[rows, :] atol=1e-14
        @test StanBlocks.stan_code(replay.model) == StanBlocks.stan_code(sb.model)
        # One row is a design of its own; the training projection applies.
        @test basis(reprocess(sb, rows_of(df, [7]))) ≈ basis(sb)[[7], :] atol=1e-14
    end
    # A fresh replay re-fits the projection on its rows.
    sb = SBBRMI(centered_domain(df); mod=@__MODULE__)
    fresh = reprocess(sb, rows_of(df, rows); freeze_constants=false)
    @test maximum(abs, column_sums(basis(fresh))) < 1e-12
    @test !(basis(fresh) ≈ basis(sb)[rows, :])
end

@testset "a model-derived axis is centered at every draw" begin
    df = (; t=collect(range(-1.0, 1.0; length=8)),
          x_obs=collect(range(-0.6, 0.9; length=8)), y=zeros(8))
    sb = SBBRMI((@brm df begin
        x ~ 1 + t
        x_obs ~ Normal(x, 0.2)
        mu ~ 1 + hsgp(x; k=4, domain=(-3.0, 3.0), orthogonal_to=:intercept)
        y ~ Normal(mu, 1)
    end); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("brm_hsgp_center_columns", code)
    @test !occursin("brm_hsgp_orthogonalize_linear", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "refused combinations name the projection" begin
    df = field_df()
    build(model) = SBBRMI(model(df); mod=@__MODULE__)
    # refused: grouped meaning (pooled vs per-level) is open (todo 1tecovn)
    @test_throws "orthogonal_to=:intercept)` is an ungrouped" build(@brm begin
        mu ~ 1 + hsgp(x, z; k=3, by=g, orthogonal_to=:intercept)
        y ~ Normal(mu, 1)
    end)
    # refused: an anchor and a centering each pin the term's level
    @test_throws "cannot be combined with `orthogonal_to=:intercept`" build(@brm begin
        mu ~ 1 + hsgp(x, z; k=3, anchor=(0.0, 0.0), orthogonal_to=:intercept)
        y ~ Normal(mu, 1)
    end)
    @test_throws "`orthogonal_to=:intercept` centers the basis of any number" build(@brm begin
        mu ~ 1 + hsgp(x, z; k=3, orthogonal_to=:linear)
        y ~ Normal(mu, 1)
    end)
    @test_throws "supports `:intercept` and `:linear`" build(@brm begin
        mu ~ 1 + hsgp(x, z; k=3, orthogonal_to=:quadratic)
        y ~ Normal(mu, 1)
    end)
    # refused: a projection fitted on no rows has no column means
    @test_throws "and this term has none" RKBRMI((@brm begin
        mu ~ 1 + hsgp(x; k=3, domain=(-1.0, 1.0), orthogonal_to=:intercept)
        y ~ Normal(mu, 1)
    end)((; x=Float64[], y=Float64[])))
end

page(r) = join((brm_description_markdown(r), r.prose..., r.equations...), "\n")

@testset "the description names the removed direction" begin
    d = brm_descriptor(centered_2d, field_df(); mod=@__MODULE__)
    text = page(brm_description(d))
    @test occursin("removes the intercept direction", text)
    @test occursin("I-P_{1}", text)
    @test !occursin("I-P_{[1,x]}", text)
    @test occursin("Richtung des Achsenabschnitts",
                   page(brm_description(d; texts=description_texts(:de))))
    linear = page(brm_description(brm_descriptor(linear_1d, field_df(); mod=@__MODULE__)))
    @test occursin("I-P_{[1,x]}", linear)
end

compiled(sb, tag) = (mkpath(CACHE);
    BRM.stan_instantiate(sb; path=joinpath(CACHE,
        tag * "_" * string(hash(BRM.stan_code(sb))) * ".stan")))
points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])

@testset "RKBRMI equals SBBRMI" begin
    if RUNTIME
        df = field_df()
        for (tag, model) in ("centered_2d" => centered_2d, "aniso" => centered_aniso,
                             "domain" => centered_domain, "rows" => centered_rows,
                             "weighted" => weighted_field)
            m = model(df)
            rk = RKBRMI(m)
            sb = SBBRMI(m; mod=@__MODULE__)
            stan = compiled(sb, tag)
            transport = brm_coordinate_transport(rk, sb, BS.param_unc_names(stan.model))
            problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for u in points(length(transport))
                stan_u = brm_rk_point_to_stan(transport, u)
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                g = similar(stan_u)
                stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, g;
                                                        propto=false, jacobian=true)
                @test value ≈ brm_stan_logdensity_to_rk(transport, u, stan_value) atol=1e-10 rtol=1e-12
                @test gradient ≈ brm_stan_gradient_to_rk(transport, u, g) atol=1e-10 rtol=1e-10
            end
        end
    else
        @info "Skipping the compiled parity sets (BRM_HSGP_PROJECTION_RUNTIME=0)"
        @test true
    end
end
