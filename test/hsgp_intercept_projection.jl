# `hsgp(x...; orthogonal_to=:intercept)`: every basis column centered over the
# term's rows, for any number of raw axes (and a model-derived axis), so the
# term sums to zero over its rows and a level elsewhere in the model is not
# shared with the field. Both projections are fitted on the training rows and
# frozen on replay; `freeze_constants=false` re-fits them on the new rows.
# With `by=g` both are fitted per level, on each group's own rows (decision
# 15hn91t), so every group's field sums to zero over its rows.
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

# `by=g`: the projection is fitted on each group's rows. The groups cover
# different stretches of `x` (an unbalanced frame), so per-level and pooled
# centering differ.
raw_grouped = @brm begin
    mu ~ 1 + hsgp(x, z; k=(3, 3), by=g)
    y ~ Normal(mu, 1)
end
grouped_2d = @brm begin
    mu ~ 1 + hsgp(x, z; k=(3, 3), by=g, orthogonal_to=:intercept)
    y ~ Normal(mu, 1)
end
grouped_cor = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    mu ~ 1 + hsgp(x, z; k=(3, 3), by=g, cor=Omega, orthogonal_to=:intercept)
    y ~ Normal(mu, 1)
end
grouped_rows = @brm begin
    mu ~ 1 + hsgp(x, z; k=(3, 3), by=g, rows=keep, orthogonal_to=:intercept)
    y ~ Normal(mu, 1)
end
grouped_linear = @brm begin
    mu ~ 1 + x + hsgp(x; k=5, by=g, orthogonal_to=:linear)
    y ~ Normal(mu, 1)
end
# The reporter's shape: one level-correlated field multiplying a positive
# weight, centred within each group.
grouped_weighted = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    u ~ 0 + hsgp(x, z; k=(3, 3), by=g, cor=Omega, orthogonal_to=:intercept)
    w ~ Normal(0, 1; lower=0)
    rate = w * exp(u)
    y ~ Normal(rate, 1)
end
# ... with a per-level amplitude (`ddc2646b`): the reporter's full target.
grouped_amplitude = @brm begin
    Omega ~ LKJCholesky(3, 2.0)
    u ~ 0 + hsgp(x, z; k=(3, 3), by=g, cor=Omega, orthogonal_to=:intercept)
    log(sd(hsgp(x, z))) ~ 1 + (1 | g)
    w ~ Normal(0, 1; lower=0)
    rate = w * exp(u)
    y ~ Normal(rate, 1)
end
grouped_df(; n=24) = merge(field_df(; n), (; g=[i <= n ÷ 3 ? "a" : i <= 2n ÷ 3 ? "b" : "c"
                                                for i in 1:n]))

column_sums(PHI) = vec(sum(PHI; dims=1))
basis(sb) = sb.data[only(k for k in keys(sb.data) if startswith(string(k), "PHI_hsgp"))]
level_rows(df) = [findall(==(level), df.g) for level in unique(df.g)]

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

@testset "a by= term is projected within each level" begin
    df = grouped_df()
    raw = basis(SBBRMI(raw_grouped(df); mod=@__MODULE__))
    for model in (grouped_2d, grouped_cor, grouped_weighted, grouped_amplitude)
        sb = SBBRMI(model(df); mod=@__MODULE__)
        PHI = basis(sb)
        for rows in level_rows(df)
            @test maximum(abs, column_sums(PHI[rows, :])) < 1e-12
            # Each group's basis is its raw basis minus that group's column means.
            @test PHI[rows, :] ≈ raw[rows, :] .- sum(raw[rows, :]; dims=1) ./ length(rows) atol=1e-14
        end
        @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model);
                                     warn_pedantic=false).ok
    end
    # On this unbalanced frame, pooled centering would leave each group a level.
    pooled = raw .- sum(raw; dims=1) ./ size(raw, 1)
    @test maximum(abs, column_sums(pooled[first(level_rows(df)), :])) > 1e-3
    # One projection per level is recorded with the basis.
    sb = SBBRMI(grouped_2d(df); mod=@__MODULE__)
    projection = sb.preproc[:PHI_hsgp_x_z_by_g].const_.projection
    @test projection.orthogonal === :intercept
    @test length(projection.per_level) == 3
    @test all(fit -> length(fit.column_means) == 9, projection.per_level)

    # `:linear`: each group's basis is orthogonal to that group's 1 and x.
    sb = SBBRMI(grouped_linear(df); mod=@__MODULE__)
    PHI = basis(sb)
    for rows in level_rows(df)
        @test maximum(abs, PHI[rows, :]' * [ones(length(rows)) df.x[rows]]) < 1e-10
    end

    # `rows=`: each group is centered over its kept rows; the others stay 0.
    sb = SBBRMI(grouped_rows(df); mod=@__MODULE__)
    PHI = basis(sb)
    @test all(iszero, PHI[.!df.keep, :])
    for rows in level_rows(df)
        kept = filter(row -> df.keep[row], rows)
        @test maximum(abs, column_sums(PHI[kept, :])) < 1e-12
    end
end

@testset "a frozen replay evaluates each level's training projection" begin
    df = grouped_df()
    # Rows of two groups only, and one row: the training levels and their
    # projections apply.
    for (model, rows) in Iterators.product((grouped_2d, grouped_cor, grouped_rows,
                                            grouped_linear, grouped_amplitude),
                                           ([2, 5, 11, 13], [7]))
        sb = SBBRMI(model(df); mod=@__MODULE__)
        replay = reprocess(sb, rows_of(df, rows))
        @test basis(replay) ≈ basis(sb)[rows, :] atol=1e-14
        @test StanBlocks.stan_code(replay.model) == StanBlocks.stan_code(sb.model)
    end
    # A fresh replay re-fits each level's projection on its new rows.
    sb = SBBRMI(grouped_2d(df); mod=@__MODULE__)
    rows = [1, 2, 3, 9, 10, 17, 18, 19, 20]
    replayed = rows_of(df, rows)
    fresh = reprocess(sb, replayed; freeze_constants=false)
    for level_index in level_rows(replayed)
        @test maximum(abs, column_sums(basis(fresh)[level_index, :])) < 1e-12
    end
    @test !(basis(fresh) ≈ basis(sb)[rows, :])
end

@testset "refused combinations name the projection" begin
    df = field_df()
    build(model) = SBBRMI(model(df); mod=@__MODULE__)
    # refused: a group with no row in `rows=` has no rows to fit its projection on
    lonely = merge(grouped_df(), (; keep=grouped_df().g .!= "c"))
    no_rows = @brm begin
        mu ~ 1 + hsgp(x, z; k=3, by=g, rows=keep, orthogonal_to=:intercept)
        y ~ Normal(mu, 1)
    end
    @test_throws "level `c` has none" SBBRMI(no_rows(lonely); mod=@__MODULE__)
    @test_throws "level `c` has none" RKBRMI(no_rows(lonely))
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
    grouped = page(brm_description(brm_descriptor(grouped_2d, grouped_df(); mod=@__MODULE__)))
    @test occursin("within each group", grouped)
    @test occursin("I-P^{(g)}_{1}", grouped)
    @test occursin("innerhalb jeder Gruppe", page(brm_description(
        brm_descriptor(grouped_2d, grouped_df(); mod=@__MODULE__); texts=description_texts(:de))))
    grouped_lin = page(brm_description(brm_descriptor(grouped_linear, grouped_df(); mod=@__MODULE__)))
    @test occursin("I-P^{(g)}_{[1,x]}", grouped_lin)
end

compiled(sb, tag) = (mkpath(CACHE);
    BRM.stan_instantiate(sb; path=joinpath(CACHE,
        tag * "_" * string(hash(BRM.stan_code(sb))) * ".stan")))
points(n) = (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])

# The grouped HSGP is not paired by the coordinate transport yet (as in
# test/level_correlation.jl): map its coordinate names onto Stan's.
function grouped_permutation(rk, sb, stan_names)
    rk_names = String.(coordinate_names(rk.model.layout))
    key = only(k for k in keys(sb.data) if startswith(string(k), "PHI_hsgp"))
    M = size(sb.data[key], 2)
    carrier = replace(string(key), "PHI_" => "")        # hsgp_<axes>_by_g
    stem = first(split(carrier, "_by_"))               # hsgp_<axes>
    function stan_name(n)
        n in stan_names && return n
        # A per-level amplitude or length scale (as in test/level_correlation_hyper.jl).
        for hyper in ("rho", "sigma")
            n == "$stem.$(hyper)_Intercept" && return "$(carrier)_beta0_$hyper"
            n == "$stem.$(hyper)_sd" && return "$(carrier)_sd_$hyper"
            x = match(Regex("^$stem\\.$(hyper)_z\\.(\\d+)\$"), n)
            x === nothing || return "$(carrier)_z_$hyper.$(x[1])"
        end
        n == "$stem.rho" && return "$(carrier)_rho_iso"
        n == "$stem.sigma" && return "$(carrier)_sigma"
        x = match(r"^hsgp_\w+?\.z_?\.(\d+)\.(\d+)$", n)
        x === nothing ||
            return "zflat_hsgpw_$(chopprefix(stem, "hsgp_"))_g.$((parse(Int, x[1]) - 1) * M + parse(Int, x[2]))"
        x = match(r"^pop_(\w+)\.beta_pop\.(\d+)$", n)
        x === nothing || return "pop_$(x[1])_beta_pop.$(x[2])"
        x = match(r"^(\w+)_Intercept$", n)        # an intercept-only predictor
        x === nothing || return "pop_$(x[1])_beta_pop.1"
        error("unmapped RK coordinate $n")
    end
    perm = [only(findall(==(stan_name(n)), stan_names)) for n in rk_names]
    @test sort(perm) == collect(eachindex(stan_names))
    perm
end

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
        # `by=`: one projection per level, on both backends.
        df = grouped_df()
        for (tag, model) in ("grouped_2d" => grouped_2d, "grouped_cor" => grouped_cor,
                             "grouped_rows" => grouped_rows,
                             "grouped_linear" => grouped_linear,
                             "grouped_weighted" => grouped_weighted,
                             "grouped_amplitude" => grouped_amplitude)
            m = model(df)
            rk = RKBRMI(m)
            sb = SBBRMI(m; mod=@__MODULE__)
            stan = compiled(sb, tag)
            perm = grouped_permutation(rk, sb, BS.param_unc_names(stan.model))
            problem = rk_logdensity_problem(rk; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            for u in points(length(perm))
                stan_u = zeros(length(u)); stan_u[perm] .= u
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                g = similar(stan_u)
                stan_value, _ = BS.log_density_gradient!(stan.model, stan_u, g;
                                                        propto=false, jacobian=true)
                @test value ≈ stan_value atol=1e-10 rtol=1e-12
                @test gradient ≈ g[perm] atol=1e-10 rtol=1e-10
            end
        end
    else
        @info "Skipping the compiled parity sets (BRM_HSGP_PROJECTION_RUNTIME=0)"
        @test true
    end
end
