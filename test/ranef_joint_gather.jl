# test/ranef_joint_gather.jl — `(… || g)` lowers as one one-margin `|ID|` block
# per term, addressed or not, and the `|ID|` slices one predictor reads through
# one grouping index are gathered ONCE: one `rows_dot_product` per row instead
# of one gather, scaling and addition per term (snag `uncorrelated-gro-ce63c15a`:
# 42 `||` terms over 43,796 rows).
#
# Run: julia --project=test test/ranef_joint_gather.jl [filter...]
# Set BRM_JOINT_GATHER_RUNTIME=0 to skip the BridgeStan density checks.
#
# The oracles are programs whose blocks stay apart because each reads its own
# copy of the grouping column: plain per-term blocks `(1 | g) + (0 + x | g2) +
# (0 + z | g3)` — the law `(1 + x + z || g)` had before it lowered through
# `|ID|` blocks — and the same `|ID|` blocks on two copies.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random
using Distributions: Exponential, LogNormal, Normal

const JOINT_RUNTIME = get(ENV, "BRM_JOINT_GATHER_RUNTIME", "1") != "0"
const JOINT_CACHE = joinpath(tempdir(), "brm-ranef-joint-gather")

Random.seed!(20261025)
const JDF = let n = 60
    g = repeat(1:6; inner=10)
    (; g, g2=copy(g), g3=copy(g), x=round.(randn(n); digits=3), z=round.(randn(n); digits=3),
       y=round.(randn(n); digits=3))
end

code(sb) = BRM.stan_code(sb)
build(b; kwargs...) = SBBRMI(b; mod=@__MODULE__, total_groups=(), kwargs...)
stanc_ok(sb) = StanBlocks.stanc_check(code(sb); warn_pedantic=false).ok
lines_with(c, needle) = [strip(l) for l in split(c, '\n') if occursin(needle, l)]
function instantiate(sb)
    mkpath(JOINT_CACHE)
    BRM.stan_instantiate(sb; path=joinpath(JOINT_CACHE, string(hash(code(sb))) * ".stan"))
end

addressed = @brm JDF begin
    mu ~ 1 + x + z + (1 + x + z || g)
    sd(:, g) ~ Exponential(2.0)
    y ~ Normal(mu, 1.0)
end
unaddressed = @brm JDF begin
    mu ~ 1 + x + z + (1 + x + z || g)
    y ~ Normal(mu, 1.0)
end
# Its law spelled with plain per-term blocks, each on its own copy of `g`: a
# lone intercept samples `log_scale ~ std_normal()`, a lone slope a half-normal
# `tau`, which are the defaults of the `|ID|` blocks `||` lowers under.
per_term = @brm JDF begin
    mu ~ 1 + x + z + (1 | g) + (0 + x | g2) + (0 + z | g3)
    y ~ Normal(mu, 1.0)
end
# Two shared blocks on one grouping, and the same blocks on two codings of it.
shared = @brm JDF begin
    mu ~ 1 + x + (1 | a | g) + (0 + x + z | b | g)
    y ~ Normal(mu, 1.0)
end
apart = @brm JDF begin
    mu ~ 1 + x + (1 | a | g) + (0 + x + z | b | g2)
    y ~ Normal(mu, 1.0)
end

@stestset "an addressed `||` block is one product per row" begin
    sb = build(addressed)
    c = code(sb)
    @test stanc_ok(sb)
    # One fused gather over the grouping index, no per-term slices.
    @test length(lines_with(c, "rows_dot_product(")) == 1
    @test only(lines_with(c, "rows_dot_product(")) ==
          "vector[g_idx_n] r_mu_g__joint = rows_dot_product(Z_mu_g__joint, E_mu_g__joint[g_idx, :]);"
    @test isempty(lines_with(c, "r_mu_mu__g__nocor__"))
    @test occursin(r"^vector\[\w+\] mu = \(pop_mu \+ r_mu_g__joint\);$", only(lines_with(c, " mu = ")))
    # The blocks themselves are one `|ID|` bucket per term.
    blocks = ranef_blocks(sb)
    @test [b.id for b in blocks] == [:mu__g__nocor__1, :mu__g__nocor__2, :mu__g__nocor__3]
    @test all(b -> b.n_terms == 1 && b.n_groups == 6 && b.group === :g, blocks)
end

@stestset "an unaddressed `||` block lowers as the addressed one" begin
    sb = build(unaddressed)
    c = code(sb)
    @test stanc_ok(sb)
    @test occursin(r"^vector\[\w+\] mu = \(pop_mu \+ r_mu_g__joint\);$", only(lines_with(c, " mu = ")))
    @test isempty(lines_with(c, "g__nocor__1_idx"))
    @test [b.id for b in ranef_blocks(sb)] == [:mu__g__nocor__1, :mu__g__nocor__2, :mu__g__nocor__3]
    # The lone intercept keeps its LogNormal(0, 1) scale, the slopes their
    # half-normal one.
    @test only(lines_with(c, "b_mu__g__nocor__1_g_tau ~ ")) == "b_mu__g__nocor__1_g_tau ~ lognormal(0.0, 1.0);"
    @test only(lines_with(c, "b_mu__g__nocor__2_g_tau ~ ")) == "b_mu__g__nocor__2_g_tau ~ std_normal();"
    # Writing the generated IDs by hand gives the identical program.
    twin = @brm JDF begin
        mu ~ 1 + x + z + (1 | mu__g__nocor__1 | g) + (x | mu__g__nocor__2 | g) +
             (z | mu__g__nocor__3 | g)
        sd(:, mu__g__nocor__1) ~ LogNormal(0.0, 1.0)
        y ~ Normal(mu, 1.0)
    end
    @test code(build(twin)) == c
    # `centered_groups` and `cv_groups` name the real grouping column (todo
    # `073p9ti` item 1: the synthetic `g__nocor__k` groups never matched it).
    centered = code(build(unaddressed; centered_groups=[:g]))
    @test !isempty(lines_with(centered, "multi_normal_cholesky0(diag_matrix(b_mu__g__nocor__2_g_tau))"))
    @test isempty(lines_with(centered, "b_mu__g__nocor__2_g_z_flat"))
    cv = code(build(unaddressed; cv_groups=[:g]))
    @test !isempty(lines_with(cv, "max(g_idx)"))
end

@stestset "two `||` declarations on one grouping stay independent" begin
    # todo `073p9ti` item 2: both used to claim `g__nocor__1` and coalesce into
    # one correlated block.
    two = @brm JDF begin
        mu ~ 1 + x + (1 || g) + (0 + x || g)
        y ~ Normal(mu, 1.0)
    end
    sb = build(two)
    @test stanc_ok(sb)
    @test [b.id for b in ranef_blocks(sb)] == [:mu__g__nocor__1, :mu__g__nocor__2]
    @test isempty(lines_with(code(sb), "lkj_corr_cholesky"))
end

@stestset "slices on one index fuse, slices on two indices do not" begin
    c = code(build(shared))
    @test stanc_ok(build(shared))
    @test length(lines_with(c, "rows_dot_product(")) == 1
    @test occursin("E_mu_g__joint = hcat(b_a_g[:, 1], b_b_g[:, 1], b_b_g[:, 2]);",
                   only(lines_with(c, "E_mu_g__joint = ")))
    c2 = code(build(apart))
    @test isempty(lines_with(c2, "__joint"))
    @test length(lines_with(c2, "rows_dot_product(")) == 1   # the two-column `|b|` slice
    @test length(lines_with(c2, "r_mu_a_g = ")) == 1
end

@stestset "a lone slice keeps its emission" begin
    lone = @brm JDF begin
        mu ~ 1 + x + (1 + x | a | g)
        y ~ Normal(mu, 1.0)
    end
    c = code(build(lone))
    @test isempty(lines_with(c, "__joint"))
    @test only(lines_with(c, "rows_dot_product(")) ==
          "vector[g_idx_n] r_mu_a_g = rows_dot_product(Z_mu_a_g, b_a_g[g_idx, col_idx_mu_a_g]);"
end

# Unconstrained coordinates of `a` mapped onto `b` by name: same values, same
# density and gradient.
function same_law(a, b, rename; rng=MersenneTwister(5), atol=1e-10)
    names_a = BridgeStan.param_unc_names(a)
    names_b = BridgeStan.param_unc_names(b)
    position = Dict(n => i for (i, n) in enumerate(names_b))
    order = [position[get(rename, n, n)] for n in names_a]
    @test sort(order) == eachindex(names_b)
    for _ in 1:3
        u_a = 0.5 .* randn(rng, length(names_a))
        u_b = similar(u_a)
        u_b[order] = u_a
        la, ga = BridgeStan.log_density_gradient(a, u_a; propto=false, jacobian=true)
        lb, gb = BridgeStan.log_density_gradient(b, u_b; propto=false, jacobian=true)
        @test la ≈ lb atol=atol rtol=1e-12
        @test ga ≈ gb[order] atol=atol rtol=1e-10
    end
end

JOINT_RUNTIME && @stestset "BridgeStan: the fused `||` program is the per-term law" begin
    # The plain block on `g`, `g2`, `g3` samples what the `|ID|` block of the
    # same term samples, on the same unconstrained scale: the intercept's
    # `log_scale` is the lowered `tau`'s unconstrained `log(tau)`.
    rename = Dict{String,String}()
    for (k, (plain_block, scale)) in enumerate((("r_mu_g", "log_scale"),
                                                ("r_mu_g2", "tau.1"), ("r_mu_g3", "tau.1")))
        lowered = "b_mu__g__nocor__$(k)_g"
        rename["$(plain_block)_$scale"] = "$(lowered)_tau.1"
        for j in 1:6
            rename["$(plain_block)_xi.$j"] = "$(lowered)_z_flat.$j"
        end
    end
    same_law(instantiate(build(per_term)).model, instantiate(build(unaddressed)).model, rename)
end

JOINT_RUNTIME && @stestset "BridgeStan: shared blocks on one index are the same law apart" begin
    rename = Dict(n => replace(n, "b_b_g_" => "b_b_g2_")
                  for n in BridgeStan.param_unc_names(instantiate(build(shared)).model))
    same_law(instantiate(build(shared)).model, instantiate(build(apart)).model, rename)
end
