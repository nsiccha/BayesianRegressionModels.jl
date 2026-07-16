using Test
using BayesianRegressionModels
using BridgeStan
using LogDensityProblems
using StanBlocks

const STRESS_CACHE = joinpath(tempdir(), "brm-plate-stress")
const RUN_BRIDGESTAN = get(ENV, "BRM_PLATE_STRESS_RUNTIME", "1") != "0"
const RUN_GAPS = get(ENV, "BRM_PLATE_STRESS_GAPS", "1") != "0"
const STRESS_CASE = get(ENV, "BRM_PLATE_STRESS_CASE", "all")

function stanc_accepts(model)
    result = StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic=false)
    result.ok || @error "stanc rejected BRM PLATE stress case" output=result.output
    result.ok
end

function bridgestan_accepts(model)
    code = StanBlocks.stan_code(model)
    path = joinpath(STRESS_CACHE, string(hash(code)) * ".stan")
    problem = StanBlocks.stan_instantiate(model; path)
    dimension = LogDensityProblems.dimension(problem)
    q = zeros(dimension)
    lp = LogDensityProblems.logdensity(problem, q)
    lp_grad, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    isfinite(lp) && isfinite(lp_grad) && length(gradient) == dimension && all(isfinite, gradient)
end

# Fixed-width correlated group effect: the first BRM migration target for
# `(1 + x | ID | group)` buckets and fixed-width Bordet parameter vectors.
StanBlocks.@slic brm_fixed_correlated_cell(
    k::int,
    L::cholesky_factor_corr[k],
    tau::vector[k],
) = begin
    z::vector[k] ~ std_normal()
    return diag_pre_multiply(tau, L) * z
end

const fixed_correlated = StanBlocks.@slic (;
    n_groups=5,
    k=3,
    y=[0.2, -0.1, 0.3, 0.0, 0.4],
) begin
    L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
    tau::vector[k] ~ normal(0.0, 1.0; lower=0.0)
    b::vector[k] ~ plate(y; outer=(n_groups,)) do yi
        cell ~ brm_fixed_correlated_cell(k, L, tau)
        yi ~ normal(cell[1], 0.5)
        cell
    end
end

# Shipped ragged BRM composition: informative ragged Cholesky parameters live
# at the top level while the called PLATE cell consumes K[g] and L[g].
StanBlocks.@slic brm_ragged_latent_cell(k::int, L::matrix[k, k]) = begin
    z::vector[k] ~ std_normal()
    return L * z
end

const ragged_correlated = StanBlocks.@slic (;
    K=[1, 2, 4],
    y=0.2,
) begin
    L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
    b ~ plate(; outer=length(K)) do g
        cell ~ brm_ragged_latent_cell(K[g], L[g])
        cell
    end
    y ~ normal(sum(b[3]), 1.0)
end

# Likelihood-bearing scalar cells exercise captured shared parameters, sliced
# data, block routing, and deterministic PLATE results.
const scalar_likelihood = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0, 0.4, -0.2],
    mu0=0.25,
) begin
    sigma ~ normal(0.0, 1.0; lower=0.0)
    theta ~ plate(y; outer=(6,)) do yi
        z ~ std_normal()
        mu = mu0 + sigma * z
        yi ~ normal(mu, sigma)
        mu
    end
end


# Dense N-D vector cells exercise logical-cell axes plus multiple PLATE axes.
const nd_vector = StanBlocks.@slic (;y=[0.2 -0.1 0.3; 0.0 0.4 -0.2], k=2) begin
    theta::vector[k] ~ plate(y; outer=(2, 3)) do yi
        z::vector[k] ~ std_normal()
        yi ~ normal(z[1], 1.0)
        z
    end
end

# Two independent grouping factors should remain two independent plates rather
# than forcing a monolithic crossed allocator.
const crossed_groups = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0],
    subject=[1, 1, 2, 2],
    item=[1, 2, 1, 2],
) begin
    b_subject ~ plate(; outer=(2,)) do s
        z_subject ~ std_normal()
        z_subject
    end
    b_item ~ plate(; outer=(2,)) do i
        z_item ~ std_normal()
        z_item
    end
    y ~ normal(b_subject[subject] + b_item[item], 1.0)
end


# Known gaps. `@test_broken` makes the suite green today, but turns a newly
# supported capability into an unexpected pass that must be promoted above.
const matrix_cell_gap = StanBlocks.@slic (;n=3, k=2) begin
    theta::matrix[k, k] ~ plate(; outer=(n,)) do g
        z::vector[k * k] ~ std_normal()
        to_matrix(z, k, k)
    end
end

const constrained_cell_gap = StanBlocks.@slic (;n=3, k=3) begin
    p ~ plate(; outer=(n,)) do g
        cell::simplex[k] ~ dirichlet(rep_vector(1.0, k))
        cell
    end
end

const vararg_cell_gap = StanBlocks.@slic (;
    x=[0.1, 0.2, 0.3],
    y=[0.3, 0.2, 0.1],
) begin
    theta ~ plate(x, y; outer=(3,)) do cells...
        z ~ std_normal()
        z + sum(cells)
    end
end

# Ragged PLATE results work, but a ragged *input slice* is currently inferred
# as `anything`, so it cannot feed the typed BRM subject/series likelihood cell.
StanBlocks.@slic brm_ragged_observed_cell(
    k::int,
    L::matrix[k, k],
    y::vector[k],
) = begin
    z::vector[k] ~ std_normal()
    b = L * z
    y ~ normal(b, 0.5)
    return b
end

const ragged_input_gap = StanBlocks.@slic (;
    K=[1, 2, 4],
    groups=[1, 2, 3],
    y=[[0.2], [-0.1, 0.3], [0.0, 0.4, -0.2, 0.1]],
) begin
    L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
    b ~ plate(groups, y; outer=length(K)) do g, yg
        cell ~ brm_ragged_observed_cell(K[g], L[g], yg)
        cell
    end
end

const crossed_name_gap = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0],
    subject=[1, 1, 2, 2],
    item=[1, 2, 1, 2],
) begin
    b_subject ~ plate(; outer=(2,)) do s
        z ~ std_normal()
        z
    end
    b_item ~ plate(; outer=(2,)) do i
        z ~ std_normal()
        z
    end
    y ~ normal(b_subject[subject] + b_item[item], 1.0)
end


@info "BRM PLATE stress environment" StanBlocks=Base.pkgversion(StanBlocks) BridgeStan=Base.pkgversion(BridgeStan) RUN_BRIDGESTAN RUN_GAPS STRESS_CASE

@testset "StanBlocks PLATE — BRM acceptance stress" begin
    @testset "transpile and stanc" begin
        for (name, model) in (
            "scalar likelihood" => scalar_likelihood,
            "fixed correlated" => fixed_correlated,
            "ragged correlated" => ragged_correlated,
            "N-D vector" => nd_vector,
            "crossed groups" => crossed_groups,
        )
            STRESS_CASE == "all" || STRESS_CASE == name || continue
            @testset "$name" begin
                @info "Running BRM PLATE compiler case" name
                transpiles = StanBlocks.transpiles(model; re=false)
                @test transpiles
                transpiles && @test stanc_accepts(model)
            end
        end
    end

    @testset "BridgeStan runtime" begin
        if RUN_BRIDGESTAN
            for (name, model) in (
                "scalar likelihood" => scalar_likelihood,
                "fixed correlated" => fixed_correlated,
                "ragged correlated" => ragged_correlated,
            )
                STRESS_CASE == "all" || STRESS_CASE == name || continue
                @info "Running BRM PLATE BridgeStan case" name
                @test bridgestan_accepts(model)
            end
        else
            @info "Skipping BridgeStan runtime gate (BRM_PLATE_STRESS_RUNTIME=0)"
        end
    end

    @testset "known capability gaps" begin
        if RUN_GAPS
            @test_broken StanBlocks.transpiles(matrix_cell_gap; re=false)
            @test_broken StanBlocks.transpiles(ragged_input_gap; re=false)
            @test_broken StanBlocks.transpiles(crossed_name_gap; re=false)
            @test StanBlocks.transpiles(constrained_cell_gap; re=false)
            @test_broken stanc_accepts(constrained_cell_gap)
            @test_broken StanBlocks.transpiles(vararg_cell_gap; re=false)
            @test_broken occursin("reduce_sum", StanBlocks.stan_code(scalar_likelihood))
        else
            @info "Skipping expected-failure probes (BRM_PLATE_STRESS_GAPS=0)"
        end
    end
end
