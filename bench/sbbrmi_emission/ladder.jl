# Isolation ladder for the SBBRMI varying-slope emission cost.
# V0 = verbatim BRM emission; V1..V5 remove ONE piece of generic machinery each;
# V6 = minimal hand non-centered; V7 = minimal hand centered (reporter's control).
using BayesianRegressionModels
using Distributions
using StanBlocks
using BridgeStan
using Random, Printf, Statistics
const BRM = BayesianRegressionModels

const CACHE = mkpath(joinpath(tempdir(), "brm-sbbrmi-emission"))

json_num(x::Integer) = string(x)
json_num(x::Real) = (isinteger(x) ? @sprintf("%.1f", x) : repr(x))
json_arr(v) = "[" * join(json_num.(v), ",") * "]"
function write_json(path, pairs)
    open(path, "w") do io
        println(io, "{")
        entries = String[]
        for (k, v) in pairs
            push!(entries, "  \"$k\": " * (v isa AbstractVector ? json_arr(v) : json_num(v)))
        end
        println(io, join(entries, ",\n"))
        println(io, "}")
    end
    path
end

function make_data(N, G; seed=11)
    rng = MersenneTwister(seed)
    x = randn(rng, N)
    g = rand(rng, 1:G, N)
    # guarantee every level is used
    for j in 1:min(G, N); g[j] = j; end
    y = randn(rng, N)
    (; N, G, x, g, y)
end

function brm_code(d)
    nt = (; x=d.x, group=d.g, y=d.y)
    m = @brm nt begin
        sigma ~ Exponential(2)
        mu ~ 0 + x + (0 + x | p | group)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    StanBlocks.stan_code(SBBRMI(m; mod=@__MODULE__).model)
end

# --- hand variants. Parameter DECLARATION ORDER matches BRM exactly, so the
# --- unconstrained coordinate vector q is elementwise comparable across V0..V6.
const HEAD = """
data {
    int n_terms_p_group;
    int n_group;
    int x_n;
    vector[x_n] x;
    int group_idx_n;
    array[group_idx_n] int group_idx;
    int y_n;
    vector[y_n] y;
}
"""

# V1: drop the LKJ / cholesky_factor_corr block (K=1 specialisation).
V1 = HEAD * """
transformed data {
    matrix[x_n, 1] X_mu = to_matrix(x, x_n, 1);
    int pop_mu_n_covariates = 1;
}
parameters {
    vector<lower=0.0>[n_terms_p_group] b_p_group_tau;
    vector[(n_terms_p_group * n_group)] b_p_group_z_flat;
    real<lower=0.0> sigma;
    vector[pop_mu_n_covariates] pop_mu_beta_pop;
}
transformed parameters {
    matrix[n_terms_p_group, n_group] b_p_group_z = to_matrix(b_p_group_z_flat, n_terms_p_group, n_group);
    matrix[n_group, n_terms_p_group] b_p_group = ((diag_matrix(b_p_group_tau) * b_p_group_z)');
    vector[x_n] pop_mu = (X_mu * pop_mu_beta_pop);
    vector[group_idx_n] r_mu_p_group = (x .* b_p_group[group_idx, 1]);
    vector[x_n] mu = (pop_mu + r_mu_p_group);
}
model {
    b_p_group_tau ~ brm_ranef_sd([1]', [1.0]');
    b_p_group_z_flat ~ std_normal();
    sigma ~ exponential((1.0 ./ 2));
    pop_mu_beta_pop ~ std_normal();
    y ~ normal(mu, sigma);
}
"""

# V2: also drop the brm_ranef_sd_lpdf UDF (branchy loop + literal vectors).
V2 = HEAD * """
transformed data {
    matrix[x_n, 1] X_mu = to_matrix(x, x_n, 1);
    int pop_mu_n_covariates = 1;
}
parameters {
    vector<lower=0.0>[n_terms_p_group] b_p_group_tau;
    vector[(n_terms_p_group * n_group)] b_p_group_z_flat;
    real<lower=0.0> sigma;
    vector[pop_mu_n_covariates] pop_mu_beta_pop;
}
transformed parameters {
    matrix[n_terms_p_group, n_group] b_p_group_z = to_matrix(b_p_group_z_flat, n_terms_p_group, n_group);
    matrix[n_group, n_terms_p_group] b_p_group = ((diag_matrix(b_p_group_tau) * b_p_group_z)');
    vector[x_n] pop_mu = (X_mu * pop_mu_beta_pop);
    vector[group_idx_n] r_mu_p_group = (x .* b_p_group[group_idx, 1]);
    vector[x_n] mu = (pop_mu + r_mu_p_group);
}
model {
    b_p_group_tau ~ exponential(1.0);
    b_p_group_z_flat ~ std_normal();
    sigma ~ exponential((1.0 ./ 2));
    pop_mu_beta_pop ~ std_normal();
    y ~ normal(mu, sigma);
}
"""

# V3: also drop reshape/transpose — keep the effect as a flat vector.
V3 = HEAD * """
transformed data {
    matrix[x_n, 1] X_mu = to_matrix(x, x_n, 1);
    int pop_mu_n_covariates = 1;
}
parameters {
    vector<lower=0.0>[n_terms_p_group] b_p_group_tau;
    vector[(n_terms_p_group * n_group)] b_p_group_z_flat;
    real<lower=0.0> sigma;
    vector[pop_mu_n_covariates] pop_mu_beta_pop;
}
transformed parameters {
    vector[n_group] b_p_group = b_p_group_tau[1] * b_p_group_z_flat;
    vector[x_n] pop_mu = (X_mu * pop_mu_beta_pop);
    vector[group_idx_n] r_mu_p_group = (x .* b_p_group[group_idx]);
    vector[x_n] mu = (pop_mu + r_mu_p_group);
}
model {
    b_p_group_tau ~ exponential(1.0);
    b_p_group_z_flat ~ std_normal();
    sigma ~ exponential((1.0 ./ 2));
    pop_mu_beta_pop ~ std_normal();
    y ~ normal(mu, sigma);
}
"""

# V4: also drop the design-matrix multiply for a 1-column population block.
V4 = HEAD * """
parameters {
    vector<lower=0.0>[n_terms_p_group] b_p_group_tau;
    vector[(n_terms_p_group * n_group)] b_p_group_z_flat;
    real<lower=0.0> sigma;
    vector[1] pop_mu_beta_pop;
}
transformed parameters {
    vector[n_group] b_p_group = b_p_group_tau[1] * b_p_group_z_flat;
    vector[x_n] pop_mu = pop_mu_beta_pop[1] * x;
    vector[group_idx_n] r_mu_p_group = (x .* b_p_group[group_idx]);
    vector[x_n] mu = (pop_mu + r_mu_p_group);
}
model {
    b_p_group_tau ~ exponential(1.0);
    b_p_group_z_flat ~ std_normal();
    sigma ~ exponential(0.5);
    pop_mu_beta_pop ~ std_normal();
    y ~ normal(mu, sigma);
}
"""

# V5: also fuse the three transformed-parameter vectors into one expression.
V5 = HEAD * """
parameters {
    vector<lower=0.0>[n_terms_p_group] b_p_group_tau;
    vector[(n_terms_p_group * n_group)] b_p_group_z_flat;
    real<lower=0.0> sigma;
    vector[1] pop_mu_beta_pop;
}
transformed parameters {
    vector[x_n] mu = (pop_mu_beta_pop[1] + b_p_group_tau[1] * b_p_group_z_flat[group_idx]) .* x;
}
model {
    b_p_group_tau ~ exponential(1.0);
    b_p_group_z_flat ~ std_normal();
    sigma ~ exponential(0.5);
    pop_mu_beta_pop ~ std_normal();
    y ~ normal(mu, sigma);
}
"""

# V6: minimal hand-written non-centered (mu as a local, not a TP).
V6 = HEAD * """
parameters {
    real<lower=0.0> tau;
    vector[n_group] z;
    real<lower=0.0> sigma;
    real beta;
}
model {
    tau ~ exponential(1.0);
    z ~ std_normal();
    sigma ~ exponential(0.5);
    beta ~ std_normal();
    y ~ normal((beta + tau * z[group_idx]) .* x, sigma);
}
"""

# V7: minimal hand-written CENTERED (different geometry; timing control only).
V7 = HEAD * """
parameters {
    real<lower=0.0> tau;
    vector[n_group] b;
    real<lower=0.0> sigma;
    real beta;
}
model {
    tau ~ exponential(1.0);
    b ~ normal(0.0, tau);
    sigma ~ exponential(0.5);
    beta ~ std_normal();
    y ~ normal((beta + b[group_idx]) .* x, sigma);
}
"""

const RANEF_SD_FN = """
functions {
real brm_ranef_sd_lpdf(vector tau, vector family, vector rate) {
    int n = dims(tau)[1];
    if (dims(family)[1] != n) reject("brm_ranef_sd_lpdf: dim mismatch");
    if (dims(rate)[1] != n) reject("brm_ranef_sd_lpdf: dim mismatch");
    real rv = 0.0;
    for(i in 1:n) {
        if((family[i] == 0)) { rv += std_normal_lpdf(tau[i]); }
        else { rv += exponential_lpdf(tau[i] | rate[i]); }
    }
    return rv;
}
}
"""

function build(name, code, datafile)
    src = joinpath(CACHE, name * "_" * string(hash(code); base=16) * ".stan")
    isfile(src) || write(src, code)
    BridgeStan.StanModel(src, datafile)
end

timeit(f, n) = (t = time_ns(); for _ in 1:n; f(); end; (time_ns() - t) / n)
function bench(f; warm=2000, reps=9, inner=20_000)
    for _ in 1:warm; f(); end
    minimum(timeit(f, inner) for _ in 1:reps)
end

function run_case(N, G; verify=true, inner=20_000)
    d = make_data(N, G)
    df = write_json(joinpath(CACHE, "data_$(N)_$(G).json"),
        ["n_terms_p_group"=>1, "n_group"=>G, "x_n"=>N, "x"=>d.x,
         "group_idx_n"=>N, "group_idx"=>d.g, "y_n"=>N, "y"=>d.y])
    v0 = brm_code(d)
    variants = ["V0 BRM emitted"=>v0,
                "V1 -LKJ/cholesky"=>RANEF_SD_FN*V1,
                "V2 -brm_ranef_sd UDF"=>V2,
                "V3 -reshape/transpose"=>V3,
                "V4 -design-matrix mul"=>V4,
                "V5 -split TP vectors"=>V5,
                "V6 hand non-centered"=>V6,
                "V7 hand centered"=>V7]
    models = [(nm, build(replace(nm, r"[^A-Za-z0-9]"=>"_"), c, df)) for (nm, c) in variants]

    println("\n" * "="^96)
    @printf("N = %d rows, G = %d groups\n", N, G)
    println("="^96)

    rng = MersenneTwister(7)
    dim0 = BridgeStan.param_unc_num(models[1][2])
    qs = [0.4 .* randn(rng, dim0) for _ in 1:3]

    if verify
        println("-- correctness (vs V0; Δlp differences between positions, and gradients) --")
        base_lp = [BridgeStan.log_density(models[1][2], q; propto=false, jacobian=true) for q in qs]
        base_gr = [BridgeStan.log_density_gradient(models[1][2], q; propto=false, jacobian=true)[2] for q in qs]
        for (nm, m) in models
            dim = BridgeStan.param_unc_num(m)
            if startswith(nm, "V7")
                @printf("  %-24s dim=%d  (CENTERED — different parameterisation, timing control only)\n", nm, dim)
                continue
            end
            if dim != dim0
                @printf("  %-24s dim=%d  (≠ %d — separate parameterisation, timing only)\n", nm, dim, dim0)
                continue
            end
            lp = [BridgeStan.log_density(m, q; propto=false, jacobian=true) for q in qs]
            gr = [BridgeStan.log_density_gradient(m, q; propto=false, jacobian=true)[2] for q in qs]
            dlp = maximum(abs.((lp .- lp[1]) .- (base_lp .- base_lp[1])))
            dgr = maximum(maximum(abs.(g .- b)) for (g, b) in zip(gr, base_gr))
            rel = maximum(maximum(abs.(g .- b)) / max(1.0, maximum(abs.(b))) for (g, b) in zip(gr, base_gr))
            @printf("  %-24s dim=%d  max|Δ(Δlp)|=%.3e  max|Δgrad|=%.3e (rel %.3e)\n", nm, dim, dlp, dgr, rel)
        end
    end

    println("-- timing (ns/call, min of 9 x $inner) --")
    @printf("  %-24s %12s %12s %10s %10s\n", "variant", "density", "gradient", "d/V6", "g/V6")
    res = Dict{String,Tuple{Float64,Float64}}()
    for (nm, m) in models
        dim = BridgeStan.param_unc_num(m)
        q = 0.4 .* randn(MersenneTwister(3), dim)
        td = bench(() -> BridgeStan.log_density(m, q), inner=inner)
        tg = bench(() -> BridgeStan.log_density_gradient(m, q), inner=inner)
        res[nm] = (td, tg)
    end
    ref = res["V6 hand non-centered"]
    for (nm, _) in models
        td, tg = res[nm]
        @printf("  %-24s %12.1f %12.1f %10.2f %10.2f\n", nm, td, tg, td/ref[1], tg/ref[2])
    end
    res
end

cases = [(4, 3), (100, 20), (1000, 50), (10_000, 200)]
all_res = Dict()
for (i, (N, G)) in enumerate(cases)
    inner = N <= 100 ? 20_000 : (N <= 1000 ? 5_000 : 500)
    all_res[(N, G)] = run_case(N, G; verify=(i <= 3), inner=inner)
end

println("\n" * "="^96)
println("SCALING: gradient ns/call, and V0/V6 ratio")
println("="^96)
@printf("  %-14s", "variant")
for (N, G) in cases; @printf("%16s", "N=$N/G=$G"); end
println()
for nm in ["V0 BRM emitted", "V1 -LKJ/cholesky", "V2 -brm_ranef_sd UDF", "V3 -reshape/transpose",
           "V4 -design-matrix mul", "V5 -split TP vectors", "V6 hand non-centered", "V7 hand centered"]
    @printf("  %-14s", first(nm, 14))
    for (N, G) in cases
        @printf("%16.1f", all_res[(N,G)][nm][2])
    end
    println()
end
@printf("  %-14s", "V0/V6 ratio")
for (N, G) in cases
    r = all_res[(N,G)]
    @printf("%16.2f", r["V0 BRM emitted"][2] / r["V6 hand non-centered"][2])
end
println()
