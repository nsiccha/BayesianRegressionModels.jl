# GLM matrix: separate (1) emission overhead, (2) link/family fusion, (3) Stan *_glm primitive.
using BayesianRegressionModels, Distributions, StanBlocks, BridgeStan, Random, Printf, Statistics
using LogExpFunctions: logit
const CACHE = mkpath(joinpath(tempdir(), "brm-glm-matrix"))

jn(x::Integer)=string(x); jn(x::Real)= isinteger(x) ? @sprintf("%.1f",x) : repr(x)
ja(v)= "["*join(jn.(v),",")*"]"
jm(M)= "["*join(["["*join(jn.(M[i,:]),",")*"]" for i in 1:size(M,1)],",")*"]"
function wjson(path, prs)
    open(path,"w") do io
        println(io,"{"); println(io, join(["  \"$k\": "*(v isa AbstractMatrix ? jm(v) : v isa AbstractVector ? ja(v) : jn(v)) for (k,v) in prs], ",\n")); println(io,"}")
    end; path
end
build(nm, code) = (p = joinpath(CACHE, nm*"_"*string(hash(code); base=16)*".stan"); isfile(p)||write(p,code); p)
inst(p, df) = BridgeStan.StanModel(p, df)
function bench(f; warm=1000, reps=7, inner=5000)
    for _ in 1:warm; f(); end
    minimum((t=time_ns(); for _ in 1:inner; f(); end; (time_ns()-t)/inner) for _ in 1:reps)
end

function run_family(tag, N, K)
    rng = MersenneTwister(5)
    X = hcat(ones(N), randn(rng, N, K-1))
    x, z, w = X[:,2], X[:,3], X[:,min(4,K)]
    eta = X * (0.3 .* randn(rng, K))
    y  = eta .+ 0.5 .* randn(rng, N)
    yb = Int.(rand(rng, N) .< 1 ./ (1 .+ exp.(-eta)))
    yc = [rand(rng, Poisson(exp(clamp(e, -3, 3)))) for e in eta]
    d = (; x, z, w, y, yb, yc)
    df = wjson(joinpath(CACHE,"d_$(tag)_$(N)_$(K).json"),
        ["N"=>N,"K"=>K,"Xd"=>X,"x_n"=>N,"x"=>x,"z_n"=>N,"z"=>z,"w_n"=>N,"w"=>w,
         "y_n"=>N,"y"=>y,"yb_n"=>N,"yb"=>yb,"yc_n"=>N,"yc"=>yc])

    HEAD = "data { int N; int K; matrix[N,K] Xd; int y_n; vector[y_n] y; int yb_n; array[yb_n] int yb; int yc_n; array[yc_n] int yc; }\n"
    if tag == :gauss
        brm = @brm d begin sigma ~ Exponential(1); mu ~ 1 + x + z + w; y ~ Normal(mu, sigma) end
        variants = [
          "1 BRM emitted"        => StanBlocks.stan_code(SBBRMI(brm; mod=@__MODULE__).model),
          "2 hand X*beta"        => HEAD*"parameters { real<lower=0.0> sigma; vector[K] b; }\nmodel { sigma ~ exponential(1.0); b ~ std_normal(); y ~ normal(Xd*b, sigma); }\n",
          "3 hand normal_id_glm" => HEAD*"parameters { real<lower=0.0> sigma; vector[K] b; }\nmodel { sigma ~ exponential(1.0); b ~ std_normal(); y ~ normal_id_glm(Xd, 0.0, b, sigma); }\n"]
    elseif tag == :bernoulli
        brm = @brm d begin logit(p) ~ 1 + x + z + w; yb ~ Bernoulli(p) end
        variants = [
          "1 BRM emitted"           => StanBlocks.stan_code(SBBRMI(brm; mod=@__MODULE__).model),
          "2 hand inv_logit+bern"   => HEAD*"parameters { vector[K] b; }\nmodel { b ~ std_normal(); yb ~ bernoulli(inv_logit(Xd*b)); }\n",
          "3 hand bernoulli_logit"  => HEAD*"parameters { vector[K] b; }\nmodel { b ~ std_normal(); yb ~ bernoulli_logit(Xd*b); }\n",
          "4 hand bern_logit_glm"   => HEAD*"parameters { vector[K] b; }\nmodel { b ~ std_normal(); yb ~ bernoulli_logit_glm(Xd, 0.0, b); }\n"]
    else
        brm = @brm d begin log(lam) ~ 1 + x + z + w; yc ~ Poisson(lam) end
        variants = [
          "1 BRM emitted"        => StanBlocks.stan_code(SBBRMI(brm; mod=@__MODULE__).model),
          "2 hand exp+poisson"   => HEAD*"parameters { vector[K] b; }\nmodel { b ~ std_normal(); yc ~ poisson(exp(Xd*b)); }\n",
          "3 hand poisson_log"   => HEAD*"parameters { vector[K] b; }\nmodel { b ~ std_normal(); yc ~ poisson_log(Xd*b); }\n",
          "4 hand poisson_log_glm"=>HEAD*"parameters { vector[K] b; }\nmodel { b ~ std_normal(); yc ~ poisson_log_glm(Xd, 0.0, b); }\n"]
    end
    models = [(nm, inst(build(replace("$(tag)_$nm", r"[^A-Za-z0-9]"=>"_"), c), df)) for (nm,c) in variants]

    @printf("\n%s  family=%s  N=%d  K=%d %s\n", "="^30, tag, N, K, "="^30)
    dim = BridgeStan.param_unc_num(models[1][2])
    qs = [0.3 .* randn(MersenneTwister(9+i), dim) for i in 1:3]
    base_lp = [BridgeStan.log_density(models[1][2], q; propto=false, jacobian=true) for q in qs]
    base_gr = [BridgeStan.log_density_gradient(models[1][2], q; propto=false, jacobian=true)[2] for q in qs]
    println("-- correctness vs BRM emitted --")
    for (nm,m) in models
        lp = [BridgeStan.log_density(m,q; propto=false, jacobian=true) for q in qs]
        gr = [BridgeStan.log_density_gradient(m,q; propto=false, jacobian=true)[2] for q in qs]
        dlp = maximum(abs.((lp .- lp[1]) .- (base_lp .- base_lp[1])))
        dgr = maximum(maximum(abs.(g .- b)) for (g,b) in zip(gr,base_gr))
        @printf("   %-24s max|Δ(Δlp)|=%.3e  max|Δgrad|=%.3e\n", nm, dlp, dgr)
    end
    println("-- timing ns/call --")
    q = 0.3 .* randn(MersenneTwister(4), dim)
    inner = N <= 200 ? 5000 : 1000
    res = [(nm, bench(()->BridgeStan.log_density(m,q); inner=inner),
                bench(()->BridgeStan.log_density_gradient(m,q); inner=inner)) for (nm,m) in models
          ]
    ref = res[end]
    @printf("   %-24s %10s %10s %8s %8s\n","variant","density","gradient","d/best","g/best")
    for (nm,td,tg) in res
        @printf("   %-24s %10.1f %10.1f %8.2f %8.2f\n", nm, td, tg, td/ref[2], tg/ref[3])
    end
    res
end

for tag in (:gauss, :bernoulli, :poisson), (N,K) in ((100,4), (2000,4))
    run_family(tag, N, K)
end
