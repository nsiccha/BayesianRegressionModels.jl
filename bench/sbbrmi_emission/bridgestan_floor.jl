using BridgeStan, Random, Printf
C = mkpath(joinpath(tempdir(), "brm-floor"))
function mk(nm, code; make_args=String[])
    p = joinpath(C, nm*"_"*string(hash((code,make_args)); base=16)*".stan"); isfile(p)||write(p,code)
    BridgeStan.StanModel(p, ""; make_args)
end
function bench(f; warm=2000, reps=9, inner=20000)
    for _ in 1:warm; f(); end
    minimum((t=time_ns(); for _ in 1:inner; f(); end; (time_ns()-t)/inner) for _ in 1:reps)
end
progs = [
 "empty (0 params)"      => "parameters { real dummy; }\nmodel { }\n",
 "1 std_normal"          => "parameters { real x; }\nmodel { x ~ std_normal(); }\n",
 "6 std_normal vector"   => "parameters { vector[6] x; }\nmodel { x ~ std_normal(); }\n",
 "6 + 1 TP vector"       => "parameters { vector[6] x; }\ntransformed parameters { vector[6] z = 2*x; }\nmodel { z ~ std_normal(); }\n",
 "6 + 1 UDF call"        => "functions { real f(real a){ return a; } }\nparameters { vector[6] x; }\nmodel { for (i in 1:6) target += std_normal_lpdf(f(x[i])); }\n",
]
@printf("%-24s %10s %10s\n", "program", "density", "gradient")
println("-"^48)
for (nm, code) in progs
    m = mk(replace(nm, r"[^A-Za-z0-9]"=>"_"), code)
    q = 0.3 .* randn(MersenneTwister(1), BridgeStan.param_unc_num(m))
    td = bench(()->BridgeStan.log_density(m,q)); tg = bench(()->BridgeStan.log_density_gradient(m,q))
    @printf("%-24s %10.1f %10.1f\n", nm, td, tg)
end
println("\n--- compiler flag sweep on the emitted varying-slope model ---")
src = read(joinpath(ENV["EMIT"], "D3_corr_K1.stan"), String)
for (lbl, ma) in ["default" => String[],
                  "STAN_CPP_OPTIMS=true" => ["STAN_CPP_OPTIMS=true"],
                  "STAN_NO_RANGE_CHECKS=true" => ["STAN_NO_RANGE_CHECKS=true"],
                  "both" => ["STAN_CPP_OPTIMS=true","STAN_NO_RANGE_CHECKS=true"]]
    try
        m = mk("flags_"*replace(lbl, r"[^A-Za-z0-9]"=>"_"), src; make_args=ma)
        q = 0.3 .* randn(MersenneTwister(1), BridgeStan.param_unc_num(m))
        @printf("  %-28s density %8.1f   gradient %8.1f\n", lbl,
                bench(()->BridgeStan.log_density(m,q); inner=10000),
                bench(()->BridgeStan.log_density_gradient(m,q); inner=10000))
    catch e
        @printf("  %-28s FAILED: %s\n", lbl, first(sprint(showerror,e), 120))
    end
end
