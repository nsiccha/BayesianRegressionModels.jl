using BridgeStan, Random, Printf
SRC = get(ENV, "FLAGS_STAN", "")   # a .stan file; e.g. one emitted by emission_survey.jl
DF  = get(ENV, "FLAGS_DATA", "")   # matching Stan JSON data file
isempty(SRC) && error("set FLAGS_STAN=<model.stan> and FLAGS_DATA=<data.json>")
C = mkpath(joinpath(tempdir(), "brm-flags"))
function bench(f; warm=1000, reps=7, inner=5000)
    for _ in 1:warm; f(); end
    minimum((t=time_ns(); for _ in 1:inner; f(); end; (time_ns()-t)/inner) for _ in 1:reps)
end
src = read(SRC, String)
println("compiler-flag sweep — BRM-emitted varying slope, N=1000 G=50")
@printf("  %-30s %10s %10s\n", "make_args", "density", "gradient")
for (lbl, ma) in ["default (BridgeStan stock)" => String[],
                  "STAN_CPP_OPTIMS=true"       => ["STAN_CPP_OPTIMS=true"],
                  "STAN_NO_RANGE_CHECKS=true"  => ["STAN_NO_RANGE_CHECKS=true"],
                  "both"                       => ["STAN_CPP_OPTIMS=true","STAN_NO_RANGE_CHECKS=true"]]
    try
        p = joinpath(C, "f_"*string(hash(ma); base=16)*".stan"); isfile(p)||write(p, src)
        m = BridgeStan.StanModel(p, DF; make_args=ma)
        q = 0.3 .* randn(MersenneTwister(1), BridgeStan.param_unc_num(m))
        @printf("  %-30s %10.1f %10.1f\n", lbl, bench(()->BridgeStan.log_density(m,q)),
                bench(()->BridgeStan.log_density_gradient(m,q)))
    catch e
        @printf("  %-30s FAILED: %s\n", lbl, first(replace(sprint(showerror,e), '\n'=>' '), 150))
    end
end
