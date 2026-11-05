using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, Enzyme
using LinearAlgebra, Serialization, Random, Dates
import DifferentiationInterface as DI
import LogDensityProblems as LDP
const BRM=BayesianRegressionModels
function timed(label,f)
    GC.gc()
    r=@timed f()
    println(now()," ",label," seconds=",r.time," allocated=",r.bytes," maxrss=",Sys.maxrss());flush(stdout)
    r.value
end
function main()
    S=deserialize(ARGS[1]);which=Symbol(ARGS[2]);C=parse(Int,ARGS[3]);d=getproperty(S,which)
    # Normal likelihood control at the actual read pairs, not the ODL target.
    n=length(d.time)*C
    data=(;east=repeat(d.east;inner=C),north=repeat(d.north;inner=C),
        time=repeat(d.time;inner=C),channel=repeat(collect(1:C);outer=length(d.time)),y=zeros(n))
    model=@brm data begin
        field ~ 0 + gmrf((east,north),time,channel;multiresolution=30)
        y ~ Normal(field,.2)
    end
    rk=timed("prepare",()->RKBRMI(model))
    query=timed("query",()->prepare_query(rk.model,BRM.rk_bound_plan(rk),:sampler))
    u=zeros(rk.model.layout.total)
    # Representative median ranges/times, SD 1. Learned parameters stay active.
    for entry in rk.model.layout.entries
        endswith(string(entry.name),".spatial_length") && (u[entry.offset]=log(which==:physical ? 94/2 : 38/2))
        endswith(string(entry.name),".temporal_length") && (u[entry.offset]=log(which==:physical ? 22.4 : 2.8))
    end
    println("CONTROL ",(;which,channels=C,read_rows=n,coordinates=length(u)));flush(stdout)
    first=timed("first primal value",()->Base.invokelatest(query,u))
    @assert isfinite(first)
    # A nonzero deterministic point exercises spatial weights and OU terms.
    for i in eachindex(u);u[i]+=.01*sin(i);end
    baseline=timed("nonzero primal value",()->Base.invokelatest(query,u))
    @assert isfinite(baseline)
    value=timed("warm primal value",()->Base.invokelatest(query,u))
    @assert isapprox(value,baseline;rtol=1e-12)
    println("native primal-only control passed; reverse is not measured");flush(stdout)
end
main()
