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
    C in (1,2) || error("This control covers one or two channels")
    # Normal likelihood control at the actual read pairs, not the ODL target.
    n=length(d.time)*C
    data=(;east=repeat(d.east;inner=C),north=repeat(d.north;inner=C),
        time=repeat(d.time;inner=C),channel=repeat(collect(1:C);outer=length(d.time)),y=zeros(n))
    model=if C==1
        @brm data begin
            field ~ 0 + gmrf((east,north),time,channel;multiresolution=30)
            y ~ Normal(field,.2)
        end
    else
        @brm data begin
            Omega ~ LKJCholesky(2,2.0)
            field ~ 0 + gmrf((east,north),time,channel;multiresolution=30,cor=Omega)
            y ~ Normal(field,.2)
        end
    end
    rk=timed("prepare",()->RKBRMI(model))
    problem=timed("problem",()->rk_logdensity_problem(rk;ad_backend=DI.AutoEnzyme(;mode=Enzyme.Reverse)))
    u=zeros(LDP.dimension(problem))
    # Representative median ranges/times, SD 1. Learned parameters stay active.
    for entry in rk.model.layout.entries
        endswith(string(entry.name),".spatial_length") && (u[entry.offset]=log(which==:physical ? 94/2 : 38/2))
        endswith(string(entry.name),".temporal_length") && (u[entry.offset]=log(which==:physical ? 22.4 : 2.8))
    end
    println("CONTROL ",(;which,channels=C,read_rows=n,coordinates=length(u)));flush(stdout)
    first=timed("first value gradient",()->LDP.logdensity_and_gradient(problem,u))
    @assert isfinite(first[1]) && all(isfinite,first[2])
    # A nonzero deterministic point exercises the basis and temporal gradients.
    for i in eachindex(u);u[i]+=.01*sin(i);end
    baseline=timed("warm value gradient",()->LDP.logdensity_and_gradient(problem,u))
    @assert isfinite(baseline[1]) && all(isfinite,baseline[2])
    value=timed("warm value",()->LDP.logdensity(problem,u))
    @assert isapprox(value,baseline[1];rtol=1e-12)
    # Central differences of the native primal along a dense direction and
    # one concentrated on the small (hyperparameter and correlation) entries.
    small=falses(length(u))
    for entry in rk.model.layout.entries
        entry.size<=16 && (small[entry.offset:entry.offset+entry.size-1].=true)
    end
    # At this scale a density's rounding error divided by a small step can
    # exceed the tolerance, so a single step cannot tell a wrong gradient from
    # cancellation. Report the ladder and require one step to agree.
    f=x->LDP.logdensity(problem,x)
    for (k,v) in enumerate(([cos(3i) for i in eachindex(u)],[small[i] ? 1.0 : 1e-3*sin(7i) for i in eachindex(u)]))
        v./=sqrt(sum(abs2,v))
        gv=sum(baseline[2].*v);best=Inf
        for h in (1e-2,1e-3,1e-4,1e-5)
            dd=timed("directional difference $k h=$h",()->(f(u.+h.*v)-f(u.-h.*v))/(2h))
            rel=abs(dd-gv)/max(1.0,abs(dd));best=min(best,rel)
            println("DIRECTION ",(;direction=k,h,small_coordinates=count(small),directional_fd=dd,gradient_dot=gv,rel_error=rel));flush(stdout)
        end
        @assert best<1e-4
    end
    println("native control passed");flush(stdout)
end
main()
