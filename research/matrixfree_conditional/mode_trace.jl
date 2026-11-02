# Trace the existing mode primitives without changing their tolerances or policy.
# RUN: julia --project=test research/matrixfree_conditional/mode_trace.jl NX NY NT [STEPS]
include("nonlinear.jl")

function trace_mode(nx,ny,nt; steps=12)
    data=lattice_data(nx,ny,nt)
    setup=@timed lattice_problem(data)
    p=setup.value
    @assert length(p.components)==1
    pop=BRM._brm_spl_population(p,fill(0.1,length(p.population)))
    n=length(p.integrated)
    z=zeros(n); fc=zeros(1); g=zeros(n); h=BRM._brm_spl_precision(p)
    BRM._brm_spl_assemble!(fc,g,h,p,pop.A,pop.C,z; derivatives=true,uA=pop.uA)
    println("setup\t",setup.time,'\t',setup.bytes,"\ttolerance=",p.tol)
    println("iteration\tvalue\tdecrement\tgradient_inf\tdirection_inf\tstep_ratio\tobserved_gain")
    for iteration in 1:steps
        delta=BRM._brm_spl_direction(p,h,g)
        if delta===nothing
            println("direction_failed\t",p.algebra.diagnostics)
            return
        end
        decrement=dot(g,delta)
        before=only(fc); gradient_inf=norm(g,Inf); direction_inf=norm(delta,Inf)
        previous=copy(z)
        accepted=BRM._brm_spl_step!(p,pop,z,delta,g,h,fc,falses(1))
        BRM._brm_spl_assemble!(fc,g,h,p,pop.A,pop.C,z; derivatives=true,uA=pop.uA)
        @printf("%d\t%.16g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
            iteration,before,decrement,gradient_inf,direction_inf,
            norm(z-previous)/norm(delta),only(fc)-before)
        flush(stdout)
        accepted || error("mode line search failed")
        decrement<=p.tol && break
    end
    println("diagnostics\t",p.algebra.diagnostics)
end

length(ARGS)>=3 || error("pass NX NY NT [STEPS]")
trace_mode(parse.(Int,ARGS[1:3])...;steps=length(ARGS)>3 ? parse(Int,ARGS[4]) : 12)
