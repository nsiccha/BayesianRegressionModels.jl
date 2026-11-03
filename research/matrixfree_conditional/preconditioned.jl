# Experiment only: fixed bounded-fill congruence for the rational determinant.
# Uses LimitedLDLFactorizations' documented L/D/P factor access, as in its
# examples/example.jl. No factor refresh as population coordinates change.
# RUN: BRM_PRECONDITIONER_PROJECT=/path/to/isolated/env julia --project=test research/matrixfree_conditional/preconditioned.jl NX NY HOURS MEMORY [NODES]
haskey(ENV,"BRM_PRECONDITIONER_PROJECT") && push!(LOAD_PATH,ENV["BRM_PRECONDITIONER_PROJECT"])
using LimitedLDLFactorizations
using Statistics
include(joinpath(@__DIR__,"scaling.jl"))

struct FixedPrecision{H,L,U}
    h::H
    lower::L
    upper::U
    permutation::Vector{Int}
    scale::Vector{Float64}
    first::Vector{Float64}
    physical::Vector{Float64}
    product::Vector{Float64}
    logdet_reference::Float64
end
function FixedPrecision(H,F)
    n=size(H,1)
    L=UnitLowerTriangular(F.L)
    @assert all(isfinite,F.D) && all(!iszero,F.D)
    FixedPrecision(H,L,L',F.P,1 ./ sqrt.(abs.(F.D)),zeros(n),zeros(n),zeros(n),sum(log,abs.(F.D)))
end
Base.size(A::FixedPrecision)=size(A.h)
Base.size(A::FixedPrecision,d::Integer)=size(A.h,d)
Base.eltype(::FixedPrecision)=Float64
Base.eltype(::Type{<:FixedPrecision})=Float64
function to_physical!(out,A,x)
    A.first .= A.scale .* x
    ldiv!(A.upper,A.first)
    out[A.permutation] .= A.first
    out
end
function from_physical!(out,A,x)
    out .= view(x,A.permutation)
    ldiv!(A.lower,out)
    out .*= A.scale
    out
end
function LinearAlgebra.mul!(out,A::FixedPrecision,x)
    to_physical!(A.physical,A,x)
    mul!(A.product,A.h,A.physical)
    from_physical!(out,A,A.product)
end

function quadrature(nodes)
    if nodes == 0
        shifts=exp.(collect(-32:32))
        weights=copy(shifts); weights[[1,end]] ./= 2
        return shifts,weights
    end
    # Gauss-Legendre in u, t=(1+u)/(1-u), with positive fixed weights.
    # A bounded dense eigendecomposition constructs only the quadrature rule.
    E=eigen(SymTridiagonal(zeros(nodes),1 ./ sqrt.(4 .* (1:nodes-1).^2 .- 1)))
    u=E.values; w=2 .* E.vectors[1,:].^2
    (1 .+ u)./(1 .- u), 2 .* w./(1 .- u).^2
end

function fixed_logdet(A,R; derivative=nothing,nodes=0)
    options=MatrixFreeLaplace(; probes=size(R,2),solve_maxiter=2048)
    shifts,weights=quadrature(nodes)
    b=zeros(size(A,1)); hx=similar(b); z=similar(b)
    workspace=BRM.Krylov.CgLanczosShiftWorkspace(A,b,shifts)
    values=zeros(size(R,2)); trace_gradient=0.0; maxiter=0; maxres=0.0
    for (q,r) in enumerate(eachcol(R))
        b .= r
        BRM.Krylov.cg_lanczos_shift!(workspace,A,b,shifts;
            check_curvature=true,atol=0.0,rtol=1e-11,itmax=options.solve_maxiter)
        info=BRM.Krylov.statistics(workspace)
        @assert info.solved info.status
        maxiter=max(maxiter,info.niter)
        for (j,x) in enumerate(BRM.Krylov.solution(workspace))
            t=shifts[j]; weight=weights[j]
            mul!(hx,A,x)
            residual2=0.0; value=0.0
            for i in eachindex(x)
                residual2+=(hx[i]+t*x[i]-b[i])^2
                value+=b[i]*(hx[i]-x[i])
            end
            residual=sqrt(residual2)/norm(b)
            @assert residual <= 1e-10 residual
            maxres=max(maxres,residual)
            values[q]+=weight/(1+t)*value
            to_physical!(z,A,x)
            contraction=derivative === nothing ? dot(z,z) : dot(z,derivative*z)
            trace_gradient+=weight/size(R,2)*contraction
        end
    end
    (; value=A.logdet_reference+sum(values)/length(values),trace_gradient,
        mcse=Statistics.std(values)/sqrt(length(values)),maxiter,maxres)
end

function measure_preconditioned(nx,ny,nt,memory,nodes=0)
    H=proxy(nx,ny,nt); n=size(H,1)
    ridge=parse(Float64,get(ENV,"BRM_REFERENCE_RIDGE","0"))
    reference=H + spdiagm(0 => ridge .* diag(H))
    build=@timed LimitedLDLFactorizations.lldl(reference;memory)
    F=build.value; A=FixedPrecision(H,F)
    println("preconditioner\t",(;n,memory,ridge,nnz=SparseArrays.nnz(F.L),shift=F.α,
        pivot_range=extrema(F.D),diagonal_range=extrema(diag(reference)),
        seconds=build.time,bytes=build.bytes,retained=Base.summarysize(A)))
    b=randn(Xoshiro(72),n); rhs=similar(b); result=similar(b)
    from_physical!(rhs,A,b)
    BRM.Krylov.cg_lanczos(A,rhs;check_curvature=true,atol=0.0,rtol=1e-13,itmax=2048)
    solve=@timed BRM.Krylov.cg_lanczos(A,rhs;check_curvature=true,atol=0.0,rtol=1e-13,itmax=2048)
    x,info=solve.value
    to_physical!(result,A,x)
    residual=norm(H*result-b)/norm(b)
    centered=similar(rhs); mul!(centered,A,x)
    println("solve_check\t",(;residual,centered_residual=norm(centered-rhs)/norm(rhs),
        iterations=info.niter,status=info.status))
    @assert info.solved info.status
    @assert residual < 1e-10 residual
    R=2.0 .* rand(Xoshiro(20261102),Bool,n,8) .- 1
    fixed_logdet(A,R;nodes)
    determinant=@timed fixed_logdet(A,R;nodes)
    # Independent separable eigenvalue oracle for this rectangular proxy.
    ls=[(1+4sinpi(j/(2nx))^2+4sinpi(k/(2ny))^2)^2 for j in 0:nx-1,k in 0:ny-1]
    rho=.9
    qt=nt==1 ? [1.0;;] : Matrix(SymTridiagonal(vcat(1.0,fill(1+rho^2,nt-2),1.0),fill(-rho,nt-1))) / (1-rho^2)
    spectrum=[t*s+.1 for t in eigvals(Symmetric(qt)),s in vec(ls)]
    println("solve\t",(;seconds=solve.time,bytes=solve.bytes,iterations=info.niter,residual))
    println("determinant\t",(;nodes,determinant.value...,seconds=determinant.time,bytes=determinant.bytes,
        exact_logdet=sum(log,spectrum),exact_trace_inverse=sum(inv,spectrum)))
end

# Noncommuting 2x2 control: preconditioner is fixed while H varies.
H0=[2.0 .3; .3 5.0]
F0=LimitedLDLFactorizations.lldl(sparse(H0);memory=0)
dH=[0.0 1;1 0]
R0=ones(2,1)
nodes=length(ARGS)==5 ? parse(Int,ARGS[5]) : 0
f(t)=fixed_logdet(FixedPrecision(sparse(H0+t*dH),F0),R0;derivative=dH,nodes)
g=f(.2).trace_gradient
fd=(f(.2+1e-5).value-f(.2-1e-5).value)/2e-5
@assert isapprox(g,fd;atol=2e-7)
println("noncommuting_control\t",(;g,fd))
length(ARGS) in (4,5) || error("pass NX NY HOURS MEMORY [NODES]")
measure_preconditioned(parse.(Int,ARGS)...)
