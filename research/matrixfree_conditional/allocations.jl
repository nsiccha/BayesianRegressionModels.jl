# Sample warmed production allocations, attributing only BRM-owned frames.
# RUN: julia --project=test research/matrixfree_conditional/allocations.jl
include("scaling.jl")
using Profile

H = proxy(8,8,8)
p = operator_problem(H)
logdet(BRM._brm_operator_factor(p,H))
F = BRM._brm_operator_factor(p,H)
Profile.Allocs.@profile sample_rate=0.001 logdet(F)
totals = Dict{Tuple{String,Int,String},Tuple{Int,Int}}()
for allocation in Profile.Allocs.fetch().allocs
    frame = findfirst(f -> occursin("/src/nlme_laplace_",String(f.file)), allocation.stacktrace)
    frame === nothing && continue
    f = allocation.stacktrace[frame]
    key = (basename(String(f.file)),Int(f.line),string(allocation.type))
    count, bytes = get(totals,key,(0,0))
    totals[key] = (count+1,bytes+allocation.size)
end
for (key,values) in sort!(collect(totals); by=x->last(x)[2], rev=true)
    println((; file=key[1], line=key[2], type=key[3], samples=values[1], bytes=values[2]))
end
