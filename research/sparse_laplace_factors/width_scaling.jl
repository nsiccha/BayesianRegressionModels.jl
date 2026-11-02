# Public planner construction, with no AD backend available: every effect of
# the sole wide term must be sampled before any derivative preparation.
# Run each point under /usr/bin/time -v in a separate process for peak RSS.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL, LinearAlgebra, SHA
width, cells = parse.(Int, ARGS[1:2])
data = (; unit=["u$i" for i in 1:width], rows=[collect(1:width) for _ in 1:cells],
    weights=ones(width), y=fill(0.2,cells))
rk = RKBRMI(@brm data begin
    eta ~ 1 + (1 | unit)
    @plate for k in eachindex(rows)
        y[k] ~ Normal(dot(eta[rows[k]], weights), 0.7)
    end
end)
plan() = brm_laplace_plan(rk; ad_backend=nothing, max_cell_width=32)
plan()
GC.gc()
measurement = @timed plan()
p = measurement.value
@assert p.integrated_dimension == 0
@assert p.prepared === nothing
@assert only(values(p.width_events)).width == width
source = join(read(joinpath(pkgdir(BayesianRegressionModels),"src",f),String)
    for f in ("nlme_laplace_width.jl","nlme_laplace_plan.jl","nlme_laplace_factors.jl"))
println("width\tcells\tinput_effect_incidence\tintegrated_dimension\tsampled_dimension\tplan_bytes\tallocated_bytes\tseconds\tprocess_maxrss_bytes\tsource_sha256")
println(join((width,cells,width*cells,p.integrated_dimension,p.sampled_dimension,
    Base.summarysize(p),measurement.bytes,measurement.time,Sys.maxrss(),bytes2hex(sha256(source))), '\t'))
