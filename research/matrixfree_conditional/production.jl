# Production operator algebra without a large exact sparse factor.
# RUN: julia --project=test research/matrixfree_conditional/production.jl NX NY NT [refine] [draw]
include("scaling.jl")

function proxy_spectrum(nx, ny, nt; rho=0.9)
    spatial = [(1 + 4sinpi(j/(2nx))^2 + 4sinpi(k/(2ny))^2)^2
        for j in 0:nx-1, k in 0:ny-1]
    temporal = nt == 1 ? [1.0] : eigvals(SymTridiagonal(
        vcat(1.0, fill(1+rho^2, nt-2), 1.0), fill(-rho, nt-1))) ./ (1-rho^2)
    [t*s + 0.1 for t in temporal, s in vec(spatial)]
end

function production_case(H, prepared; options=prepared.algebra.options)
    n = size(H,1)
    # Retain the exact same reference across quadrature/probe refinements.
    # Column-major RNG generation also retains the prefix of probe columns.
    probes = [2.0 .* rand(Xoshiro(options.seed), Bool, n, options.probes) .- 1]
    p = merge(prepared, (; algebra=merge(prepared.algebra,
        (; options, probes, diagnostics=Dict{Symbol,Any}()))))
    F = BRM._brm_operator_factor(p,H)
    b = randn(Xoshiro(72),n)
    F \ b
    solve = @timed F \ b
    residual = norm(H*solve.value-b)/norm(b)
    @assert residual <= options.solve_tolerance
    cold = @timed logdet(BRM._brm_operator_factor(p,H))
    determinant = @timed logdet(F)
    @assert cold.value == determinant.value
    trace_gradient = sum(F.sensitivity[k] for j in 1:n for k in nzrange(H,j)
        if rowvals(H)[k] == j)
    scale_gradient = dot(nonzeros(H),F.sensitivity) - 0.1trace_gradient
    println("production\t",(; n, probes=options.probes, step=options.logdet_step,
        solve_seconds=solve.time, solve_bytes=solve.bytes, residual,
        cold_determinant_seconds=cold.time, determinant_seconds=determinant.time,
        determinant_bytes=determinant.bytes, retained_bytes=Base.summarysize(F),
        logdet=determinant.value, trace_gradient, scale_gradient,
        diagnostics=p.algebra.diagnostics))
    (; F, value=determinant.value, trace_gradient, scale_gradient)
end

function measure_production(nx,ny,nt; refine=false, draw=false)
    H = proxy(nx,ny,nt)
    n = size(H,1)
    preparation = @timed operator_problem(H)
    p = preparation.value
    println("reference_preparation\t",(; n, seconds=preparation.time,
        allocated_bytes=preparation.bytes, retained_bytes=Base.summarysize(p)))
    spectrum = proxy_spectrum(nx,ny,nt)
    exact = (; logdet=sum(log,spectrum), trace_gradient=sum(inv,spectrum))
    println("exact_oracle\t",(; exact...,
        scale_gradient=n-0.1exact.trace_gradient))
    baseline = production_case(H,p)
    println("baseline_error\t",(; logdet=baseline.value-exact.logdet,
        trace_gradient=baseline.trace_gradient-exact.trace_gradient,
        scale_gradient=baseline.scale_gradient-(n-0.1exact.trace_gradient)))
    if refine
        finer = production_case(H,p; options=MatrixFreeLaplace(; logdet_step=0.5))
        more = production_case(H,p; options=MatrixFreeLaplace(; logdet_step=0.5, probes=32))
        println("quadrature_change\t",(; logdet=finer.value-baseline.value,
            trace_gradient=finer.trace_gradient-baseline.trace_gradient,
            sensitivity_relative=norm(finer.F.sensitivity-baseline.F.sensitivity) /
                norm(finer.F.sensitivity)))
        println("probe_refinement_error\t",(; logdet=more.value-exact.logdet,
            trace_gradient=more.trace_gradient-exact.trace_gradient,
            scale_gradient=more.scale_gradient-(n-0.1exact.trace_gradient)))
    end
    if draw
        cold = @timed BRM._brm_operator_draw(baseline.F,Xoshiro(47))
        warm = @timed BRM._brm_operator_draw(baseline.F,Xoshiro(47))
        @assert cold.value == warm.value && all(isfinite,warm.value)
        # For an exact Gaussian draw this statistic is chi-squared(n).
        # One value is a diagnostic, not a distributional accuracy certificate.
        quadratic = dot(warm.value,H*warm.value)
        println("draw\t",(; cold_seconds=cold.time, seconds=warm.time,
            bytes=warm.bytes, quadratic, standardized=(quadratic-n)/sqrt(2n),
            diagnostics=baseline.F.problem.algebra.diagnostics))
    end
end

length(ARGS) >= 3 || error("pass NX NY NT [refine] [draw]")
println("julia\t",VERSION)
measure_production(parse.(Int,ARGS[1:3])...; refine="refine" in ARGS, draw="draw" in ARGS)
