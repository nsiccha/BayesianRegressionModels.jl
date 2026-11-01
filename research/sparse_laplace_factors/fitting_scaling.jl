# End-to-end cost, including mode finding, pivoted factorization and selected
# inversion. Run under the repository test environment and a compute token.
empty!(ARGS); push!(ARGS, "pivoted augmented")
include(joinpath(@__DIR__, "../../test/nlme_laplace_sums.jl"))

function measure_sum_fit(n; overlapping=false, mixed=false)
    rows = overlapping ? [collect(i:(i+2)) for i in 1:(n-2)] : [collect(1:n)]
    build = @timed begin
        (; rk) = sum_cell_fixture(n; rows, nonlinear=true, mixed)
        brm_laplace_problem(rk; integrate=(:unit,), ad_backend=NLME_AD)
    end
    p = build.value; phi = zeros(length(p.population))
    initial = @timed LogDensityProblems.logdensity_and_gradient(p,phi)
    @assert isfinite(initial.value[1]) && all(isfinite,initial.value[2])
    samples = [@timed LogDensityProblems.logdensity_and_gradient(p,phi .+ 0.001j) for j in 1:3]
    state = BRM._brm_spl_mode!(p,phi)
    draws = @timed BRM._brm_spl_draw(p,state,Xoshiro(57))
    @assert all(isfinite,draws.value)
    selected = BRM._brm_spl_selected(state.F)
    # The per-level diagnostic must reuse baseline sums, not revisit every
    # summand for every effect. Include its real work in the scaling receipt.
    diagnostic = @timed BRM._brm_plan_ess(p,state,only(p.blocks).declaration,8,0.5,Xoshiro(58))
    result = (;n, overlapping, mixed, cells=length(rows), prepare_seconds=build.time,
        prepare_bytes=build.bytes, initial_seconds=initial.time,
        gradient_seconds=minimum(s.time for s in samples),
        gradient_bytes=minimum(s.bytes for s in samples),
        precision_entries=nnz(state.h.D)+nnz(state.h.B)+nnz(state.h.W),
        augmented_entries=nnz(state.F.K), selected_entries=nnz(selected),
        draw_seconds=draws.time, draw_bytes=draws.bytes,
        diagnostic_seconds=diagnostic.time, diagnostic_bytes=diagnostic.bytes)
    println(result); flush(stdout)
    result
end

measure_sum_fit(12)
measure_sum_fit(12; overlapping=true)
measure_sum_fit(12; mixed=true)
for (overlapping, mixed) in ((false,false), (true,false), (false,true))
    results = [measure_sum_fit(n;overlapping,mixed) for n in (64,512,4096)]
    @assert results[end].gradient_bytes < 12results[end-1].gradient_bytes
    @assert results[end].diagnostic_bytes < 12results[end-1].diagnostic_bytes
    @assert results[end].precision_entries < 9results[end-1].precision_entries
end
