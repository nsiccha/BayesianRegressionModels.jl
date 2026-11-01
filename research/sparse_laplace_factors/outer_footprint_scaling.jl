# Landing gate for a scalar indexed predictor read beside a reduction.
# RK 23d26ca1 repairs upstream snag sum-factor-outer-ca5cc80e.
# No AD or consumer-specific derivative code is needed.
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
for n in (16,64,4096)
    data=(;unit=["u$i" for i in 1:n], rows=[collect(1:n)], anchor=[1],y=[1.0])
    rk=RKBRMI(@brm data begin
        sigma ~ Exponential(1)
        eta ~ 1 + (1 | unit)
        @plate for k in eachindex(rows)
            y[k] ~ Normal(sum(exp.(eta[rows[k]])) + eta[anchor[k]],sigma)
        end
    end)
    factors=prepare_cell_factors(rk.model,rk_bound_plan(rk),:y;locality=:error)
    outer=prepare_cell_factor_query(factors,:outer)
    summand=prepare_cell_factor_query(factors,1)
    outer_footprint=partition_footprint_kernel(outer.kernel)
    summand_footprint=partition_footprint_kernel(summand.kernel)
    u=zeros(rk.model.layout.total)
    reads=Base.invokelatest(outer_footprint,u,zeros(length(cell_reductions(factors))),1).unconstrained
    termreads=Base.invokelatest(summand_footprint,u,1,1).unconstrained
    println((;n,total=length(u),outer=reads,summand=termreads)); flush(stdout)
    @assert length(reads) == 4 "indexed outer read acquired a model-wide footprint"
    @assert length(termreads) == 3
end
