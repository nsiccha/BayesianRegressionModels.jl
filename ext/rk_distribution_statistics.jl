# Source providers expose owned RK object endpoints, preserving the same
# ordinary emitted-source replay path as other BRM callable providers.
function BRM._rk_callable_source!(definitions, bindings, entry,
        ::BRM._RKLogNormalStatistic{Statistic,Side}) where {Statistic,Side}
    object = Side === :lower ? DistributionKernelSources.lower_truncated_lognormal :
        Side === :upper ? DistributionKernelSources.upper_truncated_lognormal :
        DistributionKernelSources.lognormal
    leaf = Symbol(entry, :_distribution)
    push!(bindings, leaf => object)
    inputs = Side === :base ? (:location, :scale) : (:location, :scale, :bound)
    receiver = Expr(:call, leaf, inputs...)
    endpoint_inputs = Statistic in (:logcdf, :logccdf) ? (:bound,) : ()
    value = Expr(:call, Expr(:., receiver, QuoteNode(Statistic)), endpoint_inputs...)
    signature = Expr(:call, entry, :location, :scale, :bound)
    push!(definitions, Expr(:macrocall, Symbol("@kernel"), LineNumberNode(0),
        Expr(:(=), signature, Expr(:block, Expr(:return, value)))))
    true
end
