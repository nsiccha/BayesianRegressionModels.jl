# Shared by the in-process and fresh-process continuation checks.
function laplace_window_builder()
    BayesianRegressionModels.brm(quote
        mu ~ 1 + (1 | probe) + hsgp(time; k=2, by=probe,
            cor=multigroup((sx,sy),2.;k=2))
        @plate for p in groups(probe)
            y[p] ~ Normal(mu[p],.7)
        end
    end;mod=@__MODULE__)
end
function laplace_window_data(labels;shift=0.,move_new=false)
    locations=Dict(1=>(-.7,.3),2=>(.2,-.8),3=>(1.1,.6),4=>(0.,0.))
    move_new && (locations[4]=(.1,0.))
    # Plain integer groups have sorted level order, independently of row
    # order. Multigroup coordinates are one entry per level, not per row.
    levels=sort(unique(labels))
    (;probe=repeat(labels;inner=3),time=repeat([.1,.8,1.6],length(labels)),
        sx=[locations[i][1] for i in levels],sy=[locations[i][2] for i in levels],
        y=collect(range(-1.,1.;length=3length(labels))).+shift)
end
function laplace_window_summary(fit)
    state=brm_svi_checkpoint(fit)
    blocks=Dict(key=>(;a=b.a,B=b.B,L=Matrix(b.F.L)) for (key,b) in state.blocks)
    (;draws=fit.draws,q=fit.diagnostics.q,elbo=fit.diagnostics.elbo,
        converged=fit.diagnostics.converged,stop_reason=fit.diagnostics.stop_reason,
        calls=fit.diagnostics.laplace.calls,numerical=fit.diagnostics.laplace.numerical,
        source_id=state.source_id,globals=state.globals,shared_coordinates=state.shared_coordinates,
        core=(;m=state.core.m,L=Matrix(state.core.F.L)),blocks,units=state.units,
        projection=state.projection,updates=state.updates)
end
