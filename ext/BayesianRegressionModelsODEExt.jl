module BayesianRegressionModelsODEExt

import BayesianRegressionModels as BRM
import ReactiveKernels

BRM._brm_ode_bridge(::typeof(BRM.ode_rk45)) = ReactiveKernels.rk_ode_rk45_tol
BRM._brm_ode_bridge(::typeof(BRM.ode_bdf)) = ReactiveKernels.rk_ode_bdf_tol

end
