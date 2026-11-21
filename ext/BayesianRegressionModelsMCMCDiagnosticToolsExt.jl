# SVI's stationarity diagnostics (src/nlme_svi_stationarity.jl): rank-normalized
# split R-hat and ESS of iterate histories, loaded with that package
# (src/fitting_extensions.jl).
module BayesianRegressionModelsMCMCDiagnosticToolsExt

import BayesianRegressionModels as BRM
import MCMCDiagnosticTools

BRM._brm_rank_ess_rhat(x) = MCMCDiagnosticTools.ess_rhat(x; kind=:rank)

end
