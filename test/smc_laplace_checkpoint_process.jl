include("smc_laplace_fixture.jl")
checkpoint = deserialize(ARGS[1])
fit = brm_smc_fit(brm_smc_resume(lsmc_problem(), checkpoint))
serialize(ARGS[2], brm_smc_checkpoint(fit))
