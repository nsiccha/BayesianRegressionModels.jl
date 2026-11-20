# Packages only fitting reaches (the sparse Laplace's factorizations and
# inertia, Krylov recurrences, the fixed preconditioner, Hessian colorings,
# SVI's R-hat) are weak dependencies (snag `brm-fitting-solv-c9af5617`; user
# pick on decision `0ilan3m`). A consumer that fits loads them, which loads
# BRM's extension for each; a consumer that only builds models and simulates
# never installs them. SelectedInversion is not one of them: every `car` term
# needs it at preparation (src/selected_inversion.jl).
#
# The code that calls package `X` is the extension `BayesianRegressionModelsXExt`
# (one trigger, `X`). BRM calls it through the functions declared below, and
# never names `X` or operates on its objects (a KLU factor, a KrylovKit state)
# itself; test/lazy_dependencies.jl checks that. Each function's fallback here
# names the package to load. The extension's methods are more specific, so with
# `X` loaded they take over.
const _BRM_FITTING_EXTENSIONS = (
    KLU = :BayesianRegressionModelsKLUExt,
    Krylov = :BayesianRegressionModelsKrylovExt,
    KrylovKit = :BayesianRegressionModelsKrylovKitExt,
    LimitedLDLFactorizations = :BayesianRegressionModelsLimitedLDLFactorizationsExt,
    MCMCDiagnosticTools = :BayesianRegressionModelsMCMCDiagnosticToolsExt,
    MUMPS = :BayesianRegressionModelsMUMPSExt,
    SparseMatrixColorings = :BayesianRegressionModelsSparseMatrixColoringsExt,
)

# What each extension defines, by package.
const _BRM_FITTING_FUNCTIONS = (
    # The sparse Laplace's augmented-system factor, its solves, scaled pivots.
    KLU = (:_brm_klu, :_brm_klu_solve, :_brm_klu_scaled_pivots),
    # The matrix-free Laplace's checked conjugate-gradient solve.
    Krylov = (:_brm_krylov_checked_cg,),
    # Lanczos recurrences (covariance root, log-determinant probes,
    # GaussianBlockLaplace quadrature) and GaussianBlockLaplace's solves.
    KrylovKit = (:_brm_krylovkit_covariance_root, :_brm_krylovkit_lanczos_run,
        :_brm_krylovkit_lanczos, :_brm_krylovkit_lanczos_probe,
        :_brm_krylovkit_logquadratic, :_brm_krylovkit_svi_laplace_solve),
    # The matrix-free Laplace's fixed reference precision.
    LimitedLDLFactorizations = (:_brm_lldl_fixed_reference,),
    # Rank-normalized ESS and split R-hat of SVI iterates.
    MCMCDiagnosticTools = (:_brm_rank_ess_rhat,),
    # The augmented system's inertia and log-determinant.
    MUMPS = (:_brm_mumps_inertia,),
    # A star coloring of a symmetric Hessian pattern.
    SparseMatrixColorings = (:_brm_star_coloring_colors,),
)

function _brm_missing_fitting_package(name::Symbol)
    missing_now = [String(n) for n in keys(_BRM_FITTING_EXTENSIONS)
        if Base.get_extension(@__MODULE__, _BRM_FITTING_EXTENSIONS[n]) === nothing]
    error("BayesianRegressionModels: this needs `$name`, a weak dependency of BRM. " *
        "Add it to your environment and load it first (`import $name`). Fitting uses " *
        "$(join(keys(_BRM_FITTING_EXTENSIONS), ", ")); to load every one this session " *
        "lacks: `import $(join(missing_now, ", "))`.")
end

# A call that reaches the fallback with `name`'s extension loaded began before
# the extension did (it was loaded during this call), so retry it once in the
# newest world. A second arrival means the extension has no method for these
# arguments, which is BRM's bug.
function _brm_fitting_fallback(name::Symbol, f, args...; kwargs...)
    Base.get_extension(@__MODULE__, _BRM_FITTING_EXTENSIONS[name]) === nothing &&
        _brm_missing_fitting_package(name)
    get(task_local_storage(), :_brm_fitting_retry, nothing) === f && error(
        "BayesianRegressionModels: internal: `$(_BRM_FITTING_EXTENSIONS[name])` defines " *
        "no method of `$f` for these arguments")
    task_local_storage(() -> Base.invokelatest(f, args...; kwargs...), :_brm_fitting_retry, f)
end

for (name, functions) in pairs(_BRM_FITTING_FUNCTIONS), f in functions
    @eval function $f end
    @eval $f(args...; kwargs...) =
        _brm_fitting_fallback($(QuoteNode(name)), $f, args...; kwargs...)
end
