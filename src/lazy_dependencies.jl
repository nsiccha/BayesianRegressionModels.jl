# Packages only fitting reaches (the sparse Laplace's factorizations and
# inertia, Krylov recurrences, the fixed preconditioner, Hessian colorings,
# selected inversion, SVI's R-hat) load on first use instead of with BRM, so a
# consumer that builds models and simulates never loads them; MUMPS alone
# brings MPI (snag `brm-fitting-solv-c9af5617`). Each stays in `[deps]`: every
# environment with BRM installs and precompiles it, and its compat is unchanged.
#
# The code that calls package `X` is the extension `BayesianRegressionModelsXExt`,
# whose one trigger is `X` itself (Julia 1.10's Pkg admits a `[deps]` package
# as a trigger), so requiring `X` from BRM loads the extension with it. Its
# methods are newer than the world of a call that triggered the load, so every
# call into one goes through `invokelatest`, and BRM never operates on an
# object of `X` outside its extension: a KLU factor, a KrylovKit state or a
# selected-inversion plan only passes between extension calls.
const _BRM_LAZY_EXTENSIONS = (
    KLU = :BayesianRegressionModelsKLUExt,
    Krylov = :BayesianRegressionModelsKrylovExt,
    KrylovKit = :BayesianRegressionModelsKrylovKitExt,
    LimitedLDLFactorizations = :BayesianRegressionModelsLimitedLDLFactorizationsExt,
    MCMCDiagnosticTools = :BayesianRegressionModelsMCMCDiagnosticToolsExt,
    MUMPS = :BayesianRegressionModelsMUMPSExt,
    SelectedInversion = :BayesianRegressionModelsSelectedInversionExt,
    SparseMatrixColorings = :BayesianRegressionModelsSparseMatrixColoringsExt,
)
const _BRM_LAZY_READY = map(_ -> Threads.Atomic{Bool}(false), _BRM_LAZY_EXTENSIONS)

function _brm_require_lazy(name::Symbol)
    ready = getfield(_BRM_LAZY_READY, name)
    ready[] && return nothing
    ext = getfield(_BRM_LAZY_EXTENSIONS, name)
    if Base.get_extension(@__MODULE__, ext) === nothing
        Base.require(@__MODULE__, name)
        Base.get_extension(@__MODULE__, ext) === nothing && error(
            "BayesianRegressionModels: loading `$name` did not load BRM's extension " *
            "`$ext`, which holds the code that calls it. An environment resolved " *
            "before BRM declared that extension records none: run `Pkg.resolve()` " *
            "there. After a failed extension load, `Base.retry_load_extensions()` " *
            "retries it.")
    end
    ready[] = true
    nothing
end

# What each extension defines; BRM calls these only through `_brm_lazy`.
# KLU: the sparse Laplace's augmented-system factor, its solves, scaled pivots.
function _brm_klu end
function _brm_klu_solve end
function _brm_klu_scaled_pivots end
# Krylov: the matrix-free Laplace's checked conjugate-gradient solve.
function _brm_krylov_checked_cg end
# KrylovKit: Lanczos recurrences (covariance root, log-determinant probes,
# GaussianBlockLaplace quadrature) and GaussianBlockLaplace's solves.
function _brm_krylovkit_covariance_root end
function _brm_krylovkit_lanczos_run end
function _brm_krylovkit_lanczos end
function _brm_krylovkit_lanczos_probe end
function _brm_krylovkit_logquadratic end
function _brm_krylovkit_svi_laplace_solve end
# LimitedLDLFactorizations: the matrix-free Laplace's fixed reference precision.
function _brm_lldl_fixed_reference end
# MCMCDiagnosticTools: rank-normalized ESS and split R-hat of SVI iterates.
function _brm_rank_ess_rhat end
# MUMPS: the augmented system's inertia and log-determinant.
function _brm_mumps_inertia end
# SelectedInversion: selected inverses of Cholesky and KLU factors, including
# a `car` term's BYM2 scaling at preparation.
function _brm_selinv_diag end
function _brm_selinv_on_pattern end
function _brm_selinv_pattern! end
function _brm_lu_selinv end
function _brm_lu_selinv! end
function _brm_lu_selinv_available end
# SparseMatrixColorings: a star coloring of a symmetric Hessian pattern.
function _brm_star_coloring_colors end

"""
    _brm_lazy(name, f, args...; kwargs...)

Call `f`, a function whose methods the extension of the fitting-time package
`name` defines, after loading that package (and with it the extension).
"""
function _brm_lazy(name::Symbol, f, args...; kwargs...)
    _brm_require_lazy(name)
    Base.invokelatest(f, args...; kwargs...)
end
