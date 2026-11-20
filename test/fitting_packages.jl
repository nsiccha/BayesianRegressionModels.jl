# BRM's fitting-time weak dependencies (src/fitting_extensions.jl). A test that
# fits loads them before it does; `import` loads each package, and BRM's
# extension for it, without bringing its exported names into scope.
import KLU, Krylov, KrylovKit, LimitedLDLFactorizations, MCMCDiagnosticTools, MUMPS,
    SparseMatrixColorings
