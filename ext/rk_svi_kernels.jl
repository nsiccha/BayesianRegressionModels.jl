# Use exactly the equations compiled for BRM's native iteration.
using LinearAlgebra: I, LowerTriangular, Symmetric, diag, diagm, dot, norm, qr
import ReactiveKernels: rk_cholesky_lower as _svi_lower_cholesky

const _SVI_KERNELS = map(BRM._BRM_SVI_KERNEL_DEFINITIONS) do definition
    Core.eval(@__MODULE__, Expr(:macrocall, GlobalRef(ReactiveKernels, Symbol("@kernel")),
        LineNumberNode(0), deepcopy(definition)))
end

BRM._rk_svi_kernels() = _SVI_KERNELS
