# Selected entries of a sparse inverse, loaded with SelectedInversion
# (src/fitting_extensions.jl): every `car` term's scaling factor at preparation,
# and the Laplace, GMRF and collapsed-SVI variances while fitting. A KLU
# factor's methods come from SelectedInversion's own KLU extension, which loads
# with it once KLU is loaded.
module BayesianRegressionModelsSelectedInversionExt

import BayesianRegressionModels as BRM
import SelectedInversion
using SparseArrays: nonzeros

BRM._brm_selinv_diag(F) = SelectedInversion.selinv_diag(F)

# The selected inverse of a sparse Cholesky factor on `pattern`.
BRM._brm_selinv_on_pattern(F, pattern) =
    SelectedInversion.selinv_extract(SelectedInversion.selinv(F; depermute=true).Z, pattern)

# Whether this SelectedInversion has the pivoted KLU API the sum-cell Laplace
# needs (a fleet revision; registered 0.2.2 lacks it).
BRM._brm_lu_selinv_available() = isdefined(SelectedInversion, :LUSelInvWorkspace)

# A KLU factor's inverse on the entries `request` holds.
BRM._brm_lu_selinv(F, request) = SelectedInversion.selinv(F, request)

# A KLU factor's inverse on its matrix's own pattern, into `selected`, reusing
# `workspace` (or building it); returns the workspace.
function BRM._brm_lu_selinv!(selected, workspace, F, K)
    workspace === nothing && (workspace = SelectedInversion.LUSelInvWorkspace(F, K))
    SelectedInversion.selinv!(selected, workspace, F)
    workspace
end

# The inverse of a Cholesky factor on `pattern`, written into `z` in the
# pattern's storage order. A supernodal result reads through an extraction
# `plan`, built once per pattern and returned for the next factor.
function BRM._brm_selinv_pattern!(z, F, pattern, destination, plan)
    selected = SelectedInversion.selinv(F; depermute=true).Z
    if selected isa SelectedInversion.SupernodalMatrix
        plan === nothing && (plan = SelectedInversion.selinv_extract_setup(selected, pattern))
        SelectedInversion.selinv_extract!(destination, selected, plan)
        copyto!(z, nonzeros(destination))
    else
        copyto!(z, nonzeros(SelectedInversion.selinv_extract(selected, pattern)))
    end
    plan
end

end
