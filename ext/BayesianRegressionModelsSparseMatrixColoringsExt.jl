# The greedy symmetric coloring of a Hessian pattern that the Laplace's η
# Hessian (src/nlme_laplace_coloring.jl) and the collapsed SVI sites
# (src/nlme_svi_collapsed.jl) read, loaded with that package (src/fitting_extensions.jl).
module BayesianRegressionModelsSparseMatrixColoringsExt

import BayesianRegressionModels as BRM
import SparseMatrixColorings

# Each column's color in a greedy (largest-first) symmetric coloring with
# direct decompression: a star coloring of `pattern`.
function BRM._brm_star_coloring_colors(pattern)
    result = SparseMatrixColorings.coloring(pattern,
        SparseMatrixColorings.ColoringProblem(; structure=:symmetric, partition=:column),
        SparseMatrixColorings.GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst();
            decompression=:direct))
    SparseMatrixColorings.column_colors(result)
end

end
