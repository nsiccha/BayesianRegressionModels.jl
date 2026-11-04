# Independent high-precision precision-matrix oracle. Tiny test supports only;
# production never constructs these space-time matrices. Use the expanded FEM
# precision, not the rescaled factorization used by any implementation.
function gmrf_big_precision(mass, stiffness, times, spatial, temporal, L)
    setprecision(256) do
        ell, theta = BigFloat(spatial), BigFloat(temporal)
        C, G = Diagonal(BigFloat.(mass)), BigFloat.(Matrix(stiffness))
        k2 = 2 / ell^2
        Qs = (k2^2 * C + 2k2 * G + G * inv(C) * G) / (4big(pi) * k2)
        Ct = [exp(-abs(BigFloat(t) - BigFloat(s)) / theta) for t in times, s in times]
        chol = BigFloat.(L)
        kron(inv(Ct), kron(inv(chol * chol'), Qs))
    end
end

function gmrf_big_logpdf(field, mass, stiffness, times, spatial, temporal, L)
    setprecision(256) do
        Q = gmrf_big_precision(mass, stiffness, times, spatial, temporal, L)
        z = BigFloat.(vec(field))
        (logdet(Symmetric(Q)) - length(z) * log(2big(pi)) - dot(z, Q * z)) / 2
    end
end
