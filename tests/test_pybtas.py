import numpy as np
import pybtas
import pytest


def test_pybtas():
    A = np.zeros((2, 2, 7))
    cnt = 0
    for i in range(2):
        for j in range(2):
            for k in range(7):
                A[i, j, k] = cnt
                cnt += 1

    A = np.ascontiguousarray(A)
    A = A + A.transpose(1, 0, 2)
    A_chol = np.zeros((4, 7))
    for i in range(2):
        for j in range(2):
            for k in range(7):
                A_chol[i * 2 + j, k] = A[i, j, k]

    np.testing.assert_allclose(A, A.transpose(1, 0, 2))

    beta, gamma, scale = pybtas.cp3(A, 7)
    Atest = np.einsum("ar,br,xr,r", beta, beta, gamma, scale.ravel())
    np.testing.assert_allclose(Atest, A, atol=1e-7, rtol=0)

    beta, gamma, scale = pybtas.cp3_from_cholesky(A_chol, 7)
    Atest = np.einsum("ar,br,xr,r", beta, beta, gamma, scale.ravel())
    np.testing.assert_allclose(Atest, A, atol=1e-7, rtol=0)

    B = np.arange(12).reshape((3, 4))
    Bt = pybtas.eigen_mat_return(B)
    np.testing.assert_allclose(Bt, B)


def test_run_thc_example():
    pytest.importorskip("openfermion")
    from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES, thc

    if not HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
        pytest.skip("Dependencies for THC factorization (pyscf & jax) not found.")
    from pyscf import gto, scf, mcscf, ao2mo
    import numpy as np

    mol = gto.M(
        atom="""O 0 0 0
      H 1 0 0
      H 0 1 0""",
        basis="cc-pvdz",
    )

    scfres = scf.HF(mol).run()

    mc = mcscf.CASCI(scfres, ncas=6, nelecas=6)

    h2 = mc.get_h2cas()
    h2 = ao2mo.restore("1", h2, norb=6)

    eri_rr, thc_leaf, thc_central, info = thc.factorize(h2, nthc=20)

    diff = h2 - eri_rr
    print(np.linalg.norm(diff))
