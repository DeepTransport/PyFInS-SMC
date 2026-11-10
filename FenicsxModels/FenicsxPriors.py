import warnings
import numpy as np
from petsc4py import PETSc

import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc as fpetsc

from BaseModels import GaussianPriorBase

# -----------------------------------------------------------------------------

def _make_zero_dirichlet_bcs_all_boundary(V: fem.FunctionSpace):
    """
    Homogeneous Dirichlet BC on the entire boundary for V.
    Assumes V is H1-conforming.
    """
    domain = V.mesh
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    bc = fem.dirichletbc(u0, dofs)
    return [bc]

def _as_owned_local_array(x, V: fem.FunctionSpace) -> np.ndarray:
    """
    Coefficients-only converter with batch support.

    Accepts:
      - np.ndarray of shape (n_owned,) or (n_batch, n_owned)

    Rejects:
      - fem.Function (by design)
      - PETSc.Vec 

    Returns:
      - np.ndarray (n_owned,) or (n_batch, n_owned)
    """
    n_owned = V.dofmap.index_map.size_local
    arr = np.asarray(x, dtype=np.float64)
    # Single vector
    if arr.ndim == 1:
        arr = arr.reshape(-1)
        if arr.size != n_owned:
            raise ValueError(f"Expected owned-local length {n_owned}, got {arr.size}")
        return arr
    # Batch: rows are samples
    if arr.ndim == 2:
        if arr.shape[1] != n_owned:
            raise ValueError(f"Expected shape (n_batch, {n_owned}), got {arr.shape}")
        return arr

    raise ValueError(f"Expected 1D or 2D coeff array, got shape {arr.shape}")

class PriorSPDE(GaussianPriorBase):
    """
    SPDE-based Matérn prior on Kh with homogeneous Dirichlet BC.

    Uses ν=1 (α=2) Matérn in 2D via:
      tau (kappa^2 - Δ) u = W

    Discrete:
      A = tau (kappa^2 M + K)
      u = A^{-1} (M^{1/2} xi), xi ~ N(0,I)

    Uses lumped mass sqrt: M^{1/2} ~ diag(M)^{1/2}.
    """

    def __init__(
        self,
        Kh: fem.FunctionSpace,
        sigma: float,
        ell: float,
        mu: float = 0.0,
        *,
        seed: int = 12345,
        bcs = None,
        ksp_type: str = "preonly",
        pc_type: str = "lu",
        mv_solver: str = "preonly",
        mv_pc: str = "cholesky",
        # for iterative alternative: ksp_type="cg", pc_type="hypre"
    ):
        self.mu = mu
        self.Kh = Kh
        self.mesh = Kh.mesh
        self.comm = self.mesh.comm
        self.seed = int(seed)
        self._rng = np.random.default_rng([self.seed, self.comm.rank])

        if self.Kh.ufl_element().degree == 0:
            self.V = fem.functionspace(self.mesh, ("CG", 1))
            self.DG0_map = True
        else:
            self.V = self.Kh
            self.DG0_map = False

        self.nV_owned = self.V.dofmap.index_map.size_local
        self.nV_global = self.V.dofmap.index_map.size_global
        self.nK_owned = self.Kh.dofmap.index_map.size_local
        self.nK_global = self.Kh.dofmap.index_map.size_global
        #self.dim = self.nV_owned

        # Dirichlet BCs (V in H_0^1)
        self.bcs = bcs if bcs is not None else _make_zero_dirichlet_bcs_all_boundary(self.V)

        # Matérn parameters (2D, ν=1)
        nu = 1.0
        self.kappa = np.sqrt(8.0 * nu) / float(ell)
        self.tau = 1.0 / (float(sigma) * np.sqrt(4.0 * np.pi) * self.kappa)

        # Build FE operators
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        aM = ufl.inner(u, v) * ufl.dx
        aK = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        self.M_form = fem.form(aM)
        self.K_form = fem.form(aK)

        # Assemble M, K with BC handling consistent with H_0^1
        self.M = fpetsc.assemble_matrix(self.M_form, bcs=self.bcs)
        self.M.assemble()

        self.K = fpetsc.assemble_matrix(self.K_form, bcs=self.bcs)
        self.K.assemble()

        # A = tau (kappa^2 M + K)
        self.A = self.K.copy()
        self.A.scale(1.0)  # ensure independent
        self.A.axpy(self.kappa**2, self.M, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
        self.A.scale(self.tau)
        self.A.assemble()

        # Lumped mass diagonal for sqrt(M)
        d = self.M.getDiagonal()              # PETSc Vec (local)
        d_arr = d.getArray(readonly=False)    # local array view
        # guard against tiny/negative diag due to BC elimination
        self.M_lumped = np.maximum(d_arr, 1e-30)
        self.sqrtM = np.sqrt(self.M_lumped).copy()
        self.invSqrtM = (1.0 / self.sqrtM).copy()

        # KSP for A solves (SPD)
        self.kspA = PETSc.KSP().create(self.comm)
        self.kspA.setOperators(self.A)
        self.kspA.setType(ksp_type)
        pc = self.kspA.getPC()
        pc.setType(pc_type)
        self.kspA.setFromOptions()
        self.kspA.setUp()

        # Local mean as owned-local array
        # scalar (optional)
        if np.isscalar(mu):
            self.mu = np.full(self.nK_owned, float(mu), dtype=np.float64)

        # If DG0 target, precompute coefficient-only Pi / Pi*
        if self.DG0_map:
            self._build_DG0_maps(mv_solver=mv_solver, mv_pc=mv_pc)

    def _build_DG0_maps(self, mv_solver: str = "preonly", mv_pc: str = "cholesky"):
        """
        Build coefficient-only maps between V (CG1) and Kh (DG0):

          Pi(uV)      = M_Q^{-1} (B uV)          in DG0 coefficients
          Pi_star(gK) = M_V^{-1} (B^T gK)        in CG1 coefficients

        where:
          M_Q is DG0 mass (diagonal cell volumes)
          M_V is CG1 mass on V (consistent)
          B_{K,i} = ∫_K φ_i dx  (coupling Q-test, V-trial)
        """
        # Coupling matrix B : Q-test, V-trial
        u = ufl.TrialFunction(self.V)
        q = ufl.TestFunction(self.Kh)
        B_form = fem.form(ufl.inner(u, q) * ufl.dx)
        self.B = fpetsc.assemble_matrix(B_form, bcs=[])
        self.B.assemble()

        # DG0 cell volumes (M_Q diagonal): vol_K = ∫ 1 * q dx
        one_form = fem.form(1.0 * q * ufl.dx)
        vol_vec = fpetsc.assemble_vector(one_form)
        vol_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.volK_owned = np.maximum(np.asarray(vol_vec.array[:self.nK_owned], dtype=np.float64), 1e-30)

        # Consistent mass matrix in V for Pi*
        #u = ufl.TrialFunction(self.V)
        #v = ufl.TestFunction(self.V)
        #MV_form = fem.form(ufl.inner(u, v) * ufl.dx)
        #self.MV = fpetsc.assemble_matrix(MV_form, bcs=self.bcs)
        #self.MV.assemble()

        #self.kspMV = PETSc.KSP().create(self.comm)
        #self.kspMV.setOperators(self.MV)
        #self.kspMV.setType(mv_solver)
        #self.kspMV.getPC().setType(mv_pc)
        #self.kspMV.setFromOptions()
        #self.kspMV.setUp()

    def _CG1_to_DG0_coef(self, uV_owned: np.ndarray) -> np.ndarray:
        """Pi: CG1 coefficients (owned) -> DG0 coefficients (owned) via cell averages."""
        # copy latent function into the one referenced by the form
        uV_owned = np.asarray(uV_owned, dtype=np.float64).reshape(-1)
        if uV_owned.size != self.nV_owned:
            raise ValueError(f"uV length {uV_owned.size} != nV_owned {self.nV_owned}")

        u_vec = self.B.createVecRight()
        u_arr = u_vec.getArray(readonly=False)
        u_arr[:self.nV_owned] = uV_owned
        u_vec.assemblyBegin(); u_vec.assemblyEnd()

        g_vec = self.B.createVecLeft()
        self.B.mult(u_vec, g_vec)

        g_arr = g_vec.getArray(readonly=False)[:self.nK_owned].copy()
        return g_arr / self.volK_owned

    def _DG0_to_CG1_coef(self, gK_owned: np.ndarray) -> np.ndarray:
        """Pi^T: DG0 coefficients (owned) -> CG1 coefficients (owned) via transpose. NOT for mapping functions"""
        gK_owned = np.asarray(gK_owned, dtype=np.float64).reshape(-1)
        if gK_owned.size != self.nK_owned:
            raise ValueError(f"gK length {gK_owned.size} != nK_owned {self.nK_owned}")
        
        g_vec = self.B.createVecLeft()
        g_arr = g_vec.getArray(readonly=False)
        g_arr[:self.nK_owned] = gK_owned / self.volK_owned
        # g_vec.assemblyBegin(); g_vec.assemblyEnd()

        # rhs lives in CG1 space (right of B)
        rhs = self.B.createVecRight()
        self.B.multTranspose(g_vec, rhs)  # rhs = B^T (M_Q^{-1} g)

        return rhs.getArray(readonly=False)[:self.nV_owned].copy()

    
    # ------------------------------------------------------------------
    # SPDE core linear algebra in V
    # ------------------------------------------------------------------
    def random(self, n: int = 1) -> np.ndarray:
        """White noise xi in the latent space V (owned-local)."""
        if n == 1:
            return self._rng.standard_normal(self.nV_owned)
        return self._rng.standard_normal((n, self.nV_owned))

    def _solve_A(self, rhsV_owned: np.ndarray) -> np.ndarray:
        """Solve A u = rhs in V, return owned-local coefficients."""
        rhsV_owned = np.asarray(rhsV_owned, dtype=np.float64).reshape(-1)
        if rhsV_owned.size != self.nV_owned:
            raise ValueError(f"rhs length {rhsV_owned.size} != nV_owned {self.nV_owned}")

        b = self.A.createVecRight()
        b_arr = b.getArray(readonly=False)
        b_arr[: self.nV_owned] = rhsV_owned
        b.assemblyBegin(); b.assemblyEnd()

        u = self.A.createVecRight()
        self.kspA.solve(b, u)
        return u.getArray(readonly=False)[: self.nV_owned].copy()

    def _mult_A(self, uV_owned: np.ndarray) -> np.ndarray:
        """Compute A u in V, return owned-local coefficients."""
        uV_owned = np.asarray(uV_owned, dtype=np.float64).reshape(-1)
        if uV_owned.size != self.nV_owned:
            raise ValueError(f"u length {uV_owned.size} != nV_owned {self.nV_owned}")

        u_vec = self.A.createVecRight()
        u_arr = u_vec.getArray(readonly=False)
        u_arr[: self.nV_owned] = uV_owned
        u_vec.assemblyBegin(); u_vec.assemblyEnd()

        Au = self.A.createVecLeft()
        self.A.mult(u_vec, Au)
        return Au.getArray(readonly=False)[: self.nV_owned].copy()
    
    def matvec_prior_L(self, x):
        """
        u = A^{-1} (sqrt(M) x)
        Input:
          x: (n_owned,) or (n_batch, n_owned)
        Output:
          u_owned: (n_owned,) or (n_batch, n_owned)
        """
        #x = np.asarray(x, dtype=np.float64)
        x_owned = _as_owned_local_array(x, self.V)
        if x_owned.ndim == 1:
            uV = self._solve_A(self.sqrtM * x_owned)
            return self._CG1_to_DG0_coef(uV) if self.DG0_map else uV
        elif x_owned.ndim == 2: 
            out = np.zeros((x_owned.shape[0], self.nK_owned), dtype=np.float64)
            for i in range(x_owned.shape[0]):
                uV = self._solve_A(self.sqrtM * x_owned[i, :])
                out[i, :] = self._CG1_to_DG0_coef(uV) if self.DG0_map else uV
            return out
        else:
            raise ValueError(f"x must be 1D or 2D, got {x.shape}")

    def matvec_prior_LT(self, y):
        """
        L^T y = sqrt(M) (A^{-T} y). A is symmetric => A^{-T}=A^{-1}.
        Input y can be ndarray or fem.Function in Kh; interpreted as owned-local DOFs.
        Output: (n_owned,) numpy
        """
        y_owned = _as_owned_local_array(y, self.Kh)
        if y_owned.ndim == 1:
            yV_owned = self._DG0_to_CG1_coef(y_owned) if self.DG0_map else y_owned
            return self.sqrtM * self._solve_A(yV_owned)
        elif y_owned.ndim == 2:
            out = np.zeros((y.shape[0], self.nV_owned), dtype=np.float64)
            for i in range(y.shape[0]):
                yV_owned = self._DG0_to_CG1_coef(y_owned[i, :]) if self.DG0_map else y_owned[i, :]
                out[i, :] = self.sqrtM * self._solve_A(yV_owned)
            return out
        else:
            raise ValueError(f"y must be 1D or 2D, got {y.shape}")
        
    def matvec_prior_invL(self, y):
        """
        invL(y) = M^{-1/2} (A y)
        Input y can be ndarray or fem.Function in Kh.
        Output: (n_owned,) numpy
        """
        y_owned = _as_owned_local_array(y, self.Kh)
        if y_owned.ndim == 1:
            yV_owned = self._DG0_to_CG1_coef(y_owned) if self.DG0_map else y_owned
            return self.invSqrtM * self._mult_A(yV_owned)
        elif y_owned.ndim == 2:
            out = np.zeros((y.shape[0], self.nV_owned), dtype=np.float64)
            for i in range(y.shape[0]):
                yV_owned = self._DG0_to_CG1_coef(y_owned[i, :]) if self.DG0_map else y_owned[i, :]
                out[i, :] = self.invSqrtM * self._mult_A(yV_owned)
            return out
        else:
            raise ValueError(f"y must be 1D or 2D, got {y.shape}")

# -----------------------------------------------------------------------------

class MultiFieldPrior(GaussianPriorBase):
    """
    Correlated Gaussian prior for multiple parameter fields.

    All fields share the same function space, but may have different means, 
    correlations and variances.

    Let

        xi_i in V

    be independent standard Gaussian fields and let

        R = L L.T

    be the field correlation matrix.

    The correlated latent fields are

        z_i = sum_j L[i, j] xi_j,

    and the physical parameter fields are

        k_i = mu_i + L_i z_i,

    where

        L_i : V -> K_i.

    Local ordering
    --------------

    Whitened coordinates:

        xi = [xi_0 | xi_1 | ... | xi_{q-1}]

    with equal-size blocks.

    Physical parameters:

        k = [k_0 | k_1 | ... | k_{q-1}]

    where the block sizes may differ.
    """

    def __init__(
        self,
        Kh,
        sigma,
        ell,
        mu,
        correlation,
        *,
        names=None,
        seed=12345,
        bcs=None,
        ksp_type="preonly",
        pc_type="lu",
        mv_solver="preonly",
        mv_pc="cholesky",
    ):
        self.Kh = Kh
        self.mesh = Kh.mesh
        self.comm = self.mesh.comm

        self.ell = list(ell)
        self.sigma = list(sigma)
        self.mu_field = list(mu)

        self.n_fields = len(self.ell)

        if len(self.sigma) != self.n_fields or len(self.mu_field) != self.n_fields:
            raise ValueError("ell, sigma, and mu must have the same number of fields.")

        # process correlation
        if correlation.shape != (self.n_fields, self.n_fields):
            raise ValueError(
                "The correlation matrix must have shape "
                f"({self.n_fields}, {self.n_fields})."
            )
        diag_error = np.max(np.abs(np.diag(correlation) - 1.0))
        eps = 1e-12
        if diag_error > eps:
            warnings.warn(
                "correlation diagonal differs from 1; resetting it to 1.",
                RuntimeWarning,
            )
        if not np.allclose(correlation, correlation.T, atol=eps, rtol=0.0):
            raise ValueError("correlation must be symmetric.")
        correlation = correlation.copy()
        np.fill_diagonal(correlation, 1.0)
        try:
            self.correlation_L = np.linalg.cholesky(correlation)
            self.correlation_invL = np.linalg.inv(self.correlation_L)
        except np.linalg.LinAlgError:
            raise ValueError("correlation must be positive definite.")
        self.correlation = correlation

        self.names = (
            [f"field_{i}" for i in range(self.n_fields)]
            if names is None
            else list(names)
        )

        # Common function space V for the whittened variable
        if self.Kh.ufl_element().degree == 0:
            self.V = fem.functionspace(self.mesh, ("CG", 1))
        else:
            self.V = self.Kh

        self.nV_field_owned = self.V.dofmap.index_map.size_local
        self.nV_field_global = self.V.dofmap.index_map.size_global

        self.nV_owned = self.n_fields * self.nV_field_owned
        self.nV_global = self.n_fields * self.nV_field_global

        # Physical parameter spaces K_i may differ
        self.nK_field_owned = self.Kh.dofmap.index_map.size_local
        self.nK_field_global = self.Kh.dofmap.index_map.size_global

        self.nK_owned = self.n_fields * self.nK_field_owned
        self.nK_global = self.n_fields * self.nK_field_global

        # Joint mean
        mu_blocks = [
            np.full(self.nK_field_owned, mu_i) if np.isscalar(mu_i) else mu_i
            for mu_i in self.mu_field
        ]
        mu_fields = np.stack(mu_blocks, axis=-2)
        self.mu = self.join_parameter(mu_fields)

        #self.dim = self.nV_owned

        self.seed = seed
        self._rng = np.random.default_rng([seed, self.comm.rank])

        self.priors = [
            PriorSPDE(
                self.Kh,
                sigma=sigma_i,
                ell=ell_i,
                mu=0.0,
                seed=seed + i,
                bcs=bcs,
                ksp_type=ksp_type,
                pc_type=pc_type,
                mv_solver=mv_solver,
                mv_pc=mv_pc,
            )
            for i, (ell_i, sigma_i) in enumerate(zip(self.ell, self.sigma))
        ]

    def split_whitened(self, xi):
        """
        Split xi = [xi_0 | xi_1 | ...] into equal-size latent field blocks.
        """
        return xi.reshape(*xi.shape[:-1], self.n_fields, self.nV_field_owned)

    def join_whitened(self, xi_fields):
        return xi_fields.reshape(*xi_fields.shape[:-2], self.nV_owned)

    def split_parameter(self, k):
        """
        Split k = [k_0 | k_1 | ...] into equal-size physical parameter blocks.
        """
        return k.reshape(*k.shape[:-1], self.n_fields, self.nK_field_owned)

    def join_parameter(self, k_fields):
        return k_fields.reshape(*k_fields.shape[:-2], self.nK_owned)

    def split_whitened_dict(self, xi):
        return dict(zip(self.names, self.split_whitened(xi)))
    
    def split_parameter_dict(self, k):
        return dict(zip(self.names, self.split_parameter(k)))

    def random(self, n=1):
        """
        Draw independent whitened coordinates.

        Cross-field correlation is introduced by matvec_prior_L().
        """
        if n == 1:
            return self._rng.standard_normal(self.nV_owned)
        return self._rng.standard_normal((n, self.nV_owned))

    def matvec_prior_L(self, xi):
        """
        Apply k = L_joint xi where
            L_joint = diag(L_0, ..., L_{q-1}) (Lcorr tensor I_V).
        """
        # (..., q*nV) -> (..., q, nV)
        xi_fields = self.split_whitened(xi)
        # Correlate the latent fields, acting on the second to the last index
        # (q, nV) or (N, q, nV)
        z_fields = np.einsum("ij,...jk->...ik", self.correlation_L, xi_fields)
        # k_fields.shape == (q, nK) or (N, q, nK)
        k_fields = np.stack(
            [prior.matvec_prior_L(z_fields[..., i, :]) for i, prior in enumerate(self.priors)],
            axis=-2,
        )
        # join the last axis (over q fields), keeping local access
        return self.join_parameter(k_fields)

    def matvec_prior_LT(self, k):
        """
        Apply L_joint.T k where
            L_joint.T = (Lcorr.T tensor I_V) diag(L_0.T, ..., L_{q-1}.T).
        """
        # (..., q*nK) -> (..., q, nK)
        k_fields = self.split_parameter(k)
        # z_fields.shape == (q, nV) or (N, q, nV)
        z_fields = np.stack(
            [prior.matvec_prior_LT(k_fields[..., i, :]) for i, prior in enumerate(self.priors)],
            axis=-2,
        )
        # note that it's ji, so L.T is used
        xi_fields = np.einsum("ji,...jk->...ik", self.correlation_L, z_fields)
        # join the last axis (over q fields), keeping local access
        return self.join_whitened(xi_fields)
    
    def matvec_prior_invL(self, k):
        """
        Apply xi = L_joint^{-1} k.
        Requires each field-specific L_i to be invertible.
        """
        # (..., q*nK) -> (..., q, nK)
        k_fields = self.split_parameter(k)
        # z_fields.shape == (q, nV) or (N, q, nV)
        z_fields = np.stack(
            [prior.matvec_prior_invL(k_fields[..., i, :]) for i, prior in enumerate(self.priors)],
            axis=-2,
        )
        # apply L^{-1}, acting on the second to the last index
        xi_fields = np.einsum("ij,...jk->...ik", self.correlation_invL, z_fields)
        # join the last two axes
        return self.join_whitened(xi_fields)

    