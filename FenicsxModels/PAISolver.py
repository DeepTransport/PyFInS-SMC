"""
MPI-compatible coupled photoacoustic imaging solver using FEniCSx/PETSc.

Observables are: p(t_i,x_j,lambda) = B(t_i,x_j) G(m,lambda)
G(m,lambda) = p0(m,lambda) = mu_a(m,lambda)*phi(m,lambda), phi(m,lambda) obtained via solving optic PDE:
mu_a phi - div(kappa(mu_a) grad(phi)) = 0 in Omega
(1/pi)*phi + 0.5*kappa(mu_a)*grad(phi)*n = s on dOmega

Unknowns are:
mu_a(m,lambda) = c_hbt(m1)*(s(m2)*1e-6*eps_ox(lambda) + (1.0 - s(m2))*1e-6*eps_de(lambda)) [absorption coeff, cm^{-1}]
kappa(mu_a,lambda) = 1/(2*(mu_a(lambda)+mu_s(lambda))), mu_s(lambda) = a*(lambda/lam_ref)**(-b) [diffusion coeff, cm]

Parameters and units:
Omega: [0,5] x [0,3] domain [cm]
lambda: wavelengt [nm]
c_hbt(m1): total concentration of hemoglobin [muM = mu mole/L (micro moles per liter)] 
eps_{X} - molar extinction coefficients [cm^-1/(mol/L)] = cm^{-1}M (= cm^{-1} 1e6 muM)
s(m2): ratio of oxygentated tissue to deoxygenated (also oxygenation) [unitless]

B: passed in matrix, takes p_0, outputs p(t_i,x_j) at 5 observation times, 50 x_js via solution of acoustic wave equation: 
p_tt - c^2 Delta p = 0 in Omega x [0,T]
p(t = 0) = p0 in Omega
p_t(t = 0) = 0 in Omega
Boundary conditions are chosen to approximate an absorbing boundary

The public PAISolver interface follows the call pattern:

    Y, Q = solver.solve(m)
    Y, Q = solver.solve_and_cache(m)

    dY = solver.linearized_observation(dm)
    g  = solver.linearized_observation_transpose(dY)
    gq = solver.linearized_qoi_transpose()

A forward solve caches all state-dependent optical quantities and the optical
operator/KSP.  Tangent and adjoint solves reuse this cache.  The linear map
from initial acoustic pressure to observations is precomputed and reused.  The
original wave solve and observation operator are retained for cross-validation.

Parameter vectors are local-owned block vectors

    m = [m1_owned, m2_owned],

where

    c_HbT = exp(m1),
    sO2   = logistic(m2).

The observation vector is replicated on all MPI ranks.
"""
import warnings
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import scipy.sparse as sp

import dolfinx
import ufl
from dolfinx import fem, io
from dolfinx.fem import petsc as fpetsc

dtype = dolfinx.default_scalar_type

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# ============================================================================
# PETSc / FEniCSx helpers
# ============================================================================

def assemble_matrix(form):
    A = fpetsc.assemble_matrix(fem.form(form))
    A.assemble()
    return A


def assemble_vector(form):
    b = fpetsc.assemble_vector(fem.form(form))
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    return b


def assemble_scalar(form, comm):
    local = fem.assemble_scalar(fem.form(form))
    return comm.allreduce(local, op=MPI.SUM)


def make_ksp(A, comm, high_accuracy=False, spd=False, debug=False):
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)

    # can't use CG anymore, not SPD anymore
    #ksp.getPC().setType("hypre")
    #ksp.setFromOptions()

    if debug:
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        return ksp

    if spd:
        ksp.setType("cg")
    else:
        ksp.setType("minres")
    ksp.getPC().setType("hypre")

    if high_accuracy:
        ksp.setTolerances(rtol=1.0e-14, atol=1.0e-14, max_it=1000)
    else:
        ksp.setTolerances(rtol=1.0e-8, atol=1.0e-10, max_it=1000)
    ksp.setFromOptions()
    return ksp



# ============================================================================
# Optical coefficients
# ============================================================================

LN10 = np.log(10.0)

_TABLE = np.array([
    [650,  368.0,   3750.12], [652,  356.8,   3642.64], [654,  345.6,   3535.16],
    [656,  335.2,   3427.68], [658,  325.6,   3320.20], [660,  319.6,   3226.56],
    [662,  314.0,   3140.28], [664,  308.4,   3053.96], [666,  302.8,   2967.68],
    [668,  298.0,   2881.40], [670,  294.0,   2795.12], [672,  290.0,   2708.84],
    [674,  285.6,   2627.64], [676,  282.0,   2554.40], [678,  279.2,   2481.16],
    [680,  277.6,   2407.92], [682,  276.0,   2334.68], [684,  274.4,   2261.48],
    [686,  272.8,   2188.24], [688,  274.4,   2115.00], [690,  276.0,   2051.96],
    [692,  277.6,   2000.48], [694,  279.2,   1949.04], [696,  282.0,   1897.56],
    [698,  286.0,   1846.08], [700,  290.0,   1794.28], [702,  294.0,   1741.00],
    [704,  298.0,   1687.76], [706,  302.8,   1634.48], [708,  308.4,   1583.52],
    [710,  314.0,   1540.48], [712,  319.6,   1497.40], [714,  325.2,   1454.36],
    [716,  332.0,   1411.32], [718,  340.0,   1368.28], [720,  348.0,   1325.88],
    [722,  356.0,   1285.16], [724,  364.0,   1244.44], [726,  372.4,   1203.68],
    [728,  381.2,   1152.80], [730,  390.0,   1102.20], [732,  398.8,   1102.20],
    [734,  407.6,   1102.20], [736,  418.8,   1101.76], [738,  432.4,   1100.48],
    [740,  446.0,   1115.88], [742,  459.6,   1161.64], [744,  473.2,   1207.40],
    [746,  487.6,   1266.04], [748,  502.8,   1333.24], [750,  518.0,   1405.24],
    [752,  533.2,   1515.32], [754,  548.4,   1541.76], [756,  562.0,   1560.48],
    [758,  574.0,   1560.48], [760,  586.0,   1548.52], [762,  598.0,   1508.44],
    [764,  610.0,   1459.56], [766,  622.8,   1410.52], [768,  636.4,   1361.32],
    [770,  650.0,   1311.88], [772,  663.6,   1262.44], [774,  677.2,   1213.00],
    [776,  689.2,   1163.56], [778,  699.6,   1114.80], [780,  710.0,   1075.44],
    [782,  720.4,   1036.08], [784,  730.8,    996.72], [786,  740.0,    957.36],
    [788,  748.0,    921.80], [790,  756.0,    890.80], [792,  764.0,    859.80],
    [794,  772.0,    828.80], [796,  786.4,    802.96], [798,  807.2,    782.36],
    [800,  816.0,    761.72], [802,  828.0,    743.84], [804,  836.0,    737.08],
    [806,  844.0,    730.28], [808,  856.0,    723.52], [810,  864.0,    717.08],
    [812,  872.0,    711.84], [814,  880.0,    706.60], [816,  887.2,    701.32],
    [818,  901.6,    696.08], [820,  916.0,    693.76], [822,  930.4,    693.60],
    [824,  944.8,    693.48], [826,  956.4,    693.32], [828,  965.2,    693.20],
    [830,  974.0,    693.04], [832,  982.8,    692.92], [834,  991.6,    692.76],
    [836, 1001.2,    692.64], [838, 1011.6,    692.48], [840, 1022.0,    692.36],
    [842, 1032.4,    692.20], [844, 1042.8,    691.96], [846, 1050.0,    691.76],
    [848, 1054.0,    691.52], [850, 1058.0,    691.32], [852, 1062.0,    691.08],
    [854, 1066.0,    690.88], [856, 1072.8,    690.64], [858, 1082.4,    692.44],
    [860, 1092.0,    694.32], [862, 1101.6,    696.20], [864, 1111.2,    698.04],
    [866, 1118.4,    699.92], [868, 1123.2,    701.80], [870, 1128.0,    705.84],
    [872, 1132.8,    709.96], [874, 1137.6,    714.08], [876, 1142.8,    718.20],
    [878, 1148.4,    722.32], [880, 1154.0,    726.44], [882, 1159.6,    729.84],
    [884, 1165.2,    733.20], [886, 1170.0,    736.60], [888, 1174.0,    739.96],
    [890, 1178.0,    743.60], [892, 1182.0,    747.24], [894, 1186.0,    750.88],
    [896, 1190.0,    754.52], [898, 1194.0,    758.16], [900, 1198.0,    761.84],
    [902, 1202.0,    765.04], [904, 1206.0,    767.44], [906, 1209.2,    769.80],
    [908, 1211.6,    772.16], [910, 1214.0,    774.56], [912, 1216.4,    776.92],
    [914, 1218.8,    778.40], [916, 1220.8,    778.04], [918, 1222.4,    777.72],
    [920, 1224.0,    777.36], [922, 1225.6,    777.04], [924, 1227.2,    776.64],
    [926, 1226.8,    772.36], [928, 1224.4,    768.08], [930, 1222.0,    763.84],
    [932, 1219.6,    752.28], [934, 1217.2,    737.56], [936, 1215.6,    722.88],
    [938, 1214.8,    708.16], [940, 1214.0,    693.44], [942, 1213.2,    678.72],
    [944, 1212.4,    660.52], [946, 1210.4,    641.08], [948, 1207.2,    621.64],
    [950, 1204.0,    602.24], [952, 1200.8,    583.40], [954, 1197.6,    568.92],
    [956, 1194.0,    554.48], [958, 1190.0,    540.04], [960, 1186.0,    525.56],
    [962, 1182.0,    511.12], [964, 1178.0,    495.36], [966, 1173.2,    473.32],
    [968, 1167.6,    451.32], [970, 1162.0,    429.32], [972, 1156.4,    415.28],
    [974, 1150.8,    402.28], [976, 1144.0,    389.288],[978, 1136.0,    374.944],
    [980, 1128.0,    359.656],[982, 1120.0,    344.372],[984, 1112.0,    329.084],
    [986, 1102.4,    313.796],[988, 1091.2,    298.508],[990, 1080.0,    283.220],
    [992, 1068.8,    267.932],[994, 1057.6,    252.648],[996, 1046.4,    237.360],
    [998, 1035.2,    222.072],[1000,1024.0,    206.784],
])

LAMBDA_TABLE = _TABLE[:, 0]
EPS_HBO2_TABLE = _TABLE[:, 1]
EPS_HB_TABLE = _TABLE[:, 2]


def molar_extinction(lam):
    eps_ox = LN10 * np.interp(lam, LAMBDA_TABLE, EPS_HBO2_TABLE)
    eps_de = LN10 * np.interp(lam, LAMBDA_TABLE, EPS_HB_TABLE)
    return eps_ox, eps_de


def reduced_scattering(lam, a=18.9, b=1.286, lam_ref=500.0):
    return a * (lam / lam_ref) ** (-b)

# ============================================================================
# Optical diffusion model
# ============================================================================

class OpticalDiffusion:
    """
    Optical diffusion equation

        mu_a phi - div(kappa grad(phi)) = 0,

        kappa = 1 / (2 (mu_a + mu_s')).

    The mass matrix/KSP are fixed and assembled once.  The state-dependent
    optical matrix/KSP are assembled by solve() and cached for tangent and
    adjoint solves.
    """

    def __init__(
        self,
        mesh,
        Vh,
        roi=None,
        scat_a=18.9,
        scat_b=1.286,
        scat_lam_ref=500.0,
        high_accuracy=False,
    ):
        self.mesh = mesh
        self.comm = mesh.comm
        self.Vh = Vh

        self.n_owned = Vh.dofmap.index_map.size_local
        self.n_global = Vh.dofmap.index_map.size_global

        self.scat_a = scat_a
        self.scat_b = scat_b
        self.scat_lam_ref = scat_lam_ref
        self.high_accuracy = high_accuracy

        self.source = fem.Function(Vh, name="source")

        self.chi = None
        if roi is not None:
            self.chi = fem.Function(Vh, name="roi")
            self.chi.interpolate(roi)
            self.chi.x.scatter_forward()

        self.u = ufl.TrialFunction(Vh)
        self.v = ufl.TestFunction(Vh)

        self.M = assemble_matrix(ufl.inner(self.u, self.v) * ufl.dx)
        self.mass_ksp = make_ksp(self.M, self.comm, high_accuracy, spd=True)

        self.A = None
        self.ksp = None

        self.phi = None
        self.p0 = None
        self.c = None
        self.s = None
        self.mu_a = None

    def set_source(self, xs, amplitude=3.0, sigma=0.75, yb=0.0):
        def source_expression(x):
            on_boundary = np.isclose(x[1], yb)
            profile = amplitude * np.exp(-0.5 * (x[0] - xs) ** 2 / sigma**2)
            return np.where(on_boundary, profile, 0.0)
        self.source.interpolate(source_expression)
        self.source.x.scatter_forward()

    def split_parameters(self, m):
        m = m.squeeze()
        if m.ndim != 1:
            raise ValueError(f"The input parameter should be a 1 dimension array")

        c = fem.Function(self.Vh, name="cHbT")
        c.x.array[:self.n_owned] = np.exp(m[:self.n_owned])
        c.x.scatter_forward()

        s = fem.Function(self.Vh, name="sO2")
        s.x.array[:self.n_owned] = 1.0 / (1.0 + np.exp(-m[self.n_owned:]))
        s.x.scatter_forward()

        return c, s

    def mu_s_prime(self, lam):
        return reduced_scattering(
            lam,
            a=self.scat_a,
            b=self.scat_b,
            lam_ref=self.scat_lam_ref,
        )

    def absorption(self, c, s, lam):
        eps_ox, eps_de = molar_extinction(lam)
        eps_ox = fem.Constant(self.mesh, dtype(eps_ox * 1.0e-6))
        eps_de = fem.Constant(self.mesh, dtype(eps_de * 1.0e-6))
        E1 = s * eps_ox + (1.0 - s) * eps_de
        E2 = c * (eps_ox - eps_de)
        mu_a = c * E1
        return mu_a, E1, E2

    def bilinear_form(self, mu_a, mu_sp):
        mu_sp = fem.Constant(self.mesh, dtype(mu_sp))
        kappa = 1.0 / (2.0 * (mu_a + mu_sp))
        return (
            ufl.inner(kappa * ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
            + ufl.inner(mu_a * self.u, self.v) * ufl.dx
            + (2.0 / np.pi) * ufl.inner(self.u, self.v) * ufl.ds
        )

    def _assemble_operator(self, mu_a, mu_sp):
        if self.ksp is not None:
            self.ksp.destroy()
        if self.A is not None:
            self.A.destroy()
        self.A = assemble_matrix(self.bilinear_form(mu_a, mu_sp))
        self.ksp = make_ksp(self.A, self.comm, self.high_accuracy, spd=True)

    def _solve_cached_operator(self, rhs):
        sol = fem.Function(self.Vh)
        self.ksp.solve(rhs, sol.x.petsc_vec)
        sol.x.scatter_forward()
        return sol

    def _solve_cached_mass(self, rhs, name=None):
        sol = fem.Function(self.Vh)
        self.mass_ksp.solve(rhs, sol.x.petsc_vec)
        sol.x.scatter_forward()
        if name is not None:
            sol.name = name
        return sol

    def _project(self, expr, name=None):
        rhs = assemble_vector(ufl.inner(expr, self.v) * ufl.dx)
        return self._solve_cached_mass(rhs, name=name)

    def solve(self, m, lam):
        """
        Forward optical solve.  This method caches the operator and all optical
        state needed by linearized() and adjoint operations.
        """
        c, s = self.split_parameters(m)

        mu_sp = self.mu_s_prime(lam)
        mu_a, _, _ = self.absorption(c, s, lam)

        self._assemble_operator(mu_a, mu_sp)
        rhs = assemble_vector(2.0 * ufl.inner(self.source, self.v) * ufl.ds)
        phi = self._solve_cached_operator(rhs)

        p0 = self._project(mu_a * phi, "p0")
        mu_a_fun = self._project(mu_a, "mu_a")

        self.phi = phi
        self.p0 = p0
        self.c = c
        self.s = s
        self.mu_a = mu_a_fun
        self.lam = lam
        self.mu_sp = mu_sp

        return p0

    def linearized(self, dm):
        """
        Tangent optical solve at the cached forward state.
        """

        # local physical perturbations
        n = self.n_owned

        dc = fem.Function(self.Vh)
        dc.x.array[:n] = self.c.x.array[:n] * dm[:n]
        dc.x.scatter_forward()

        ds = fem.Function(self.Vh)
        ds.x.array[:n] = self.s.x.array[:n] * (1.0 - self.s.x.array[:n]) * dm[n:]
        ds.x.scatter_forward()

        mu_a, E1, E2 = self.absorption(self.c, self.s, self.lam)
        dmu_a = E1 * dc + E2 * ds

        mu_sp = fem.Constant(self.mesh, dtype(self.mu_sp))
        dkappa = -dmu_a / (2.0 * (mu_a + mu_sp) ** 2)


        rhs = assemble_vector(
            -ufl.inner(dmu_a * self.phi, self.v) * ufl.dx
            - ufl.inner(dkappa * ufl.grad(self.phi), ufl.grad(self.v)) * ufl.dx
        )
        dphi = self._solve_cached_operator(rhs)
        dp0 = self._project(mu_a * dphi + dmu_a * self.phi)

        return dp0

    def observation_adjoint(self, p0_gradient):
        """
        Optical adjoint for a coefficient-space gradient with respect to p0.
        """
        # G = self.mass_inverse_local(p0_gradient, "G")
        b = self.M.createVecRight()
        b.array[:] = p0_gradient
        G = self._solve_cached_mass(b, name="G")

        mu_a, _, _ = self.absorption(self.c, self.s, self.lam)
        rhs = assemble_vector(ufl.inner(mu_a * G, self.v) * ufl.dx)
        lam = self._solve_cached_operator(rhs)
        lam.name = "lambda_obs"

        return lam, G

    def parameter_gradient(self, lam, G):
        """
        Return the local-owned block gradient with respect to

            [m1, m2].
        """
        mu_a, E1, E2 = self.absorption(self.c, self.s, self.lam)
        mu_sp = fem.Constant(self.mesh, dtype(self.mu_sp))

        K = (
            self.phi * (G - lam)
            + ufl.inner(ufl.grad(self.phi), ufl.grad(lam))
            / (2.0 * (mu_a + mu_sp) ** 2)
        )

        g1_vec = assemble_vector(E1 * K * self.v * ufl.dx)
        g2_vec = assemble_vector(E2 * K * self.v * ufl.dx)

        c_arr = self.c.x.array[:self.n_owned]
        s_arr = self.s.x.array[:self.n_owned]

        grad = np.empty(2 * self.n_owned)
        grad[:self.n_owned] = c_arr * g1_vec.array[:self.n_owned]
        grad[self.n_owned:] = (s_arr * (1.0 - s_arr) * g2_vec.array[:self.n_owned])

        g1_vec.destroy()
        g2_vec.destroy()

        return grad

    def qoi(self, qoi_type="sO2", s_thr=0.5, c_thr=40.0, quad_deg=6):
        dx_q = ufl.dx(metadata={"quadrature_degree": quad_deg})
        c, s = self.c, self.s

        if qoi_type == "sO2":
            D = assemble_scalar(self.chi * c * dx_q, self.comm)
            Q = assemble_scalar(self.chi * c * s * dx_q, self.comm) / D
            return np.array([Q])

        if qoi_type == "HF":
            threshold = fem.Constant(self.mesh, dtype(s_thr))
            chi = ufl.conditional(ufl.lt(s, threshold), 1.0, 0.0)

            D = assemble_scalar(c * dx_q, self.comm)
            Q = assemble_scalar(c * chi * dx_q, self.comm) / D
            return np.array([Q])

        if qoi_type == "vsO2":
            threshold = fem.Constant(self.mesh, dtype(c_thr))
            weight = c * ufl.conditional(ufl.gt(c, threshold), 1.0, 0.0)

            D = assemble_scalar(weight * dx_q, self.comm)
            Q = assemble_scalar(weight * s * dx_q, self.comm) / D
            return np.array([Q])

        return np.concatenate(
            [
                self.qoi("sO2", s_thr, c_thr, quad_deg),
                self.qoi("HF", s_thr, c_thr, quad_deg),
                self.qoi("vsO2", s_thr, c_thr, quad_deg),
            ]
        )

    def qoi_gradient(self, qoi_type="sO2", quad_deg=6):
        if qoi_type != "sO2":
            raise NotImplementedError("QoI gradient is implemented only for sO2.")

        c, s = self.c, self.s
        dx_q = ufl.dx(metadata={"quadrature_degree": quad_deg})

        D = assemble_scalar(self.chi * c * dx_q, self.comm)
        Q = assemble_scalar(self.chi * c * s * dx_q, self.comm) / D

        w0 = assemble_vector(self.chi * self.v * dx_q)
        w1 = assemble_vector(self.chi * s * self.v * dx_q)
        w2 = assemble_vector(self.chi * c * self.v * dx_q)

        c_arr = c.x.array[:self.n_owned]
        s_arr = s.x.array[:self.n_owned]

        grad = np.empty(2 * self.n_owned)
        grad[:self.n_owned] = c_arr * (w1.array[:self.n_owned] - Q * w0.array[:self.n_owned]) / D
        grad[self.n_owned:] = (s_arr * (1.0 - s_arr) * w2.array[:self.n_owned] / D)

        w0.destroy()
        w1.destroy()
        w2.destroy()

        return grad

    def plot_solution(self, filename=None, show=True):
        """
        Plot the cached optical solution:

            cHbT, sO2, mu_a, fluence, p0.

        The distributed fields are gathered to rank 0 for plotting.
        """

        fields = [
            ("cHbT", self.c),
            ("sO2", self.s),
            ("mu_a", self.mu_a),
            ("fluence", self.phi),
            ("p0", self.p0),
            ("source", self.source),
        ]

        X_local = self.Vh.tabulate_dof_coordinates()[:self.n_owned, :2]
        X_all = self.comm.gather(X_local, root=0)
        values_all = []
        for _, field in fields:
            values_all.append(
                self.comm.gather(field.x.array[:self.n_owned].copy(), root=0)
            )

        if self.comm.rank != 0:
            return

        X = np.concatenate(X_all, axis=0)
        triangulation = mtri.Triangulation(X[:, 0], X[:, 1])

        ncols = 2
        nrows = (len(fields) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 4 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()
        for i, (name, _) in enumerate(fields):
            values = np.concatenate(values_all[i], axis=0)
            im = axes[i].tripcolor(
                triangulation,
                values,
                shading="gouraud",
            )
            axes[i].set_title(name)
            axes[i].set_aspect("equal")
            fig.colorbar(im, ax=axes[i])
        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ============================================================================
# Acoustic wave model
# ============================================================================

class AcousticWave:
    """
    Newmark scheme with gamma=1/2 and beta=1/4.

    All spatial matrices and KSPs are assembled once.  The wave solver is
    retained for constructing and cross-validating the precomputed acoustic
    observation operator, and for debugging.

    States are returned as local-owned arrays

        states.shape = (n_times, n_owned).
    """

    def __init__(
        self,
        mesh,
        Vh,
        simulation_times,
        sound_speed=1510.0e2,
        high_accuracy=False,
    ):
        self.mesh = mesh
        self.comm = mesh.comm
        self.Vh = Vh

        self.simulation_times = simulation_times
        self.sound_speed = sound_speed
        self.dt = simulation_times[1] - simulation_times[0]
        self.n_times = simulation_times.size

        self.n_owned = Vh.dofmap.index_map.size_local
        self.n_global = Vh.dofmap.index_map.size_global

        u = ufl.TrialFunction(Vh)
        v = ufl.TestFunction(Vh)

        c = fem.Constant(mesh, dtype(sound_speed))
        dt = self.dt

        m_form = ufl.inner(u / c, v / c) * ufl.dx
        s_form = (1.0 / c) * ufl.inner(u, v) * ufl.ds
        k_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

        l_form = (4.0 / (dt*dt)) * m_form + (2.0 / dt) * s_form + k_form 

        self.M = assemble_matrix(m_form)
        self.S = assemble_matrix(s_form)
        self.K = assemble_matrix(k_form)
        self.L = assemble_matrix(l_form)

        self.mass_ksp = make_ksp(self.M, self.comm, high_accuracy, spd=True)
        self.newmark_ksp = make_ksp(self.L, self.comm, high_accuracy, spd=True)

    def _new_vec(self):
        return self.M.createVecRight()

    def solve(self, p0):
        """
        Acoustic forward solve from an initial pressure Function or local-owned
        coefficient vector.
        """
        p = self._new_vec()

        if hasattr(p0, "x"):
            p.array[:] = p0.x.array[:self.n_owned]
        else:
            p.array[:] = p0

        pt = self._new_vec()
        ptt = self._new_vec()
        pt.set(0.0)

        tmp = self._new_vec()
        self.K.mult(p, tmp)
        tmp.scale(-1.0)
        self.mass_ksp.solve(tmp, ptt)

        states = np.empty((self.simulation_times.size, self.n_owned))
        states[0] = p.array_r

        dt = self.dt
        a = 4.0 / dt**2
        b = 4.0 / dt
        c = 2.0 / dt

        Mrhs = self._new_vec()
        Srhs = self._new_vec()
        rhsM = self._new_vec()
        rhsS = self._new_vec()
        rhs = self._new_vec()

        p_new = self._new_vec()
        dp = self._new_vec()
        pt_new = self._new_vec()
        ptt_new = self._new_vec()

        for j in range(1, self.simulation_times.size):
            p.copy(Mrhs)
            Mrhs.scale(a)
            Mrhs.axpy(b, pt)
            Mrhs.axpy(1.0, ptt)

            p.copy(Srhs)
            Srhs.scale(c)
            Srhs.axpy(1.0, pt)

            self.M.mult(Mrhs, rhsM)
            self.S.mult(Srhs, rhsS)

            rhsM.copy(rhs)
            rhs.axpy(1.0, rhsS)

            self.newmark_ksp.solve(rhs, p_new)

            p_new.copy(dp)
            dp.axpy(-1.0, p)

            dp.copy(pt_new)
            pt_new.scale(c)
            pt_new.axpy(-1.0, pt)

            dp.copy(ptt_new)
            ptt_new.scale(a)
            ptt_new.axpy(-b, pt)
            ptt_new.axpy(-1.0, ptt)

            p_new.copy(p)
            pt_new.copy(pt)
            ptt_new.copy(ptt)

            states[j] = p.array_r

        for vec in (
            p, pt, ptt, tmp, Mrhs, Srhs, rhsM, rhsS, rhs,
            p_new, dp, pt_new, ptt_new,
        ):
            vec.destroy()

        return states

    def transpose(self, Z):
        """
        Exact discrete transpose of p0 -> acoustic states.

        Used to construct the precomputed acoustic observation operator.

        Z has shape (n_times, n_owned).  The returned coefficient gradient has
        shape (n_owned,).
        """
        dt = self.dt

        a = 4.0 / dt**2
        b = 4.0 / dt
        c = 2.0 / dt

        r1 = self._new_vec()
        r2 = self._new_vec()
        r3 = self._new_vec()

        r1.set(0.0)
        r2.set(0.0)
        r3.set(0.0)

        rhs = self._new_vec()
        z = self._new_vec()
        Mz = self._new_vec()
        Sz = self._new_vec()

        r1_new = self._new_vec()
        r2_new = self._new_vec()
        r3_new = self._new_vec()

        for j in range(self.simulation_times.size - 1, 0, -1):
            r1.array[:] += Z[j]

            r1.copy(rhs)
            rhs.axpy(c, r2)
            rhs.axpy(a, r3)

            self.newmark_ksp.solve(rhs, z)

            self.M.mult(z, Mz)
            self.S.mult(z, Sz)

            Mz.copy(r1_new)
            r1_new.scale(a)
            r1_new.axpy(c, Sz)
            r1_new.axpy(-c, r2)
            r1_new.axpy(-a, r3)

            Mz.copy(r2_new)
            r2_new.scale(b)
            r2_new.axpy(1.0, Sz)
            r2_new.axpy(-1.0, r2)
            r2_new.axpy(-b, r3)

            Mz.copy(r3_new)
            r3_new.axpy(-1.0, r3)

            r1_new.copy(r1)
            r2_new.copy(r2)
            r3_new.copy(r3)

        r1.array[:] += Z[0]

        Minv_r3 = self._new_vec()
        KMinv_r3 = self._new_vec()

        self.mass_ksp.solve(r3, Minv_r3)
        self.K.mult(Minv_r3, KMinv_r3)

        out = r1.array_r.copy()
        out -= KMinv_r3.array_r

        for vec in (
            r1, r2, r3, rhs, z, Mz, Sz,
            r1_new, r2_new, r3_new, Minv_r3, KMinv_r3,
        ):
            vec.destroy()

        return out


# ============================================================================
# Acoustic observation
# ============================================================================


def boundary_observation_matrix(V, sensor_x, sensor_y, tol=1.0e-10):
    """
    Build a distributed column block of the boundary interpolation operator.

    The returned matrix has shape

        (n_sensors, n_owned)

    on each rank.  Forward application is completed with an MPI sum.

    This preserves the original nodal linear interpolation used by the PAI
    model.  It assumes a scalar nodal Lagrange space and sensors on a straight
    horizontal boundary.
    """
    comm = V.mesh.comm
    imap = V.dofmap.index_map

    n_owned = imap.size_local
    coords = V.tabulate_dof_coordinates()[:n_owned]
    gids = imap.local_to_global(np.arange(n_owned, dtype=np.int32))

    local_data = np.column_stack((coords[:, 0], coords[:, 1], gids))
    all_data = np.concatenate(comm.allgather(local_data), axis=0)

    rows = []
    cols = []
    vals = []

    for i in range(sensor_x.size):
        on = np.flatnonzero(np.abs(all_data[:, 1] - sensor_y[i]) < tol)
        boundary = all_data[on]

        order = np.argsort(boundary[:, 0])
        boundary = boundary[order]

        xs = boundary[:, 0]
        j = np.clip(np.searchsorted(xs, sensor_x[i]), 1, xs.size - 1)

        x0, x1 = xs[j - 1], xs[j]
        g0 = int(boundary[j - 1, 2])
        g1 = int(boundary[j, 2])

        t = (sensor_x[i] - x0) / (x1 - x0)

        for gid, weight in ((g0, 1.0 - t), (g1, t)):
            local = np.flatnonzero(gids == gid)
            if local.size:
                rows.append(i)
                cols.append(local[0])
                vals.append(weight)

    return sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(sensor_x.size, n_owned),
    )


def time_gate_matrix(simulation_times, n_gates, t_start, t_end, window="boxcar"):
    edges = np.linspace(t_start, t_end, n_gates + 1)
    R = np.zeros((n_gates, simulation_times.size))

    for i in range(n_gates):
        ind = np.flatnonzero(
            (simulation_times >= edges[i])
            & (simulation_times < edges[i + 1])
        )

        weights = np.hanning(ind.size + 2)[1:-1] if window == "hann" else np.ones(ind.size)
        R[i, ind] = weights / weights.sum()

    return R, edges


class AcousticObservation:
    """
    Space-time acoustic observation operator.

    The full observation map is

        p0 -> acoustic states -> sensor traces -> time-gated observations.

    Since the acoustic wave equation and observation operator are linear, the
    composed map is precomputed as

        Y = H p0,

    where

        H_local.shape = (n_data, n_owned)

    is stored by distributed columns.

    The original wave solve and space-time observation are retained for
    debugging and cross-validation.  Whenever the observation setup changes,
    H is rebuilt and checked against a full wave solve.
    """

    def __init__(
        self,
        acoustic,
        sensor_x,
        sensor_y,
        n_gates=5,
        t_start=None,
        t_end=None,
        window="boxcar",
        *,
        check_tolerance=1.0e-8,
        check_seed=12345,
    ):
        self.acoustic = acoustic
        self.V = acoustic.Vh
        self.comm = acoustic.comm
        self.simulation_times = acoustic.simulation_times
        self.sound_speed = acoustic.sound_speed

        if t_start is None:
            t_start = self.simulation_times[0]

        if t_end is None:
            t_end = self.simulation_times[-1]

        self.R, self.gate_edges = time_gate_matrix(
            self.simulation_times,
            n_gates,
            t_start,
            t_end,
            window,
        )

        self.n_gates = n_gates
        self.gate_ranges = self.sound_speed * self.gate_edges

        self.n_owned = acoustic.n_owned
        self.check_tolerance = check_tolerance
        self.check_seed = check_seed

        self.H_local = None

        self._set_sensor_locations(sensor_x, sensor_y)
        self.build()

    @property
    def n_data(self):
        return self.n_sensors * self.n_gates

    def _set_sensor_locations(self, sensor_x, sensor_y):
        self.sensor_x = sensor_x
        self.sensor_y = sensor_y
        self.C_local = boundary_observation_matrix(
            self.V,
            sensor_x,
            sensor_y,
        )
        self.n_sensors = sensor_x.size

    def set_sensor_locations(self, sensor_x, sensor_y):
        self._set_sensor_locations(sensor_x, sensor_y)
        self.build()

    def observe(self, states):
        """
        Apply the original space-time observation operator to acoustic states.
        """
        traces_local = self.C_local @ states.T
        traces = np.empty_like(traces_local)
        self.comm.Allreduce(traces_local, traces, op=MPI.SUM)

        return traces @ self.R.T

    def transpose_to_states(self, delta_y):
        """
        Apply the transpose space-time observation operator.
        """
        delta_y = delta_y.reshape(self.n_sensors, self.n_gates)
        trace_grad = delta_y @ self.R
        return (self.C_local.T @ trace_grad).T

    def build(self):
        """
        Construct H by applying the exact discrete wave transpose to each
        observation basis vector.  A full forward wave solve is then used to
        cross-validate the resulting matrix.
        """
        H_local = np.empty((self.n_data, self.n_owned))

        for j in range(self.n_data):
            delta_y = np.zeros(self.n_data)
            delta_y[j] = 1.0

            Z = self.transpose_to_states(delta_y)
            H_local[j] = self.acoustic.transpose(Z)

        self.H_local = H_local

        self._check_against_full_solve()

    def apply(self, p0):
        if hasattr(p0, "x"):
            p0_local = p0.x.array[:self.n_owned]
        else:
            p0_local = p0

        Y_local = self.H_local @ p0_local
        Y = np.empty_like(Y_local)
        self.comm.Allreduce(Y_local, Y, op=MPI.SUM)

        return Y

    def apply_transpose(self, delta_y):
        return self.H_local.T @ delta_y.reshape(-1)

    def apply_full(self, p0):
        """
        Apply the original wave solve followed by the space-time observation.
        """
        states = self.acoustic.solve(p0)
        return self.observe(states).reshape(-1)

    def check(self):
        return self._check_against_full_solve()

    def _check_against_full_solve(self):
        rng = np.random.default_rng([self.check_seed, self.comm.rank])
        p0 = rng.standard_normal(self.n_owned)

        Y_matrix = self.apply(p0)
        Y_full = self.apply_full(p0)

        scale = max(
            1.0e-15,
            np.linalg.norm(Y_matrix),
            np.linalg.norm(Y_full),
        )
        error = np.linalg.norm(Y_matrix - Y_full) / scale

        if self.comm.rank == 0:
            print(
                "Acoustic observation operator: "
                f"relative check error={error:.6e}, "
                f"passed={error <= self.check_tolerance}"
            )

        if not np.isfinite(error):
            raise RuntimeError(
                "Precomputed acoustic observation operator produced a non-finite "
                "full-wave check error."
            )

        if error > self.check_tolerance and self.comm.rank == 0:
            warnings.warn(
                "Precomputed acoustic observation operator does not satisfy the "
                f"requested check tolerance: error={error:.6e}, "
                f"tolerance={self.check_tolerance:.6e}.",
                RuntimeWarning,
            )

        return error

# ============================================================================
# Coupled PAI solver
# ============================================================================

class PAISolver:
    """
    Coupled optical-acoustic photoacoustic solver.

    solve(m) caches the state-dependent optical solution.  The acoustic
    observation map is precomputed and reused by forward, tangent, and adjoint
    operations.
    """

    def __init__(
        self,
        mesh,
        Vh,
        simulation_times,
        *,
        wavelength=750.0,
        roi=None,
        n_gates=5,
        t_start=None,
        t_end=None,
        gate_window="boxcar",
        sound_speed=1510.0e2,
        design_probe=False,
        xs_default=2.0,
        source_amplitude=3.0,
        source_sigma=0.75,
        probe_y=2.0,
        qoi_type="sO2",
        s_thr=0.5,
        c_thr=40.0,
        quad_deg=6,
        scat_a=18.9,
        scat_b=1.286,
        scat_lam_ref=500.0,
        high_accuracy=False,
        acoustic_check_tolerance=1.0e-10,
        acoustic_check_seed=12345,
    ):
        self.mesh = mesh
        self.Vh = Vh
        self.Kh = Vh

        self.comm = mesh.comm

        self.n_field_owned = Vh.dofmap.index_map.size_local
        self.n_field_global = Vh.dofmap.index_map.size_global

        self.nK_owned = 2 * self.n_field_owned
        self.nK_global = 2 * self.n_field_global

        self.wavelength = wavelength
        self.design_probe = design_probe
        self.xs_default = xs_default
        self.probe_x = xs_default
        self.probe_y = probe_y
        self.source_amplitude = source_amplitude
        self.source_sigma = source_sigma

        self.optical = OpticalDiffusion(
            mesh,
            Vh,
            roi=roi,
            scat_a=scat_a,
            scat_b=scat_b,
            scat_lam_ref=scat_lam_ref,
            high_accuracy=high_accuracy,
        )

        self.optical.set_source(
            self.probe_x,
            amplitude=self.source_amplitude,
            sigma=self.source_sigma,
            yb=self.probe_y,
        )
        
        self.acoustic = AcousticWave(
            mesh,
            Vh,
            simulation_times,
            sound_speed=sound_speed,
            high_accuracy=high_accuracy,
        )

        sensor_x = np.array([self.probe_x])
        sensor_y = np.array([self.probe_y])
        self.observation = AcousticObservation(
            self.acoustic,
            sensor_x,
            sensor_y,
            n_gates=n_gates,
            t_start=t_start,
            t_end=t_end,
            window=gate_window,
            check_tolerance=acoustic_check_tolerance,
            check_seed=acoustic_check_seed,
        )

        self.sensor2data = None
        self.qoi_type = qoi_type
        self.s_thr = s_thr
        self.c_thr = c_thr
        self.quad_deg = quad_deg

        self.n_parameter = self.nK_global
        self.n_data = self.observation.n_data
        self.n_qoi = 1 if qoi_type != "all" else 3

    def set_design(self, wavelength, xs=None):
        self.wavelength = wavelength

        if not self.design_probe or xs is None:
            xs = self.xs_default

        if xs == self.probe_x:
            return

        self.probe_x = xs
        self.optical.set_source(
            xs,
            amplitude=self.source_amplitude,
            sigma=self.source_sigma,
            yb=self.probe_y,
        )
        self.observation.set_sensor_locations(
            np.array([xs]),
            np.array([self.probe_y]),
        )
        self.n_data = self.observation.n_data

    def solve(self, m):
        """
        Solve and cache the coupled forward model.
        """
        p0 = self.optical.solve(m, self.wavelength)
        Y = self.observation.apply(p0)
        Q = self.apply_QoI()
        return Y, Q

    def solve_and_cache(self, m):
        """
        Compatibility wrapper for generic Jacobian/SVD code.

        The optical operator depends on m, so a new forward state necessarily
        rebuilds it.  Tangent and adjoint solves then reuse that cached operator.
        """
        return self.solve(m)

    def apply_QoI(self):
        Q = self.optical.qoi(
            qoi_type=self.qoi_type,
            s_thr=self.s_thr,
            c_thr=self.c_thr,
            quad_deg=self.quad_deg,
        )

        return Q[0] if Q.size == 1 else Q

    def linearized_observation(self, dm):
        """
        Apply the Jacobian at the cached forward state.
        """
        dp0 = self.optical.linearized(dm)
        return self.observation.apply(dp0)

    def linearized_observation_transpose(self, delta_y):
        """
        Apply the transpose Jacobian at the cached forward state.
        """
        p0_gradient = self.observation.apply_transpose(delta_y)
        lam, G = self.optical.observation_adjoint(p0_gradient)
        return self.optical.parameter_gradient(lam, G)

    def linearized_qoi_transpose(self):
        return self.optical.qoi_gradient(
            qoi_type=self.qoi_type,
            quad_deg=self.quad_deg,
        )
