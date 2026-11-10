from typing import Optional
import numpy as np
from BaseModels import GaussianPriorBase
from BaseModels import SolverAdjointCache

def relative_error(
    actual: np.ndarray,
    reference: np.ndarray,
) -> float:
    scale = max(1E-15, np.linalg.norm(actual), np.linalg.norm(reference))
    return float(np.linalg.norm(actual - reference) / scale)

def scalar_relative_error(lhs: float, rhs: float) -> float:
    """Symmetric scalar relative error with protection near zero."""
    return abs(lhs - rhs) / max(1E-15, abs(lhs), abs(rhs))

def normalized_random_vector(
    rng: np.random.Generator,
    dim: int,
) -> np.ndarray:
    """Generate a standard normal vector and normalise it."""
    vector = rng.standard_normal(dim)
    norm = np.linalg.norm(vector)
    return vector / norm

def apply_whitened_jacobian(
    prior: GaussianPriorBase,
    solver: SolverAdjointCache,
    delta_xi: np.ndarray,
) -> np.ndarray:
    """
    Apply the Jacobian with respect to whitened coordinates.

    If x = L xi and J_x is the solver Jacobian with respect to x, then

        J_xi = J_x L.
    """
    delta_x = prior.matvec_prior_L(delta_xi)
    return solver.linearized_observation(delta_x)

def apply_whitened_jacobian_transpose(
    prior: GaussianPriorBase,
    solver: SolverAdjointCache,
    delta_y: np.ndarray,
) -> np.ndarray:
    """
    Apply the transpose Jacobian with respect to whitened coordinates.

    If x = L xi, then

        J_xi.T = L.T J_x.T.
    """
    grad_x = solver.linearized_observation_transpose(delta_y)
    return prior.matvec_prior_LT(grad_x)

def test_prior_L_adjoint(
    prior: GaussianPriorBase,
    *,
    tolerance: float = 1.0e-12,
    verbose: bool = True,
):
    """
    Test the prior adjoint identity

        <L xi, x> = <xi, L.T x>.
    """
    rng = np.random.default_rng()
    dim = int(prior.dim)

    xi = rng.standard_normal(dim)
    x_dual = rng.standard_normal(dim)

    lhs = float(np.dot(prior.matvec_prior_L(xi), x_dual))
    rhs = float(np.dot(xi, prior.matvec_prior_LT(x_dual)))

    error = scalar_relative_error(lhs, rhs)
    passed = error <= tolerance

    if verbose:
        print("Prior L/L.T adjoint test")
        print(f"  <L xi, x>       : {lhs:.16e}")
        print(f"  <xi, L.T x>     : {rhs:.16e}")
        print(f"  relative error  : {error:.6e}")
        print(f"  passed          : {passed}")

    return error

def test_prior_inverse(
    prior: GaussianPriorBase,
    *,
    tolerance: float = 1.0e-12,
    verbose: bool = True,
):
    """
    Test

        invL(L(xi)) = xi.

    The prior must implement ``matvec_prior_invL``.
    """
    rng = np.random.default_rng()
    dim = int(prior.dim)

    xi = rng.standard_normal(dim)
    x = prior.matvec_prior_L(xi)
    xi_recovered = prior.matvec_prior_invL(x)

    error = relative_error(xi_recovered, xi)
    passed = error <= tolerance

    if verbose:
        print("Prior inverse test")
        print(f"  relative error  : {error:.6e}")
        print(f"  passed          : {passed}")

    return error

def test_solver_forward_derivative(
    solver: SolverAdjointCache,
    x: np.ndarray,
    *,
    direction: Optional[np.ndarray] = None,
    epsilon: float = 1.0e-6,
    tolerance: float = 1.0e-5,
    verbose: bool = True,
):
    """
    Test the physical-parameter Jacobian action ``J_x @ direction``.

    The solver must implement

        solve(x)
        solve_and_cache(x)
        linearized_observation(direction)
    """
    rng = np.random.default_rng() 

    if direction is None:
        direction = normalized_random_vector(rng, x.size)

    _ = solver.solve_and_cache(x)
    jv = solver.linearized_observation(direction)
    y_plus, _ = solver.solve(x + epsilon * direction)
    y_minus, _ = solver.solve(x - epsilon * direction)
    finite_difference = (y_plus - y_minus) / (2.0 * epsilon)
    error = relative_error(finite_difference, jv)
    passed = error <= tolerance

    if verbose:
        print("Physical-parameter forward derivative test")
        print(f"  epsilon          : {epsilon:.3e}")
        print(
            f"  ||finite diff||  : "
            f"{np.linalg.norm(finite_difference):.6e}"
        )
        print(f"  ||Jv||           : {np.linalg.norm(jv):.6e}")
        print(f"  relative error   : {error:.6e}")
        print(f"  passed           : {passed}")

    return error

def test_solver_adjoint(
    solver: SolverAdjointCache,
    x: np.ndarray,
    *,
    parameter_direction: Optional[np.ndarray] = None,
    observation_direction: Optional[np.ndarray] = None,
    tolerance: float = 1.0e-12,
    verbose: bool = True,
):
    """
    Test the physical-space adjoint identity

        <J_x v, z> = <v, J_x.T z>.
    """
    rng = np.random.default_rng() 

    y, _ = solver.solve_and_cache(x)

    if parameter_direction is None:
        parameter_direction = rng.standard_normal(x.size)

    if observation_direction is None:
        observation_direction = rng.standard_normal(y.size)

    jv = solver.linearized_observation(parameter_direction)
    jtz = solver.linearized_observation_transpose(observation_direction)

    lhs = float(np.dot(jv, observation_direction))
    rhs = float(np.dot(parameter_direction, jtz))

    error = scalar_relative_error(lhs, rhs)
    passed = error <= tolerance

    if verbose:
        print("Physical-parameter adjoint test")
        print(f"  <Jv, z>          : {lhs:.16e}")
        print(f"  <v, J.T z>       : {rhs:.16e}")
        print(f"  relative error   : {error:.6e}")
        print(f"  passed           : {passed}")

    return error

def test_whitened_forward_derivative(
    prior: GaussianPriorBase,
    solver: SolverAdjointCache,
    xi: np.ndarray,
    *,
    direction: Optional[np.ndarray] = None,
    epsilon: float = 1.0e-6,
    tolerance: float = 1.0e-5,
    verbose: bool = True,
):
    """
    Test the derivative of

        xi -> solver.solve(prior.matvec_prior_L(xi)).
    """
    rng = np.random.default_rng() 

    if direction is None:
        direction = normalized_random_vector(rng, xi.size)

    x = prior.transform(xi)
    _ = solver.solve_and_cache(x)
    jv = apply_whitened_jacobian(prior, solver, direction)

    x_plus = prior.transform(xi + epsilon * direction)
    x_minus = prior.transform(xi - epsilon * direction)

    y_plus, _ = solver.solve(x_plus)
    y_minus, _ = solver.solve(x_minus)

    finite_difference = (y_plus - y_minus) / (2.0 * epsilon)
    
    error = relative_error(finite_difference, jv)
    passed = error <= tolerance

    if verbose:
        print("Whitened-coordinate forward derivative test")
        print(f"  epsilon          : {epsilon:.3e}")
        print(
            f"  ||finite diff||  : "
            f"{np.linalg.norm(finite_difference):.6e}"
        )
        print(f"  ||J_xi v||       : {np.linalg.norm(jv):.6e}")
        print(f"  relative error   : {error:.6e}")
        print(f"  passed           : {passed}")

    return error

def test_whitened_adjoint(
    prior: GaussianPriorBase,
    solver: SolverAdjointCache,
    xi: np.ndarray,
    *,
    whitened_direction: Optional[np.ndarray] = None,
    observation_direction: Optional[np.ndarray] = None,
    tolerance: float = 1.0e-12,
    verbose: bool = True,
):
    """
    Test the whitened adjoint identity

        <J_xi v, z> = <v, J_xi.T z>.
    """
    rng = np.random.default_rng() 

    x = prior.transform(xi)
    y, _ = solver.solve_and_cache(x)

    if whitened_direction is None:
        whitened_direction = rng.standard_normal(xi.size)

    if observation_direction is None:
        observation_direction = rng.standard_normal(y.size)

    jv = apply_whitened_jacobian(prior, solver, whitened_direction)
    jtz = apply_whitened_jacobian_transpose(prior, solver, observation_direction)

    lhs = float(np.dot(jv, observation_direction))
    rhs = float(np.dot(whitened_direction, jtz))

    error = scalar_relative_error(lhs, rhs)
    passed = error <= tolerance

    if verbose:
        print("Whitened-coordinate adjoint test")
        print(f"  <J_xi v, z>      : {lhs:.16e}")
        print(f"  <v, J_xi.T z>    : {rhs:.16e}")
        print(f"  relative error   : {error:.6e}")
        print(f"  passed           : {passed}")

    return error

def test_combined(
    prior: GaussianPriorBase,
    solver: SolverAdjointCache,
    *,
    n_tests: int = 50,
    epsilon: float = 1.0e-6,
    prior_tolerance: float = 1.0e-12,
    derivative_tolerance: float = 1.0e-5,
    adjoint_tolerance: float = 1.0e-12,
    verbose: bool = True,
):
    """
    Test a Gaussian prior and an adjoint solver using multiple random
    realisations.

    Tests
    -----
    1. Prior L/L.T adjoint identity.
    2. Prior inverse: invL(L xi) = xi.
    3. Physical-parameter Jacobian by central differences.
    4. Physical-parameter adjoint identity.
    5. Whitened-coordinate Jacobian by central differences.
    6. Whitened-coordinate adjoint identity.
    """
    prior_adjoint_errors = np.empty(n_tests)
    prior_inverse_errors = np.empty(n_tests)
    jacobian_errors = np.empty(n_tests)
    adjoint_errors = np.empty(n_tests)
    whitened_jacobian_errors = np.empty(n_tests)
    whitened_adjoint_errors = np.empty(n_tests)

    for i in range(n_tests):
        xi = prior.random()
        x = prior.transform(xi)

        prior_adjoint_errors[i] = test_prior_L_adjoint(
            prior,
            tolerance=prior_tolerance,
            verbose=False,
        )

        prior_inverse_errors[i] = test_prior_inverse(
            prior,
            tolerance=prior_tolerance,
            verbose=False,
        )

        jacobian_errors[i] = test_solver_forward_derivative(
            solver,
            x,
            epsilon=epsilon,
            tolerance=derivative_tolerance,
            verbose=False,
        )

        adjoint_errors[i] = test_solver_adjoint(
            solver,
            x,
            tolerance=adjoint_tolerance,
            verbose=False,
        )

        whitened_jacobian_errors[i] = test_whitened_forward_derivative(
                prior,
                solver,
                xi,
                epsilon=epsilon,
                tolerance=derivative_tolerance,
                verbose=False,
            )

        whitened_adjoint_errors[i] = test_whitened_adjoint(
            prior,
            solver,
            xi,
            tolerance=adjoint_tolerance,
            verbose=False,
        )

    if verbose:
        print("Solver and prior derivative tests")
        print(f"  number of tests : {n_tests}")
        print(f"  epsilon         : {epsilon:.3e}")
        print()
        print(
            f"{'test':30s}"
            f"{'mean relative error':>22s}"
            f"{'max relative error':>22s}"
        )
        print("-" * 74)

        print(
            f"{'prior L/L.T adjoint':30s}"
            f"{np.mean(prior_adjoint_errors):22.6e}"
            f"{np.max(prior_adjoint_errors):22.6e}"
        )

        print(
            f"{'prior inverse':30s}"
            f"{np.mean(prior_inverse_errors):22.6e}"
            f"{np.max(prior_inverse_errors):22.6e}"
        )

        print(
            f"{'physical Jacobian':30s}"
            f"{np.mean(jacobian_errors):22.6e}"
            f"{np.max(jacobian_errors):22.6e}"
        )

        print(
            f"{'physical adjoint':30s}"
            f"{np.mean(adjoint_errors):22.6e}"
            f"{np.max(adjoint_errors):22.6e}"
        )

        print(
            f"{'whitened Jacobian':30s}"
            f"{np.mean(whitened_jacobian_errors):22.6e}"
            f"{np.max(whitened_jacobian_errors):22.6e}"
        )

        print(
            f"{'whitened adjoint':30s}"
            f"{np.mean(whitened_adjoint_errors):22.6e}"
            f"{np.max(whitened_adjoint_errors):22.6e}"
        )

