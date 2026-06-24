# I have three problem formats - my own custom, sif2jax, and cyipopt.

import jax.numpy as jnp
import jax

def custom_to_cyipopt_format(f, c, d, x_L, x_U, d_L, d_U, x0):
    obj = jax.jit(lambda x: f(x))
    obj_grad = jax.jit(jax.grad(obj))
    obj_hess = jax.jit(jax.hessian(obj))

    constraints = []

    try: 
        cond_c = c(x0).shape[0] != 0
    except: 
        cond_c = c(x0) is not None

    if cond_c:
        if c(x0).shape[0] == 1:
            _c = lambda x: c(x)[0]
        else:
            _c = c  
        con_eq = _c
        con_eq_jac = jax.jacobian(con_eq)
        con_eq_hess = jax.hessian(con_eq)
        if c(x0).shape[0] == 1:
            def con_eq_hessvp(x, v):
                # v is scalar or (1,) for single constraint, con_eq_hess(x) is (n, n)
                out = v * con_eq_hess(x)
                # jax.debug.print("out shape in eq hessvp {}", out.shape)
                return out
        else:
            def con_eq_hessvp(x, v):
                out = jnp.einsum("ijk,i->jk", con_eq_hess(x), v)
                # jax.debug.print("out shape in eq hessvp {}", out.shape)
                return out
            
        constraints.append({
            'type': 'eq',
            'fun' : jax.jit(con_eq),
            'jac' : jax.jit(con_eq_jac),
            'hess': jax.jit(con_eq_hessvp),
        })

    try: 
        cond_d = d(x0).shape[0] != 0
    except: 
        cond_d = d(x0) is not None

    if cond_d:
        if d(x0).shape[0] == 1:
            _d = lambda x: d(x)[0]
        else:
            _d = d
        con_ineq = _d
        con_ineq_jac = jax.jacobian(con_ineq)
        con_ineq_hess = jax.hessian(con_ineq)
        if d(x0).shape[0] == 1:
            def con_ineq_hessvp(x, v):
                # v is scalar or (1,) for single constraint, con_ineq_hess(x) is (n, n)
                out = v * con_ineq_hess(x)
                # jax.debug.print("out shape in con_ineq_hessvp: {}", out.shape)
                return out
        else:
            def con_ineq_hessvp(x, v):
                out = jnp.einsum("ijk,i->jk", con_ineq_hess(x), v)
                # jax.debug.print("out shape in con_ineqx_hessvp: {}", out.shape)
                return out
        constraints.append({
            'type': 'ineq',
            'fun' : jax.jit(con_ineq),
            'jac' : jax.jit(con_ineq_jac),
            'hess': jax.jit(con_ineq_hessvp),
        })

    bounds = []
    for lb, ub in zip(x_L, x_U):
        bounds.append((lb, ub))

    return obj, obj_grad, obj_hess, constraints, bounds
    


def sif2jax_to_cyipopt_format(p):
    """
    Map a sif2jax problem `p` →  (obj, grad, hess, constraints, bounds)
    ready for cyipopt.minimize_ipopt.
    """

    # 1. Objective and its derivatives
    obj = lambda x: p.objective(x, p.args)
    obj_grad = jax.grad(obj)
    obj_hess = jax.hessian(obj)

    # 2. Constraints
    # Check for constraints using an initial point
    x0 = p.y0
    eq_x0, ineq_x0 = p.constraint(x0)

    constraints = []

    # Equality constraints h(x) = 0
    if eq_x0 is not None and eq_x0.size > 0:
        con_eq = lambda x: p.constraint(x)[0]
        con_eq_jac = jax.jacobian(con_eq)
        def con_eq_hessvp(x, v):
            hess = jax.hessian(con_eq)(x)
            hvp = jnp.einsum("ijk,i1->jk", hess, v)
            return hvp # jax.hessian(lambda x_inner: jnp.dot(v, con_eq(x_inner)))(x)

        constraints.append({
            'type': 'eq',
            'fun' : con_eq,
            'jac' : con_eq_jac,
            'hess': con_eq_hessvp,
        })

    # Inequality constraints g(x) >= 0
    # sif2jax problems define them as c(x) <= 0
    if ineq_x0 is not None and ineq_x0.size > 0:
        con_ineq = lambda x: p.constraint(x)[1]
        con_ineq_jac = jax.jacobian(con_ineq)
        def con_ineq_hessvp(x, v):
            hess = jax.hessian(con_ineq)(x)
            hvp = jnp.einsum("ijk,i1->jk", hess, v)
            return hvp # jax.hessian(lambda x_inner: jnp.dot(v, con_ineq(x_inner)))(x)

        constraints.append({
            'type': 'ineq',
            'fun' : con_ineq,
            'jac' : con_ineq_jac,
            'hess': con_ineq_hessvp,
        })

    # 3. Bounds
    if p.bounds is None:
        num_vars = len(p.y0)
        bounds = [(None, None)] * num_vars
    else:
        bounds = p.bounds

    return obj, obj_grad, obj_hess, constraints, bounds


def sif2jax_to_custom_format(p):
    """
    Fetches a problem from the sif2jax collection and adapts it to the
    custom format: (f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux).

    Args:
        problem_name (str): The name of the CUTEst problem to load.
        **sif_params: Additional parameters to pass to the SIF decoder.

    Returns:
        A tuple containing:
        - f: Objective function.
        - c: Equality constraints function (returns empty array if none).
        - d: Inequality constraints function (d(z) >= 0, returns empty array if none).
        - x_L: Lower bounds on variables.
        - x_U: Upper bounds on variables.
        - d_L: Lower bounds on inequality constraints.
        - d_U: Upper bounds on inequality constraints.
        - x0: Initial guess.
        - gt: Ground truth solution (if available).
        - aux: Auxiliary data (always None for this adapter).
    """
    # p = sif2jax.cutest.get_problem(problem_name)

    # 1. Objective function
    #f = lambda z: p.objective(z, p.args)
    def f(x_):
        out = p.objective(x_, p.args)
        has_inf = jnp.any(jnp.isinf(out))
        has_nan = jnp.any(jnp.isnan(out))
        # jax.debug.print("shape: {}", out.shape)
        # jax.debug.print("has inf/NaN: {}", has_inf | has_nan)
        return out

    # 2. Constraint functions
    # Evaluate at x0 to determine the existence and size of constraints
    eq_x0, ineq_x0 = p.constraint(p.y0)

    if eq_x0 is not None and eq_x0.size > 0:
        def c(x_):
            out = p.constraint(x_.flatten())[0].flatten()
            has_inf = jnp.any(jnp.isinf(out))
            has_nan = jnp.any(jnp.isnan(out))
            # jax.debug.print("shape: {}", out.shape)
            # jax.debug.print("equality constraint has inf or nan: {}", has_inf | has_nan)
            return out
        #c = lambda z: p.constraint(z.flatten())[0].flatten()
    else:
        c = lambda z: None

    if ineq_x0 is not None and ineq_x0.size > 0:
        def d(x_):
            out = p.constraint(x_.flatten())[1].flatten()
            has_inf = jnp.any(jnp.isinf(out))
            has_nan = jnp.any(jnp.isnan(out))
            # jax.debug.print("shape: {}", out.shape)
            # jax.debug.print("inequality constraint has inf or nan: {}", has_inf | has_nan)
            return out
        #d = lambda z: p.constraint(z.flatten())[1].flatten()
        num_ineq = ineq_x0.size
    else:
        d = lambda z: None
        num_ineq = 0

    # 3. Variable bounds
    if p.bounds is not None:
        x_L, x_U = p.bounds
    else:
        x_L = jnp.full_like(p.y0, -jnp.inf)
        x_U = jnp.full_like(p.y0, jnp.inf)
    # if p.bounds:
    #     x_L = jnp.array([b[0] if b[0] is not None else -jnp.inf for b in p.bounds])
    #     x_U = jnp.array([b[1] if b[1] is not None else jnp.inf for b in p.bounds])
    # else:
    #     # If no bounds are specified, assume they are infinite
    #     x_L = jnp.full_like(p.y0, -jnp.inf)
    #     x_U = jnp.full_like(p.y0, jnp.inf)

    # 4. Constraint bounds (for d(z) >= 0)
    d_L = jnp.zeros(num_ineq)
    d_U = jnp.full(num_ineq, jnp.inf)

    # 5. Initial guess
    x0 = p.y0

    # 6. Ground truth solution
    gt = p.expected_result

    return f, c, d, x_L, x_U, d_L, d_U, x0, gt
