"""
Minimal loader for the jaxipm-aligned IPOPT state dump (see
src/jaxipm/tests/LOGGING_SCHEMA.md).

The patched IPOPT now writes one file per OptimizationState sub-struct under
``ipopt_logs/iter_<n>/`` in a sectioned ``@name count [cols]`` format, with
section names equal to jaxipm field names. That makes the reader a dumb
deserializer: parse the sections, drop them into the eqx.Modules. This replaces
the ~900-line translation layer in validation_utils.py / load_ipopt_state_*.py.

Anything IPOPT does not emit is zero-filled from ``cp`` dimensions (the schema's
"gaps"), so this loader degrades gracefully as the producer-side logging is
completed (e.g. step_aff_full / soc_taken are currently not emitted by mk2).
"""
import os
import numpy as np
import jax.numpy as jnp

from jaxipm.structures import (
    OptimizationState,
    Iterate,
    CalculatedQuantitiesPreMu,
    CalculatedQuantitiesPostMu,
    IterateFlags,
    WatchdogState,
    LineSearchFilterState,
    LineSearchState,
    InertiaCorrectionState,
)

save_dir = "ipopt_logs"

# helpers
def load_vector(filename):
    """Loads a vector from a file, returning an empty array for non-existent or empty files."""
    try:
        # Check if file exists and is not empty
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            return np.loadtxt(filename)
        else:
            return np.array([])
    except (IOError, ValueError):
        return np.array([])
    
def load_scalars(filename):
    """Loads key-value scalar pairs from a file."""
    scalars = {}
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        scalars[parts[0]] = float(parts[1])
    except IOError:
        pass  # File not found, return empty dict
    return scalars


def load_iterate_x_from_components(save_dir, iter_num, cp, prefix="iterate"):
    """Load the x component of an iterate/delta, handling restoration phase components.

    During restoration (iteration 1), IPOPT generates component files:
    - {prefix}_x_{iter}_comp0.txt: x_orig (original problem variables)
    - {prefix}_x_{iter}_comp1.txt: nc (negative constraint slack)
    - {prefix}_x_{iter}_comp2.txt: pc (positive constraint slack)
    - {prefix}_x_{iter}_comp3.txt: nd (negative inequality slack)
    - {prefix}_x_{iter}_comp4.txt: pd (positive inequality slack)

    During regular iterations, only comp0 exists.

    Args:
        save_dir: directory containing log files
        iter_num: iteration number
        cp: CommonProblem instance with problem dimensions
        prefix: file prefix, e.g., "iterate" or "delta" (default: "iterate")

    Returns:
        x vector with shape (cp.nx + 2*cp.nyc + 2*cp.nyd, 1) for restoration,
        or (cp.nx, 1) padded with zeros for regular iterations
    """
    # Check if this is a restoration iteration (comp1 exists)
    comp1_path = f"{save_dir}/{prefix}_x_{iter_num}_comp1.txt"

    if os.path.exists(comp1_path):
        # Restoration phase - load all 5 components
        components = []
        for i in range(5):
            comp_path = f"{save_dir}/{prefix}_x_{iter_num}_comp{i}.txt"
            if os.path.exists(comp_path):
                vec = load_vector(comp_path)
                if vec.size == 0:
                    vec = jnp.zeros([0, 1])
                else:
                    vec = jnp.array(vec).reshape(-1, 1)
                components.append(vec)
            else:
                # If a component file is missing, use zeros
                # Infer size from cp dimensions
                if i == 0:  # x_orig
                    components.append(jnp.zeros([cp.nx, 1]))
                elif i in [1, 2]:  # nc, pc
                    components.append(jnp.zeros([cp.nyc, 1]))
                else:  # nd, pd
                    components.append(jnp.zeros([cp.nyd, 1]))

        # Concatenate all components: [x_orig; nc; pc; nd; pd]
        x_full = jnp.vstack(components)
        return x_full
    else:
        # Regular iteration - load comp0 and pad with zeros
        comp0_path = f"{save_dir}/{prefix}_x_{iter_num}_comp0.txt"
        if os.path.exists(comp0_path):
            x_orig = load_vector(comp0_path)
            if x_orig.size == 0:
                x_orig = jnp.zeros([cp.nx, 1])
            else:
                x_orig = jnp.array(x_orig).reshape(-1, 1)
        else:
            x_orig = jnp.zeros([cp.nx, 1])

        # Pad with zeros for restoration variables (even though we're not in restoration)
        resto_dim = 2 * cp.nyc + 2 * cp.nyd
        padding = jnp.zeros([resto_dim, 1])
        return jnp.vstack([x_orig, padding])


def load_iterate_z_L_from_components(save_dir, iter_num, cp, prefix="iterate"):
    """Load the z_L component of an iterate/delta, handling restoration phase components.

    During restoration, IPOPT generates component files:
    - {prefix}_z_L_{iter}_comp0.txt: z_L_orig (original lower bound multipliers, size nxL)
    - {prefix}_z_L_{iter}_comp1.txt: z_L_nc (lower bounds for nc, size nyc)
    - {prefix}_z_L_{iter}_comp2.txt: z_L_pc (lower bounds for pc, size nyc)
    - {prefix}_z_L_{iter}_comp3.txt: z_L_nd (lower bounds for nd, size nyd)
    - {prefix}_z_L_{iter}_comp4.txt: z_L_pd (lower bounds for pd, size nyd)

    During regular iterations, only comp0 exists.

    The structure matches: [z_L_orig; z_L_nc; z_L_pc; z_L_nd; z_L_pd]
    which corresponds to: [nxL; nyc; nyc; nyd; nyd]

    Args:
        save_dir: directory containing log files
        iter_num: iteration number
        cp: CommonProblem instance with problem dimensions
        prefix: file prefix, e.g., "iterate" or "delta" (default: "iterate")

    Returns:
        z_L vector with shape (nxL + nx + nyc, 1) - padded for consistency
    """
    # Check if this is a restoration iteration (comp1 exists)
    comp1_path = f"{save_dir}/{prefix}_z_L_{iter_num}_comp1.txt"

    if os.path.exists(comp1_path):
        # Restoration phase - load all 5 components
        components = []
        for i in range(5):
            comp_path = f"{save_dir}/{prefix}_z_L_{iter_num}_comp{i}.txt"
            if os.path.exists(comp_path):
                vec = load_vector(comp_path)
                if vec.size == 0:
                    vec = jnp.zeros([0, 1])
                else:
                    vec = jnp.array(vec).reshape(-1, 1)
                components.append(vec)
            else:
                # If a component file is missing, use zeros
                # Infer size from cp dimensions
                if i == 0:  # z_L_orig
                    components.append(jnp.zeros([cp.nxL, 1]))
                elif i in [1, 2]:  # z_L_nc, z_L_pc
                    components.append(jnp.zeros([cp.nyc, 1]))
                else:  # z_L_nd, z_L_pd
                    components.append(jnp.zeros([cp.nyd, 1]))

        # Concatenate all components: [z_L_orig; z_L_nc; z_L_pc; z_L_nd; z_L_pd]
        z_L_full = jnp.vstack(components)
        return z_L_full
    else:
        # Regular iteration - load comp0 and pad with zeros
        comp0_path = f"{save_dir}/{prefix}_z_L_{iter_num}_comp0.txt"
        if os.path.exists(comp0_path):
            z_L_orig = load_vector(comp0_path)
            if z_L_orig.size == 0:
                z_L_orig = jnp.zeros([cp.nxL, 1])
            else:
                z_L_orig = jnp.array(z_L_orig).reshape(-1, 1)
        else:
            z_L_orig = jnp.zeros([cp.nxL, 1])

        # Pad with zeros: [z_L_orig; zeros(nyc*2 + nyd*2)]
        # The resto-padding represents bound mults on the resto vars (nc, pc, nd, pd).
        return jnp.vstack([z_L_orig, jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])])


# --------------------------------------------------------------------------- #
# Parser                                                                       #
# --------------------------------------------------------------------------- #
def parse_sections(path):
    """Parse a ``@name count [cols]`` file -> {name: jnp.ndarray}.

    1-D section -> shape (count,);  2-D section -> shape (rows, cols).
    Missing file -> {} (caller zero-fills).

    Duplicate keys: FIRST occurrence wins. At the restoration-EXIT boundary IPOPT
    rewinds the iteration counter (PerformRestoration: Set_iter_count(resto-1)), so
    the boundary iter dir gets a SECOND dump of fl/ls/wd (+cqpo.alpha_pr) from the
    resumed REGULAR line search that merely re-accepts the final resto point. Only
    the FIRST dump is aligned with that iter's it/cqpr/state (all logged once, from
    the resto step computation); taking the last dump instead seeds the harness with
    in_restoration=0 over a mid-resto iterate, so post_process never runs the exit
    check and the saved_* restore is skipped (the old iter-34/35 divergence).
    """
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        lines = f.read().split("\n")
    i, n = 0, len(lines)
    while i < n:
        head = lines[i].strip()
        i += 1
        if not head.startswith("@"):
            continue
        parts = head[1:].split()
        name = parts[0]
        rows = int(parts[1])
        cols = int(parts[2]) if len(parts) > 2 else None
        block = lines[i : i + rows]
        i += rows
        if name in out:
            continue
        if cols is None:  # 1-D
            out[name] = jnp.array([float(x) for x in block]) if block else jnp.zeros(0)
        else:  # 2-D (e.g. filter.F, adfs)
            data = [[float(x) for x in ln.split()] for ln in block if ln.strip()]
            out[name] = jnp.array(data) if data else jnp.zeros((0, cols))
    return out


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def _col(a):
    """-> column vector (n, 1)."""
    return jnp.asarray(a, dtype=float).reshape(-1, 1)


def _sc(sections, name, default=0.0):
    """Scalar section -> (1, 1)."""
    v = sections.get(name)
    if v is None or jnp.asarray(v).size == 0:
        return jnp.atleast_2d(float(default))
    return jnp.atleast_2d(float(jnp.asarray(v).reshape(-1)[0]))


def _vec(sections, name, dim):
    """Vector section -> (dim, 1), zero-filled if absent/empty."""
    v = sections.get(name)
    if v is None or jnp.asarray(v).size == 0:
        return jnp.zeros([dim, 1])
    return _col(v)


def _filter_pad(F2, filter_size):
    """(n, 2) entries -> (filter_size, 3) = [index, theta, phi], inf-padded.

    IPOPT's filter_.GetEntries() emits each entry as (phi, theta); jaxipm stores
    [index, theta, phi] (augment_raw_filter writes [index, val_theta, val_phi];
    is_acceptable_to_current_filter reads col1=theta, col2=phi). So the two value
    columns are swapped here. Shared by adfs and ls.filter.F (same IPOPT source +
    same jaxipm convention); the swap is invisible for an empty/all-inf filter.
    """
    F2 = jnp.asarray(F2)
    if F2.ndim != 2 or F2.shape[0] == 0:
        F2 = jnp.zeros((0, 2))
    F2 = jnp.vstack([F2, jnp.full([filter_size - F2.shape[0], 2], jnp.inf)])
    F2 = F2[:, ::-1]  # (phi, theta) -> (theta, phi)
    return jnp.hstack([jnp.arange(filter_size)[:, None], F2])


def _stitch(sections, base, sizes):
    """Restoration-aware x / z_L: base section (= comp0) plus the augmented
    .nc/.pc/.nd/.pd blocks when present, else zero-padded to restoration size.

    sizes = [base_dim, nyc, nyc, nyd, nyd].  Returns (sum(sizes), 1).
    """
    parts = [_vec(sections, base, sizes[0])]
    suffixes = ["nc", "pc", "nd", "pd"]
    in_resto = (base + ".nc") in sections
    for suf, sz in zip(suffixes, sizes[1:]):
        parts.append(_vec(sections, f"{base}.{suf}", sz) if in_resto else jnp.zeros([sz, 1]))
    return jnp.vstack(parts)


def _load_iterate(sections, prefix, cp):
    """Build an Iterate from sections named ``<prefix>x``, ``<prefix>s``, ...

    x and z_L are stitched/padded to restoration size; the rest are static.
    """
    xs = [cp.nx, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    zs = [cp.nxL, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    return Iterate(
        x=_stitch(sections, prefix + "x", xs),
        s=_vec(sections, prefix + "s", cp.nyd),
        y_c=_vec(sections, prefix + "y_c", cp.nyc),
        y_d=_vec(sections, prefix + "y_d", cp.nyd),
        z_L=_stitch(sections, prefix + "z_L", zs),
        z_U=_vec(sections, prefix + "z_U", cp.nxU),
        v_L=_vec(sections, prefix + "v_L", cp.ndL),
        v_U=_vec(sections, prefix + "v_U", cp.ndU),
    )


def _load_step_full(sections, prefix, cp):
    """Flatten an iterate-shaped step (``<prefix>x``, ``<prefix>s``, ...) into
    jaxipm's flat ``step_aff_full`` / ``step_cen_full`` layout

        [x(nx+aug); s(nyd); y_c(nyc); y_d(nyd); z_L(nxL+aug); z_U(nxU); v_L(ndL); v_U(ndU)]

    matching quantities.calc_transform_aug_to_full. x and z_L carry the resto
    block (stitched/zero-padded).

    The affine/centrality predictor steps are produced only by the adaptive-mu
    oracle (QualityFunctionMuOracle), so IPOPT omits them in fixed/monotone-mu mode
    (free_mu_mode=0). Zero-fill when the producer section is absent — analyze_diff
    skips these where IPOPT didn't log them (see _SKIP_IF_ABSENT)."""
    xs = [cp.nx, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    zs = [cp.nxL, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    if (prefix + "x") not in sections:
        total = sum(xs) + cp.nyd + cp.nyc + cp.nyd + sum(zs) + cp.nxU + cp.ndL + cp.ndU
        return jnp.zeros([total, 1])
    return jnp.vstack([
        _stitch(sections, prefix + "x", xs),
        _vec(sections, prefix + "s", cp.nyd),
        _vec(sections, prefix + "y_c", cp.nyc),
        _vec(sections, prefix + "y_d", cp.nyd),
        _stitch(sections, prefix + "z_L", zs),
        _vec(sections, prefix + "z_U", cp.nxU),
        _vec(sections, prefix + "v_L", cp.ndL),
        _vec(sections, prefix + "v_U", cp.ndU),
    ])


def _empty_iterate(cp):
    return Iterate(
        x=jnp.zeros([cp.nx + 2 * cp.nyc + 2 * cp.nyd, 1]),
        s=jnp.zeros([cp.nyd, 1]),
        y_c=jnp.zeros([cp.nyc, 1]),
        y_d=jnp.zeros([cp.nyd, 1]),
        z_L=jnp.zeros([cp.nxL + 2 * cp.nyc + 2 * cp.nyd, 1]),
        z_U=jnp.zeros([cp.nxU, 1]),
        v_L=jnp.zeros([cp.ndL, 1]),
        v_U=jnp.zeros([cp.ndU, 1]),
    )


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #
def load_state(iter_num, cp, base_dir=None):
    """Reconstruct an OptimizationState from ``<base_dir>/iter_<iter_num>/``.

    base_dir defaults to the module-level ``save_dir`` ("ipopt_logs").
    """
    base = base_dir if base_dir is not None else save_dir
    d = f"{base}/iter_{iter_num}"
    # Fail loudly rather than silently zero-filling a non-existent dump.
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"no IPOPT iter dump at '{d}' (cwd={os.getcwd()}). "
            f"Run from a directory containing '{base}/', or pass base_dir=."
        )
    if not os.path.exists(f"{d}/it.txt"):
        raise FileNotFoundError(f"missing '{d}/it.txt' — incomplete IPOPT dump.")
    S = {name: parse_sections(f"{d}/{name}.txt") for name in
         ("it", "cqpr", "cqpo", "fl", "wd", "ls", "ic", "state", "soc")}
    static = parse_sections(f"{base}/_static/proj.txt")
    scaling = parse_sections(f"{base}/_static/nlp_scaling.txt")
    # Frozen restoration reference mu (rmu = max(orig_mu, ||c||_inf, ||d-s||_inf)),
    # logged once to _static by IPOPT's RestoIterateInitializer. Constant for the whole
    # restoration phase; distinct from the current (updated) restoration mu.
    resto_ref = parse_sections(f"{base}/_static/resto_ref_mu.txt")
    # Frozen restoration reference point (x_ref = entry iterate, dr_x = min(1,1/|x_ref|)),
    # logged once to _static by IPOPT's RestoIpoptNLP::InitializeStructures. jaxipm freezes
    # these at restoration entry (initialization.py:1651); the current iterate drifts away
    # from them as restoration progresses, so reconstructing x_ref/dr_x from it.x is only
    # correct at the entry iter. Read the true frozen values when in restoration.
    resto_ref_x = parse_sections(f"{base}/_static/resto_ref_x.txt")

    fs = cp.p["filter_size"]
    resto_dim = 2 * cp.nyc + 2 * cp.nyd
    aug = resto_dim  # padding length for the resto-variable blocks

    # ---- it : Iterate ----
    it = _load_iterate(S["it"], "", cp)

    # ---- fl : IterateFlags (7 logged, 6 internal gaps -> 0) ----
    fls = S["fl"]
    fl = IterateFlags(
        in_watchdog=_sc(fls, "in_watchdog"),
        in_soft_resto_phase=_sc(fls, "in_soft_resto_phase"),
        in_restoration=_sc(fls, "in_restoration"),
        theta_max_instantiated=_sc(fls, "theta_max_instantiated"),
        fallback_activated=_sc(fls, "fallback_activated"),
        tiny_step_last_iter=_sc(fls, "tiny_step_last_iter"),
        skip_first_trial=jnp.array([[0]]),
        soft_resto_entry_requested=jnp.array([[0]]),
        free_mu_mode=_sc(fls, "free_mu_mode"),
        tiny_step_flag=_sc(fls, "tiny_step_flag"),
        needs_resto_init=jnp.array([[0]]),
        needs_regular_init=jnp.array([[0]]),
        should_exit_resto=jnp.array([[0]]),
    )

    # ---- state : top-level scalars (mu/tau/mu_max/init_inf) + adfs ----
    st = S["state"]
    mu = _sc(st, "mu")
    tau = _sc(st, "tau")
    mu_max = _sc(st, "mu_max", 1e10)
    init_dual_inf = _sc(st, "init_dual_inf", 1.0)
    init_primal_inf = _sc(st, "init_primal_inf", 1.0)
    adfs = _filter_pad(st.get("adfs", jnp.zeros((0, 2))), fs)

    # ---- args (scaling) — needed before jac_f ----
    df = float(jnp.asarray(scaling.get("df", jnp.array([1.0]))).reshape(-1)[0])
    dc = scaling.get("dc")
    dd = scaling.get("dd")
    dc = jnp.ones([cp.nyc]) if dc is None or jnp.asarray(dc).size == 0 else jnp.asarray(dc).reshape(-1)
    dd = jnp.ones([cp.nyd]) if dd is None or jnp.asarray(dd).size == 0 else jnp.asarray(dd).reshape(-1)
    in_resto = bool(jnp.asarray(fl.in_restoration).squeeze())
    # x_ref / dr_x: in restoration jaxipm FREEZES these at the entry iterate, so read the
    # frozen values IPOPT logged to _static (the current iterate has drifted away by later
    # resto iters). Outside restoration these slots are dummies, so reconstruct from it.x.
    if in_resto and "x_ref" in resto_ref_x:
        x_ref = jnp.asarray(resto_ref_x["x_ref"]).reshape(-1)
        dr_x = jnp.asarray(resto_ref_x["dr_x"]).reshape(-1)
    else:
        x_ref = it.x.flatten()
        dr_x = jnp.minimum(1.0, 1.0 / jnp.abs(x_ref))
    # args[0][0] = the FROZEN restoration reference mu while in restoration (jaxipm holds
    # rmu in args[0][0], frozen at entry; it is NOT the current iter's mu). Outside
    # restoration this slot is a dummy, so the current mu is fine. Fall back to the
    # current mu if the _static dump is absent (e.g. a run with no restoration).
    arg0_mu = _sc(resto_ref, "ref_mu", float(jnp.asarray(mu).reshape(-1)[0])) if in_resto else mu
    args = (
        (arg0_mu, x_ref[: cp.nx], dr_x[: cp.nx], jnp.array([df])),  # (1,1)
        (dc,),
        (dd,),
    )

    # ---- cqpr : CalculatedQuantitiesPreMu ----
    cq = S["cqpr"]
    # jac_f: gradient over the original x, padded by the resto-variable block to
    # the optimizer's (nx + resto_dim, 1) shape (f is independent of resto vars).
    jac_f = jnp.vstack([
        cp.nstqf.calc_jac_f(it.x[: cp.nx], *args[0]).todense().reshape(-1, 1),
        jnp.zeros([aug, 1]),
    ])

    # grad_lag_x and the x-slack are padded by the resto-variable block when at
    # regular (non-resto) size, matching the optimizer's static shapes.
    def _pad_resto(v, base_dim):
        return jnp.vstack([v, jnp.zeros([aug, 1])]) if v.shape[0] == base_dim else v

    grad_lag_x = _pad_resto(_vec(cq, "grad_lag_x", cp.nx), cp.nx)
    sxL = _pad_resto(_vec(cq, "slacks.sxL", cp.nxL), cp.nxL)

    cqpr = CalculatedQuantitiesPreMu(
        f=_sc(cq, "f"),
        c=_vec(cq, "c", cp.nyc),
        d=_vec(cq, "d", cp.nyd),
        jac_f=jac_f,
        dms=_vec(cq, "dms", cp.nyd),
        y_nrminf=jnp.asarray(_sc(cq, "y_nrminf")).reshape(()),
        grad_lag_x=grad_lag_x,
        grad_lag_s=_vec(cq, "grad_lag_s", cp.nyd),
        slacks=(
            sxL,
            _vec(cq, "slacks.sxU", cp.nxU),
            _vec(cq, "slacks.sdL", cp.ndL),
            _vec(cq, "slacks.sdU", cp.ndU),
        ),
        avrg_compl=_sc(cq, "avrg_compl"),
        theta=_sc(cq, "theta"),
        grad_lag_x_nrm2=jnp.asarray(_sc(cq, "grad_lag_x_nrm2")).reshape(()),
        grad_lag_s_nrm2=jnp.asarray(_sc(cq, "grad_lag_s_nrm2")).reshape(()),
        c_nrm2=jnp.asarray(_sc(cq, "c_nrm2")).reshape(()),
        d_minus_s_nrm2=jnp.asarray(_sc(cq, "d_minus_s_nrm2")).reshape(()),
        nlp_error=jnp.asarray(_sc(cq, "nlp_error")).reshape(()),
        nlp_constr_viol=_sc(cq, "nlp_constr_viol"),
        barrier_constr_viol=_sc(cq, "barrier_constr_viol"),
        primal_inf=_sc(cq, "primal_inf"),
        step_aff_full=_load_step_full(cq, "step_aff_full.", cp),
        step_cen_full=_load_step_full(cq, "step_cen_full.", cp),
        Sigma_nc_inv=_vec(cq, "Sigma_nc_inv", cp.nyc),
        Sigma_pc_inv=_vec(cq, "Sigma_pc_inv", cp.nyc),
        Sigma_nd_inv=_vec(cq, "Sigma_nd_inv", cp.nyd),
        Sigma_pd_inv=_vec(cq, "Sigma_pd_inv", cp.nyd),
        # y_c_init / y_d_init: least-square multiplier estimate. IPOPT only emits
        # these where it actually calls CalculateMultipliers (instantiation, and
        # potentially resto/regular re-init), so zero-fill when absent rather than
        # require it — and exclude them from analyze_diff for iters that lack them.
        y_c_init=_vec(cq, "y_c_init", cp.nyc),
        y_d_init=_vec(cq, "y_d_init", cp.nyd),
    )

    # ---- cqpo : CalculatedQuantitiesPostMu ----
    co = S["cqpo"]
    # jaxipm's cqpo.rhs is calc_aug_pd_RHS = -[mod_rhs_x, mod_rhs_s, c, dms,
    # rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U]: NEGATED, with CONDENSED x/s rows
    # (bound-complementarity folded in) but ORIGINAL z/v rows. IPOPT logs the
    # un-negated, UNREDUCED rhs.* plus the condensed rhs_aug.x/.s — so rebuild
    # jaxipm's convention: x/s from rhs_aug (condensed), the rest from rhs.*, all
    # negated. (cqpo.step needs no transform — IPOPT's delta already matches.)
    _rhs_u = _load_iterate(co, "rhs.", cp)
    rhs = Iterate(
        x=-jnp.vstack([_vec(co, "rhs_aug.x", cp.nx), jnp.zeros([aug, 1])]),
        s=-_vec(co, "rhs_aug.s", cp.nyd),
        y_c=-_rhs_u.y_c,
        y_d=-_rhs_u.y_d,
        z_L=-_rhs_u.z_L,
        z_U=-_rhs_u.z_U,
        v_L=-_rhs_u.v_L,
        v_U=-_rhs_u.v_U,
    )
    step = _load_iterate(co, "step.", cp)
    # slack derivatives, derived from the step and the static projection indices
    ind_x_L = jnp.asarray(static.get("ind_x_L", jnp.zeros(0)), dtype=int).reshape(-1)
    ind_x_U = jnp.asarray(static.get("ind_x_U", jnp.zeros(0)), dtype=int).reshape(-1)
    ind_d_L = jnp.asarray(static.get("ind_d_L", jnp.zeros(0)), dtype=int).reshape(-1)
    ind_d_U = jnp.asarray(static.get("ind_d_U", jnp.zeros(0)), dtype=int).reshape(-1)
    dsxL = step.x[ind_x_L] if ind_x_L.size else jnp.zeros([cp.nxL, 1])
    if dsxL.shape[0] == cp.nxL:
        dsxL = jnp.vstack([dsxL, jnp.zeros([aug, 1])])
    slack_derivatives = (
        dsxL,
        -step.x[ind_x_U] if ind_x_U.size else jnp.zeros([cp.nxU, 1]),
        step.s[ind_d_L] if ind_d_L.size else jnp.zeros([cp.ndL, 1]),
        -step.s[ind_d_U] if ind_d_U.size else jnp.zeros([cp.ndU, 1]),
    )
    # SOC-ACCEPTED iterations: jaxipm's post-LS cqpo.step / ls.trial_step hold the
    # ACCEPTED (SOC-corrected) delta — execute_search's final merge writes the
    # direction actually taken — while IPOPT's cqpo.txt step.* is dumped in
    # PDSearchDirCalc BEFORE the line search and keeps the raw direction (the SOC
    # re-solve never re-dumps). Override `step` with the accepted attempt's delta
    # from soc.txt so accepted-vs-accepted is compared; the raw direction remains
    # validated via step_aug (raw on both sides). NOTE: slack_derivatives above
    # are intentionally derived from the RAW step BEFORE this override (jaxipm's
    # are computed in post_process from the raw direction, never re-derived).
    socS = S["soc"]
    _soc_attempts = sorted(int(k[3:k.index(".")]) for k in socS if k.endswith(".attempt")) if socS else []
    _soc_pre = f"soc{_soc_attempts[-1]}." if _soc_attempts else None
    _soc_accepted = False
    if _soc_pre is not None:
        _soc_accepted = int(jnp.asarray(socS[_soc_pre + "accept"]).reshape(-1)[0]) == 1
        if _soc_accepted and (_soc_pre + "delta.x") in socS:
            step = _load_iterate(socS, _soc_pre + "delta.", cp)

    # augmented (condensed) rhs/step: concat the [x, s, y_c, y_d] sections IPOPT
    # now logs from PDFullSpaceSolver. jaxipm's rhs convention is negated vs
    # IPOPT's (jaxipm solves +K x = rhs, IPOPT step = -K^-1 rhs), so rhs_aug is
    # negated; step_aug already matches (the two negations cancel).
    def _aug(prefix, sign):
        return sign * jnp.vstack([
            _vec(co, prefix + "x", cp.nx),
            _vec(co, prefix + "s", cp.nyd),
            _vec(co, prefix + "y_c", cp.nyc),
            _vec(co, prefix + "y_d", cp.nyd),
        ])

    cqpo = CalculatedQuantitiesPostMu(
        rhs_aug=_aug("rhs_aug.", -1.0),
        step_aug=_aug("step_aug.", 1.0),
        rhs=rhs,
        step=step,
        alpha_pr=_sc(co, "alpha_pr"),
        barr=_sc(co, "barr"),
        gBD=_sc(co, "gBD"),
        slack_derivatives=slack_derivatives,
    )

    # ---- wd : WatchdogState ----
    wds = S["wd"]
    wd_it = _load_iterate(wds, "it.", cp) if "it.x" in wds else _empty_iterate(cp)
    wd_delta = _load_iterate(wds, "delta.", cp) if "delta.x" in wds else _empty_iterate(cp)
    wd = WatchdogState(
        shortened_iter=_sc(wds, "shortened_iter"),
        trial_iter=_sc(wds, "trial_iter"),
        alpha_pr_test=_sc(wds, "alpha_pr_test"),
        it=wd_it,
        delta=wd_delta,
        last_mu=_sc(wds, "last_mu", -1.0),
        theta=_sc(wds, "theta"),
        barr=_sc(wds, "barr"),
        gBD=_sc(wds, "gBD"),
    )

    # ---- ls : LineSearchState (+ nested filter) ----
    lss = S["ls"]
    ls_filter = LineSearchFilterState(
        theta_min=_sc(lss, "filter.theta_min", -1.0),
        theta_max=_sc(lss, "filter.theta_max", -1.0),
        last_rejection_due_to_filter=_sc(lss, "filter.last_rejection_due_to_filter"),
        count_successive_filter_rejections=_sc(lss, "filter.count_successive_filter_rejections"),
        n_filter_resets=_sc(lss, "filter.n_filter_resets"),
        F=_filter_pad(lss.get("filter.F", jnp.zeros((0, 2))), fs),
        ref_theta=_sc(lss, "filter.ref_theta"),
        ref_barr=_sc(lss, "filter.ref_barr"),
        ref_gBD=_sc(lss, "filter.ref_gBD"),
    )
    ls_it_trial = _load_iterate(lss, "it_trial.", cp) if "it_trial.x" in lss else _empty_iterate(cp)
    ls_acceptable = _load_iterate(lss, "acceptable_point.", cp) if "acceptable_point.x" in lss else _empty_iterate(cp)

    # ---- soc : per-attempt SOC dump (parsed above, before the cqpo build) ----
    # jaxipm's execute_search now carries the LAST attempt's loop state into the
    # post-LS result (search.py final merge): c_soc/dms_soc (accumulated SOC rhs,
    # same recurrence c_soc = trial_c + alpha*c_soc_prev on both sides),
    # theta_soc_old (attempt-entry trial theta, zero-initialized at LS start like
    # IPOPT's function-local), count_soc (+1 per REJECTED attempt only — matching
    # IPOPT's loop counter), and trial_step == cqpo.step == the direction actually
    # taken (`step` already holds the accepted SOC delta via the override above).
    ls_count_soc = _sc(lss, "count_soc")
    ls_c_soc = _vec(lss, "c_soc", cp.nyc)
    ls_dms_soc = _vec(lss, "dms_soc", cp.nyd)
    ls_theta_soc_old = _sc(lss, "theta_soc_old")
    ls_trial_step = step  # == cqpo.step (accepted delta; raw when no SOC accepted)
    if _soc_pre is not None:
        _last = _soc_attempts[-1]
        ls_count_soc = jnp.array([[_last - 1 if _soc_accepted else _last]])
        ls_c_soc = jnp.asarray(socS[_soc_pre + "c_soc"]).reshape(-1, 1)
        ls_dms_soc = jnp.asarray(socS[_soc_pre + "dms_soc"]).reshape(-1, 1)
        ls_theta_soc_old = _sc(socS, _soc_pre + "theta_soc_old")

    ls = LineSearchState(
        acceptable_point=ls_acceptable,
        n_steps=_sc(lss, "n_steps"),
        accept=_sc(lss, "accept"),
        trial_step=ls_trial_step,
        it_trial=ls_it_trial,
        alpha_pr=_sc(lss, "alpha_pr"),
        alpha_min=_sc(lss, "alpha_min"),
        n_filter_resets=_sc(lss, "filter.n_filter_resets"),
        trial_theta=_sc(lss, "trial_theta"),
        last_obj_val=jnp.atleast_2d(0.0),
        count_soc=ls_count_soc,
        theta_soc_old=ls_theta_soc_old,
        c_soc=ls_c_soc,
        dms_soc=ls_dms_soc,
        soft_resto_phase_counter=_sc(lss, "soft_resto_phase_counter"),
        satisfies_original_criterion=jnp.atleast_2d(0),
        count_restorations=_sc(lss, "count_restorations"),
        required_infeasibility_reduction=jnp.atleast_2d(0.0),
        filter=ls_filter,
    )

    # ---- ic : InertiaCorrectionState ----
    ics = S["ic"]
    inertia = jnp.array([
        float(jnp.asarray(_sc(ics, "inertia_num_neg_evals")).reshape(-1)[0]),
        float(jnp.asarray(_sc(ics, "inertia_expected_neg_evals")).reshape(-1)[0]),
    ])
    ic = InertiaCorrectionState(
        dxs=jnp.asarray(_sc(ics, "dxs")).reshape(()),
        dcd=jnp.asarray(_sc(ics, "dcd")).reshape(()),
        dxs_old=jnp.asarray(_sc(ics, "dxs_old")).reshape(()),
        dcd_old=jnp.asarray(_sc(ics, "dcd_old")).reshape(()),
        jac_degen=jnp.asarray(_sc(ics, "jac_degen")).reshape(()),
        hess_degen=jnp.asarray(_sc(ics, "hess_degen")).reshape(()),
        test_status=jnp.asarray(_sc(ics, "test_status")).reshape(()),
        degen_iters=jnp.asarray(_sc(ics, "degen_iters")).reshape(()),
        inertia=inertia,
        # debug-only in jaxipm; IPOPT doesn't log it -> zeros of the KKT-triu length
        perturbed_data=jnp.zeros(cp.nnz_triu),
    )

    # ---- assemble. Save slots = current state (regular path); resto-entry save
    # slots are an internal/deferred concern (see LOGGING_SCHEMA.md §5). ----
    return OptimizationState(
        it=it, cqpr=cqpr, cqpo=cqpo, fl=fl, wd=wd, ls=ls, ic=ic, adfs=adfs,
        mu=mu, tau=tau, mu_max=mu_max,
        init_dual_inf=init_dual_inf, init_primal_inf=init_primal_inf,
        saved_fl=fl, saved_wd=wd, saved_ls=ls, saved_ic=ic, saved_adfs=adfs,
        saved_mu=mu, saved_tau=tau, saved_mu_max=mu_max,
        saved_init_dual_inf=init_dual_inf, saved_init_primal_inf=init_primal_inf,
        saved_orig_inf_pr=jnp.zeros_like(init_primal_inf),
        saved_z_L=jnp.zeros_like(it.z_L[: cp.nxL]),
        saved_z_U=jnp.zeros_like(it.z_U),
        saved_v_L=jnp.zeros_like(it.v_L),
        saved_v_U=jnp.zeros_like(it.v_U),
        saved_slacks=(
            jnp.zeros_like(it.z_L[: cp.nxL]),
            jnp.zeros_like(it.z_U),
            jnp.zeros_like(it.v_L),
            jnp.zeros_like(it.v_U),
        ),
        resto_tol=jnp.atleast_2d(cp.p["tol"]),
        iter_count=jnp.atleast_2d(iter_num),
        args=args,
    )


def analyze_diff(jaxipm_state, ipopt_state, rtol=1e-10, atol=1e-10, verbose_skips=False,
                 print_diffs=True, exiting_resto=False):
    """Print every leaf where jaxipm's state and the mk3-loaded IPOPT state
    disagree in shape or value. Self-contained — no dependency on the old loader.
    Arg order matches the labels: (jaxipm_state, ipopt_state).

    Real discrepancies (SHAPE / max_abs_diff) always print. The SKIPPED/WARNING
    notes for gated/benign fields are suppressed by default to keep the output
    readable; pass verbose_skips=True to see them.

    exiting_resto: set True when this compare pass is the restoration-EXIT
    boundary (in_restoration 1 -> 0 between this iter and the next; the caller
    must determine this — it is not knowable from the pair alone). Gates the
    it.* family exactly like the entry boundary gates it: the compare pairs
    jaxipm's PRE-exit resto trial iterate with IPOPT's POST-exit iter-(idx+1)
    iterate (resto-slack x padding, LSQ/zero-reset y_c/y_d, bound-multiplier-
    stepped z_L/v_L), and post_process applies those identical exit updates
    right after the compare — validated transitively by the next iter
    comparing clean. Everything else at the exit pass (cqpr/cqpo/ls/wd/mu)
    keeps its real counterpart and stays compared.

    Returns a list of per-leaf records ``(path, status, max_abs_diff, scale,
    isclose_dev)`` with status in {"compared", "skipped", "shape"};
    max_abs_diff is the RAW (un-banded) inf-norm of the difference for
    compared leaves (None otherwise), and scale is the REFERENCE (IPOPT-side)
    magnitude for relative-error use: the leaf's own inf-norm, or IPOPT's
    ||step_aug||_inf for the coupled _STEP_FAMILY solve outputs (the printed
    banding keeps the jaxipm-side step scale; a relative error must never be
    normalized by the system under test). isclose_dev is the ELEMENT-WISE
    numpy.isclose score max_i |a_i - b_i| / (1 + |b_i|), b the IPOPT side —
    the smallest rtol (= atol) at which every element of the leaf would pass
    numpy.isclose(a, b, rtol, atol). This is the single source of comparison
    semantics (which leaves have an IPOPT counterpart, filter-table
    column/sort alignment, inf-safe deltas, banding scales) — offline
    consumers (plot_validation_state_diffs.py's full-state ECDF) call this
    with print_diffs=False instead of re-implementing the gating.
    """
    import jax.tree_util as jtu

    records = []

    def _note(msg):
        if verbose_skips:
            print(msg)

    # Fields with NO IPOPT counterpart at this comparison point (state BETWEEN
    # iterations, post-execute_search) — applies at every iteration, not just
    # iter 0. These are NOT silently dropped: each is printed once as
    # "SKIPPED (<reason>)" so the gap stays visible in the diff output. None is
    # consumed by jaxipm's algorithm at this point, so there is genuinely nothing
    # to validate here.
    _SKIP = {
        "ls.acceptable_point.":
            "IPOPT acceptable_iterate_ is NULL until near convergence (loader "
            "zero-fills); jaxipm parks the initial iterate here, never updates/reads it",
        "ls.it_trial.y_c":
            "IPOPT logs the stepped primal-dual trial; jaxipm's trial steps the "
            "primal only, holding duals at current -> trial duals inert (x/s ARE compared)",
        "ls.it_trial.y_d": "see ls.it_trial.y_c",
        "ls.it_trial.z_L": "see ls.it_trial.y_c",
        "ls.it_trial.z_U": "see ls.it_trial.y_c",
        "ls.it_trial.v_L": "see ls.it_trial.y_c",
        "ls.it_trial.v_U": "see ls.it_trial.y_c",
        "ls.last_obj_val":
            "jaxipm field is dead (its only reader, calc_current_is_acceptable, is "
            "never invoked); IPOPT's analog curr_obj_val_ lives in the ConvCheck, unlogged",
        "ls.required_infeasibility_reduction":
            "jaxipm reads this from params (search.py:106), not the ls field, which "
            "is dead state; IPOPT has no filter-state field for it",
        "ic.inertia":
            "jaxipm field is a debug-only placeholder [0,0], never populated/consumed "
            "(structures.py:321); IPOPT logs real eigenvalue counts. The perturbation "
            "decision that flows from inertia (ic.dxs/ic.dcd) IS compared",
        "ic.perturbed_data":
            "jaxipm holds the perturbed KKT-triu values fed to linear_solve; IPOPT "
            "doesn't log its matrix (loader zero-fills). Transitively validated: ic.dxs/"
            "ic.dcd compared + cqpo.step_aug validated => same operator on solved subspace",
        "ic.jac_degen":
            "inertia-correction degeneracy bookkeeping; jaxipm's linear solver cannot "
            "report zero-eigenvalue inertia (assumes zero), so this flag differs from "
            "IPOPT's by design. The final inertia correction (ic.dxs/ic.dcd) always agrees",
        "ic.test_status": "see ic.jac_degen (inertia-correction bookkeeping differs by design)",
        # Line-search TRIAL bookkeeping: jaxipm records the full-step trial, IPOPT the
        # accepted one. The ACCEPTED step (ls.alpha_pr) and iterate (it.*) are NOT
        # skipped and DO match — these three only differ when backtracking occurs:
        "cqpo.alpha_pr":
            "jaxipm's cqpo.alpha_pr is the initial/full step (fraction-to-boundary "
            "alpha_max, the BT start point); IPOPT logs the ACCEPTED alpha_primal here. "
            "The accepted alpha is validated via ls.alpha_pr (which matches)",
        "ls.n_steps":
            "counting convention differs: jaxipm's (n_steps>0) guard (search.py:327) "
            "evaluates the full step twice before reducing, so its count is +1 vs IPOPT's "
            "#rejections. Same accepted step (ls.alpha_pr matches) => same effective backtrack",
        "ls.trial_theta":
            "jaxipm stores theta at the full-step trial (alpha_max); IPOPT at the accepted "
            "trial. Nearby points -> close-but-unequal. Accepted outcome validated via it.*",
        "init_dual_inf":
            "adaptive-mu safeguard is off (adaptive_mu_safeguard_factor=0, the default): "
            "IPOPT never computes init_dual_inf_ (lower_mu_safeguard early-returns), and "
            "jaxipm multiplies it by 0 in calc_lower_mu_safeguard -> dead on both sides",
        "init_primal_inf": "see init_dual_inf (same factor=0 dead-state reasoning)",
    }
    # c_soc/dms_soc hold live values ONLY during an active SOC loop. IPOPT now logs
    # every SOC attempt to soc.txt (TrySecondOrderCorrection), and the loader fills
    # ls.c_soc/dms_soc/count_soc/trial_step from the LAST attempt — so on SOC iters
    # these compare element-wise like everything else. On non-SOC iters (no soc.txt)
    # the loader zero-fills while jaxipm's slots hold the curr_c/curr_dms seed
    # (already validated via cqpr.c/cqpr.dms) -> SKIP. Detect via the IPOPT side:
    # a nonzero loaded c_soc means soc.txt was present.
    _SOC = ("ls.c_soc", "ls.dms_soc")

    # The whole wd (WatchdogState) substruct is lazily maintained: IPOPT only
    # populates it once the watchdog activates; while dormant its members hold
    # uninitialized memory (note the denormal floats like 4.9e-322 in the IPOPT
    # column). jaxipm maintains wd eagerly. Gate on jaxipm's fl.in_watchdog flag:
    # SKIP the entire substruct (one line) while the watchdog is OFF, compare
    # leaf-by-leaf once it is ON.
    _WD = "wd."

    def _flag(getter):
        try:
            return int(jnp.asarray(getter()).squeeze())
        except Exception:
            return 0

    in_watchdog = _flag(lambda: jaxipm_state.fl.in_watchdog)
    in_restoration = _flag(lambda: jaxipm_state.fl.in_restoration)
    # SOC ran on the IPOPT side iff the loader filled c_soc from soc.txt (nonzero);
    # count_soc alone cannot detect a first-attempt-accepted SOC (it stays 0).
    try:
        ipopt_soc_ran = bool(float(jnp.max(jnp.abs(jnp.asarray(ipopt_state.ls.c_soc)))) > 0.0)
    except Exception:
        ipopt_soc_ran = False

    # Restoration ENTRY iteration. jaxipm raises fl.fallback_activated on the single
    # iter where it bails the regular line search into restoration. On that iter IPOPT
    # runs PerformRestoration() *inline* (IpBacktrackingLineSearch.cpp:611), which:
    #   (a) bumps iter_count for the whole restoration sub-solve, so the fl/state/ls-
    #       result logs for THIS main iter land at a later iter number -> iter_<n>/
    #       {fl,state}.txt are absent and ls.txt holds only filter.* (the line-search
    #       result block at line 706 never fires here). The loader zero-fills these
    #       (fallback_activated->0, mu_max->1e10, adfs->empty/inf, alpha/it_trial->0).
    #   (b) re-initializes the iterate (multipliers->0, bound mults->bound_mult_init,
    #       filter cleared). jaxipm instead DETECTS here and applies the reset on the
    #       NEXT execute_search, so result.it is still pre-reset -> a one-sub-step
    #       offset vs IPOPT's already-reset iter_<idx+1>.it.
    # So on the entry iter these transition-bookkeeping fields have no aligned IPOPT
    # counterpart. cqpr/cqpo (the search direction) ARE logged and stay compared;
    # full leaf-by-leaf comparison resumes at the first restoration sub-iter, where
    # IPOPT logs in_restoration=1 completely. Gate on jaxipm fl.fallback_activated.
    entering_resto = _flag(lambda: jaxipm_state.fl.fallback_activated)
    _RESTO_ENTRY = (
        "fl.fallback_activated",  # IPOPT logs it normally (line 689) but fl.txt absent this iter
        "fl.needs_resto_init",    # jaxipm-internal; IPOPT never logs it (loader hardcodes 0)
        "mu_max",                 # state.txt absent -> loader default 1e10
        "adfs",                   # state.txt absent -> empty filter -> inf pad
        "ls.alpha_pr",            # ls.txt holds only filter.* this iter
        "ls.alpha_min",
        "ls.it_trial",            # covers it_trial.x / .s (.y_c etc. already in _SKIP)
        "it.",                    # iterate offset: result.it pre-reset vs IPOPT post-reset
    )

    # The whole saved_* family (saved_fl/wd/ls/ic/adfs/mu/.../z_L/v_L/slacks) only
    # holds meaningful values during restoration: jaxipm freezes the pre-restoration
    # state into these slots on entry and reads them back on exit. Outside
    # restoration they hold dead/init values, and IPOPT has no saved-state log (the
    # loader just mirrors current state into saved_*). Gate on fl.in_restoration:
    # SKIP the whole family (one line) while OFF, compare leaf-by-leaf once ON.
    _SAVED = "saved_"

    # args[0][0..2] = the restoration objective's reference args, frozen at restoration
    # entry: [0] = the restoration reference mu rmu = max(orig_mu, ||c||_inf, ||dms||_inf)
    # (initialization.py:1673-1681, mirrors IPOPT's IpRestoIterateInitializer.cpp:58
    # resto_mu); [1],[2] = x_ref, dr_x (proximity reference, consumed by new_f_resto).
    # These are dummy zeros outside restoration. The live function-scaling factors
    # (args[0][3]=df, args[1][0]=dc, args[2][0]=dd) are NOT gated. Gate on
    # fl.in_restoration: SKIP while OFF, compare once ON.
    # args[0][0] is the FROZEN rmu, distinct from the current iter mu; the loader reads
    # it from _static/resto_ref_mu.txt (logged by IPOPT's RestoIterateInitializer) when
    # in restoration, so it compares correctly. args[0][1..2] (x_ref, dr_x) are likewise
    # frozen at entry; the loader now reads them from _static/resto_ref_x.txt (logged by
    # IPOPT's RestoIpoptNLP::InitializeStructures) when in restoration, so they too
    # compare correctly throughout the phase (previously reconstructed from the current
    # iterate, which drifted at later resto iters).
    _RESTO_ARGS = ("args.[0].[0]", "args.[0].[1]", "args.[0].[2]")

    # adfs / ls.filter.F are (filter_size, 3) = [iter_index, theta, phi]. Column 0
    # is a per-entry iteration index that IPOPT does not log (L.pairs emits only the
    # (phi, theta) pair; the loader fabricates arange for col 0, with a different
    # empty-slot convention than jaxipm), so compare only the two value columns.
    _FILTER = ("adfs", "filter.F")

    # Fields IPOPT computes only in certain modes, so its cqpr.txt logs them only on
    # those iters (absent -> loader zero-fills). jaxipm computes them every iter.
    # Compare only where IPOPT logged them (detected by a non-all-zero IPOPT side);
    # SKIP otherwise. Keyed by path -> reason:
    #   y_c_init/y_d_init  least-squares multiplier estimate; IPOPT computes it only
    #     at instantiation + resto/regular re-init (CalculateMultipliers). jaxipm
    #     recomputes every iter (quantities.py:1137) but only CONSUMES it on re-init
    #     (search.py:1860) -> recomputed-but-unused on a non-re-init iter.
    #   step_aff_full/step_cen_full  affine + centrality predictor steps; computed
    #     only by the adaptive-mu oracle, so IPOPT omits them in fixed/monotone-mu
    #     mode (free_mu_mode=0). jaxipm computes them every iter but the mu update
    #     only uses them in free-mu mode.
    _SKIP_IF_ABSENT = {
        "cqpr.y_c_init": "LSQ multiplier estimate; IPOPT logs it only at instantiation/re-init",
        "cqpr.y_d_init": "LSQ multiplier estimate; IPOPT logs it only at instantiation/re-init",
        "cqpr.step_aff_full": "affine predictor step; IPOPT logs it only in free/adaptive-mu mode (free_mu_mode=1)",
        "cqpr.step_cen_full": "centrality predictor step; IPOPT logs it only in free/adaptive-mu mode (free_mu_mode=1)",
    }

    # Same "absent -> zero-filled" idea as _SKIP_IF_ABSENT, but matched EXACTLY (not
    # by substring) so single-token top-level paths don't swallow siblings like
    # mu_max / free_mu_mode / last_mu / saved_mu. IPOPT's state.txt mu/tau are
    # written ONLY by the QualityFunctionMuOracle (IpQualityFunctionMuOracle.cpp:499),
    # i.e. only on free/adaptive-mu iters. In fixed/monotone-mu mode (free_mu_mode=0)
    # the oracle never runs, so neither @mu nor @tau is logged and the loader fills 0
    # (mu_max/adfs come from a different, every-iter site and stay compared). jaxipm
    # always carries a mu/tau, so compare only where IPOPT logged them (IPOPT side != 0).
    _ABSENT_EXACT = {
        "mu": "barrier parameter; IPOPT's state.txt mu is logged only by the free-mu oracle (free_mu_mode=1)",
        "tau": "fraction-to-boundary tau; IPOPT's state.txt tau is logged only by the free-mu oracle (free_mu_mode=1)",
    }

    # Outputs of the single augmented-system solve (quantities.py:1299) viewed in
    # different bases: the primal-dual step and everything transformed from it.
    # A direct solve returns x at ~1e-11 RELATIVE RESIDUAL, but the component-wise
    # forward error is ~cond(A)*eps*||x||_inf, set by the LARGEST block (here y_c ~
    # 1e7), not by each element's own size. So the element-wise tol = atol + rtol*|b|
    # wrongly flags the small blocks (y_d, v_L) whose admissible error is governed by
    # ||step||, not by their tiny magnitude. Band these by ||step_aug||_inf instead:
    # conditioning noise clears, while a real O(||step||) divergence still trips.
    # step_aff_full/step_cen_full are predictor SOLVE outputs: like the main step,
    # their forward error is set by the solve's large intermediates, not their own
    # (tiny) magnitude — so a per-leaf norm under-bands them. Band by ||step_aug||_inf.
    _STEP_FAMILY = (
        "cqpo.step", "slack_derivatives", "trial_step",
        "step_aff_full", "step_cen_full",
    )

    # The d-slack bound-multiplier STEP blocks (v_L/v_U) are not solved directly: they
    # are recovered from the primal step via complementarity, Δv_L ≈ μ/s_L − v_L −
    # (V_L/S_L)·Δs_L. The μ/s_L division ties their forward-error floor to the recovered
    # magnitude (a ~1e-7 relative wobble in the slack s_L propagates straight through),
    # which the primal-dual step_scale band doesn't see. Band these by a per-leaf
    # RELATIVE tolerance instead — it scales with |Δv_L| (so it clears the recovery
    # floor) while a genuine O(magnitude) divergence still trips. z_L/z_U (x-bound mults)
    # are NOT here: their recovery isn't μ/s_L-amplified and they pass the step_scale band.
    _RECOVERED_MULT = ("step.v_L", "step.v_U")
    _RECOVERED_MULT_RTOL = 1e-6  # vs the ~2e-7 relative recovery floor -> ~5x margin

    # During restoration jaxipm solves a CONDENSED (Schur-reduced) restoration KKT,
    # while IPOPT logs the FULL augmented restoration system (the p_c/n_c/p_d/n_d
    # slack blocks explicit). These fields therefore have no element-wise counterpart
    # in jaxipm's reduced representation -> SKIP while in_restoration=1. What IS still
    # validated: cqpo.step (the step in the ORIGINAL variables, matched), the iterate
    # it.*, grad_lag_x/s, and every scalar (f/theta/nlp_error/primal_inf/...). Members:
    #   cqpr.jac_f   loader recomputes the REGULAR objective gradient (cp.nstqf); the
    #                restoration objective (rho*sum(p+n)+zeta/2||D_R(x-x_ref)||^2) differs.
    #   cqpr.Sigma   Sigma_{nc,pc,nd,pd}_inv: scalings for the resto slacks; IPOPT logs
    #                no Sigma keys at all -> loader default 0 vs jaxipm's real values.
    #   cqpo.rhs     full (IPOPT rhs.x=nx+resto, condensation-modified y_c/y_d rows) vs
    #                jaxipm's reduced rhs -> shape + value mismatch (covers rhs_aug too).
    #   cqpo.step_aug   reduced aug step (nx) vs full (nx+resto) -> shape mismatch.
    #   cqpo.slack_derivatives  reconstructed from regular bound-indices (ind_x_L/...),
    #                which select the wrong entries once the bounds are on resto slacks.
    # NOTE: cqpo.step_aug is NOT here — the harness expands jaxipm's reduced step to
    # the full augmented restoration layout (calc_transform_red_to_aug) before this
    # runs, so it compares element-wise (banded by _STEP_FAMILY) instead of skipping.
    _RESTO_REDUCED = (
        "cqpr.jac_f", "cqpr.Sigma",
        "cqpo.rhs", "cqpo.slack_derivatives",
    )

    # Natural scale of the augmented solve: max |block| over the full step vector
    # (dominated by the largest block). Used to band the _STEP_FAMILY fields.
    step_scale = float(jnp.max(jnp.nan_to_num(jnp.abs(jaxipm_state.cqpo.step_aug))))
    # REFERENCE-side step scale, recorded for the _STEP_FAMILY leaves (used by
    # the offline scaled-deviation ECDF): relative error must be measured
    # against the reference (IPOPT), never the system under test — a jaxipm
    # step blowup dividing by its own norm would self-normalize to ~1 instead
    # of reading huge. Printing/banding above keeps the jaxipm-side scale so
    # harness output is unchanged.
    step_scale_ref = float(jnp.max(jnp.nan_to_num(jnp.abs(ipopt_state.cqpo.step_aug))))

    seen = set()
    for (path, a), (_, b) in zip(
        jtu.tree_leaves_with_path(jaxipm_state), jtu.tree_leaves_with_path(ipopt_state)
    ):
        if not hasattr(a, "shape"):
            continue
        # collapse repeated dots from GetAttrKey's leading-dot str() so the
        # patterns match regardless of path formatting.
        name = ".".join(str(k) for k in path)
        norm = name.replace("..", ".").strip(".")

        skip_key = next((k for k in _SKIP if k in norm), None)
        if skip_key is not None:
            records.append((norm, "skipped", None, None, None))
            if skip_key not in seen:  # one line per field, not per leaf
                seen.add(skip_key)
                _note(f"{skip_key}: SKIPPED — {_SKIP[skip_key]}")
            continue

        soc_key = next((k for k in _SOC if k in norm), None)
        if soc_key is not None and not ipopt_soc_ran:
            # No soc.txt this iter -> IPOPT side zero-filled while jaxipm holds the
            # curr_c/curr_dms seed; nothing to validate. When SOC ran, fall through
            # to the normal element-wise comparison against the logged attempt.
            records.append((norm, "skipped", None, None, None))
            if soc_key not in seen:
                seen.add(soc_key)
                _note(
                    f"{soc_key}: SKIPPED — no SOC this iter (no soc.txt); "
                    "seed == curr_c/curr_dms, validated via cqpr.c/cqpr.dms"
                )
            continue

        if norm.startswith(_WD) and in_watchdog == 0:
            records.append((norm, "skipped", None, None, None))
            if _WD not in seen:  # one line for the whole dormant substruct
                seen.add(_WD)
                _note(
                    "wd.*: SKIPPED — watchdog inactive (in_watchdog=0); IPOPT's "
                    "WatchdogState is dormant/uninitialized until the watchdog "
                    "activates. Compared leaf-by-leaf once in_watchdog=1."
                )
            continue

        if norm.startswith(_SAVED):
            records.append((norm, "skipped", None, None, None))
            if _SAVED not in seen:  # one line for the whole save-slot family
                seen.add(_SAVED)
                _note(
                    "saved_*: SKIPPED (always) — jaxipm's save slots freeze the "
                    "pre-restoration state on entry and read it back on exit; this is "
                    "internal restoration bookkeeping. IPOPT runs a SEPARATE restoration "
                    "NLP and never logs a saved state, so the loader can only mirror the "
                    "current iterate (+ zeros) into saved_* — there is no real counterpart "
                    "to compare against, in or out of restoration. (The freeze is validated "
                    "indirectly: the RESTORED live state on exit is compared via it.*/fl.*.)"
                )
            continue

        if any(k in norm for k in _RESTO_ARGS) and in_restoration == 0:
            records.append((norm, "skipped", None, None, None))
            if "_RESTO_ARGS" not in seen:  # one line for the three reference args
                seen.add("_RESTO_ARGS")
                _note(
                    "args[0][0..2]: SKIPPED — restoration inactive (in_restoration=0); "
                    "the restoration objective's reference args (rmu, x_ref, dr_x) are "
                    "dummy zeros outside restoration. Function-scaling args (df/dc/dd) "
                    "stay compared. Compared once in_restoration=1."
                )
            continue

        if in_restoration:
            rr_key = next((k for k in _RESTO_REDUCED if k in norm), None)
            if rr_key is not None:
                records.append((norm, "skipped", None, None, None))
                if ("_RESTO_REDUCED", rr_key) not in seen:  # one line per family
                    seen.add(("_RESTO_REDUCED", rr_key))
                    _note(
                        f"{rr_key}*: SKIPPED — restoration: jaxipm solves a condensed "
                        "(Schur-reduced) restoration KKT while IPOPT logs the full "
                        "augmented system, so this field has no element-wise counterpart "
                        "(reduced-vs-full shape/condensation, or a resto-slack scaling "
                        "IPOPT never logs). cqpo.step / it.* / grad_lag / scalars stay "
                        "compared."
                    )
                continue

        absent_key = next((k for k in _SKIP_IF_ABSENT if k in norm), None)
        if absent_key is not None and not bool(jnp.any(b != 0)):
            records.append((norm, "skipped", None, None, None))
            if absent_key not in seen:  # one line per field
                seen.add(absent_key)
                _note(
                    f"{absent_key}: SKIPPED — {_SKIP_IF_ABSENT[absent_key]} "
                    "(absent here -> zero-filled); jaxipm computes it every iter. "
                    "Compared at iters where IPOPT logs it."
                )
            continue

        if norm in _ABSENT_EXACT and not bool(jnp.any(b != 0)):
            records.append((norm, "skipped", None, None, None))
            if norm not in seen:  # one line per field
                seen.add(norm)
                _note(
                    f"{norm}: SKIPPED — {_ABSENT_EXACT[norm]} "
                    "(absent here -> zero-filled); jaxipm always carries it. "
                    "Compared at iters where IPOPT logs it."
                )
            continue

        # Restoration EXIT boundary (caller-flagged): the it.* family compares
        # jaxipm's pre-exit resto trial against IPOPT's post-exit iterate — a
        # pipeline-stage offset, not a deviation (mirror of the entry gate
        # below). post_process applies the same exit updates right after this
        # compare; validated transitively by the next iter comparing clean.
        if exiting_resto and norm.startswith("it."):
            records.append((norm, "skipped", None, None, None))
            if "_RESTO_EXIT" not in seen:
                seen.add("_RESTO_EXIT")
                _note(
                    "it.*: SKIPPED — restoration-EXIT boundary: compare sees the "
                    "pre-exit resto trial while IPOPT's iter idx+1 iterate is "
                    "post-exit (x-pad resto slacks, LSQ/zero-reset y_c/y_d, "
                    "bound-mult-stepped z_L/v_L). post_process applies the same "
                    "updates right after this compare; the exit-updated state is "
                    "validated at the next iter. cqpr/cqpo/ls/wd/mu stay compared."
                )
            continue

        if entering_resto:
            re_key = next(
                (k for k in _RESTO_ENTRY if norm == k or norm.startswith(k)), None
            )
            if re_key is not None:
                records.append((norm, "skipped", None, None, None))
                marker = ("_RESTO_ENTRY", re_key)
                if marker not in seen:  # one line per affected field/family
                    seen.add(marker)
                    _note(
                        f"{re_key}*: SKIPPED — restoration-entry iter "
                        "(jaxipm fl.fallback_activated=1); IPOPT ran PerformRestoration "
                        "inline so this iter's fl/state/ls-result logs are absent "
                        "(loader defaults) and the iterate is offset one sub-step by the "
                        "resto re-init. cqpr/cqpo stay compared; full comparison resumes "
                        "at the first restoration sub-iter (in_restoration=1)."
                    )
                continue

        # filter tables: drop col 0 (iter-index, not logged by IPOPT)
        if any(k in norm for k in _FILTER) and a.ndim == 2 and a.shape[-1] == 3:
            a = a[:, 1:]
            b = b[:, 1:]
            # The filter is an UNORDERED SET of (theta, phi) constraints, not an ordered
            # list: is_acceptable_to_current_filter does jnp.all over every row, and
            # eviction is by domination -- row position carries no meaning. But the two
            # sides REACH a given set via different row layouts: on eviction IPOPT's
            # GetEntries() COMPACTS (drops the dominated entry, shifts the rest up, appends
            # the new one at the end) while jaxipm's augment_raw_filter CLEARS the dominated
            # slot in place to inf and writes the new entry into that first empty slot. Same
            # multiset, permuted rows -> a positional diff (e.g. adfs iter 27: identical 8
            # entries, max_abs_diff 327.8 purely from one swapped pair). Sort both sides by
            # (phi, theta) so row order stops masking equality; a genuine multiset
            # difference (a wrongly-computed or missing entry) still trips the diff below.
            # inf pads sort to the end on both sides, so padding stays aligned.
            a = a[jnp.lexsort((a[:, 0], a[:, 1]))]
            b = b[jnp.lexsort((b[:, 0], b[:, 1]))]

        if a.shape != b.shape:
            records.append((norm, "shape", None, None, None))
            if print_diffs:
                print(f"{name}: SHAPE jaxipm {a.shape} vs ipopt {b.shape}")
            continue
        # inf-safe diff: positions where both sides are exactly equal contribute 0.
        # This covers the +/-inf sentinels padding the filter tables (adfs, filter.F):
        # a plain a-b there gives inf-inf = nan, which masks the real max. Mismatched
        # inf -> inf (flagged); a genuine nan on either side -> not-equal -> flagged.
        delta = jnp.where(a == b, 0.0, jnp.abs(a - b))
        dmax = float(jnp.max(delta)) if delta.size else 0.0
        # Element-wise numpy.isclose score, max over the leaf:
        #   max_i |a_i - b_i| / (1 + |b_i|),  b = IPOPT (reference) side
        # = the smallest rtol (= atol) at which EVERY element passes
        # numpy.isclose's  |a-b| <= atol + rtol*|b|  band. Non-finite
        # reference entries are masked out of the denominator: a matched inf
        # pad has delta 0 -> contributes 0; a mismatched one keeps delta inf
        # -> flags. (delta==0 short-circuit also keeps bit-identical entries
        # exactly 0 regardless of magnitude.)
        if delta.size:
            e_denom = 1.0 + jnp.where(jnp.isfinite(b), jnp.abs(b), 0.0)
            emax = float(jnp.max(jnp.where(delta == 0, 0.0, delta / e_denom)))
        else:
            emax = 0.0
        if any(k in norm for k in _RECOVERED_MULT):
            # μ/s_L-recovered bound-mult step (see _RECOVERED_MULT note): per-leaf
            # RELATIVE band, since the floor scales with the recovered magnitude.
            leaf = jnp.max(jnp.where(jnp.isfinite(b), jnp.abs(b), 0.0)) if b.size else 0.0
            records.append((norm, "compared", dmax, float(leaf), emax))
            tol = atol + _RECOVERED_MULT_RTOL * leaf
            if print_diffs and not bool(jnp.all(delta <= tol)):
                d = jnp.max(delta)
                print(f"{name}: max_abs_diff {d:.3e}  (jaxipm {a.flatten()[:3]} | ipopt {b.flatten()[:3]})")
            continue
        if any(k in norm for k in _STEP_FAMILY):
            # coupled augmented-solve output: band by ||step_aug||_inf (cross-leaf:
            # a small block's error is driven by the largest block in the SAME solve,
            # which lives in a different leaf -> a per-leaf norm wouldn't see it).
            scale = step_scale
            rec_scale = step_scale_ref  # reference-side for the records
        else:
            # per-leaf forward-error band: the error in a computed vector is ~
            # eps*||vector||_inf, so judge each leaf by its OWN infinity norm rather
            # than each element's magnitude. This clears conditioning/cancellation
            # noise in the small elements of a leaf that also holds a large element
            # (e.g. it.y_c's 0.25 entry beside a 2.6e3 entry, or grad_lag_x / the
            # affine+centrality steps), while a genuine O(magnitude) divergence in
            # any element still trips. Inf sentinels (filter pads) contribute 0.
            # Empty leaf (size-0 block, e.g. z_U when nxU=0) -> scale 0; delta is
            # also empty so the jnp.all check below is vacuously True.
            scale = (
                jnp.max(jnp.where(jnp.isfinite(b), jnp.abs(b), 0.0))
                if b.size else 0.0
            )
            rec_scale = scale  # already the reference (IPOPT) side
        records.append((norm, "compared", dmax, float(rec_scale), emax))
        tol = atol + rtol * scale
        if print_diffs and not bool(jnp.all(delta <= tol)):
            d = jnp.max(delta)
            print(f"{name}: max_abs_diff {d:.3e}  (jaxipm {a.flatten()[:3]} | ipopt {b.flatten()[:3]})")

    return records


if __name__ == "__main__":
    import datetime
    import json
    import os
    import pickle
    import subprocess

    import yaml

    with open("/home/john/code/jaxipm/src/jaxipm/params.yaml", "r") as fh:
        p = yaml.safe_load(fh)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(p.get("gpu_id", 0))

    import jax
    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import numpy as np

    from jaxipm.initialization import initialize_common_problem, initialize_problem_regular
    from jaxipm.search import execute_search, post_process
    from jaxipm.solver import TerminationCode
    from problems.redundant.quadcopter_nmpc_nav.jaxipm_quad_nav import quadcopter_nav

    f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux = quadcopter_nav()
    f_args, c_args, d_args = (), (), ()
    _x0_flat = x0.squeeze()
    calc_next_problem = lambda key, sol: (_x0_flat, (), (), ())

    cp = initialize_common_problem(
        f, c, d, x_L, x_U, d_L, d_U, x0, p, [f_args, c_args, d_args],
        calc_next_problem=calc_next_problem,
    )
    state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
    state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[0]]))

    # ======================================================================= #
    # Regenerate the IPOPT logs (./ipopt_logs/iter_<n>/) by solving the SAME
    # problem with the patched cyipopt. Comment this block out to reuse the
    # logs already on disk.
    # ======================================================================= #
    # import numpy as np
    # from cyipopt import minimize_ipopt
    # from jaxipm.utils.sif_adapter import custom_to_cyipopt_format

    # obj, obj_grad, obj_hess, constraints, bounds = custom_to_cyipopt_format(
    #     f, c, d, x_L, x_U, d_L, d_U, x0
    # )
    # minimize_ipopt(
    #     fun=obj,
    #     x0=np.asarray(x0.squeeze(), dtype=np.float64),
    #     jac=obj_grad,
    #     hess=obj_hess,
    #     constraints=constraints,
    #     bounds=bounds,
    #     options={"tol": 1e-8, "max_iter": 500, "mu_strategy": "adaptive", "print_level": 5},
    # )
    # ======================================================================= #

    # ======================================================================= #
    # OPTIONAL E2E FREE RUN (no per-step injection). jaxipm propagates its OWN
    # state from the shared iter-0 start; at each iteration the result is
    # compared against IPOPT's logged state with the SAME comparison model and
    # gating as the injection loop (analyze_diff: bookkeeping from iter idx,
    # iterate from iter idx+1, entry gate via fl.fallback_activated, exit gate
    # via exiting_resto from the IPOPT log timeline). Unlike the injection
    # loop, single-step errors ACCUMULATE here — this validates the whole
    # pipeline end-to-end: expect machine-band agreement early, slow drift
    # through restoration (~eps*rho reduced-form floor), and possible
    # divergence at the ill-conditioned tail (cuDSS transients).
    #
    # States are pure pytrees, so this does NOT touch the injection loop's
    # `state` — both start from the identical iter-0 point. Comment the call
    # in/out as needed; archives land under data/validation_runs/freerun_*
    # (the "run_" glob of the ECDF script ignores them by design).
    # ======================================================================= #
    def free_run_compare(state0, max_iter=500, save_states_free=True):
        run_dir_f = os.path.join(
            STATES_ROOT, datetime.datetime.now().strftime("freerun_%Y%m%d_%H%M%S"))
        state_f = state0
        kdx = 0
        term_f = jnp.array([[0]])
        while kdx < max_iter:
            print(f"\n===== FREE-RUN compare iter {kdx} =====")
            result_f = _execute_search(state_f, cp)
            result_f_cmp = result_f
            if int(jnp.asarray(result_f.fl.in_restoration).squeeze()) == 1:
                _saf = cp.nstqfr.kkt.calc_transform_red_to_aug(
                    result_f.cqpo.step_aug, result_f.cqpo.rhs,
                    result_f.cqpr.Sigma_nc_inv, result_f.cqpr.Sigma_pc_inv,
                    result_f.cqpr.Sigma_nd_inv, result_f.cqpr.Sigma_pd_inv,
                )
                result_f_cmp = eqx.tree_at(lambda t: t.cqpo.step_aug, result_f, _saf)
            if not os.path.isdir(os.path.join(save_dir, f"iter_{kdx + 1}")):
                print(f"  (no IPOPT iter_{kdx + 1} dump — IPOPT log ends; "
                      "comparison stops here)")
                break
            ip_state_f = load_state(kdx, cp)
            ip_next_f = load_state(kdx + 1, cp)
            ip_cmp_f = eqx.tree_at(
                lambda t: (
                    t.it,
                    t.ls.filter.count_successive_filter_rejections,
                    t.ls.filter.last_rejection_due_to_filter,
                ),
                ip_state_f,
                (
                    ip_next_f.it,
                    ip_next_f.ls.filter.count_successive_filter_rejections,
                    ip_next_f.ls.filter.last_rejection_due_to_filter,
                ),
            )
            _exiting_f = (
                int(jnp.asarray(ip_state_f.fl.in_restoration).squeeze()) == 1
                and int(jnp.asarray(ip_next_f.fl.in_restoration).squeeze()) == 0
            )
            analyze_diff(result_f_cmp, ip_cmp_f, verbose_skips=VERBOSE_SKIPS,
                         exiting_resto=_exiting_f)
            if save_states_free:
                _save_state_pair(run_dir_f, kdx, (result_f_cmp, ip_cmp_f))
            # natural (non-injected) advance — solver.py's loop verbatim
            state_f, term_f = _post_process(state_f, result_f, cp)
            kdx += 1
            if int(jnp.asarray(term_f).squeeze()) != 0:
                print(f"\nFREE-RUN: jaxipm terminated at iter {kdx} with "
                      f"TerminationCode={int(jnp.asarray(term_f).squeeze())} "
                      "(1=CONVERGED 2=MAX_ITER 3=TINY_STEP 4=RESTO_FAILURE "
                      "5=ACCEPTABLE_POINT)")
                break
            if kdx == 40:
                pass
        n_ipopt = 0
        while os.path.isdir(os.path.join(save_dir, f"iter_{n_ipopt}")):
            n_ipopt += 1
        print(f"\nFREE-RUN summary: jaxipm ran {kdx} iters "
              f"(term={int(jnp.asarray(term_f).squeeze())}); "
              f"IPOPT logged {n_ipopt} iters.")
        return state_f, term_f
    # NOTE: free_run_compare is CALLED below, after the loop preamble — it
    # needs _execute_search/_post_process/VERBOSE_SKIPS/_save_state_pair,
    # which are defined there (shared with the injection loop).

    cont = 1
    # idx MUST start at 0: `state` is the iter-0 starting point, and the
    # comparison model (result.it == IPOPT iter idx+1; cqpr/cqpo/ls/wd == IPOPT
    # iter idx) only holds when idx equals the iteration `state` represents.
    # Starting at idx=1 shifts every leaf by one iteration (jaxipm iter-k vs IPOPT
    # iter-(k+1)). Each loop pass validates one iteration in order; iter 1 lands on
    # the 2nd pass once post_process has advanced `state` to the iter-1 point.
    idx = 0
    # Tracks the last REGULAR iter before restoration began (the iter whose state
    # IPOPT freezes on entry and restores on exit). Set when entering_resto fires;
    # used by the exit-boundary diagnostic to validate jaxipm's own entry-freeze.
    resto_entry_idx = None
    # Set True to also print the SKIPPED/WARNING notes for gated/benign fields
    # (verbose). Default False -> only real discrepancies (SHAPE / max_abs_diff).
    VERBOSE_SKIPS = False
    _execute_search = eqx.filter_jit(execute_search)
    _post_process = eqx.filter_jit(post_process)

    # ---- state-pair archiving (consumed by problems/correctness_test/
    # plot_validation_state_diffs.py) ----
    # Every compare pass serializes the COMPLETE (result_cmp, ipopt_cmp) pair —
    # the exact pytrees analyze_diff walks — with eqx.tree_serialise_leaves, so
    # the full structure (it/cqpr/cqpo/ls/wd/fl/ic/saved_*/mu/...) survives and
    # the jaxipm-vs-IPOPT difference of ANY leaf (e.g. a linear-solve
    # discrepancy in cqpo.step_aug) can be recomputed offline later. Each run
    # gets a fresh timestamped dir under data/validation_runs so runs
    # ACCUMULATE — nondeterministic events (the iter-86 cuDSS transient) need a
    # population of runs, not a single trace. A per-iter pickled skeleton
    # (zeroed numpy leaves, structure preserved) is saved alongside because
    # tree_deserialise_leaves needs a like-tree and leaf shapes vary between
    # regular and restoration iters (1128 vs 2140 step_aug expansion).
    SAVE_STATES = True
    STATES_ROOT = "/home/john/code/jaxipm/data/validation_runs"
    _run_dir = os.path.join(
        STATES_ROOT, datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))

    def _save_state_pair(run_dir, k, pair):
        os.makedirs(run_dir, exist_ok=True)
        meta_path = os.path.join(run_dir, "meta.json")
        if not os.path.exists(meta_path):
            try:
                _rev = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                ).stdout.strip() or "unknown"
            except Exception:
                _rev = "unknown"
            with open(meta_path, "w") as _fh:
                json.dump({
                    "ipopt_logs": os.path.abspath(save_dir),
                    "git_rev": _rev,
                    "ir_nsteps": p.get("ir_nsteps"),
                    "gpu_id": p.get("gpu_id"),
                    "pair_order": ["jaxipm(result_cmp)", "ipopt(ipopt_cmp)"],
                }, _fh, indent=1)
        eqx.tree_serialise_leaves(
            os.path.join(run_dir, f"iter_{k:03d}.eqx"), pair)
        _skel = jax.tree_util.tree_map(
            lambda l: np.zeros(l.shape, l.dtype) if eqx.is_array(l) else l,
            pair)
        with open(os.path.join(run_dir, f"iter_{k:03d}.skel.pkl"), "wb") as _fh:
            pickle.dump(_skel, _fh)

    def _conform(loaded, template):
        """Cast every array leaf of `loaded` to the (dtype, shape) of the matching
        leaf in `template`, leaving the pytree structure untouched.

        This is what makes a freshly-loaded IPOPT state injectable WHOLESALE into the
        jitted post_process. post_process's init-branch selects (filter_select ->
        jax.lax.select, search.py:1434) trace BOTH branches regardless of the
        condition, so the loaded (it, ic, mu, args) must share jaxipm's dtypes or the
        trace dies with `lax.select requires arguments to have the same dtypes`. The
        loader, building from text dumps, lands ints where jaxipm holds float64 (and
        vice versa); conforming against a genuine jaxipm state (`result`) fixes ALL of
        them structurally in one pass instead of patching field-by-field. reshape()
        (not just astype) also absorbs benign (n,) vs (n,1) / scalar vs (1,1) layout
        differences; it raises only if element COUNTS truly differ -- the signal we
        want surfaced rather than silently reshaped away.
        """
        def _cast(l, t):
            if hasattr(t, "dtype") and hasattr(t, "shape"):
                return jnp.asarray(l).astype(t.dtype).reshape(t.shape)
            return l
        return jax.tree_util.tree_map(_cast, loaded, template)

    # ---- optional E2E free run (defined above, after the cyipopt block) ----
    # Runs jaxipm WITHOUT per-step injection from the same iter-0 start and
    # compares every iteration against the IPOPT logs. Pure-pytree states, so
    # the injection loop's `state` below is untouched. Comment in/out.
    # free_run_compare(state)

    while cont == 1:
        orig = state
        print(f"\n===== compare iter {idx} =====")
        result = _execute_search(state, cp)
        # Restoration: jaxipm PERSISTS only the reduced (1128) aug step — its
        # cqpo.step_aug field is a static (1128,1) slot that can't hold the full
        # augmented restoration vector. IPOPT logs the full (2140) system. Expand
        # jaxipm's reduced step to the augmented layout with the SAME transform the
        # solver uses internally (calc_transform_red_to_aug; see search.py:465), so
        # step_aug compares element-wise against IPOPT instead of being skipped.
        # Use a comparison-only copy: the unmodified `result` (static 1128 shape)
        # must still flow into the jitted _post_process below.
        result_cmp = result
        if int(jnp.asarray(result.fl.in_restoration).squeeze()) == 1:
            _step_aug_full = cp.nstqfr.kkt.calc_transform_red_to_aug(
                result.cqpo.step_aug, result.cqpo.rhs,
                result.cqpr.Sigma_nc_inv, result.cqpr.Sigma_pc_inv,
                result.cqpr.Sigma_nd_inv, result.cqpr.Sigma_pd_inv,
            )
            result_cmp = eqx.tree_at(lambda t: t.cqpo.step_aug, result, _step_aug_full)
        # ===================== PER-STEP INJECTION ============================
        # This is a PER-STEP validator: every iteration starts jaxipm from IPOPT's
        # EXACT iterate (no free-run accumulation). jaxipm takes one step out of
        # IPOPT[idx], we compare that step against IPOPT, then RE-SEED jaxipm with
        # IPOPT[idx+1] for the next step. iter 0 is the only free start (it IS
        # IPOPT's cold start). Stop once IPOPT has no iter_<idx+1> dump.
        if not os.path.isdir(os.path.join(save_dir, f"iter_{idx + 1}")):
            break
        ipopt_next = load_state(idx + 1, cp)

        # Comparison (unchanged split model): `result` carries THIS iteration's
        # search direction + line search computed FROM IPOPT[idx]
        # (cqpr/cqpo/ls/wd/mu == IPOPT iter idx); result.it is the accepted trial
        # point (== IPOPT iter idx+1). Unify the IPOPT comparison state: `it` from
        # iter idx+1, every other sub-struct from iter idx.
        ipopt_state = load_state(idx, cp)                 # iter-idx, regular .it
        # The filter-rejection counter (+ its flag) advances DURING the line search
        # (FilterLSAcceptor increments it at the end of a successful acceptability
        # check, IpFilterLSAcceptor.cpp:443-456), but IPOPT's ls.txt dump happens in
        # InitThisLineSearch — BEFORE the iteration's trials. jaxipm's result.ls is
        # post-LS, so like `.it` these two fields must come from iter idx+1 (pre-LS
        # of k+1 == post-LS of k). Only visible when a filter rejection actually
        # occurs (e.g. the SOC iteration 47).
        ipopt_cmp = eqx.tree_at(
            lambda t: (
                t.it,
                t.ls.filter.count_successive_filter_rejections,
                t.ls.filter.last_rejection_due_to_filter,
            ),
            ipopt_state,
            (
                ipopt_next.it,
                ipopt_next.ls.filter.count_successive_filter_rejections,
                ipopt_next.ls.filter.last_rejection_due_to_filter,
            ),
        )
        # Exit-boundary flag for the it.* gate (see analyze_diff docstring):
        # IPOPT in resto at idx but regular at idx+1 -> this pass IS the exit.
        _exiting_now = (
            int(jnp.asarray(ipopt_state.fl.in_restoration).squeeze()) == 1
            and int(jnp.asarray(ipopt_next.fl.in_restoration).squeeze()) == 0
        )
        analyze_diff(result_cmp, ipopt_cmp, verbose_skips=VERBOSE_SKIPS,
                     exiting_resto=_exiting_now)

        # Archive the exact compare pair for offline diffing/plotting (see the
        # SAVE_STATES block above the loop).
        if SAVE_STATES:
            _save_state_pair(_run_dir, idx, (result_cmp, ipopt_cmp))

        # SOC iterations: IPOPT logs every SOC attempt to soc.txt, and the loader
        # fills ls.c_soc/dms_soc/count_soc/trial_step from the LAST attempt, so the
        # SOC path is validated element-wise (accumulated rhs, corrected delta,
        # rejected-attempt count; the accepted alpha via ls.alpha_pr; the accepted
        # point via it.*). IPOPT's cqpo.step stays the PRE-LS raw direction (the
        # PDSearchDirCalc dump; the SOC re-solve never re-dumps) — compared against
        # jaxipm's raw direction as on every other iter.
        if float(jnp.max(jnp.abs(jnp.asarray(ipopt_state.ls.c_soc)))) > 0.0:
            print(f"  NOTE idx {idx}: SOC iteration (IPOPT logged soc.txt) — "
                  "ls.trial_step / ls.c_soc / ls.dms_soc / ls.count_soc above are "
                  "compared against IPOPT's logged SOC attempt data.")

        # ---- KKT-residual probe for step divergences (regular mode only) ----
        # cqpo.rhs_aug/step_aug hold the EXACT (rhs, x) pair of cp.linear_solve
        # (quantities.py ~1397: rhs_aug=rhs, step_aug=step, "just for debugging").
        # Assemble K from the solver's upper-tri pattern (cp.coo_indices) + the
        # perturbed values actually factorized (result.ic.perturbed_data),
        # symmetrize, and measure ||K@s - rhs||inf for jaxipm's step vs IPOPT's
        # logged step. Same K + same rhs has a unique solution, so the side with
        # the (much) larger residual owns the divergence (cuDSS IR stall vs a
        # genuinely different system). Skipped in restoration (reduced form needs
        # the Sigma expansion; resto steps already validated via cqpo.step).
        if int(jnp.asarray(result.fl.in_restoration).squeeze()) == 0:
            _sj = jnp.asarray(result.cqpo.step_aug).flatten()
            _si = jnp.asarray(ipopt_state.cqpo.step_aug).flatten()
            _sdiff = float(jnp.max(jnp.abs(_sj - _si)))
            _snrm = max(float(jnp.max(jnp.abs(_si))), 1e-30)
            if _sdiff / _snrm > 1e-8:
                from jax.experimental import sparse as _jsparse
                _n = _sj.shape[0]
                _Ktri = _jsparse.BCOO(
                    (jnp.asarray(result.ic.perturbed_data).flatten(),
                     jnp.asarray(cp.coo_indices)),
                    shape=(_n, _n),
                ).todense()
                _K = _Ktri + _Ktri.T - jnp.diag(jnp.diag(_Ktri))
                _rhs = jnp.asarray(result.cqpo.rhs_aug).flatten()
                _rj = float(jnp.max(jnp.abs(_K @ _sj - _rhs)))
                _ri = float(jnp.max(jnp.abs(_K @ _si - _rhs)))
                _wloc = int(jnp.argmax(jnp.abs(_sj - _si)))
                print(f"  [STEP-RESIDUAL idx {idx}] max|step_jax-step_ipopt|={_sdiff:.3e}"
                      f" (rel {_sdiff/_snrm:.3e}) at row {_wloc}"
                      f" (x<{cp.nx}<=s<{cp.nx+cp.nyd}<=y_c<{cp.nx+cp.nyd+cp.nyc}<=y_d)")
                print(f"      ||K@step_jax   - rhs||inf = {_rj:.3e}")
                print(f"      ||K@step_ipopt - rhs||inf = {_ri:.3e}")
                print(f"      ||rhs||inf={float(jnp.max(jnp.abs(_rhs))):.3e}"
                      f"  max|K|={float(jnp.max(jnp.abs(_K))):.3e}"
                      f"  ic.dxs={float(jnp.asarray(result.ic.dxs).squeeze()):.3e}")

        # ---- [MU-ORACLE PROBE] probing-step solver diagnostic (regular mode) ----
        # The affine/centering probing steps (cqpr.step_aff_full/step_cen_full) feed
        # the adaptive-mu quality oracle. Their rhs are mu-INDEPENDENT and built only
        # from already-validated injected quantities, so a divergence here can only
        # be the linear solve itself (iter 86: jaxipm ~1e20 vs IPOPT ~1e2, then the
        # oracle's mu cascades into rhs/step/LS). When the affine step diverges:
        # rebuild both probing rhs exactly as calc_values_pre_mu does (quantities.py
        # :1118-1119 -> kkt.calc_aug_pd_RHS_aff/_cen, first 1128 rows = the solver
        # rhs), residual-test BOTH sides' steps against the same K, dense-LU solve
        # the system as a backend-independent reference, and re-solve with cuDSS at
        # several IR counts. cuDSS runs a FIXED ir_nsteps (params.yaml: 100) with no
        # divergence guard, and quantities.py:1233 records IR diverging to infinity
        # on a reused factorization — the sweep separates "bad factorization" (ir=0
        # already garbage) from "IR divergence" (ir=0 sane, residual grows with ir).
        # Uses `orig` (the state execute_search consumed): orig.it/cqpr/ic are the
        # EXACT iterate, quantities and factorized values the probing solves ran at.
        if int(jnp.asarray(result.fl.in_restoration).squeeze()) == 0:
            def _aug_blocks(_v):
                # step_aff_full/step_cen_full are FLAT full-space vectors in the
                # padded layout {x[+pads], s, y_c, y_d, z_L[+pads], z_U, v_L, v_U};
                # split with the same helper quantities.py uses, keep the aug rows.
                _itv = cp.stqf.calc_vector_to_iterate(jnp.asarray(_v).reshape(-1, 1))
                return jnp.concatenate([
                    jnp.asarray(_itv.x).flatten()[:cp.nx],
                    jnp.asarray(_itv.s).flatten(),
                    jnp.asarray(_itv.y_c).flatten(),
                    jnp.asarray(_itv.y_d).flatten(),
                ])
            _aff_j = _aug_blocks(orig.cqpr.step_aff_full)
            _aff_i = _aug_blocks(ipopt_state.cqpr.step_aff_full)
            _adiff = float(jnp.max(jnp.abs(_aff_j - _aff_i)))
            _anrm = max(float(jnp.max(jnp.abs(_aff_i))), 1e-30)
            # IPOPT dumps step_aff/cen_full ONLY when the quality oracle runs
            # (QualityFunctionMuOracle::CalculateMu — free mu mode). At monotone
            # iters the loader holds zeros and jaxipm's always-computed probing
            # step is unused — skip, or the gate fires on diff/1e-30.
            _oracle_logged = float(jnp.max(jnp.abs(_aff_i))) > 0.0
            if _oracle_logged and _adiff / _anrm > 1e-6:
                from jax.experimental import sparse as _jsparse
                _n = _aff_j.shape[0]
                _kdata = jnp.asarray(orig.ic.perturbed_data).flatten()
                _Ktri = _jsparse.BCOO(
                    (_kdata, jnp.asarray(cp.coo_indices)), shape=(_n, _n)
                ).todense()
                _K = _Ktri + _Ktri.T - jnp.diag(jnp.diag(_Ktri))
                _rhs_aff = jnp.asarray(cp.nstqf.kkt.calc_aug_pd_RHS_aff(
                    orig.it, orig.cqpr.slacks,
                    jnp.asarray(orig.cqpr.grad_lag_x), jnp.asarray(orig.cqpr.grad_lag_s),
                    jnp.asarray(orig.cqpr.c), jnp.asarray(orig.cqpr.dms))).flatten()[:_n]
                _rhs_cen = jnp.asarray(cp.nstqf.kkt.calc_aug_pd_RHS_cen(
                    orig.it, orig.cqpr.slacks,
                    jnp.asarray(orig.cqpr.avrg_compl))).flatten()[:_n]
                _cen_j = _aug_blocks(orig.cqpr.step_cen_full)
                _cen_i = _aug_blocks(ipopt_state.cqpr.step_cen_full)
                def _res(_x, _b):
                    return float(jnp.max(jnp.abs(_K @ _x - _b)))
                def _mx(_x):
                    return float(jnp.max(jnp.abs(_x)))
                print(f"  [MU-ORACLE PROBE idx {idx}] affine probing step diverged"
                      f" (max|aff_jax-aff_ipopt|={_adiff:.3e}, rel {_adiff/_anrm:.3e})")
                print(f"      aff : max|x_jax|={_mx(_aff_j):.3e} res_jax={_res(_aff_j,_rhs_aff):.3e}"
                      f" | max|x_ipopt|={_mx(_aff_i):.3e} res_ipopt={_res(_aff_i,_rhs_aff):.3e}"
                      f" | ||rhs||inf={_mx(_rhs_aff):.3e}")
                print(f"      cen : max|x_jax|={_mx(_cen_j):.3e} res_jax={_res(_cen_j,_rhs_cen):.3e}"
                      f" | max|x_ipopt|={_mx(_cen_i):.3e} res_ipopt={_res(_cen_i,_rhs_cen):.3e}"
                      f" | ||rhs||inf={_mx(_rhs_cen):.3e}")
                # backend-independent reference: dense LU on the SAME K
                _x_lu = jnp.linalg.solve(_K, _rhs_aff)
                print(f"      dense-LU(aff): max|x|={_mx(_x_lu):.3e} res={_res(_x_lu,_rhs_aff):.3e}"
                      f" ||x_lu - aff_ipopt||inf={_mx(_x_lu - _aff_i):.3e}")
                # cuDSS re-solve sweep on the CURRENT factorization (refac=0): if the
                # ir=0 solve is sane and the residual GROWS with ir, IR is diverging.
                # cp.linear_solve is ft.partial(CuDSSSolver_instance, ...); .func is
                # the instance, callable with explicit signals.
                _i32 = lambda _v: jnp.array([_v], dtype=jnp.int32)
                for _ir in (0, 1, 5, 10, 100):
                    _xs = jnp.asarray(cp.linear_solve.func(
                        _rhs_aff, _kdata, refactorize_signal=_i32(0),
                        solve_signal=_i32(1), ir_nsteps_signal=_i32(_ir))[0]).flatten()
                    print(f"      cuDSS re-solve(aff)    ir={_ir:3d}:"
                          f" max|x|={_mx(_xs):.3e} res={_res(_xs,_rhs_aff):.3e}")
                # fresh refactorize + solve (handle state is washed out by the next
                # pass's post_process, which refactorizes at the next iterate anyway)
                for _ir in (0, 5, 100):
                    _xs = jnp.asarray(cp.linear_solve.func(
                        _rhs_aff, _kdata, refactorize_signal=_i32(1),
                        solve_signal=_i32(1), ir_nsteps_signal=_i32(_ir))[0]).flatten()
                    print(f"      cuDSS refac+solve(aff) ir={_ir:3d}:"
                          f" max|x|={_mx(_xs):.3e} res={_res(_xs,_rhs_aff):.3e}")
                # main step on the exact logged (rhs_aug, step_aug) pair for context
                _rhs_m = jnp.asarray(result.cqpo.rhs_aug).flatten()
                _x_lu_m = jnp.linalg.solve(_K, _rhs_m)
                print(f"      dense-LU(main): max|x|={_mx(_x_lu_m):.3e}"
                      f" res={_res(_x_lu_m,_rhs_m):.3e}"
                      f" (jax step_aug max={_mx(jnp.asarray(result.cqpo.step_aug)):.3e})")
                for _ir in (0, 100):
                    _xs = jnp.asarray(cp.linear_solve.func(
                        _rhs_m, _kdata, refactorize_signal=_i32(0),
                        solve_signal=_i32(1), ir_nsteps_signal=_i32(_ir))[0]).flatten()
                    print(f"      cuDSS re-solve(main)   ir={_ir:3d}:"
                          f" max|x|={_mx(_xs):.3e} res={_res(_xs,_rhs_m):.3e}")

        # ===================== RE-SEED (FULL STATE) =========================
        # Inject IPOPT's COMPLETE state — but with the SAME pre/post split the
        # comparison uses, because that split is exactly the shape of the `result`
        # post_process consumes. In the natural loop, execute_search advances only the
        # ITERATE (result.it == iterate_{k+1}); it leaves mu/tau/adfs/fl/args/ic at
        # their PRE-update (iter-k) values, and post_process's calc_updated_mu is what
        # advances mu_k -> mu_{k+1} and the adaptive-mu state (adfs, free_mu_mode)
        # along with it. So the injected "result" must be: iterate from iter idx+1,
        # ALL bookkeeping from iter idx. That is precisely `ipopt_state` (built above:
        # load_state(idx) with .it <- load_state(idx+1).it).
        #
        # Injecting the whole iter-(idx+1) state instead (the previous attempt) feeds
        # post_process an ALREADY-updated mu/adfs and makes it update a SECOND time:
        # the double-updated adaptive filter rejects, free_mu_mode flips free->fixed,
        # and the monotone branch returns a 6-orders-too-large mu that cascades into
        # barr/gBD/duals/step. iter-0's pre-update (mu=0.987, free_mu_mode=1, adfs=
        # [366,19.9]) is what the oracle must advance to iter-1's mu=1.74e-6.
        #
        # cqpr/cqpo are the ONLY fields taken from jaxipm's `result`: they are
        # post_process OUTPUTS (recomputed AT inj.it inside post_process), and in
        # restoration jaxipm's Schur-reduced aug blocks (1128) have a different SHAPE
        # than IPOPT's full system (2140) — they physically cannot be cross-loaded.
        # Swapping them in first also gives _conform a template whose every leaf
        # already matches jaxipm's shapes.
        #
        # RESTORATION-ENTRY BOUNDARY: when IPOPT is regular at iter idx but in
        # restoration at iter idx+1, jaxipm must ENTER restoration here, not continue
        # regular over iter-(idx+1)'s resto iterate. post_process's init_resto branch
        # BUILDS the restoration iterate (p/n slacks, resto mu, resto args) FROM the
        # regular entry iterate via initialize_iterate_resto/initialize_resto_args --
        # exactly as IPOPT's RestoIterateInitializer does. So at the boundary we
        #   (a) step FROM the regular entry iterate (load_state(idx).it, NOT idx+1's
        #       resto iterate -- that is precisely what jaxipm is about to construct), and
        #   (b) raise needs_resto_init=1 instead of forcing it to 0.
        # The resulting state IS jaxipm's first resto iterate; the next loop pass
        # compares it against IPOPT iter idx+1. Detect via the loaded fl flags (iter_5,
        # the trigger iter, has no fl.txt -> defaults in_restoration=0, which is correct).
        in_resto_now = int(jnp.asarray(ipopt_state.fl.in_restoration).squeeze())
        in_resto_next = int(jnp.asarray(ipopt_next.fl.in_restoration).squeeze())
        entering_resto = (in_resto_now == 0) and (in_resto_next == 1)
        exiting_resto = (in_resto_now == 1) and (in_resto_next == 0)
        if entering_resto:
            resto_entry_idx = idx  # last regular iter before restoration

        # base = the iterate jaxipm steps FROM: regular entry point (iter idx) at the
        # boundary, else the post-step iterate (iter idx+1).
        base = ipopt_state if entering_resto else ipopt_cmp
        resto_init_flag = jnp.array([[1]]) if entering_resto else jnp.array([[0]])

        ipopt_inj = eqx.tree_at(
            lambda t: (t.cqpr, t.cqpo), base, (result.cqpr, result.cqpo)
        )
        # Conform every injected leaf to jaxipm's exact (dtype, shape). This resolves
        # the float64/int64 lax.select crash at the SOURCE, for all leaves at once.
        inj = _conform(ipopt_inj, result)
        # needs_regular_init always off; needs_resto_init raised only at the entry
        # boundary so post_process re-initializes restoration over the entry iterate.
        inj = eqx.tree_at(
            lambda t: (t.fl.needs_resto_init, t.fl.needs_regular_init),
            inj,
            (resto_init_flag, jnp.array([[0]])),
        )

        # RESTORATION CONTINUATION: preserve jaxipm's OWN frozen saved_* slots instead
        # of the per-step loader's mirror. IPOPT logs no saved_* — they are jaxipm's
        # internal freeze of the pre-restoration regular state, set ONCE by post_process's
        # init_resto branch at the entry boundary (saved_orig_inf_pr = the ENTRY original
        # infeasibility, saved_ls.filter = the saved ORIGINAL filter, saved_mu, saved
        # bound mults). The loader can only mirror them to the CURRENT iterate, which
        # clobbers those frozen values every step. Invisible during restoration (saved_*
        # is SKIPPED in the comparison and the resto STEP never reads it) UNTIL the EXIT:
        # should_exit_resto fires only when orig_trial_inf_pr <= orig_inf_pr_max =
        # max(kappa_resto * saved_orig_inf_pr, min(tol,cvtol)) AND the trial is acceptable
        # to the saved ORIGINAL filter. With the mirror (saved_orig_inf_pr collapsed to the
        # small current resto infeasibility) orig_inf_pr_max is tiny, the reduction test can
        # never pass, and jaxipm never exits (iter 34: jaxipm in_restoration=1 vs IPOPT 0).
        # post_process froze the CORRECT values at entry and carries them forward via its
        # `saved_* = cond(init_resto, fresh, result.saved_*)` lines, so once jaxipm is in
        # restoration we re-inject ITS OWN saved_* (from `orig`, the prior post_process
        # output) and let the exit check see the true entry state. At the entry boundary
        # `orig` is still regular (in_restoration=0) -> skip, so post_process does the freeze.
        if int(jnp.asarray(orig.fl.in_restoration).squeeze()) == 1:
            inj = eqx.tree_at(
                lambda t: (
                    t.saved_fl, t.saved_wd, t.saved_ls, t.saved_ic, t.saved_adfs,
                    t.saved_mu, t.saved_tau, t.saved_mu_max,
                    t.saved_init_dual_inf, t.saved_init_primal_inf, t.saved_orig_inf_pr,
                    t.saved_z_L, t.saved_z_U, t.saved_v_L, t.saved_v_U,
                    # saved_slacks is the DIVISOR of the exit bound-multiplier step;
                    # the loader mirror holds zeros there, which NaN-poisons v_L
                    # (hence the LSQ y, grad_lag, the IC and the mu-oracle mode
                    # switch) on the exit pass.
                    t.saved_slacks,
                ),
                inj,
                (
                    orig.saved_fl, orig.saved_wd, orig.saved_ls, orig.saved_ic, orig.saved_adfs,
                    orig.saved_mu, orig.saved_tau, orig.saved_mu_max,
                    orig.saved_init_dual_inf, orig.saved_init_primal_inf, orig.saved_orig_inf_pr,
                    orig.saved_z_L, orig.saved_z_U, orig.saved_v_L, orig.saved_v_U,
                    orig.saved_slacks,
                ),
            )

        # ===================== EXIT-BOUNDARY DIAGNOSTIC ======================
        # At the restoration EXIT boundary (IPOPT resto at idx, regular at idx+1)
        # jaxipm must fire should_exit_resto in THIS post_process so state_{idx+1} is
        # regular. The exit semantics were validated against IpRestoConvCheck.cpp:
        # IPOPT exits iff orig_trial_inf_pr <= max(kappa*orig_curr_inf_pr, min(tol,
        # cvtol)) AND filter/iterate acceptable, where orig_curr_inf_pr is evaluated
        # at orig_ip_data->curr() -- which NOTHING advances during restoration, i.e.
        # it IS the frozen entry infeasibility. jaxipm's saved_orig_inf_pr matches
        # IPOPT exactly. (quad_nav exit: trial 0.8293 at IPOPT[35].it <= 0.9*0.9549.)
        # Two things to surface, both non-jitted (eager jax, so prints work):
        #   (A) VALIDATE jaxipm's own entry-freeze: orig.saved_* should match the LAST
        #       regular iter before resto (load_state(resto_entry_idx)). A mismatch here
        #       means post_process froze the wrong state at entry.
        #   (B) REPLICATE check_resto_convergence's sub-conditions on inj.it with
        #       orig.saved_*, to see WHICH of infeas / filter / iterate blocks the exit.
        #       With first-wins section parsing this fires at the TRUE boundary (inj.it
        #       = the final resto point) and all three checks should print True.
        if exiting_resto:
            print(f"\n----- EXIT-BOUNDARY DIAGNOSTIC @ loop idx {idx} "
                  f"(IPOPT resto->regular at {idx}->{idx+1}) -----")
            def _f(v):
                return float(jnp.asarray(v).squeeze())
            # ---- (A) freeze validation vs the pre-resto regular IPOPT state ----
            if resto_entry_idx is not None:
                ipe = load_state(resto_entry_idx, cp)
                print(f"  [A] entry-freeze vs IPOPT[{resto_entry_idx}] (jaxipm saved_* | ipopt):")
                print(f"      saved_mu            {_f(orig.saved_mu):.10e} | {_f(ipe.mu):.10e}")
                print(f"      saved_tau           {_f(orig.saved_tau):.10e} | {_f(ipe.tau):.10e}")
                print(f"      saved_mu_max        {_f(orig.saved_mu_max):.10e} | {_f(ipe.mu_max):.10e}")
                print(f"      saved theta_max     {_f(orig.saved_ls.filter.theta_max):.10e} | {_f(ipe.ls.filter.theta_max):.10e}")
                print(f"      saved theta_min     {_f(orig.saved_ls.filter.theta_min):.10e} | {_f(ipe.ls.filter.theta_min):.10e}")
                print(f"      saved ref_barr      {_f(orig.saved_ls.filter.ref_barr):.10e} | {_f(ipe.ls.filter.ref_barr):.10e}")
                print(f"      saved ref_theta     {_f(orig.saved_ls.filter.ref_theta):.10e} | {_f(ipe.ls.filter.ref_theta):.10e}")
                print(f"      saved_z_L[:3]       {jnp.asarray(orig.saved_z_L).flatten()[:3]} | {jnp.asarray(ipe.it.z_L).flatten()[:3]}")
                print(f"      saved_v_L[:3]       {jnp.asarray(orig.saved_v_L).flatten()[:3]} | {jnp.asarray(ipe.it.v_L).flatten()[:3]}")
                # saved_orig_inf_pr is COMPUTED at entry (not loaded) -> cross-check it
                # against the original infeasibility actually at IPOPT[entry].it.
                try:
                    ef_args, ec_args, ed_args = ipe.args
                    exnx = ipe.it.x[:cp.nx]
                    e_c = cp.nstqf.calc_c(exnx, *ec_args)
                    e_dms = cp.nstqf.calc_d(exnx, *ed_args) - ipe.it.s[:cp.nyd]
                    e_oip = jnp.maximum(jnp.linalg.norm(e_c, ord=jnp.inf),
                                        jnp.linalg.norm(e_dms, ord=jnp.inf))
                    print(f"      saved_orig_inf_pr   {_f(orig.saved_orig_inf_pr):.10e} | "
                          f"{_f(e_oip):.10e} (orig inf at IPOPT[{resto_entry_idx}].it)")
                except Exception as e:
                    print(f"      saved_orig_inf_pr   {_f(orig.saved_orig_inf_pr):.10e} | <recompute failed: {e!r}>")
            # ---- (B) replicate the exit-convergence sub-conditions ----
            try:
                f_args, c_args, d_args = inj.args
                xnx = inj.it.x[:cp.nx]
                reg_c = cp.nstqf.calc_c(xnx, *c_args)
                reg_d = cp.nstqf.calc_d(xnx, *d_args)
                reg_f = jnp.atleast_2d(cp.nstqf.calc_f(xnx, *f_args))
                reg_dms = reg_d - inj.it.s[:cp.nyd]
                orig_trial_inf_pr = jnp.maximum(jnp.linalg.norm(reg_c, ord=jnp.inf),
                                                jnp.linalg.norm(reg_dms, ord=jnp.inf))
                # mirror check_resto_convergence's defensive default (key may be absent)
                if hasattr(cp.p, "get"):
                    kappa_resto = cp.p.get("required_infeasibility_reduction", 0.9)
                elif "required_infeasibility_reduction" in getattr(cp.p, "_dict", {}):
                    kappa_resto = cp.p["required_infeasibility_reduction"]
                else:
                    kappa_resto = 0.9
                saved_oip = orig.saved_orig_inf_pr
                orig_inf_pr_max = jnp.maximum(kappa_resto * saved_oip,
                                              jnp.minimum(cp.p["tol"], cp.p["constr_viol_tol"]))
                orig_inf_pr_max = jnp.where(kappa_resto == 0.0, 0.0, orig_inf_pr_max)
                infeas_ok = orig_trial_inf_pr <= orig_inf_pr_max
                orig_trial_theta = cp.nstqf.calc_theta(reg_c, reg_dms)
                orig_slacks = cp.nstqf.calc_slacks(inj.it)
                orig_trial_barr = cp.nstqf.calc_barrier_obj(reg_f, orig.saved_mu, orig_slacks)
                filt_ok = cp.stqf.is_acceptable_to_current_filter(
                    orig_trial_barr, orig_trial_theta, orig.saved_ls.filter.F)
                iter_ok = cp.stqf.is_acceptable_to_current_iterate(
                    orig_trial_barr, orig_trial_theta,
                    orig.saved_ls.filter.ref_barr, orig.saved_ls.filter.ref_theta,
                    jnp.array([[1]]))
                print("  [B] should_exit_resto sub-conditions on inj.it (= IPOPT[idx+1].it):")
                print(f"      saved_orig_inf_pr   {_f(saved_oip):.10e}")
                print(f"      orig_trial_inf_pr   {_f(orig_trial_inf_pr):.10e}")
                print(f"      orig_inf_pr_max     {_f(orig_inf_pr_max):.10e}  (kappa_resto={_f(kappa_resto):.3f})")
                print(f"      -> infeas_check     {bool(jnp.asarray(infeas_ok).squeeze())}")
                print(f"      orig_trial_theta    {_f(orig_trial_theta):.10e}")
                print(f"      orig_trial_barr     {_f(orig_trial_barr):.10e}")
                print(f"      saved ref_barr/theta {_f(orig.saved_ls.filter.ref_barr):.6e} / {_f(orig.saved_ls.filter.ref_theta):.6e}")
                print(f"      -> filter_accept    {bool(jnp.asarray(filt_ok).squeeze())}")
                print(f"      -> iterate_accept   {bool(jnp.asarray(iter_ok).squeeze())}")
                print(f"      => filter_checks_pass = {bool(jnp.asarray(infeas_ok & filt_ok & iter_ok).squeeze())}")
            except Exception as e:
                print(f"  [B] sub-condition replication failed: {e!r}")
            print("  NOTE: it.* is GATED at this idx (analyze_diff exiting_resto=True): "
                  "the compare sees the pre-exit resto trial while IPOPT["
                  f"{idx+1}].it is post-exit (x-pad resto slacks, LSQ/zero-reset "
                  "y_c/y_d, bound-mult-stepped z_L/v_L) -- post_process applies the "
                  "same updates right after this compare (search.py exit branch); "
                  "the exit-updated state is validated at the next iter.")
            print("----- END EXIT-BOUNDARY DIAGNOSTIC -----\n")

        state, term = _post_process(orig, inj, cp)
        idx += 1
        if idx == 39:
            pass