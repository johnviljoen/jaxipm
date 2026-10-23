"""MadNLP (GPU, cuDSS) receding-horizon expert for track-avoid -- the paper's
MadNLP baseline (tests/quad_track_avoid/madnlp_track_avoid.jl) as a DAgger
labeller, with the same interface as tests/imitation_track/expert.Expert.

A Julia server (madnlp_expert.jl) keeps one ExaModel of the identical NLP
(measured state, reference horizon and obstacle centres as parameters) and
solves the pool sequentially with a fresh MadNLPSolver per instance (paper
options, tol 1e-6). Requests/results travel as npz files; `elapsed` is the
server's solve-loop wall (no file I/O), `elapsed_io` includes the transfer.
Labels are the solver's own x[0] / u[0] of SOLVE_SUCCEEDED and
SOLVED_TO_ACCEPTABLE_LEVEL instances, like the jaxipm expert keeps codes 1/5.
"""

import os
import subprocess
import tempfile
import uuid
from time import time

import numpy as np
import jax
import jax.numpy as jnp

from tests.quad_track_avoid.jaxipm_track_avoid import N_HORIZON
from tests.imitation_track.sim import NX, NU, ref_horizon, obstacle_horizon, valid_states
from tests.imitation_track.expert import warm_from_ref, U0_SLICE

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JULIA = os.environ.get("JULIA", "julia")
SCRIPT = os.path.join(REPO, "tests", "imitation_track", "madnlp_expert.jl")
NZ = N_HORIZON * NX + (N_HORIZON - 1) * NU
# MadNLP.Status -> census slot in the jaxipm TerminationCode sense
# (1 converged, 5 acceptable, 2 max iterations, 3 tiny step, 4 any other failure)
_STATUS_TO_TERM = {1: 1, 2: 5, 6: 2, 3: 3}


def _default_work_dir():
    base = "/dev/shm" if os.access("/dev/shm", os.W_OK) else tempfile.gettempdir()
    return os.path.join(base, f"jaxipm_madnlp_{os.getpid()}")


class MadNLPExpert:
    def __init__(self, pool_size, v, margin=0.0, k_start=0, r_u=0.0, tol=1e-6, batch_size=None,
                 debug=True, work_dir=None, log_path=None):
        self.pool_size, self.v, self.nz = int(pool_size), float(v), NZ
        self.tol, self.debug = float(tol), bool(debug)
        self._xr = jax.jit(jax.vmap(ref_horizon, in_axes=(0, 0, None)), static_argnums=2)
        self._obs = jax.jit(jax.vmap(obstacle_horizon))                 # (P, N, N_OBS, 2)
        self._warm = jax.jit(jax.vmap(warm_from_ref))
        self.work_dir = work_dir or _default_work_dir()
        os.makedirs(self.work_dir, exist_ok=True)
        self.log = open(log_path or os.path.join(self.work_dir, "madnlp_server.log"), "a")
        t0 = time()
        self.proc = subprocess.Popen([JULIA, SCRIPT, str(margin), str(k_start), str(r_u), str(tol)],
                                     cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1)
        line = self._readline()
        if line != "READY":
            raise RuntimeError(f"madnlp server: unexpected '{line}' (see {self.log.name})")
        self.startup_s = time() - t0
        print(f"    [madnlp expert] server ready (build + JIT warmup) in {self.startup_s:.1f}s", flush=True)
        self.compiled = True

    # ------------------------------------------------------------------ #
    def _readline(self):
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError(f"madnlp server exited (see {self.log.name})")
        return line.strip()

    def _run(self, x0, xr, obs, warm):
        req = os.path.join(self.work_dir, f"req_{uuid.uuid4().hex}.npz")
        np.savez(req, x0=x0, xr=xr, obs=obs, warm=warm)
        t0 = time()
        self.proc.stdin.write(req + "\n"); self.proc.stdin.flush()
        line = self._readline()
        assert line.startswith("DONE "), line
        out_path = line[5:]
        with np.load(out_path) as f:
            res = {k: f[k] for k in f.files}
        elapsed_io = time() - t0
        os.remove(req); os.remove(out_path)
        return res, elapsed_io

    def label(self, x0s, s0s, ts, warm=None, return_z=False):
        """Labels for pool members (x0s (P,13), s0s (P,), ts (P,) sim time), same
        return dict as Expert.label plus elapsed_io, madnlp_status, solve_times
        and pool_index (the pool row of every returned label)."""
        x0s = np.asarray(x0s, dtype=np.float64); s0s = np.asarray(s0s, dtype=np.float64)
        ts = np.asarray(ts, dtype=np.float64)
        P = x0s.shape[0]
        assert x0s.shape == (P, NX) and s0s.shape == (P,) and ts.shape == (P,)
        xr = np.asarray(self._xr(jnp.asarray(s0s), jnp.asarray(ts), self.v))
        obs = np.asarray(self._obs(jnp.asarray(ts)))
        ok_in = valid_states(x0s)
        warm_ref = np.asarray(self._warm(jnp.asarray(xr)))
        if warm is None:
            warm = warm_ref
        else:
            warm = np.asarray(warm, dtype=np.float64)
            assert warm.shape == (P, self.nz), warm.shape
            warm = np.where(ok_in[:, None], warm, warm_ref)
        idx = np.flatnonzero(ok_in)
        empty = dict(X0=np.zeros((0, NX)), U0=np.zeros((0, NU)), terms=np.zeros(0, np.int32),
                     iters=np.zeros(0, np.int32), n=0, elapsed=0.0, elapsed_io=0.0,
                     n_invalid=int((~ok_in).sum()), term_hist=np.zeros(6, int),
                     madnlp_status=np.zeros(0, np.int32), solve_times=np.zeros(0), pool_index=idx[:0])
        if idx.size == 0:
            if return_z: empty["Z"] = np.zeros((0, self.nz))
            return empty
        res, elapsed_io = self._run(x0s[idx], xr[idx], obs[idx], warm[idx])
        status = res["status"].astype(np.int32)
        succ = (status == 1) | (status == 2)
        Z = res["Z"][succ]
        term_hist = np.zeros(6, int)
        for s in status:
            term_hist[_STATUS_TO_TERM.get(int(s), 4)] += 1
        out = dict(X0=Z[:, :NX], U0=Z[:, U0_SLICE],
                   terms=np.array([_STATUS_TO_TERM.get(int(s), 4) for s in status[succ]], np.int32),
                   iters=res["iters"][succ].astype(np.int32), n=int(succ.sum()),
                   elapsed=float(res["elapsed"][0]), elapsed_io=elapsed_io,
                   n_invalid=int((~ok_in).sum()), term_hist=term_hist, madnlp_status=status,
                   solve_times=res["times"], pool_index=idx[succ])
        if return_z:
            out["Z"] = Z
        return out

    def label_ordered(self, x0s, s0s, ts, warm=None):
        """Ordered plans per pool member: Z (P, nz) (NaN rows for failures), ok (P,), res."""
        res = self.label(x0s, s0s, ts, warm=warm, return_z=True)
        P = np.asarray(x0s).shape[0]
        Z = np.full((P, self.nz), np.nan); ok = np.zeros(P, bool)
        Z[res["pool_index"]] = res["Z"]; ok[res["pool_index"]] = True
        return Z, ok, res

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write("QUIT\n"); proc.stdin.flush()
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        try:
            self.log.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
