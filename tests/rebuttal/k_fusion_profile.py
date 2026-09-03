#!/usr/bin/env python
"""Run K — fusion cost breakdown of ONE fused jaxipm iteration (manifest run K).

Profiles the fused iteration body (vmapped execute_search + post_process +
hot-restart splice, i.e. exactly the `solve_throughput` while-loop body) on a
warm batch state, with the JAX/XLA device profiler.  The solver source carries
`jax.named_scope("K_<region>")` markers (jaxipm/utils/kscope.py) so every GPU
kernel can be attributed to a region:

  pp_derivs               derivative evaluation (f, c, d, Jacobians, Hessians)
  kkt_assembly            KKT RHS/LHS assembly, BCSR conversion
  ls_mult_factor_solve    least-squares multiplier system (factorize + solve)
  ic_loop                 inertia-correction loop (refactorizations, inertia queries, step_aff solve)
  solve_cen / solve_newton  triangular solves against the IC factorization
  pp_quantities / pp_mu_update / pre_mu_misc / post_mu_*  scalar quantities, mu update, step transforms
  search_sfr_setup        soft-feasibility-restoration trial (branch-unique)
  search_ls_first         first line-search trial evaluation (needed by every branch)
  search_wd_socinit       watchdog bookkeeping, SOC initial residuals (branch-unique)
  search_soc / search_bt  second-order-correction / backtracking while-loops (branch-unique)
  search_select / pp_select_term / pp_resto_init  selects and state writes
  hr_restart              solution scatter + hot-restart problem injection

cuDSS custom calls are additionally counted per iteration by name.

Fusion tax: the fused body computes every branch for every member; an oracle
body that knows a member takes the full step needs derivatives + assembly +
factorization + solves + ONE trial evaluation + state write, but none of the
SFR / watchdog / SOC / backtracking work.  tax = 1 - t(oracle) / t(fused).

Usage:
  python -m tests.rebuttal.k_fusion_profile --scenario nav --variants 90 --batch 500 --gpu 4
  python -m tests.rebuttal.k_fusion_profile --analyze      # tables + figure from results/K_*.json
"""
import argparse, collections, glob, json, os, re, socket, subprocess, sys
from time import time, perf_counter

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "tests", "rebuttal", "results")
FIG = os.path.join(REPO, "tests", "rebuttal", "figures")

REGION_ORDER = [
    ("pp_derivs", "derivative evaluation"),
    ("kkt_assembly", "KKT assembly"),
    ("ls_mult_factor_solve", "LS-multiplier factor+solve"),
    ("ic_loop", "inertia-corr. loop (factorizations)"),
    ("solve_cen", "centering solve"),
    ("solve_newton", "Newton solve"),
    ("pp_quantities", "quantities"),
    ("pp_mu_update", "mu update"),
    ("pre_mu_misc", "pre-mu misc"),
    ("post_mu_rhs", "post-mu RHS"),
    ("post_mu_transform", "step transform"),
    ("search_ls_first", "first trial evaluation"),
    ("search_sfr_setup", "SFR trial (branch-unique)"),
    ("search_wd_socinit", "WD/SOC init (branch-unique)"),
    ("search_soc", "SOC loop (branch-unique)"),
    ("search_bt", "backtracking loop (branch-unique)"),
    ("search_select", "search select/state write"),
    ("pp_select_term", "post select/termination"),
    ("pp_resto_init", "resto/init bookkeeping"),
    ("hr_restart", "scatter + hot-restart inject"),
    ("cudss_factorize", "cuDSS factorization kernels"),
    ("cudss_solve", "cuDSS triangular-solve kernels"),
    ("cudss_analysis", "cuDSS symbolic-analysis kernels"),
    ("memcpy", "memcpy / memset"),
    ("unscoped", "unscoped XLA kernels"),
]
BRANCH_UNIQUE = {"search_sfr_setup", "search_wd_socinit", "search_soc", "search_bt"}
LINALG = {"cudss_factorize", "cudss_solve", "cudss_analysis",
          "ls_mult_factor_solve", "ic_loop", "solve_cen", "solve_newton"}
CUDSS_ANALYSIS = ("offsets_par_ker", "define_superpanel", "csc_rows_ker", "nnz_per_col_ker",
                  "dependency_map_ker", "etree", "postorder", "symbolic", "reorder", "colamd", "metis")


def _git(cwd, *a):
    try:
        return subprocess.check_output(["git", *a], cwd=cwd, text=True).strip()
    except Exception:
        return "?"


# ----------------------------------------------------------------------------
def profile(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir(REPO); sys.path.insert(0, REPO)
    from tests.rebuttal.problems import (DEFAULT_VARIANTS, build_scenario, load_params,
                                         paper_batch_size, parse_variant)
    import jax, jax.numpy as jnp, equinox as eqx
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    from jaxipm.solver import solve_throughput, make_batch_state
    from jaxipm.search import execute_search, post_process
    from spineax import cudss as _cudss
    def spx(tag):
        st = dict(registry_size=_cudss.registry_size(), cache_capacity=_cudss.cache_capacity(),
                  rebuild_count=_cudss.rebuild_count())
        print(f"[SPINEAX] {tag}: {st}", flush=True); return st

    p = load_params(); p["DEBUG_MODE"] = False; p["hot_restarting"] = True
    variants = ([parse_variant(args.scenario, s) for s in args.variants.split(",")]
                if args.variants else DEFAULT_VARIANTS[args.scenario])
    for variant in variants:
        jax.clear_caches()
        N = args.batch or paper_batch_size(args.scenario)
        sc = build_scenario(args.scenario, variant, p, N); cp = sc.cp
        label = f"{sc.label}_b{N}{('_' + args.out_suffix) if args.out_suffix else ''}"
        print(f"\n===== K profile {label} =====", flush=True)
        idx0 = np.arange(N) % sc.N_RUNS
        bt = sc.inject(make_batch_state(cp, [sc.state] * N), idx0)
        bt = eqx.tree_at(lambda s: s.fl.needs_regular_init, bt,
                         jnp.full_like(bt.fl.needs_regular_init, 1))
        key = jax.random.PRNGKey(0)

        # -- warm state: run the real throughput loop for a few pipeline fills
        warm = args.warm_solves or max(2 * N, 8)
        _solve = eqx.filter_jit(solve_throughput, donate="none")
        t0 = time()
        out = _solve(key, cp, bt, warm)
        state = out[0]; jax.block_until_ready(state)
        print(f"[warm] solve_throughput(max_solves={warm}) took {time()-t0:.1f}s "
              f"(incl. JIT); iter_count range "
              f"{int(state.iter_count.min())}-{int(state.iter_count.max())}", flush=True)

        vs = eqx.filter_vmap(execute_search, in_axes=(0, None))
        vp = eqx.filter_vmap(post_process, in_axes=(0, 0, None))

        def step(state, rng_key):
            orig = state
            with jax.named_scope("K_search"):
                result = vs(state, cp)
            with jax.named_scope("K_post"):
                state, terminate = vp(orig, result, cp)
            with jax.named_scope("K_hr_restart"):
                term_codes = terminate.reshape(-1)
                should_restart = (terminate > 0).reshape(-1)
                rng_key, *subkeys = jax.random.split(rng_key, N + 1)
                subkeys = jnp.stack(subkeys)
                next_x0, next_f, next_c, next_d = jax.vmap(cp.calc_next_problem)(
                    subkeys, state.it.x[:, :cp.nx, 0])
                f_curr, c_curr, d_curr = state.args
                next_args = ((*f_curr[:4], *next_f), (c_curr[0], *next_c), (d_curr[0], *next_d))
                new_args = jax.tree.map(
                    lambda n, o: jnp.where(jnp.reshape(should_restart, (-1,) + (1,) * (n.ndim - 1)), n, o),
                    next_args, state.args)
                new_x = jnp.where(should_restart[:, None, None],
                                  state.it.x.at[:, :cp.nx].set(next_x0[:, :, None]), state.it.x)
                state = eqx.tree_at(lambda t: (t.it.x, t.args), state, (new_x, new_args))
                # the solver's scatter into the solution buffer
                sol = state.it.x[:, :cp.nx, 0]
                solved = (term_codes == 1) | (term_codes == 5)
                pos = jnp.where(solved, jnp.cumsum(solved) - 1, N)
                buf = jnp.zeros((N, cp.nx)).at[pos].set(sol, mode="drop")
            return state, rng_key, term_codes, buf

        _step = eqx.filter_jit(step, donate="none")
        t0 = time()
        st, k2, tc, _ = _step(state, key); jax.block_until_ready(st)
        print(f"[jit] step compiled+run in {time()-t0:.1f}s", flush=True)
        spx_after_jit = spx('after warm+jit')
        # instruction name -> (op_name scope path, custom_call_target) from the compiled HLO,
        # so trace events can be attributed even when the profiler drops the tf_op stat
        hlo_txt = _step.lower(state, key).compile().compiled.as_text()
        opmap = hlo_map(hlo_txt)
        _td = os.path.join(REPO, "tests", "rebuttal", "logs", f"ktrace_{label}")
        os.makedirs(_td, exist_ok=True)
        with open(os.path.join(_td, "compiled_hlo.txt"), "w") as _f:
            _f.write(hlo_txt)  # kept so the trace can be re-attributed offline
        n_scoped = sum(1 for v in opmap.values() if "K_" in v[0])
        print(f"[hlo] {len(opmap)} instructions, {n_scoped} carry a K_ scope, "
              f"{sum(1 for v in opmap.values() if v[1])} custom calls", flush=True)

        # -- wall timing of the fused body (no profiler)
        walls = []
        s_, k_ = state, key
        for i in range(args.timing_iters):
            t0 = perf_counter()
            s_, k_, tc, _ = _step(s_, k_); jax.block_until_ready(s_)
            walls.append(perf_counter() - t0)
        walls = np.array(walls)
        spx_after_timing = spx('after timing loop')
        print(f"[wall] fused iteration: median {np.median(walls)*1e3:.2f} ms, "
              f"min {walls.min()*1e3:.2f}, max {walls.max()*1e3:.2f} (n={len(walls)})", flush=True)

        # -- profiler trace
        tracedir = os.path.join(REPO, "tests", "rebuttal", "logs", f"ktrace_{label}")
        os.makedirs(tracedir, exist_ok=True)
        s_, k_ = state, key
        with jax.profiler.trace(tracedir):
            for i in range(args.profile_iters):
                s_, k_, tc, _ = _step(s_, k_)
            jax.block_until_ready(s_)
        xs = sorted(glob.glob(os.path.join(tracedir, "**", "*.xplane.pb"), recursive=True),
                    key=os.path.getmtime)
        assert xs, f"no xplane.pb under {tracedir}"
        spx_after_profile = spx('after profile loop')
        if spx_after_profile['rebuild_count'] != spx_after_timing['rebuild_count']:
            print('[SPINEAX] WARNING: registry rebuilds (cuDSS re-analysis) occurred inside the profiled iterations', flush=True)
        agg, custom, n_events, dev_total, top = parse_trace(xs[-1], args.profile_iters, opmap)
        # per-iteration ms
        regions = {k: v / args.profile_iters / 1e6 for k, v in agg.items()}
        custom_ms = {k: v[0] / args.profile_iters / 1e6 for k, v in custom.items()}
        custom_n = {k: v[1] / args.profile_iters for k, v in custom.items()}
        dev_ms = dev_total / args.profile_iters / 1e6
        bu = sum(regions.get(k, 0.0) for k in BRANCH_UNIQUE)
        la = sum(regions.get(k, 0.0) for k in LINALG)
        print(f"[dev] device time per fused iteration {dev_ms:.2f} ms "
              f"({n_events/args.profile_iters:.0f} kernels/iter); wall {np.median(walls)*1e3:.2f} ms")
        for k, _ in REGION_ORDER:
            if k in regions:
                print(f"   {k:24s} {regions[k]:8.3f} ms  {100*regions[k]/dev_ms:5.1f}%")
        print(f"   linear algebra (LS+IC+solves) {la:.3f} ms = {100*la/dev_ms:.1f}%")
        print(f"   branch-unique work            {bu:.3f} ms = {100*bu/dev_ms:.1f}%  -> fusion tax "
              f"{100*bu/dev_ms:.1f}% of device time")
        for k in sorted(custom_ms, key=lambda k: -custom_ms[k])[:12]:
            print(f"   custom-call {k:40s} {custom_n[k]:6.2f}/iter  {custom_ms[k]:8.3f} ms")

        out = dict(label=label, scenario=args.scenario, variant=str(variant), N_batch=N,
                   warm_solves=warm, profile_iters=args.profile_iters,
                   timing_iters=args.timing_iters,
                   wall_ms=dict(median=float(np.median(walls)), min=float(walls.min()),
                                max=float(walls.max()), all=(walls * 1e3).tolist()),
                   device_ms_per_iter=dev_ms, kernels_per_iter=n_events / args.profile_iters,
                   regions_ms=regions, custom_calls_ms=custom_ms, custom_calls_per_iter=custom_n,
                   branch_unique_ms=bu, linalg_ms=la,
                   fusion_tax_frac=bu / dev_ms if dev_ms else None,
                   top_kernels=top[:40],
                   iter_count_range=[int(state.iter_count.min()), int(state.iter_count.max())],
                   spineax=dict(after_jit=spx_after_jit, after_timing=spx_after_timing, after_profile=spx_after_profile),
                   metadata=dict(jaxipm_commit=_git(REPO, "rev-parse", "HEAD"),
                                 jaxipm_dirty=_git(REPO, "status", "--porcelain") != "",
                                 spineax_commit=_git("/home/john/code/spineax", "rev-parse", "HEAD"),
                                 hostname=socket.gethostname(), gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                                 jax=jax.__version__, params=p, argv=sys.argv, xplane=xs[-1]))
        os.makedirs(RES, exist_ok=True)
        fn = os.path.join(RES, f"K_fusion_profile_{label}.json")
        with open(fn, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print(f"saved {fn}", flush=True)


def hlo_map(txt):
    """{instruction_name: (op_name, custom_call_target)} from HLO text."""
    out = {}
    pat = re.compile(r'^\s*%?([\w.\-]+)\s*=\s*(.*)$')
    for line in txt.splitlines():
        m = pat.match(line)
        if not m:
            continue
        name, rest = m.group(1), m.group(2)
        mo = re.search(r'op_name="([^"]+)"', rest)
        mc = re.search(r'custom_call_target="([^"]+)"', rest)
        out[name] = (mo.group(1) if mo else "", mc.group(1) if mc else "")
    return out


def parse_trace(path, n_iters, opmap=None):
    """Aggregate GPU kernel time by region.

    XLA kernels are named after their HLO fusion instruction (e.g. loop_concatenate_fusion_23)
    and are attributed through the compiled-HLO op_name metadata (K_ scopes). cuDSS library
    kernels (cudss::factorize_*, fwd_ker/bwd_ker, symbolic-analysis kernels) carry no HLO
    metadata and are classified by name."""
    from jax.profiler import ProfileData
    opmap = opmap or {}
    pd = ProfileData.from_file(path)
    agg = collections.Counter(); custom = {}; n_events = 0; dev_total = 0
    per_kernel = collections.Counter()
    gpu_planes = [pl for pl in pd.planes if "GPU" in pl.name and "Host" not in pl.name]
    if not gpu_planes:
        raise RuntimeError("no GPU plane in " + path + " planes=" + str([pl.name for pl in pd.planes]))

    def lookup(name):
        # XLA names the fusion instruction `loop_fusion.52` but the launched kernel
        # `loop_fusion_52` (dots are not legal in kernel symbols): map `_N` -> `.N`.
        # (Earlier version stripped the suffix, which either fell through to
        # "unscoped" or matched the *unsuffixed* instruction -- a different fusion.)
        for cand in (name, re.sub(r"_(\d+)$", r".\1", name)):
            if cand in opmap:
                return opmap[cand]
        return None

    for pl in gpu_planes:
        for ln in pl.lines:
            for ev in ln.events:
                name = ev.name; dur = ev.duration_ns
                dev_total += dur; n_events += 1
                low = name.lower()
                if "cudss::" in name or any(k in low for k in CUDSS_ANALYSIS) or "cusparse" in low or "cusolver" in low:
                    if "factorize" in low:
                        region = "cudss_factorize"
                    elif "fwd_ker" in low or "bwd_ker" in low or "solve" in low or "trsm" in low:
                        region = "cudss_solve"
                    else:
                        region = "cudss_analysis"
                    short = re.sub(r"<.*", "", name).replace("void ", "")
                    t, c = custom.get(short, (0, 0)); custom[short] = (t + dur, c + 1)
                elif low.startswith(("memcpy", "memset")) or "memcpy" in low:
                    region = "memcpy"
                else:
                    hm = lookup(name)
                    m = re.findall(r"K_([a-z_]+)", hm[0]) if hm else []
                    inner = [x for x in m if x not in ("search", "post")]
                    region = inner[-1] if inner else (m[-1] if m else "unscoped")
                agg[region] += dur
                per_kernel[(region, name[:80])] += dur
    top = [dict(region=r, kernel=k, ms_per_iter=v / n_iters / 1e6)
           for (r, k), v in per_kernel.most_common(60)]
    return agg, custom, n_events, dev_total, top


# ----------------------------------------------------------------------------
def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(RES, "K_fusion_profile_*.json")))
    if not files:
        print("no K_fusion_profile_*.json"); return
    runs = [json.load(open(f)) for f in files]
    lines = ["# Run K — fusion cost profile of one fused iteration", "",
             "Device (GPU) time per fused iteration attributed to solver regions via "
             "`jax.named_scope` markers (jaxipm/utils/kscope.py) and the XLA profiler; "
             "cuDSS custom calls counted per iteration. Wall = median wall-clock of the "
             "jitted fused body (block_until_ready). Fusion tax = share of device time spent "
             "on branch-unique work (SFR trial, WD/SOC init, SOC loop, backtracking loop) that an "
             "oracle body told 'full step' would skip.", ""]
    hdr = "| region | " + " | ".join(f"{r['label']}" for r in runs) + " |"
    lines += [hdr, "|---|" + "---|" * len(runs)]
    for key, name in REGION_ORDER:
        vals = []
        for r in runs:
            v = r["regions_ms"].get(key)
            vals.append("–" if v is None else f"{v:.3f} ms ({100*v/r['device_ms_per_iter']:.1f}%)")
        if any(v != "–" for v in vals):
            lines.append(f"| {name} (`{key}`) | " + " | ".join(vals) + " |")
    lines.append("| **device total / iteration** | " + " | ".join(f"**{r['device_ms_per_iter']:.2f} ms**" for r in runs) + " |")
    lines.append("| wall / iteration (median) | " + " | ".join(f"{r['wall_ms']['median']*1e3:.2f} ms" for r in runs) + " |")
    lines.append("| linear algebra (LS + IC loop + solves) | " + " | ".join(f"{r['linalg_ms']:.2f} ms ({100*r['linalg_ms']/r['device_ms_per_iter']:.1f}%)" for r in runs) + " |")
    lines.append("| branch-unique work = fusion tax | " + " | ".join(f"{r['branch_unique_ms']:.2f} ms ({100*r['fusion_tax_frac']:.1f}%)" for r in runs) + " |")
    lines.append("| kernels / iteration | " + " | ".join(f"{r['kernels_per_iter']:.0f}" for r in runs) + " |")
    lines += ["", "## cuDSS custom calls per fused iteration (count, device ms)", ""]
    keys = sorted({k for r in runs for k in r["custom_calls_per_iter"]})
    lines += ["| target | " + " | ".join(r["label"] for r in runs) + " |", "|---|" + "---|" * len(runs)]
    for k in keys:
        lines.append(f"| `{k}` | " + " | ".join(
            f"{r['custom_calls_per_iter'].get(k, 0):.2f} × ({r['custom_calls_ms'].get(k, 0):.3f} ms)" for r in runs) + " |")
    lines += ["", "Warm state: `solve_throughput` run for `warm_solves` solutions before profiling "
              "(members at mixed iteration counts, range " +
              ", ".join(f"{r['label']}: {r['iter_count_range']}" for r in runs) + ")."]
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "K_fusion_profile.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # stacked bar
    groups = [("derivatives", ["pp_derivs"]), ("KKT assembly", ["kkt_assembly"]),
              ("factorization (IC loop) + LS", ["ic_loop", "ls_mult_factor_solve"]),
              ("triangular solves", ["solve_cen", "solve_newton"]),
              ("scalar quantities / mu / transforms", ["pp_quantities", "pp_mu_update", "pre_mu_misc", "post_mu_rhs", "post_mu_transform"]),
              ("first trial evaluation", ["search_ls_first"]),
              ("branch-unique (SFR/WD/SOC/BT)", sorted(BRANCH_UNIQUE)),
              ("select / state write / restart", ["search_select", "pp_select_term", "pp_resto_init", "hr_restart", "unscoped"])]
    cols = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#F0E442", "#CC79A7", "#999999"]
    fig, ax = plt.subplots(figsize=(1.2 + 1.1 * len(runs), 3.6))
    bottom = np.zeros(len(runs))
    for (gname, keys_), col in zip(groups, cols):
        vals = np.array([sum(r["regions_ms"].get(k, 0.0) for k in keys_) for r in runs])
        ax.bar(range(len(runs)), vals, bottom=bottom, color=col, label=gname, width=0.6)
        bottom += vals
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([r["label"].replace("_", "\n") for r in runs], fontsize=7)
    ax.set_ylabel("device time per fused iteration [ms]")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"K_fusion_profile.{ext}"), dpi=200)
    print("figure -> tests/rebuttal/figures/K_fusion_profile.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["nav", "track", "multi"])
    ap.add_argument("--variants", type=str, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--warm-solves", type=int, default=None)
    ap.add_argument("--profile-iters", type=int, default=10)
    ap.add_argument("--timing-iters", type=int, default=30)
    ap.add_argument("--gpu", type=str, default="1")
    ap.add_argument("--out-suffix", type=str, default="")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        assert args.scenario, "--scenario required"
        profile(args)


if __name__ == "__main__":
    main()
