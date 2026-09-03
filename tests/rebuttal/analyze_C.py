"""Run C -- the ablation ladder: N=1 rung, ILB-off (WS) and ILB-on (HR) batch
sweeps, peak memory; plus the fresh-process B repeats. Reads the harness npz
files (jaxipm_ablation_<tag>_<mode>_ir0_b<N>[_fresh_rep*]_results.npz) and the
2026-08-10/12 ILB-on sweeps. Run from the repo root: python -m tests.rebuttal.analyze_C
"""
import glob, os, re
import numpy as np
RES = os.path.join(os.path.dirname(__file__), "results")
SC = [("nav 90°", "tests/quad_nav_circle/logs", "sector90", "jaxipm_batch_sweep_sector90_results.fullsweep-bak.npz"),
      ("nav 180°", "tests/quad_nav_circle/logs", "sector180", "jaxipm_batch_sweep_sector180_results.fullsweep-bak.npz"),
      ("track v̄=1.0", "tests/quad_track_avoid/logs", "v1.0", "jaxipm_batch_sweep_v1.0_results.npz"),
      ("track v̄=1.5", "tests/quad_track_avoid/logs", "v1.5", "jaxipm_batch_sweep_v1.5_results.npz"),
      ("track v̄=2.0", "tests/quad_track_avoid/logs", "v2.0", "jaxipm_batch_sweep_v2.0_results.npz"),
      ("multi N=2", "tests/quad_multi_swap/logs", "2", "jaxipm_batch_sweep_2_results.npz"),
      ("multi N=4", "tests/quad_multi_swap/logs", "4", "jaxipm_batch_sweep_4_results.npz")]
def ms(v): v = np.asarray(v, float); return f"{v.mean():.2f} ± {v.std(ddof=1) if v.size > 1 else 0:.2f}" if v.size else "–"
def main():
    os.makedirs(RES, exist_ok=True)
    md = ["# Run C — ablation ladder and batch sweeps\n"]
    md.append("## Ladder (solves/s): (0) IPOPT P=1 pinned · (1) jaxipm N=1 · (2) fusion + solve-level batching (ILB off, paper N) · (3) + ILB (paper N)\n")
    md.append("| config | (0) IPOPT P=1 (tol 1e-6) | (1) N=1 ILB-on | (1') N=1 ILB-off | (2) ILB-off in-proc / fresh | (3) ILB-on (A) | (2)/(1) | (3)/(2) | (3)/(1) | peak mem (3) |\n|---|---|---|---|---|---|---|---|---|---|")
    sweep_rows = ["\n## Batch sweeps: throughput [solves/s] (peak device MiB) — ILB-off from this campaign (3 reps), ILB-on from 2026-08-10/12 (1 rep) and this campaign where present\n"]
    for label, logs, tag, old_sweep in SC:
        Nfile = {"sector90": 500, "sector180": 500, "2": 75, "4": 75}.get(tag, 250)
        def thr(pattern, key="rep_throughput"):
            fs = sorted(glob.glob(os.path.join(logs, pattern)))
            return np.concatenate([np.asarray(np.load(f, allow_pickle=True)[key]).reshape(-1) for f in fs]) if fs else np.array([])
        def mem(pattern):
            fs = sorted(glob.glob(os.path.join(logs, pattern)))
            return max([int(np.load(f, allow_pickle=True)["peak_mem_bytes"][0]) for f in fs if "peak_mem_bytes" in np.load(f, allow_pickle=True).files] + [-1])
        ip = (glob.glob(os.path.join(logs, f"casadi_cpu_throughput_{tag}_phys_c001_tol1e-08_results.npz"))
              or glob.glob(os.path.join(logs, f"casadi_cpu_throughput_{tag}_phys_c001_results.npz")))  # tol 1e-8 rerun preferred
        ip = float(np.load(ip[0])["throughput"][0]) if ip else float("nan")
        n1h = thr(f"jaxipm_ablation_{tag}_hr_ir0_b1_results.npz"); n1w = thr(f"jaxipm_ablation_{tag}_ws_ir0_b1_results.npz")
        B = thr(f"jaxipm_ablation_{tag}_ws_ir0_b{Nfile}_results.npz"); Bf = thr(f"jaxipm_ablation_{tag}_ws_ir0_b{Nfile}_fresh_rep*_results.npz")
        A = thr(f"jaxipm_variance_{tag}_fresh_rep*_results.npz")
        r21 = f"{B.mean()/n1h.mean():.2f}" if B.size and n1h.size else "–"; r32 = f"{A.mean()/B.mean():.2f}" if A.size and B.size else "–"; r31 = f"{A.mean()/n1h.mean():.1f}" if A.size and n1h.size else "–"
        md.append(f"| {label} | {ip:.2f} | {ms(n1h)} | {ms(n1w)} | {ms(B)} / {ms(Bf)} | {ms(A)} | {r21} | {r32} | {r31} | – |")
        # sweeps
        rows = {}
        for f in sorted(glob.glob(os.path.join(logs, f"jaxipm_ablation_{tag}_ws_ir0_b*_results.npz"))):
            m = re.search(r"_b(\d+)_results\.npz$", f)
            if not m: continue
            z = np.load(f, allow_pickle=True); N = int(m.group(1))
            rows.setdefault(N, {})["ws"] = (ms(z["rep_throughput"]), int(z["peak_mem_bytes"][0]) // 2**20 if "peak_mem_bytes" in z.files else -1, int(np.asarray(z["rep_n_collected"]).min()), int(z["N_RUNS"][0]))
        for f in sorted(glob.glob(os.path.join(logs, f"jaxipm_ablation_{tag}_hr_ir0_b*_results.npz"))):
            m = re.search(r"_b(\d+)_results\.npz$", f)
            if not m: continue
            z = np.load(f, allow_pickle=True); N = int(m.group(1))
            rows.setdefault(N, {})["hr"] = (ms(z["rep_throughput"]), int(z["peak_mem_bytes"][0]) // 2**20 if "peak_mem_bytes" in z.files else -1)
        old = os.path.join(logs, old_sweep)
        if os.path.exists(old):
            z = np.load(old)
            for N, t in zip(z["batch_sizes"], z["throughput"]): rows.setdefault(int(N), {})["hr_aug"] = f"{t:.2f}"
        sweep_rows.append(f"\n### {label}\n\n| N | ILB-off (WS) solves/s | WS collected/target | WS peak MiB | ILB-on (HR) this campaign | HR peak MiB | ILB-on Aug-10/12 |\n|---|---|---|---|---|---|---|")
        for N in sorted(rows):
            r = rows[N]; ws = r.get("ws"); hr = r.get("hr")
            sweep_rows.append(f"| {N} | {ws[0] if ws else '–'} | {f'{ws[2]}/{ws[3]}' if ws else '–'} | {ws[1] if ws else '–'} | {hr[0] if hr else '–'} | {hr[1] if hr else '–'} | {r.get('hr_aug', '–')} |")
    open(os.path.join(RES, "C_ladder_and_sweeps.md"), "w").write("\n".join(md + sweep_rows) + "\n")
    print("\n".join(md)); print("\n".join(sweep_rows))
if __name__ == "__main__":
    main()
