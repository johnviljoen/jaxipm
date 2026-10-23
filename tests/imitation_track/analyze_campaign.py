"""Aggregate the DAgger expert campaign (tests/imitation_track/logs/runs/<expert>/run<k>/).

Per run: total wall, labelling wall (sum of expert calls), training wall, final
held-out success / tail RMS / min clearance (student and expert), expert
label throughput per round, timing.json inference numbers. Prints a Markdown
summary (mean +- std over completed runs) and writes runs/campaign_summary.{md,json}.
"""

import glob
import json
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "runs")
EXPERTS = ["il", "madnlp", "sl"]
NAMES = dict(il="jaxipm-IL", madnlp="MadNLP", sl="jaxipm-SL")


def load_run(d):
    m = json.load(open(os.path.join(d, "metrics.json")))
    r = m["rounds"]; last = r[-1]
    out = dict(run=os.path.basename(d), seed=m["args"]["seed"], rounds_done=last["round"],
               aborted=m.get("aborted_round"), wall_total_s=m.get("wall_total_s"),
               label_wall_s=sum(x["label_wall_s"] for x in r), train_wall_s=sum(x["train_wall_s"] for x in r),
               round0_label_s=r[0]["label_wall_s"],
               dagger_label_s=sum(x["label_wall_s"] for x in r[1:]),
               solves_per_s=float(np.mean([x["solves_per_s"] for x in r[1:]])) if len(r) > 1 else np.nan,
               iters_mean=float(np.mean([x["iters_mean"] for x in r[1:] if "iters_mean" in x])) if len(r) > 1 else np.nan,
               labels=last["n_labels"], expert_success=r[0]["expert_success"],
               expert_rms=r[0]["expert_median_rms_tail"], expert_labelled_frac=r[0].get("expert_labelled_frac"),
               student_success=last["student_success"], student_rms=last["student_median_rms_tail"],
               student_clear=last["student_min_clearance"],
               best_success=max(x["student_success"] for x in r),
               success_by_round=[x["student_success"] for x in r])
    p = os.path.join(d, "figures", "render_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        out["render_student_success"] = s["student"]["success"]; out["render_expert_success"] = s["expert"]["success"]
    p = os.path.join(d, "timing.json")
    if os.path.exists(p):
        t = json.load(open(p))
        out["mlp_batched_ms"] = 1e3 * t["mlp_batched"]["mean_s"]; out["mlp_single_us"] = 1e6 * t["mlp_single"]["mean_s"]
        out["jaxipm_batched_s"] = t["jaxipm_batched"]["mean_s"]; out["jaxipm_single_ms"] = 1e3 * t["jaxipm_single"]["mean_s"]
    p = os.path.join(d, "done.json")
    if os.path.exists(p):
        out["done"] = json.load(open(p))
    return out


def ms(vals, fmt="{:.0f}"):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0: return "–"
    return (fmt + " ± " + fmt).format(v.mean(), v.std(ddof=1)) if v.size > 1 else fmt.format(v[0])


def main():
    table = {}
    for ex in EXPERTS:
        runs = []
        for d in sorted(glob.glob(os.path.join(RUNS, ex, "run*"))):
            if os.path.exists(os.path.join(d, "metrics.json")):
                try: runs.append(load_run(d))
                except Exception as e: print(f"skip {d}: {e}")
        table[ex] = runs
    lines = ["# DAgger track-avoid expert campaign", "",
             "| expert | runs | total wall [s] | labelling [s] | round-0 (closed-loop) [s] | DAgger rounds label [s] | training [s] | solves/s (rounds 1+) | iters | labels | expert held-out succ. | student held-out succ. /32 (final) | best round succ. | student tail RMS [m] | min clearance [m] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ex in EXPERTS:
        r = table[ex]
        g = lambda k, f="{:.0f}": ms([x.get(k) for x in r], f)
        lines.append(f"| {NAMES[ex]} | {len(r)} | {g('wall_total_s')} | {g('label_wall_s')} | {g('round0_label_s')} | {g('dagger_label_s')} | "
                     f"{g('train_wall_s')} | {g('solves_per_s', '{:.1f}')} | {g('iters_mean', '{:.1f}')} | {g('labels')} | {g('expert_success', '{:.1f}')} | "
                     f"{g('student_success', '{:.1f}')} | {g('best_success', '{:.1f}')} | {g('student_rms', '{:.2f}')} | {g('student_clear', '{:+.2f}')} |")
    lines += ["", "Per run (final student success by round):"]
    for ex in EXPERTS:
        for x in table[ex]:
            lines.append(f"- {NAMES[ex]} {x['run']} (seed {x['seed']}): wall {x['wall_total_s']:.0f}s, label {x['label_wall_s']:.0f}s, "
                         f"success by round {x['success_by_round']}, aborted={x['aborted']}")
    if any("mlp_single_us" in x for r in table.values() for x in r):
        lines += ["", "Inference timing (timing.json, MLP vs jaxipm-IL receding-horizon solve):"]
        for ex in EXPERTS:
            r = [x for x in table[ex] if "mlp_single_us" in x]
            if r:
                lines.append(f"- {NAMES[ex]} runs: MLP batched-1000 {ms([x['mlp_batched_ms'] for x in r], '{:.2f}')} ms, single {ms([x['mlp_single_us'] for x in r], '{:.0f}')} us; "
                             f"jaxipm batched-1000 {ms([x['jaxipm_batched_s'] for x in r], '{:.1f}')} s, single {ms([x['jaxipm_single_ms'] for x in r], '{:.0f}')} ms")
    md = "\n".join(lines)
    print(md)
    with open(os.path.join(RUNS, "campaign_summary.md"), "w") as fh: fh.write(md + "\n")
    with open(os.path.join(RUNS, "campaign_summary.json"), "w") as fh: json.dump(table, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
