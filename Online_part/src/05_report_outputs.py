"""Results table for the report, in Markdown and LaTeX.

Usage:  python src/05_report_outputs.py
"""
import json
import os

import common

RES = os.path.join(common.ROOT, "results")


if __name__ == "__main__":
    p = os.path.join(RES, "estimation.json")
    if not os.path.exists(p):
        raise SystemExit("results/estimation.json missing: run "
                         "src/04_estimate_angle.py first")
    data = json.load(open(p))

    hdr = ["Observation", "Estimated flexion", "RMSE [mm]"]
    rows = [[name,
             f"{r['flexion_best']:.0f}°",
             f"{r['rmse_best']*1000:.2f}"]
            for name, r in data.items()]

    md = ["| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(RES, "tabella_risultati.md"), "w").write("\n".join(md))

    tex = ["\\begin{tabular}{lrr}", "\\toprule",
           " & ".join(hdr) + " \\\\", "\\midrule"]
    tex += [" & ".join(c.replace("°", "$^\\circ$") for c in r) + " \\\\"
            for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(RES, "tabella_risultati.tex"), "w").write("\n".join(tex))

    print("\n".join(md))
