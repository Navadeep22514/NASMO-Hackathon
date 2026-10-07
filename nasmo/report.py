"""Hardware-aware performance report matching the certificates.

Generates a self-contained HTML document binding together:

* the device under test (profile + classifier-head overheads),
* the certified Pareto front (every point, all five objectives),
* the certificate digests (profile/space SHA-256, points digest),
* the independent verification result (per-check pass/fail).

The report reads the same in-memory structures that produced the
certificate, and re-states the digests so any drift from
``certs/front.json`` / ``certs/coverage.json`` is visible to a reader —
the report is evidence *about* the certificate, never a substitute for it.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from . import OBJECTIVE_ORDER, SENSE

_OBJ_LABELS = {
    "accuracy_bp": ("Accuracy", "%", 100.0),
    "latency_us": ("Latency", "ms", 1000.0),
    "energy_uj": ("Energy", "mJ", 1000.0),
    "peak_mem_bytes": ("Peak memory", "KB", 1024.0),
    "fairness_bp": ("Fairness gap", "pp", 100.0),
}


def _fmt(name: str, value) -> str:
    label, unit, div = _OBJ_LABELS[name]
    if isinstance(value, int):
        return f"{value / div:.2f} {unit}"
    return f"{value} {unit}"


def _row(cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


_CSS = """
:root{--navy:#0b1f3b;--gold:#f0a500;--ink:#1b2430;--mut:#5b6b7f;--bg:#f4f7fb;--ok:#0a7d3f;--bad:#b3261e}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
header{background:linear-gradient(105deg,var(--navy) 0%,#123a6b 70%,#0b1f3b 100%);color:#fff;padding:26px 40px;border-bottom:5px solid var(--gold)}
header h1{margin:0;font-size:30px;letter-spacing:3px}
header .sub{color:#ffd777;font-weight:600;letter-spacing:1px}
header .meta{margin-top:8px;font-size:13px;color:#c9d6ea}
main{max-width:1100px;margin:24px auto;padding:0 20px}
section{background:#fff;border:1px solid #dde5ef;border-radius:10px;padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 3px rgba(11,31,59,.06)}
h2{margin:0 0 14px;font-size:17px;color:var(--navy);border-left:4px solid var(--gold);padding-left:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th{background:var(--navy);color:#fff;text-align:left;padding:7px 9px;font-weight:600}
td{padding:6px 9px;border-bottom:1px solid #e7edf4}
tr:nth-child(even) td{background:#f8fafd}
.kv td:first-child{font-weight:600;color:var(--mut);width:280px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.card{background:#f8fafd;border:1px solid #e2e9f2;border-radius:8px;padding:12px 14px}
.card .big{font-size:22px;font-weight:700;color:var(--navy)}
.card .lab{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.valid{background:#e8f7ee;border:1px solid #9ed9b7;color:var(--ok);border-radius:8px;padding:12px 16px;font-weight:700;font-size:16px}
.invalid{background:#fdecea;border:1px solid #f2b8b5;color:var(--bad);border-radius:8px;padding:12px 16px;font-weight:700;font-size:16px}
.mono{font-family:Consolas,Menlo,monospace;font-size:12px;word-break:break-all}
.scroll{max-height:420px;overflow:auto;border:1px solid #e2e9f2;border-radius:8px}
.checks li{list-style:none;padding:3px 0}
.checks{padding-left:0;margin:0}
.p{font-weight:700;color:var(--ok)}.f{font-weight:700;color:var(--bad)}
footer{text-align:center;color:var(--mut);font-size:12px;padding:18px 0 30px}
"""


def generate_report(prob, points, meta, coverage, verify_result, out_path) -> Path:
    """Write the HTML report to ``out_path`` and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = prob.profile
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # summary cards over the front
    cards = []
    for name in OBJECTIVE_ORDER:
        vals = [p["objectives"][name] for p in points]
        lo, hi = min(vals), max(vals)
        label, unit, div = _OBJ_LABELS[name]
        direction = "higher is better" if SENSE[name] == "max" else "lower is better"
        cards.append(
            f'<div class="card"><div class="lab">{label} range</div>'
            f'<div class="big">{lo / div:.2f} – {hi / div:.2f} {unit}</div>'
            f'<div class="lab">{direction}</div></div>')

    # device table
    dev_rows = [
        _row(["Device", prof["device"]]),
        _row(["SoC", prof["soc"]]),
        _row(["Memory", prof["memory"]]),
        _row(["Runtime", prof["runtime"]]),
    ]
    head = prof["head"]
    dev_rows += [
        _row(["Classifier-head latency", _fmt("latency_us", head["latency_us"])]),
        _row(["Classifier-head energy", _fmt("energy_uj", head["energy_uj"])]),
        _row(["Classifier-head params", _fmt("peak_mem_bytes", head["params_bytes"])]),
        _row(["Classifier-head fairness", _fmt("fairness_bp", head["fairness_bp"])]),
    ]

    # lemma / coverage
    lem = coverage["lemma"]
    cov_rows = [
        _row(["Search space", prob.space["space_name"]]),
        _row(["Layer types", coverage["layer_type_count"]]),
        _row(["Capacity saturation c_sat", f'{lem["capacity_saturation"]} units']),
        _row(["a_min / a_max", f'{lem["a_min"]} / {lem["a_max"]}']),
        _row(["Certified cutoff D*", str(lem["d_star"])]),
        _row(["Stop rule", f'{meta["stop_reason"]} at depth {meta["stop_depth"]}']),
        _row(["Rounds executed", str(len(coverage["rounds"]))]),
        _row(["Front size (globally optimal architectures)", str(len(points))]),
    ]

    # verification
    if verify_result and verify_result.get("ok"):
        v_block = ('<div class="valid">✔ CERTIFICATE VALID — '
                   f'{verify_result["passed"]}/{verify_result["total"]} independent checks passed</div>')
    else:
        passed = verify_result["passed"] if verify_result else 0
        total = verify_result["total"] if verify_result else 0
        v_block = f'<div class="invalid">✖ CERTIFICATE NOT YET VERIFIED — {passed}/{total} checks passed</div>'
    v_items = "".join(
        f'<li><span class="{"p" if c["ok"] else "f"}">'
        f'{"PASS" if c["ok"] else "FAIL"}</span> — {html.escape(c["name"])}'
        + (f' <span class="mono">({html.escape(c["detail"])})</span>' if c["detail"] else "")
        + "</li>"
        for c in (verify_result or {}).get("checks", []))

    # full front table
    head_cells = ["ID", "Depth"] + [
        f'{_OBJ_LABELS[n][0]} ({_OBJ_LABELS[n][1]})' for n in OBJECTIVE_ORDER] + ["Architecture"]
    front_rows = []
    for p in points:
        o = p["objectives"]
        cells = [p["id"], str(p["depth"])]
        cells += [_fmt(n, o[n]) for n in OBJECTIVE_ORDER]
        cells.append(f'<span class="mono">{html.escape(" → ".join(p["arch"]))}</span>')
        front_rows.append(_row(cells))

    dig = coverage
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>NASMO Hardware-Aware Performance Report</title>
<style>{_CSS}</style></head><body>
<header>
  <h1>NASMO</h1>
  <div class="sub">Neural Architecture Search with Machine-checkable Optimality — Hardware-Aware Performance Report</div>
  <div class="meta">Generated {now} · objectives: {", ".join(OBJECTIVE_ORDER)}</div>
</header>
<main>

<section><h2>1. Certificate verification (independent re-derivation)</h2>
{v_block}
<ul class="checks">{v_items}</ul></section>

<section><h2>2. Certified front — objective ranges</h2>
<div class="grid">{''.join(cards)}</div></section>

<section><h2>3. Target hardware profile</h2>
<table>{''.join(dev_rows)}</table></section>

<section><h2>4. Search-space &amp; certificate coverage</h2>
<table>{''.join(cov_rows)}</table>
<p class="mono">points digest: {dig.get('points_digest')}<br>
profile sha256: {dig.get('profile_sha256')}<br>
space sha256: {dig.get('space_sha256')}</p></section>

<section><h2>5. Complete Pareto front — {len(points)} globally optimal architectures</h2>
<p>Every point below is non-dominated on all five objectives; the certificate
proves no architecture of any length outside this set dominates any point
below (see docs/THEORY.md).</p>
<div class="scroll"><table>
<thead>{_row(head_cells, "th")}</thead>
<tbody>{''.join(front_rows)}</tbody>
</table></div></section>

</main>
<footer>NASMO · report derived from certs/front.json + certs/coverage.json · verify independently with <span class="mono">python -m nasmo.verify certs</span></footer>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
