"""Generate a print-friendly HTML visualization of reasoning traces.

Designed for browser Print > Save as PDF. No JavaScript — pure CSS with
@media print rules and page breaks between problems.
"""

import argparse
import json
from pathlib import Path
from html import escape

from visualize_trace import (
    TAG_COLORS,
    TAG_SHORT,
    TAG_DISPLAY,
    METRICS,
    FLIP_SIGN_METRICS,
    compute_z_scores,
    discover_problems,
    load_problem_data,
)

DISPLAY_METRICS = [
    ("resampling_importance_accuracy",      "Resamp"),
    ("counterfactual_importance_accuracy",   "Countf"),
    ("forced_importance_accuracy",           "Forced"),
]


def _metric_label(key):
    for k, label, _ in METRICS:
        if k == key:
            return label
    return key


def _get_importance(chunk, metric_key):
    """Get importance value with sign flip applied."""
    val = chunk.get(metric_key, 0.0) or 0.0
    if metric_key in FLIP_SIGN_METRICS:
        val = -val
    return val


def generate_print_html(all_problems):
    # Collect used tags for legend
    used_tags = set()
    for p in all_problems:
        for c in p["chunks"]:
            for t in c.get("function_tags", ["unknown"]):
                used_tags.add(t)

    # Legend HTML
    legend_items = ""
    for tag in sorted(used_tags):
        color = TAG_COLORS.get(tag, TAG_COLORS["unknown"])
        display = TAG_DISPLAY.get(tag, tag)
        legend_items += (
            f'<span class="legend-item">'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f'{escape(display)}</span>\n'
        )

    # Problem sections
    problem_sections = ""
    for pi, p in enumerate(all_problems):
        chunks = p["chunks"]
        idx = p["idx"]
        trace_type = p["trace_type"]
        nickname = p.get("nickname", "")
        problem_text = p.get("problem_text", "")
        gt_answer = p.get("gt_answer", "N/A")
        model_answer = p.get("model_answer", "N/A")
        is_correct = p.get("is_correct", False)
        benchmark_accuracy = p.get("benchmark_accuracy", None)

        title_nick = f" — {escape(nickname)}" if nickname else ""
        title = f"Problem {idx}{title_nick} ({trace_type})"

        correct_class = "" if is_correct else " wrong"

        bench_html = ""
        if benchmark_accuracy is not None:
            bench_html = (
                f'<div class="meta-box bench">'
                f'<span class="meta-label">Benchmark Accuracy</span>'
                f'<span class="meta-value">{benchmark_accuracy * 100:.0f}%</span>'
                f'</div>'
            )

        # Chunk cards
        cards_html = ""
        for i, c in enumerate(chunks):
            acc = c.get("accuracy", 0.0) or 0.0
            tags = c.get("function_tags", ["unknown"])
            primary_tag = tags[0] if tags else "unknown"
            color = TAG_COLORS.get(primary_tag, TAG_COLORS["unknown"])
            text = c.get("chunk", "")

            # Tag badges
            tag_badges = ""
            for t in tags:
                tc = TAG_COLORS.get(t, TAG_COLORS["unknown"])
                ts = TAG_SHORT.get(t, t)
                tag_badges += f'<span class="tag-badge" style="background:{tc};">{ts}</span> '

            # Build stats for all three metrics
            stats_html = ""
            for metric_key, short_label in DISPLAY_METRICS:
                val = _get_importance(c, metric_key)
                cls = "pos" if val >= 0 else "neg"
                sign = "+" if val >= 0 else ""
                stats_html += (
                    f'<span class="stat {cls}">'
                    f'{short_label} {sign}{val * 100:.1f}%</span>'
                )
            stats_html += f'<span class="stat">acc {acc * 100:.0f}%</span>'

            # Anchor badge based on resampling importance (primary metric)
            resamp_val = _get_importance(c, "resampling_importance_accuracy")
            resamp_z_chunks = []
            for cc in chunks:
                resamp_z_chunks.append({"v": _get_importance(cc, "resampling_importance_accuracy")})
            z_vals = compute_z_scores(
                [{"resampling_importance_accuracy": cc["v"]} for cc in resamp_z_chunks],
                "resampling_importance_accuracy",
            )
            z = z_vals[i]
            is_anchor = abs(z) > 1.5

            anchor_html = ""
            if is_anchor:
                sign = "+" if resamp_val >= 0 else "-"
                anchor_cls = "pos" if resamp_val >= 0 else "neg"
                anchor_html = f'<span class="anchor-badge {anchor_cls}">{sign} ANCHOR</span>'

            anchor_card_cls = " is-anchor" if is_anchor else ""

            cards_html += f'''<div class="chunk-card{anchor_card_cls}">
  <div class="accent-bar" style="background:{color};"></div>
  <div class="card-body">
    <div class="card-header">
      <span class="card-idx">#{c.get("chunk_idx", i)}</span>
      {tag_badges}{anchor_html}
      <span class="card-stats">
        {stats_html}
      </span>
    </div>
    <div class="card-text">{escape(text)}</div>
  </div>
</div>
'''

        page_break = ' class="page-break"' if pi > 0 else ""
        problem_sections += f'''<section{page_break}>
  <h2>{escape(title)}</h2>
  <div class="problem-text">{escape(problem_text)}</div>
  <div class="meta-row">
    <div class="meta-box correct-ans">
      <span class="meta-label">Correct Answer</span>
      <span class="meta-value">{escape(str(gt_answer))}</span>
    </div>
    <div class="meta-box model-ans{correct_class}">
      <span class="meta-label">Model Answer</span>
      <span class="meta-value">{escape(str(model_answer))}</span>
    </div>
    {bench_html}
  </div>
  <div class="chunks">{cards_html}</div>
</section>
'''

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Reasoning Traces (Print)</title>
<style>
/* ── Base ─────────────────────────────────────────────── */
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #202124; font-size: 11pt; line-height: 1.5;
    padding: 24px 32px;
}}
h1 {{ font-size: 16pt; font-weight: 600; margin-bottom: 2px; }}
.subtitle {{ font-size: 9pt; color: #5F6368; margin-bottom: 12px; }}

/* ── Legend ────────────────────────────────────────────── */
.legend {{
    display: flex; flex-wrap: wrap; gap: 8px 14px;
    margin-bottom: 6px; padding: 8px 12px;
    border: 1px solid #DADCE0; border-radius: 6px;
    background: #F8F9FA;
}}
.legend-item {{
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 8.5pt; color: #5F6368;
}}
.legend-dot {{
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; flex-shrink: 0;
}}
.legend-note {{
    font-size: 8pt; color: #5F6368; margin-bottom: 16px;
    line-height: 1.4;
}}

/* ── Sections / page breaks ───────────────────────────── */
section {{ margin-bottom: 28px; }}
.page-break {{ page-break-before: always; }}

h2 {{
    font-size: 13pt; font-weight: 600;
    margin-bottom: 6px; padding-bottom: 4px;
    border-bottom: 2px solid #202124;
}}

.problem-text {{
    font-size: 9.5pt; line-height: 1.55; color: #5F6368;
    background: #F8F9FA; padding: 8px 12px; border-radius: 4px;
    border: 1px solid #DADCE0; margin-bottom: 8px;
    white-space: pre-wrap;
}}

/* ── Metadata row ─────────────────────────────────────── */
.meta-row {{
    display: flex; gap: 12px; margin-bottom: 10px; flex-wrap: wrap;
}}
.meta-box {{
    font-size: 9pt; padding: 4px 10px; border-radius: 4px;
    border: 1px solid #DADCE0; background: #FFF;
}}
.meta-label {{
    font-weight: 600; color: #5F6368; text-transform: uppercase;
    font-size: 7.5pt; letter-spacing: 0.3px; margin-right: 6px;
}}
.meta-box.correct-ans {{ border-left: 3px solid #34A853; }}
.meta-box.model-ans {{ border-left: 3px solid #4285F4; }}
.meta-box.model-ans.wrong {{ border-left-color: #EA4335; }}
.meta-box.bench {{ border-left: 3px solid #FA7B17; }}

/* ── Chunk cards ──────────────────────────────────────── */
.chunks {{ display: flex; flex-direction: column; gap: 3px; }}
.chunk-card {{
    display: flex; border: 1px solid #DADCE0; border-radius: 4px;
    overflow: hidden; break-inside: avoid;
}}
.chunk-card.is-anchor {{ border-color: #EA433588; }}
.accent-bar {{ width: 4px; flex-shrink: 0; }}
.card-body {{ padding: 6px 10px; flex: 1; min-width: 0; }}
.card-header {{
    display: flex; align-items: center; flex-wrap: wrap;
    gap: 4px; margin-bottom: 3px;
}}
.card-idx {{
    font-weight: 600; font-size: 9pt; color: #5F6368;
    min-width: 24px;
}}
.tag-badge {{
    font-size: 7pt; padding: 1px 7px; border-radius: 8px;
    color: white; font-weight: 600;
}}
.anchor-badge {{
    font-size: 7pt; padding: 1px 7px; border-radius: 8px;
    color: white; font-weight: 700;
}}
.anchor-badge.pos {{ background: #34A853; }}
.anchor-badge.neg {{ background: #EA4335; }}
.card-stats {{
    margin-left: auto; display: flex; gap: 8px;
    font-size: 8.5pt; font-family: 'Menlo', 'Consolas', monospace;
    color: #5F6368;
}}
.stat.pos {{ color: #1B7D2F; font-weight: 600; }}
.stat.neg {{ color: #C5221F; font-weight: 600; }}
.card-text {{
    font-size: 9.5pt; line-height: 1.55; color: #202124;
    white-space: pre-wrap;
}}

/* ── Print ────────────────────────────────────────────── */
@media print {{
    body {{ padding: 0; font-size: 10pt; }}
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
    color-adjust: exact !important;
    .page-break {{ page-break-before: always; }}
    .chunk-card {{ break-inside: avoid; }}
    .legend {{ break-inside: avoid; }}
}}
</style>
</head>
<body>

<h1>Reasoning Trace Visualization</h1>
<p class="subtitle">{len(all_problems)} trace(s)</p>

<div class="legend">
{legend_items}
</div>
<p class="legend-note">
  <strong>+ ANCHOR</strong> (green) = this chunk helps accuracy (|z| &gt; 1.5 on resampling importance) &nbsp;
  <strong>- ANCHOR</strong> (red) = this chunk hurts accuracy &nbsp;<br/>
  <strong>Resamp</strong> = Resampling importance: accuracy change when this chunk is kept vs regenerated &nbsp;<br/>
  <strong>Countf</strong> = Counterfactual importance: accuracy impact measured by generating semantically different alternatives &nbsp;<br/>
  <strong>Forced</strong> = Forced importance: how much forced-answer accuracy changes after seeing this chunk &nbsp;<br/>
  Sign convention: positive = chunk helps, negative = chunk hurts (consistent across all metrics)
</p>

{problem_sections}

</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(
        description="Generate print-friendly HTML reasoning trace visualization"
    )
    parser.add_argument("-ic", "--correct_dir", type=str, default=None)
    parser.add_argument("-ii", "--incorrect_dir", type=str, default=None)
    parser.add_argument(
        "-p", "--problems", type=str, default=None,
        help="Comma-separated problem indices (default: auto-discover all)",
    )
    parser.add_argument("-o", "--output_dir", type=str, default="analysis/visualizations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    correct_dir = Path(args.correct_dir) if args.correct_dir else None
    incorrect_dir = Path(args.incorrect_dir) if args.incorrect_dir else None

    if args.problems:
        problem_indices = [int(x.strip()) for x in args.problems.split(",")]
    else:
        problem_indices = discover_problems(correct_dir, incorrect_dir)
        print(f"Auto-discovered {len(problem_indices)} problems: {problem_indices}")

    all_problems = []

    for idx in problem_indices:
        for trace_type, rdir in [("correct", correct_dir), ("incorrect", incorrect_dir)]:
            if rdir is None:
                continue
            chunks, problem, base_solution = load_problem_data(rdir, idx)
            if chunks is None:
                continue
            problem = problem or {}
            base_solution = base_solution or {}

            all_problems.append({
                "idx": idx,
                "trace_type": trace_type,
                "chunks": chunks,
                "problem_text": problem.get("problem", problem.get("question", "")),
                "gt_answer": problem.get("gt_answer", problem.get("answer", "")),
                "model_answer": base_solution.get("answer", ""),
                "is_correct": base_solution.get("is_correct", False),
                "benchmark_accuracy": problem.get("accuracy", None),
                "nickname": problem.get("nickname", ""),
            })

    if not all_problems:
        print("No data found.")
        return

    html = generate_print_html(all_problems)
    out_file = output_dir / "traces_print.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_file} ({len(all_problems)} traces)")


if __name__ == "__main__":
    main()
