"""Generate interactive HTML visualizations of reasoning traces.

Two views:
  1. Overview — horizontal bead sequences for all problems, compare at a glance
  2. Detail  — click a problem row to read the full trace chunk-by-chunk
"""

import argparse
import json
import numpy as np
from pathlib import Path
from html import escape

# ── Google-style Material palette for function tags ─────────────────────────

TAG_COLORS = {
    "case_representation":        "#4285F4",
    "knowledge_activation":       "#34A853",
    "hypothesis_generation":      "#FBBC04",
    "evidence_evaluation":        "#FA7B17",
    "reconsideration_uncertainty": "#EA4335",
    "diagnostic_verification":    "#A142F4",
    "diagnostic_commitment":      "#00897B",
    "final_answer_emission":      "#E91E63",
    "problem_setup":              "#4285F4",
    "plan_generation":            "#FBBC04",
    "fact_retrieval":             "#34A853",
    "active_computation":         "#FA7B17",
    "result_consolidation":       "#00897B",
    "uncertainty_management":     "#EA4335",
    "self_checking":              "#A142F4",
    "unknown":                    "#9E9E9E",
}

TAG_SHORT = {
    "case_representation":        "Case",
    "knowledge_activation":       "Knowledge",
    "hypothesis_generation":      "Hypothesis",
    "evidence_evaluation":        "Evidence",
    "reconsideration_uncertainty": "Uncertainty",
    "diagnostic_verification":    "Verification",
    "diagnostic_commitment":      "Commitment",
    "final_answer_emission":      "Answer",
    "problem_setup":              "Setup",
    "plan_generation":            "Plan",
    "fact_retrieval":             "Fact",
    "active_computation":         "Compute",
    "result_consolidation":       "Consolidate",
    "uncertainty_management":     "Uncertainty",
    "self_checking":              "Check",
    "unknown":                    "Unknown",
}

TAG_DISPLAY = {
    "case_representation":        "Case Representation",
    "knowledge_activation":       "Knowledge Activation",
    "hypothesis_generation":      "Hypothesis Generation",
    "evidence_evaluation":        "Evidence Evaluation",
    "reconsideration_uncertainty": "Reconsideration / Uncertainty",
    "diagnostic_verification":    "Diagnostic Verification",
    "diagnostic_commitment":      "Diagnostic Commitment",
    "final_answer_emission":      "Final Answer",
    "problem_setup":              "Problem Setup",
    "plan_generation":            "Plan Generation",
    "fact_retrieval":             "Fact Retrieval",
    "active_computation":         "Active Computation",
    "result_consolidation":       "Result Consolidation",
    "uncertainty_management":     "Uncertainty Management",
    "self_checking":              "Self Checking",
    "unknown":                    "Unknown",
}

# All metrics available per chunk, with human descriptions
METRICS = [
    ("forced_importance_accuracy",      "Forced Importance (Accuracy)",
     "Accuracy change at the next position when the model is forced to commit to a final answer after each chunk. "
     "Computed as forced_acc(chunk_i+1) - forced_acc(chunk_i)."),
    ("forced_importance_kl",            "Forced Importance (KL)",
     "KL divergence between the forced-answer distributions at adjacent chunks. "
     "Measures how much the distribution of final answers shifts between this step and the next under forced answering."),
    ("resampling_importance_accuracy",  "Resampling Importance (Accuracy)",
     "Accuracy change at the next position after resampling (regenerating) this chunk. "
     "Computed as accuracy(chunk_i+1) - accuracy(chunk_i)."),
    ("resampling_importance_kl",        "Resampling Importance (KL)",
     "KL divergence between the answer distributions of resampled vs original continuations. "
     "Measures how much regenerating this step shifts the distribution of final answers."),
    ("counterfactual_importance_accuracy", "Counterfactual Importance (Accuracy)",
     "Accuracy difference between rollouts where the resampled chunk is semantically different vs similar to the original. "
     "Measures how much changing this step (to something 'different') affects accuracy."),
    ("counterfactual_importance_kl",    "Counterfactual Importance (KL)",
     "KL divergence between answer distributions of semantically dissimilar vs similar resampled continuations. "
     "Measures how much changing this step to something different shifts the final answer distribution."),
    ("different_trajectories_fraction", "Different Trajectories Fraction",
     "Fraction of rollouts where the resampled chunk is semantically different from the removed chunk "
     "(cosine similarity below threshold). Higher = the model often generates a different reasoning path."),
    ("overdeterminedness",              "Overdeterminedness",
     "1 - (unique resampled chunks / total resampled chunks). "
     "High values mean the model almost always regenerates the same text, suggesting this step is highly determined by context."),
    ("accuracy",                        "Accuracy",
     "Fraction of rollout continuations from this chunk onward that produce the correct final answer. "
     "This is the baseline performance at each position in the reasoning trace."),
]


def compute_z_scores(chunks, metric):
    values = [c.get(metric, 0.0) or 0.0 for c in chunks]
    arr = np.array(values)
    std = np.std(arr)
    if std < 1e-9:
        return [0.0] * len(values)
    return ((arr - np.mean(arr)) / std).tolist()


def load_problem_data(rollouts_dir, problem_idx):
    problem_dir = rollouts_dir / f"problem_{problem_idx}"
    labeled_file = problem_dir / "chunks_labeled.json"
    problem_file = problem_dir / "problem.json"
    base_file = problem_dir / "base_solution.json"
    if not labeled_file.exists():
        return None, None, None
    with open(labeled_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    problem = {}
    if problem_file.exists():
        with open(problem_file, "r", encoding="utf-8") as f:
            problem = json.load(f)
    base_solution = {}
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            base_solution = json.load(f)
    return chunks, problem, base_solution


def bead_radius(z_score):
    mag = min(abs(z_score), 4.0)
    return 6 + mag * 3


def _build_js(js_data, metrics_js, default_metric):
    """Return pure JS code as a string. No f-string — avoids brace/quote escaping."""
    return r"""
(function() {
    var blob = JSON.parse(document.getElementById('data-blob').textContent);
    var DATA = blob.data;
    var METRICS = blob.metrics;
    var TAG_COLORS = blob.tagColors;
    var TAG_SHORT = blob.tagShort;
    var DEFAULT_METRIC = blob.defaultMetric;

    var currentFilter = 'correct';
    var currentMetric = DEFAULT_METRIC;
    var activeDetail = null;

    // Populate global metric selector
    (function() {
        var sel = document.getElementById('global-metric');
        METRICS.forEach(function(m) {
            var opt = document.createElement('option');
            opt.value = m.key;
            opt.textContent = m.label;
            if (m.key === DEFAULT_METRIC) opt.selected = true;
            sel.appendChild(opt);
        });
        updateGlobalMetricDesc();
    })();

    function updateGlobalMetricDesc() {
        var desc = document.getElementById('global-metric-desc');
        for (var i = 0; i < METRICS.length; i++) {
            if (METRICS[i].key === currentMetric) {
                desc.textContent = METRICS[i].desc;
                return;
            }
        }
        desc.textContent = '';
    }

    window.changeGlobalMetric = function(metricKey) {
        currentMetric = metricKey;
        updateGlobalMetricDesc();
        renderOverview();
        if (activeDetail) {
            renderDetail(activeDetail.idx, activeDetail.traceType);
        }
    };

    function escapeHtml(s) {
        var d = document.createElement('div'); d.textContent = s; return d.innerHTML;
    }

    function computeZScores(chunks, metricKey) {
        var vals = chunks.map(function(c) { return c[metricKey] || 0; });
        var mean = vals.reduce(function(a,b) { return a+b; }, 0) / vals.length;
        var variance = vals.reduce(function(a,b) { return a + Math.pow(b-mean, 2); }, 0) / vals.length;
        var std = Math.sqrt(variance);
        if (std < 1e-9) return vals.map(function() { return 0; });
        return vals.map(function(v) { return (v - mean) / std; });
    }

    function beadRadius(z) {
        var mag = Math.min(Math.abs(z), 4);
        return 6 + mag * 3;
    }

    function renderOverview() {
        var container = document.getElementById('overview');
        container.innerHTML = '';
        DATA.forEach(function(p) {
            if (p.trace_type !== currentFilter) return;
            var zScores = computeZScores(p.chunks, currentMetric);
            var nAnchors = zScores.filter(function(z) { return Math.abs(z) > 1.5; }).length;

            var row = document.createElement('div');
            row.className = 'overview-row';
            row.addEventListener('click', function() { showDetail(p.idx, p.trace_type); });

            var label = document.createElement('div');
            label.className = 'row-label';
            var nick = p.nickname ? ' ' + p.nickname : '';
            label.innerHTML = 'P' + p.idx + nick +
                '<span class="row-meta">' + p.chunks.length + ' steps' +
                (nAnchors ? ', ' + nAnchors + ' anchors' : '') + '</span>';

            var track = document.createElement('div');
            track.className = 'bead-track';

            p.chunks.forEach(function(c, i) {
                var z = zScores[i];
                var r = beadRadius(z);
                var isAnchor = Math.abs(z) > 1.5;
                var bead = document.createElement('div');
                bead.className = 'bead';
                var imp = c[currentMetric] || 0;
                var glowColor = isAnchor ? (imp >= 0 ? '#34A85366' : '#EA433566') : '';
                var glow = isAnchor ? 'box-shadow: 0 0 ' + Math.round(r*0.7) + 'px ' + Math.round(r*0.4) + 'px ' + glowColor + ';' : '';
                bead.style.cssText = 'width:' + (r*2) + 'px; height:' + (r*2) + 'px; background:' + c.color + ';' + glow;
                bead.title = '#' + c.chunk_idx + ' ' + c.tag_short + ': ' + c.summary;
                (function(idx, tt, ci) {
                    bead.addEventListener('click', function(e) {
                        e.stopPropagation();
                        showDetail(idx, tt, ci);
                    });
                })(p.idx, p.trace_type, i);
                track.appendChild(bead);
            });

            row.appendChild(label);
            row.appendChild(track);
            container.appendChild(row);
        });
    }

    function showDetail(idx, traceType, chunkIndex) {
        activeDetail = {idx: idx, traceType: traceType};
        renderDetail(idx, traceType, chunkIndex);
    }

    function renderDetail(idx, traceType, scrollToChunk) {
        var container = document.getElementById('detail-container');
        var p = null;
        for (var i = 0; i < DATA.length; i++) {
            if (DATA[i].idx === idx && DATA[i].trace_type === traceType) { p = DATA[i]; break; }
        }
        if (!p) return;

        var key = idx + '-' + traceType;
        var selectedMetric = currentMetric;
        var zScores = computeZScores(p.chunks, selectedMetric);

        var gtAnswer = p.gt_answer || 'N/A';
        var modelAnswer = p.model_answer || 'N/A';
        var isCorrect = p.is_correct;

        // Chunk cards
        var cards = p.chunks.map(function(c, i) {
            var z = zScores[i];
            var imp = c[selectedMetric] || 0;
            var acc = c.accuracy || 0;
            var isAnchor = Math.abs(z) > 1.5;
            var barWidth = Math.max(3, Math.min(Math.round(Math.abs(z) * 3 + 2), 14));

            var tagBadges = c.tags.map(function(t) {
                var tc = TAG_COLORS[t] || TAG_COLORS['unknown'];
                var ts = TAG_SHORT[t] || t;
                return '<span class="detail-tag" style="background:' + tc + ';">' + ts + '</span>';
            }).join('');

            var anchorDir = imp >= 0 ? 'pos' : 'neg';
            var anchorHtml = isAnchor ? '<span class="detail-anchor ' + anchorDir + '">' + (imp >= 0 ? '+' : '-') + ' ANCHOR</span>' : '';
            var anchorClass = isAnchor ? ' is-anchor' : '';

            return '<div class="detail-card' + anchorClass + '" id="card-' + key + '-' + i + '">' +
                '<div class="accent-bar" style="background:' + c.color + '; width:' + barWidth + 'px;"></div>' +
                '<div class="card-body">' +
                '<div class="card-header">' +
                '<span class="card-idx">#' + c.chunk_idx + '</span>' +
                tagBadges + anchorHtml +
                '<div class="card-metrics">' +
                '<span class="cm imp ' + (imp >= 0 ? 'pos' : 'neg') + '">' + (imp >= 0 ? '+' : '') + (imp * 100).toFixed(1) + '%</span>' +
                '<span class="cm">acc ' + (acc * 100).toFixed(0) + '%</span>' +
                '<span class="cm">z ' + (z >= 0 ? '+' : '') + z.toFixed(2) + '</span>' +
                '</div></div>' +
                '<div class="card-text">' + escapeHtml(c.text) + '</div>' +
                '</div></div>';
        }).join('');

        var nickname = p.nickname ? ' — ' + escapeHtml(p.nickname) : '';
        var benchAcc = p.benchmark_accuracy != null ? '<div class="answer-box benchmark-acc"><div class="label">Benchmark Accuracy</div><div class="value">' + (p.benchmark_accuracy * 100).toFixed(0) + '%</div></div>' : '';

        container.innerHTML =
            '<div class="detail-panel active">' +
            '<div class="detail-header">' +
            '<h2>Problem ' + idx + nickname + ' <span class="trace-label">(' + traceType + ')</span></h2>' +
            '<div class="problem-text">' + escapeHtml(p.problem_text) + '</div>' +
            '<div class="answer-row">' +
            '<div class="answer-box correct-answer"><div class="label">Correct Answer</div><div class="value">' + escapeHtml(gtAnswer) + '</div></div>' +
            '<div class="answer-box model-answer' + (isCorrect ? '' : ' wrong') + '"><div class="label">Model Answer (' + traceType + ')</div><div class="value">' + escapeHtml(modelAnswer) + '</div></div>' +
            benchAcc +
            '</div>' +
            '</div>' +
            '<div class="detail-chunks">' + cards + '</div>' +
            '</div>';

        if (scrollToChunk !== undefined) {
            setTimeout(function() {
                var card = document.getElementById('card-' + key + '-' + scrollToChunk);
                if (card) {
                    card.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    card.classList.add('highlight');
                    setTimeout(function() { card.classList.remove('highlight'); }, 1500);
                }
            }, 50);
        } else {
            container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    window.filterTraces = function(filter) {
        currentFilter = filter;
        document.querySelectorAll('#trace-filter .toggle-btn').forEach(function(b) {
            b.classList.toggle('active', b.dataset.filter === filter);
        });
        renderOverview();
        if (activeDetail && activeDetail.traceType !== filter) {
            document.getElementById('detail-container').innerHTML = '';
            activeDetail = null;
        }
    };

    renderOverview();
})();
"""


def generate_html(all_problems, default_metric):
    # Collect all used tags
    used_tags = set()
    for p in all_problems:
        for c in p["chunks"]:
            for t in c.get("function_tags", ["unknown"]):
                used_tags.add(t)

    # Legend
    tag_items = ""
    for tag in sorted(used_tags):
        color = TAG_COLORS.get(tag, TAG_COLORS["unknown"])
        display = TAG_DISPLAY.get(tag, tag)
        tag_items += (
            f'<div class="legend-item">'
            f'<span class="legend-dot" style="background:{color};"></span>'
            f'<span>{display}</span></div>\n'
        )
    legend_items = (
        f'<div class="legend-section">{tag_items}</div>'
        f'<div class="legend-rule"><strong>Bead size</strong> = importance magnitude for selected metric</div>'
        f'<div class="legend-rule"><strong>Green glow</strong> = positive anchor (removing this step hurts accuracy — the step helps)</div>'
        f'<div class="legend-rule"><strong>Red glow</strong> = negative anchor (removing this step helps accuracy — the step hurts)</div>'
        f'<div class="legend-rule">Anchors have |z-score| &gt; 1.5 on the selected metric</div>'
    )

    # Serialize all problem data to JSON for JS reactivity
    js_data = []
    for p in all_problems:
        chunks_js = []
        for c in p["chunks"]:
            tags = c.get("function_tags", ["unknown"])
            primary_tag = tags[0] if tags else "unknown"
            chunk_js = {
                "chunk_idx": c.get("chunk_idx", 0),
                "text": c.get("chunk", ""),
                "summary": c.get("summary", ""),
                "tags": tags,
                "primary_tag": primary_tag,
                "color": TAG_COLORS.get(primary_tag, TAG_COLORS["unknown"]),
                "tag_short": TAG_SHORT.get(primary_tag, primary_tag),
            }
            # Add all metric values
            for metric_key, _, _ in METRICS:
                chunk_js[metric_key] = c.get(metric_key, 0.0) or 0.0
            chunks_js.append(chunk_js)

        js_data.append({
            "idx": p["idx"],
            "trace_type": p["trace_type"],
            "problem_text": p["problem_text"],
            "gt_answer": p["gt_answer"],
            "model_answer": p["model_answer"],
            "is_correct": p["is_correct"],
            "benchmark_accuracy": p["benchmark_accuracy"],
            "nickname": p["nickname"],
            "chunks": chunks_js,
        })

    # Metric definitions for JS
    metrics_js = []
    for key, label, desc in METRICS:
        metrics_js.append({"key": key, "label": label, "desc": desc})

    # Build JS code separately (no f-string to avoid escaping nightmares)
    js_code = _build_js(js_data, metrics_js, default_metric)
    data_blob = {
        "data": js_data,
        "metrics": metrics_js,
        "tagColors": TAG_COLORS,
        "tagShort": TAG_SHORT,
        "tagDisplay": TAG_DISPLAY,
        "defaultMetric": default_metric,
    }
    data_blob_json = json.dumps(data_blob)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reasoning Trace Visualization</title>
<style>
:root {{
    --bg: #FAFAFA; --card-bg: #FFFFFF; --text: #202124;
    --text-secondary: #5F6368; --border: #DADCE0;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.4em; font-weight: 500; margin-bottom: 4px; }}
.subtitle {{ font-size: 0.85em; color: var(--text-secondary); margin-bottom: 16px; }}

/* Legend */
.legend {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px; }}
.legend-section {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.78em; color: var(--text-secondary); }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
.legend-rule {{ font-size: 0.82em; color: var(--text-secondary); line-height: 1.5; }}
.legend-rule strong {{ color: var(--text); }}

/* Controls bar */
.controls {{ display: flex; align-items: center; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
.toggle-group {{ display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
.toggle-btn {{
    padding: 6px 16px; border: none; background: var(--card-bg);
    font-size: 0.82em; cursor: pointer; transition: all 0.15s;
    border-right: 1px solid var(--border); color: var(--text);
}}
.toggle-btn:last-child {{ border-right: none; }}
.toggle-btn.active {{ background: #202124; color: white; }}
.toggle-btn:hover:not(.active) {{ background: #E8F0FE; }}

/* Overview */
.section-title {{ font-size: 0.9em; font-weight: 500; color: var(--text-secondary); margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.overview-row {{
    display: flex; align-items: center; gap: 12px;
    padding: 14px 12px; margin-bottom: 6px;
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
    cursor: pointer; transition: background 0.15s;
    overflow: visible;
}}
.overview-row:hover {{ background: #E8F0FE; }}
.overview-row.hidden {{ display: none; }}
.row-label {{ min-width: 160px; font-size: 0.85em; font-weight: 500; display: flex; flex-direction: column; }}
.row-meta {{ font-size: 0.75em; color: var(--text-secondary); font-weight: 400; }}
.bead-track {{ display: flex; align-items: center; gap: 3px; flex-wrap: nowrap; overflow-x: auto; overflow-y: visible; padding: 10px 0; }}
.bead {{ border-radius: 50%; flex-shrink: 0; transition: transform 0.15s; }}
.bead:hover {{ transform: scale(1.4); z-index: 10; }}

/* Detail panel */
.detail-panel {{ display: none; margin-bottom: 40px; }}
.detail-panel.active {{ display: block; }}
.detail-header {{ margin-bottom: 16px; }}
.detail-header h2 {{ font-size: 1.2em; font-weight: 500; margin-bottom: 8px; }}
.trace-label {{ color: var(--text-secondary); font-weight: 400; font-size: 0.85em; }}
.problem-text {{
    font-size: 0.88em; line-height: 1.6; color: var(--text-secondary);
    background: var(--card-bg); padding: 12px 16px; border-radius: 8px;
    border: 1px solid var(--border); max-height: 160px; overflow-y: auto;
    margin-bottom: 10px;
}}
.answer-row {{
    display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap;
}}
.answer-box {{
    font-size: 0.85em; padding: 8px 14px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--card-bg);
}}
.answer-box .label {{ font-weight: 600; color: var(--text-secondary); font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.3px; }}
.answer-box .value {{ margin-top: 2px; }}
.answer-box.correct-answer {{ border-left: 3px solid #34A853; }}
.answer-box.model-answer {{ border-left: 3px solid #4285F4; }}
.answer-box.model-answer.wrong {{ border-left-color: #EA4335; }}
.answer-box.benchmark-acc {{ border-left: 3px solid #FA7B17; }}

/* Metric selector */
.metric-control {{ display: flex; flex-direction: column; gap: 2px; }}
.metric-select-wrap {{ margin-bottom: 12px; }}
.metric-select {{
    padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.85em; background: var(--card-bg); color: var(--text);
    cursor: pointer; min-width: 280px;
}}
.metric-desc {{ font-size: 0.8em; color: var(--text-secondary); margin-top: 4px; font-style: italic; }}

/* Chunk cards */
.detail-chunks {{ display: flex; flex-direction: column; gap: 2px; }}
.detail-card {{
    display: flex; background: var(--card-bg);
    border: 1px solid var(--border); border-radius: 6px;
    overflow: hidden; transition: box-shadow 0.15s;
}}
.detail-card:hover {{ box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
.detail-card.is-anchor {{ border: 1px solid #EA433588; }}
.accent-bar {{ flex-shrink: 0; border-radius: 6px 0 0 6px; }}
.card-body {{ padding: 14px 18px; flex: 1; min-width: 0; }}
.card-header {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }}
.card-idx {{ font-weight: 600; font-size: 0.88em; color: var(--text-secondary); min-width: 30px; }}
.detail-tag {{ font-size: 0.72em; padding: 3px 10px; border-radius: 10px; color: white; font-weight: 500; }}
.detail-anchor {{ font-size: 0.72em; padding: 3px 10px; border-radius: 10px; color: white; font-weight: 700; }}
.detail-anchor.pos {{ background: #34A853; }}
.detail-anchor.neg {{ background: #EA4335; }}
.card-metrics {{ margin-left: auto; display: flex; gap: 10px; }}
.cm {{ font-size: 0.82em; font-family: 'Roboto Mono', monospace; color: var(--text-secondary); }}
.cm.imp {{ font-weight: 500; }}
.cm.imp.pos {{ color: #1B7D2F; }}
.cm.imp.neg {{ color: #C5221F; }}
.card-text {{ font-size: 0.95em; line-height: 1.65; color: var(--text); }}
.detail-card.highlight {{ background: #FFF9C4; transition: background 0.6s; }}
</style>
</head>
<body>
<div class="container">
    <h1>Reasoning Trace Visualization</h1>
    <p class="subtitle">Click a row to read the full trace. Hover beads for details.</p>
    <div class="legend">{legend_items}</div>

    <div class="controls">
        <div class="toggle-group" id="trace-filter">
            <button class="toggle-btn active" data-filter="correct" onclick="filterTraces('correct')">Correct</button>
            <button class="toggle-btn" data-filter="incorrect" onclick="filterTraces('incorrect')">Incorrect</button>
        </div>
        <div class="metric-control">
            <select class="metric-select" id="global-metric" onchange="changeGlobalMetric(this.value)"></select>
            <div class="metric-desc" id="global-metric-desc"></div>
        </div>
    </div>

    <div class="section-title">Overview</div>
    <div class="overview" id="overview"></div>

    <div class="section-title">Detail</div>
    <div id="detail-container"></div>
</div>

<script id="data-blob" type="application/json">{data_blob_json}</script>
<script>{js_code}</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reasoning trace visualization")
    parser.add_argument("-ic", "--correct_dir", type=str, default=None)
    parser.add_argument("-ii", "--incorrect_dir", type=str, default=None)
    parser.add_argument("-p", "--problems", type=str, required=True, help="Comma-separated problem indices")
    parser.add_argument("-o", "--output_dir", type=str, default="analysis/visualizations")
    parser.add_argument("-im", "--importance_metric", type=str, default="resampling_importance_accuracy")
    args = parser.parse_args()

    problem_indices = [int(x.strip()) for x in args.problems.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    correct_dir = Path(args.correct_dir) if args.correct_dir else None
    incorrect_dir = Path(args.incorrect_dir) if args.incorrect_dir else None

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
            problem_text = problem.get("problem", problem.get("question", ""))
            gt_answer = problem.get("gt_answer", problem.get("answer", ""))
            model_answer = base_solution.get("answer", "")
            is_correct = base_solution.get("is_correct", False)
            benchmark_accuracy = problem.get("accuracy", None)
            nickname = problem.get("nickname", "")

            z_scores = compute_z_scores(chunks, args.importance_metric)
            all_problems.append({
                "idx": idx,
                "trace_type": trace_type,
                "chunks": chunks,
                "z_scores": z_scores,
                "problem_text": problem_text,
                "gt_answer": gt_answer,
                "model_answer": model_answer,
                "is_correct": is_correct,
                "benchmark_accuracy": benchmark_accuracy,
                "nickname": nickname,
            })

    if not all_problems:
        print("No data found.")
        return

    html = generate_html(all_problems, args.importance_metric)
    out_file = output_dir / "traces.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_file} ({len(all_problems)} traces)")


if __name__ == "__main__":
    main()
