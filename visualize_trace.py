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
# Sign convention: positive = chunk helps accuracy, negative = chunk hurts.
# Counterfactual metrics have the opposite raw sign, so they are negated at display time.
METRICS = [
    ("forced_importance_accuracy",      "Forced Importance (Accuracy)",
     "How much the forced-answer accuracy changes after seeing this chunk. "
     "Positive (green) = this chunk moves the model toward the correct answer. "
     "Negative (red) = this chunk moves the model away from the correct answer."),
    ("forced_importance_kl",            "Forced Importance (KL)",
     "KL divergence between forced-answer distributions at adjacent chunks. "
     "Measures how much the model's answer distribution shifts after seeing this chunk. "
     "Higher magnitude = more influential step (always non-negative)."),
    ("resampling_importance_accuracy",  "Resampling Importance (Accuracy)",
     "How much accuracy changes when this chunk is kept vs resampled (regenerated). "
     "Positive (green) = keeping this chunk helps accuracy; resampling it would hurt. "
     "Negative (red) = keeping this chunk hurts accuracy; resampling it would help."),
    ("resampling_importance_kl",        "Resampling Importance (KL)",
     "KL divergence between answer distributions when resampling at this position vs the next. "
     "Measures how much regenerating this step shifts the final answer distribution. "
     "Higher magnitude = more influential step (always non-negative)."),
    ("counterfactual_importance_accuracy", "Counterfactual Importance (Accuracy)",
     "Accuracy impact of this chunk, measured by generating semantically different alternatives "
     "and comparing final-answer accuracy against the original trace. "
     "Positive (green) = this chunk helps accuracy; replacing it with something different would hurt. "
     "Negative (red) = this chunk hurts accuracy; replacing it with something different would help. "
     "(Sign is flipped from raw metric so that positive = helps, consistent with other metrics.)"),
    ("counterfactual_importance_kl",    "Counterfactual Importance (KL)",
     "KL divergence between answer distributions of semantically different vs similar resampled continuations. "
     "Measures how much replacing this chunk with something different shifts the final answer distribution. "
     "Higher magnitude = more influential step (always non-negative)."),
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

# Metrics where the raw sign means "chunk hurts" — negate at display time
# so that positive = helps, consistent with all other metrics.
FLIP_SIGN_METRICS = {"counterfactual_importance_accuracy"}


def discover_problems(*dirs):
    """Find all problem indices across the given directories."""
    indices = set()
    for d in dirs:
        if d is None or not d.exists():
            continue
        for child in d.iterdir():
            if child.is_dir() and child.name.startswith("problem_"):
                try:
                    indices.add(int(child.name.split("_", 1)[1]))
                except ValueError:
                    pass
    return sorted(indices)


def compute_z_scores(chunks, metric):
    values = [c.get(metric, 0.0) or 0.0 for c in chunks]
    arr = np.array(values)
    std = np.std(arr)
    if std < 1e-9:
        return [0.0] * len(values)
    return ((arr - np.mean(arr)) / std).tolist()


def load_problem_data(rollouts_dir, problem_idx, forced_dir=None):
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

    # Load solution samples for each chunk
    forced_problem_dir = (forced_dir / f"problem_{problem_idx}") if forced_dir else None
    for c in chunks:
        cidx = c.get("chunk_idx", 0)
        # Regular rollout samples (resampled chunks)
        sol_file = problem_dir / f"chunk_{cidx}" / "solutions.json"
        c["_sample_resampled"] = []
        c["_sample_dissimilar"] = []
        if sol_file.exists():
            with open(sol_file, "r", encoding="utf-8") as f:
                sols = json.load(f)
            # Collect unique resampled chunks
            seen = set()
            for s in sols:
                txt = s.get("chunk_resampled", "")
                if txt and txt not in seen:
                    seen.add(txt)
                    c["_sample_resampled"].append(txt)
                    if len(c["_sample_resampled"]) >= 5:
                        break
        # Forced answer samples
        c["_sample_forced"] = []
        if forced_problem_dir:
            forced_sol_file = forced_problem_dir / f"chunk_{cidx}" / "solutions.json"
            if forced_sol_file.exists():
                with open(forced_sol_file, "r", encoding="utf-8") as f:
                    forced_sols = json.load(f)
                seen_f = set()
                for s in forced_sols:
                    ans = s.get("answer", "")
                    if ans and ans not in seen_f:
                        seen_f.add(ans)
                        c["_sample_forced"].append(ans)
                        if len(c["_sample_forced"]) >= 5:
                            break

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
        return 5;
    }

    var tooltipEl = document.getElementById('bead-tooltip');
    var tooltipTimeout = null;

    function truncate(s, n) { return s.length > n ? s.substring(0, n) + '...' : s; }

    function showTooltip(e, chunk) {
        clearTimeout(tooltipTimeout);
        var html = '<strong>#' + chunk.chunk_idx + ' ' + escapeHtml(chunk.tag_short) + '</strong><br>' + escapeHtml(truncate(chunk.text, 200));
        tooltipEl.innerHTML = html;
        tooltipEl.classList.add('visible');
        positionTooltip(e);
    }

    function positionTooltip(e) {
        var x = e.clientX + 16;
        var y = e.clientY + 16;
        var w = tooltipEl.offsetWidth;
        var h = tooltipEl.offsetHeight;
        if (x + w > window.innerWidth - 20) x = e.clientX - w - 16;
        if (y + h > window.innerHeight - 20) y = e.clientY - h - 16;
        tooltipEl.style.left = x + 'px';
        tooltipEl.style.top = y + 'px';
    }

    function hideTooltip() {
        tooltipTimeout = setTimeout(function() {
            tooltipEl.classList.remove('visible');
        }, 100);
    }

    function buildBeadRow(p) {
        var zScores = computeZScores(p.chunks, currentMetric);
        var nAnchors = zScores.filter(function(z) { return Math.abs(z) > 1.5; }).length;

        var row = document.createElement('div');
        row.className = 'overview-row';
        row.addEventListener('click', function() { showDetail(p.idx, p.trace_type); });

        var label = document.createElement('div');
        label.className = 'row-label';
        var nick = p.nickname ? ' ' + p.nickname : '';
        var typeTag = currentFilter === 'both' ? ' <span class="row-type ' + p.trace_type + '">' + p.trace_type.charAt(0).toUpperCase() + '</span>' : '';
        label.innerHTML = 'P' + p.idx + nick + typeTag +
            '<span class="row-meta">' + p.chunks.length + ' steps' +
            (nAnchors ? ', ' + nAnchors + ' anchors' : '') + '</span>';

        var track = document.createElement('div');
        track.className = 'bead-track';

        p.chunks.forEach(function(c, i) {
            var z = zScores[i];
            var isAnchor = Math.abs(z) > 1.5;
            var imp = c[currentMetric] || 0;
            var mag = Math.min(Math.abs(z), 3);
            var d = 10 + Math.round(mag * 4);
            var bead = document.createElement('div');
            bead.className = 'bead';
            var glow = '';
            if (isAnchor) {
                var glowColor = imp >= 0 ? 'rgba(52,168,83,0.5)' : 'rgba(234,67,53,0.5)';
                glow = 'box-shadow: 0 0 6px 2px ' + glowColor + ';';
            }
            bead.style.cssText = 'width:' + d + 'px; height:' + d + 'px; background:' + c.color + ';' + glow;
            (function(idx, tt, ci, chunk) {
                bead.addEventListener('click', function(e) {
                    e.stopPropagation();
                    showDetail(idx, tt, ci);
                });
                bead.addEventListener('mouseenter', function(e) { showTooltip(e, chunk); });
                bead.addEventListener('mousemove', function(e) { positionTooltip(e); });
                bead.addEventListener('mouseleave', hideTooltip);
            })(p.idx, p.trace_type, i, c);
            track.appendChild(bead);
        });

        var trackWrap = document.createElement('div');
        trackWrap.className = 'bead-track-wrap';
        var hint = document.createElement('div');
        hint.className = 'scroll-hint';
        trackWrap.appendChild(track);
        trackWrap.appendChild(hint);

        (function(tw, tr) {
            function checkOverflow() {
                if (tr.scrollWidth > tw.offsetWidth + 2) {
                    tw.classList.add('has-overflow');
                } else {
                    tw.classList.remove('has-overflow');
                }
                if (tr.scrollLeft + tr.offsetWidth >= tr.scrollWidth - 4) {
                    tw.classList.add('scrolled-end');
                } else {
                    tw.classList.remove('scrolled-end');
                }
            }
            tr.addEventListener('scroll', checkOverflow);
            setTimeout(checkOverflow, 50);
        })(trackWrap, track);

        row.appendChild(label);
        row.appendChild(trackWrap);
        return row;
    }

    function renderOverview() {
        var container = document.getElementById('overview');
        container.innerHTML = '';

        if (currentFilter === 'both') {
            var problemIndices = [];
            var seen = {};
            DATA.forEach(function(p) {
                if (!seen[p.idx]) { seen[p.idx] = true; problemIndices.push(p.idx); }
            });
            problemIndices.forEach(function(idx) {
                var group = document.createElement('div');
                group.className = 'overview-group';

                // Problem label on the left
                var problemLabel = document.createElement('div');
                problemLabel.className = 'group-label';
                var nick = '';
                DATA.forEach(function(p) { if (p.idx === idx && p.nickname) nick = p.nickname; });
                problemLabel.innerHTML = '<strong>P' + idx + '</strong>' + (nick ? '<br><span class="group-nick">' + escapeHtml(nick) + '</span>' : '');

                var tracesCol = document.createElement('div');
                tracesCol.className = 'group-traces';

                ['correct', 'incorrect'].forEach(function(tt) {
                    var p = null;
                    DATA.forEach(function(d) { if (d.idx === idx && d.trace_type === tt) p = d; });
                    if (!p) return;

                    var traceRow = document.createElement('div');
                    traceRow.className = 'group-trace-row';
                    traceRow.addEventListener('click', function() { showDetail(p.idx, p.trace_type); });

                    var zScores = computeZScores(p.chunks, currentMetric);
                    var nAnchors = zScores.filter(function(z) { return Math.abs(z) > 1.5; }).length;

                    var badge = document.createElement('div');
                    badge.className = 'group-type-badge ' + tt;
                    badge.innerHTML = (tt === 'correct' ? 'C' : 'I') + '<span class="group-steps">' + p.chunks.length + ' chunks<br>' + nAnchors + ' anchors</span>';
                    var track = document.createElement('div');
                    track.className = 'bead-track';

                    p.chunks.forEach(function(c, i) {
                        var z = zScores[i];
                        var isAnchor = Math.abs(z) > 1.5;
                        var imp = c[currentMetric] || 0;
                        var mag = Math.min(Math.abs(z), 3);
                        var d = 10 + Math.round(mag * 4);
                        var bead = document.createElement('div');
                        bead.className = 'bead';
                        var glow = '';
                        if (isAnchor) {
                            var glowColor = imp >= 0 ? 'rgba(52,168,83,0.5)' : 'rgba(234,67,53,0.5)';
                            glow = 'box-shadow: 0 0 6px 2px ' + glowColor + ';';
                        }
                        bead.style.cssText = 'width:' + d + 'px; height:' + d + 'px; background:' + c.color + ';' + glow;
                        (function(pidx, ptt, ci, chunk) {
                            bead.addEventListener('click', function(e) {
                                e.stopPropagation();
                                showDetail(pidx, ptt, ci);
                            });
                            bead.addEventListener('mouseenter', function(e) { showTooltip(e, chunk); });
                            bead.addEventListener('mousemove', function(e) { positionTooltip(e); });
                            bead.addEventListener('mouseleave', hideTooltip);
                        })(p.idx, p.trace_type, i, c);
                        track.appendChild(bead);
                    });

                    var trackWrap = document.createElement('div');
                    trackWrap.className = 'bead-track-wrap';
                    var hint = document.createElement('div');
                    hint.className = 'scroll-hint';
                    trackWrap.appendChild(track);
                    trackWrap.appendChild(hint);
                    (function(tw, tr) {
                        function checkOverflow() {
                            tw.classList.toggle('has-overflow', tr.scrollWidth > tw.offsetWidth + 2);
                            tw.classList.toggle('scrolled-end', tr.scrollLeft + tr.offsetWidth >= tr.scrollWidth - 4);
                        }
                        tr.addEventListener('scroll', checkOverflow);
                        setTimeout(checkOverflow, 50);
                    })(trackWrap, track);

                    traceRow.appendChild(badge);
                    traceRow.appendChild(trackWrap);
                    tracesCol.appendChild(traceRow);
                });

                group.appendChild(problemLabel);
                group.appendChild(tracesCol);
                container.appendChild(group);
            });
        } else {
            DATA.forEach(function(p) {
                if (p.trace_type !== currentFilter) return;
                container.appendChild(buildBeadRow(p));
            });
        }
    }

    function showDetail(idx, traceType, chunkIndex) {
        activeDetail = {idx: idx, traceType: traceType};
        renderDetail(idx, traceType, chunkIndex);
    }

    var selectedCardIdx = null;

    function populateSamplesPanel(panel, chunk) {
        var isForced = currentMetric.indexOf('forced') !== -1;
        var isCounterfactual = currentMetric.indexOf('counterfactual') !== -1;

        var html = '<div class="sp-title">#' + chunk.chunk_idx + ' ' + escapeHtml(chunk.tag_short) + '</div>';

        if (isForced && chunk.sample_forced && chunk.sample_forced.length > 0) {
            html += '<div class="sp-section">Sample forced answers</div>';
            chunk.sample_forced.forEach(function(s) {
                html += '<div class="sp-sample">' + escapeHtml(s) + '</div>';
            });
        }

        if (chunk.sample_resampled && chunk.sample_resampled.length > 0) {
            var label = isCounterfactual ? 'Sample alternative chunks' : 'Sample resampled chunks';
            html += '<div class="sp-section">' + label + '</div>';
            chunk.sample_resampled.forEach(function(s) {
                html += '<div class="sp-sample">' + escapeHtml(s) + '</div>';
            });
        }

        if ((!chunk.sample_resampled || chunk.sample_resampled.length === 0) &&
            (!chunk.sample_forced || chunk.sample_forced.length === 0)) {
            html += '<div class="sp-section">No rollout samples available</div>';
        }

        panel.innerHTML = html;
        panel.classList.add('visible');
        var layout = panel.closest('.detail-layout');
        if (layout) layout.classList.add('has-panel');
    }

    function alignPanelToCard(cardEl) {
        var panel = document.getElementById('samples-panel');
        var layout = cardEl.closest('.detail-layout');
        if (!layout || !panel) return;
        var layoutRect = layout.getBoundingClientRect();
        var cardRect = cardEl.getBoundingClientRect();
        panel.style.top = (cardRect.top - layoutRect.top) + 'px';
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
        selectedCardIdx = null;

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

            return '<div class="detail-card' + anchorClass + '" data-chunk-idx="' + i + '" id="card-' + key + '-' + i + '">' +
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
            '<div class="detail-layout">' +
            '<div class="detail-chunks-col"><div class="detail-chunks">' + cards + '</div></div>' +
            '<div class="samples-panel" id="samples-panel"></div>' +
            '</div>' +
            '</div>';

        // Wire up hover + click handlers on detail cards
        var detailCards = container.querySelectorAll('.detail-card');
        var hoverTimeout = null;
        detailCards.forEach(function(cardEl) {
            var ci = parseInt(cardEl.getAttribute('data-chunk-idx'));
            cardEl.addEventListener('mouseenter', function() {
                if (selectedCardIdx !== null) return; // pinned — don't override
                clearTimeout(hoverTimeout);
                var panel = document.getElementById('samples-panel');
                populateSamplesPanel(panel, p.chunks[ci]);
                alignPanelToCard(cardEl);
            });
            cardEl.addEventListener('mouseleave', function() {
                if (selectedCardIdx !== null) return;
                hoverTimeout = setTimeout(function() {
                    var panel = document.getElementById('samples-panel');
                    panel.classList.remove('visible');
                    var layout = panel.closest('.detail-layout');
                    if (layout) layout.classList.remove('has-panel');
                }, 200);
            });
            cardEl.addEventListener('click', function() {
                clearTimeout(hoverTimeout);
                detailCards.forEach(function(c) { c.classList.remove('selected'); });
                var panel = document.getElementById('samples-panel');
                if (selectedCardIdx === ci) {
                    selectedCardIdx = null;
                    panel.classList.remove('visible');
                    var layout = panel.closest('.detail-layout');
                    if (layout) layout.classList.remove('has-panel');
                    return;
                }
                selectedCardIdx = ci;
                cardEl.classList.add('selected');
                populateSamplesPanel(panel, p.chunks[ci]);
                alignPanelToCard(cardEl);
                cardEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });

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
        if (filter !== 'both' && activeDetail && activeDetail.traceType !== filter) {
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
        f'<div class="legend-rule"><strong>Green glow / + ANCHOR</strong> = this chunk helps accuracy (positive importance)</div>'
        f'<div class="legend-rule"><strong>Red glow / - ANCHOR</strong> = this chunk hurts accuracy (negative importance)</div>'
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
            # Add all metric values (negate flipped metrics so positive = helps)
            for metric_key, _, _ in METRICS:
                val = c.get(metric_key, 0.0) or 0.0
                if metric_key in FLIP_SIGN_METRICS:
                    val = -val
                chunk_js[metric_key] = val
            # Add solution samples for tooltips
            chunk_js["sample_resampled"] = c.get("_sample_resampled", [])
            chunk_js["sample_forced"] = c.get("_sample_forced", [])
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
    display: flex; align-items: center; gap: 8px;
    padding: 6px 12px; margin-bottom: 6px;
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
    cursor: pointer; transition: background 0.15s;
    overflow: visible;
}}
.overview-row:hover {{ background: #E8F0FE; }}
.overview {{ display: flex; flex-direction: column; }}
.overview-group {{
    display: flex; align-items: stretch; gap: 0;
    margin-bottom: 6px; background: var(--card-bg);
    border: 1px solid var(--border); border-radius: 8px;
    cursor: pointer;
}}
.overview-group:hover {{ background: #E8F0FE; }}
.group-label {{
    width: 140px; flex-shrink: 0; padding: 10px 12px;
    font-size: 0.85em; display: flex; flex-direction: column; justify-content: center;
    border-right: 1px solid var(--border); line-height: 1.4;
}}
.group-nick {{ font-size: 0.85em; color: var(--text-secondary); font-weight: 400; }}
.group-traces {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
.group-trace-row {{
    display: flex; align-items: center; gap: 6px;
    padding: 4px 8px; cursor: pointer;
}}
.group-trace-row + .group-trace-row {{ border-top: 1px solid #ECECEC; }}
.group-trace-row:hover {{ background: rgba(66,133,244,0.06); }}
.group-type-badge {{
    flex-shrink: 0; text-align: center;
    font-size: 0.75em; font-weight: 700; padding: 4px 8px;
    border-radius: 4px; color: white; line-height: 1.3;
    min-width: 60px;
}}
.group-type-badge.correct {{ background: #81C995; }}
.group-type-badge.incorrect {{ background: #E8938A; }}
.group-steps {{ display: block; font-size: 0.85em; font-weight: 400; opacity: 0.85; }}
.overview-row.hidden {{ display: none; }}
.row-label {{ width: 140px; flex-shrink: 0; font-size: 0.85em; font-weight: 500; display: flex; flex-direction: column; }}
.row-meta {{ font-size: 0.75em; color: var(--text-secondary); font-weight: 400; }}
.bead-track-wrap {{
    position: relative; flex: 1; min-width: 0; overflow-x: clip; overflow-y: visible;
}}
.bead-track {{
    display: flex; align-items: center; gap: 3px; flex-wrap: nowrap;
    overflow-x: auto; padding: 6px 20px 6px 0;
    scrollbar-width: none;
}}
.bead-track::-webkit-scrollbar {{ display: none; }}
.scroll-hint {{
    position: absolute; right: 0; top: 0; bottom: 0; width: 20px;
    background: linear-gradient(to right, transparent, rgba(0,0,0,0.06));
    display: flex; align-items: center; justify-content: center;
    pointer-events: none; opacity: 0; transition: opacity 0.2s;
}}
.scroll-hint::after {{
    content: '\u203A'; font-size: 18px; color: #5F6368; font-weight: 700;
}}
.bead-track-wrap.has-overflow .scroll-hint {{ opacity: 1; }}
.bead-track-wrap.scrolled-end .scroll-hint {{ opacity: 0; }}
.bead {{ border-radius: 50%; flex: none; transition: transform 0.15s; cursor: pointer; position: relative; }}
.bead:hover {{ transform: scale(1.4); z-index: 10; }}

/* Tooltip */
.bead-tooltip {{
    display: none; position: fixed; z-index: 1000;
    background: white; border: 1px solid var(--border); border-radius: 8px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15); padding: 10px 14px;
    max-width: 420px; font-size: 0.82em; line-height: 1.5;
    pointer-events: none;
}}
.bead-tooltip.visible {{ display: block; }}

/* Detail panel */
.detail-panel {{ display: none; margin-bottom: 40px; }}
.detail-panel.active {{ display: block; }}
.detail-header {{ margin-bottom: 16px; }}
.detail-header h2 {{ font-size: 1.2em; font-weight: 500; margin-bottom: 8px; }}
.trace-label {{ color: var(--text-secondary); font-weight: 400; font-size: 0.85em; }}
.problem-text {{
    font-size: 0.88em; line-height: 1.6; color: var(--text-secondary);
    background: var(--card-bg); padding: 12px 16px; border-radius: 8px;
    border: 1px solid var(--border);
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
.detail-card.selected {{ background: #E8F0FE; }}

/* Detail layout: chunks + samples panel */
.detail-layout {{ position: relative; padding-right: 356px; }}
.detail-layout:not(.has-panel) {{ padding-right: 0; }}
.detail-chunks-col {{ }}
.samples-panel {{
    position: absolute; top: 0; right: 0;
    width: 340px;
    background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px; font-size: 0.85em; max-height: 50vh; overflow-y: auto;
    display: none;
}}
.samples-panel.visible {{ display: block; }}
.samples-panel .sp-title {{ font-weight: 600; margin-bottom: 8px; font-size: 0.95em; }}
.samples-panel .sp-section {{ font-weight: 600; color: var(--text-secondary); font-size: 0.8em; margin: 10px 0 4px; text-transform: uppercase; letter-spacing: 0.3px; }}
.samples-panel .sp-sample {{
    padding: 6px 8px; margin: 4px 0; background: #F8F9FA; border-radius: 4px;
    border-left: 3px solid var(--border); font-size: 0.92em; line-height: 1.5;
}}
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
            <button class="toggle-btn" data-filter="both" onclick="filterTraces('both')">Both</button>
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
<div class="bead-tooltip" id="bead-tooltip"></div>

<script id="data-blob" type="application/json">{data_blob_json}</script>
<script>{js_code}</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML reasoning trace visualization")
    parser.add_argument("-ic", "--correct_dir", type=str, default=None)
    parser.add_argument("-ii", "--incorrect_dir", type=str, default=None)
    parser.add_argument("-icf", "--correct_forced_dir", type=str, default=None)
    parser.add_argument("-iif", "--incorrect_forced_dir", type=str, default=None)
    parser.add_argument("-p", "--problems", type=str, default=None, help="Comma-separated problem indices (default: auto-discover all)")
    parser.add_argument("-o", "--output_dir", type=str, default="analysis/visualizations")
    parser.add_argument("-im", "--importance_metric", type=str, default="counterfactual_importance_accuracy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    correct_dir = Path(args.correct_dir) if args.correct_dir else None
    incorrect_dir = Path(args.incorrect_dir) if args.incorrect_dir else None
    correct_forced_dir = Path(args.correct_forced_dir) if args.correct_forced_dir else None
    incorrect_forced_dir = Path(args.incorrect_forced_dir) if args.incorrect_forced_dir else None

    if args.problems:
        problem_indices = [int(x.strip()) for x in args.problems.split(",")]
    else:
        problem_indices = discover_problems(correct_dir, incorrect_dir)
        print(f"Auto-discovered {len(problem_indices)} problems: {problem_indices}")

    all_problems = []

    for idx in problem_indices:
        for trace_type, rdir, fdir in [("correct", correct_dir, correct_forced_dir), ("incorrect", incorrect_dir, incorrect_forced_dir)]:
            if rdir is None:
                continue
            chunks, problem, base_solution = load_problem_data(rdir, idx, forced_dir=fdir)
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

    # Also generate print version
    from visualize_trace_print import generate_print_html
    print_html = generate_print_html(all_problems)
    print_file = output_dir / "traces_print.html"
    with open(print_file, "w", encoding="utf-8") as f:
        f.write(print_html)
    print(f"Wrote {print_file} ({len(all_problems)} traces)")


if __name__ == "__main__":
    main()
