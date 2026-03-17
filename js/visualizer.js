// ============================================================
//  visualizer.js — All Chart/Canvas Drawing Logic
//  Uses Chart.js (loaded via CDN in index.html)
// ============================================================

const Viz = (() => {
    // ─── COLOR PALETTE ─────────────────────────────────────
    const COLORS = {
        blue: '#4f8ef7',
        purple: '#9c6ef7',
        cyan: '#22d3ee',
        green: '#10b981',
        orange: '#f97316',
        pink: '#f472b6',
        yellow: '#fbbf24',
        red: '#f87171',
    };

    const CLUSTER_COLORS = {
        animals: '#10b981',
        food: '#f97316',
        tech: '#4f8ef7',
        emotion: '#f472b6',
        nature: '#22d3ee',
        actions: '#9c6ef7',
        words: '#94a3b8',
    };

    // Keep chart instances so we can destroy/redraw
    const _charts = {};

    // ─── HELPER: destroy old chart if exists ────────────────
    function destroyChart(id) {
        if (_charts[id]) {
            _charts[id].destroy();
            delete _charts[id];
        }
    }

    // ─── CHART 1: EMBEDDING SPACE (2D Scatter) ──────────────
    // Maps a curated set of words to 2D using projectTo2D()
    function renderEmbeddingSpace(canvasId, highlightTokens = []) {
        const E = window.EmbeddingEngine;
        const wordGroups = {
            animals: ['cat', 'dog', 'bird', 'lion', 'tiger', 'bear', 'wolf', 'fox', 'horse', 'rabbit'],
            food: ['pizza', 'burger', 'pasta', 'salad', 'soup', 'sushi', 'bread', 'rice', 'cake', 'coffee'],
            tech: ['computer', 'laptop', 'phone', 'robot', 'software', 'data', 'code', 'algorithm', 'model', 'neural'],
            emotion: ['happy', 'sad', 'angry', 'excited', 'calm', 'good', 'bad', 'great', 'fast', 'slow'],
            nature: ['sun', 'moon', 'star', 'sky', 'ocean', 'river', 'mountain', 'forest', 'tree', 'flower'],
            actions: ['run', 'walk', 'eat', 'sleep', 'read', 'write', 'speak', 'think', 'love', 'learn'],
        };

        const datasets = [];

        for (const [group, words] of Object.entries(wordGroups)) {
            const points = [];
            const labels = [];
            for (const word of words) {
                const vec = E.EMBEDDING_TABLE[word];
                if (!vec) continue;
                const { x, y } = E.projectTo2D(vec);
                points.push({ x, y });
                labels.push(word);
            }
            const color = CLUSTER_COLORS[group];
            datasets.push({
                label: group.charAt(0).toUpperCase() + group.slice(1),
                data: points,
                backgroundColor: color + 'bb',
                borderColor: color,
                borderWidth: 1.5,
                pointRadius: 7,
                pointHoverRadius: 11,
                wordLabels: labels,
            });
        }

        // Highlighted words from user input
        if (highlightTokens.length > 0) {
            const hPoints = [];
            const hLabels = [];
            for (const word of highlightTokens) {
                const vec = E.EMBEDDING_TABLE[word] || E.EMBEDDING_TABLE['[UNK]'];
                const { x, y } = E.projectTo2D(vec);
                hPoints.push({ x, y });
                hLabels.push('▶ ' + word);
            }
            datasets.push({
                label: 'Your Words',
                data: hPoints,
                backgroundColor: '#ffffff',
                borderColor: '#fbbf24',
                borderWidth: 2.5,
                pointRadius: 12,
                pointHoverRadius: 16,
                wordLabels: hLabels,
            });
        }

        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId).getContext('2d');
        _charts[canvasId] = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 600 },
                plugins: {
                    legend: {
                        labels: {
                            color: '#94a3b8',
                            boxWidth: 12,
                            padding: 16,
                            font: { family: 'Inter', size: 12 },
                        }
                    },
                    tooltip: {
                        backgroundColor: '#0d1526',
                        borderColor: 'rgba(255,255,255,0.12)',
                        borderWidth: 1,
                        padding: 10,
                        callbacks: {
                            label(context) {
                                const ds = context.dataset;
                                const word = ds.wordLabels ? ds.wordLabels[context.dataIndex] : '';
                                return ` ${word}  (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                            }
                        }
                    },
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#475569', font: { size: 10 } },
                        title: { display: true, text: 'Dimension 1 (Animal/Food/Tech axis)', color: '#475569', font: { size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#475569', font: { size: 10 } },
                        title: { display: true, text: 'Dimension 2 (Emotion/Nature/Action axis)', color: '#475569', font: { size: 11 } }
                    }
                }
            },
            plugins: [{
                // Draw word labels on the points
                afterDatasetsDraw(chart) {
                    const ctx = chart.ctx;
                    ctx.save();
                    chart.data.datasets.forEach((dataset, di) => {
                        const meta = chart.getDatasetMeta(di);
                        if (meta.hidden) return;
                        meta.data.forEach((point, index) => {
                            const label = dataset.wordLabels ? dataset.wordLabels[index] : '';
                            ctx.fillStyle = dataset.borderColor || '#94a3b8';
                            ctx.font = di === chart.data.datasets.length - 1
                                ? 'bold 12px Inter' : '10px Inter';
                            ctx.textAlign = 'center';
                            ctx.fillText(label, point.x, point.y - 13);
                        });
                    });
                    ctx.restore();
                }
            }]
        });
    }

    // ─── CHART 2: SIMILARITY BAR CHART ──────────────────────
    // Shows similarity breakdown by semantic cluster dimensions
    function renderSimilarityBar(canvasId, vecA, vecB) {
        const E = window.EmbeddingEngine;
        if (!vecA || !vecB) return;

        // Compute dimension-wise contributions to similarity
        const dims = ['Animal/Food', 'Tech', 'Emotion', 'Nature', 'Action', 'Other'];
        const contributions = [
            Math.abs(vecA[0] * vecB[0] + vecA[1] * vecB[1]),
            Math.abs(vecA[4] * vecB[4] + vecA[5] * vecB[5]),
            Math.abs(vecA[6] * vecB[6] + vecA[7] * vecB[7]),
            Math.abs(vecA[8] * vecB[8] + vecA[9] * vecB[9]),
            Math.abs(vecA[10] * vecB[10] + vecA[11] * vecB[11]),
            Math.abs(vecA[12] * vecB[12] + vecA[13] * vecB[13]),
        ];
        const clrs = [COLORS.green, COLORS.blue, COLORS.pink, COLORS.cyan, COLORS.purple, COLORS.yellow];

        destroyChart(canvasId);
        const ctx = document.getElementById(canvasId).getContext('2d');
        _charts[canvasId] = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: dims,
                datasets: [{
                    label: 'Dimension Similarity Contribution',
                    data: contributions.map(v => parseFloat(v.toFixed(4))),
                    backgroundColor: clrs.map(c => c + 'aa'),
                    borderColor: clrs,
                    borderWidth: 2,
                    borderRadius: 6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#0d1526',
                        borderColor: 'rgba(255,255,255,0.12)',
                        borderWidth: 1,
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#94a3b8', font: { size: 11 } }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.06)' },
                        ticks: { color: '#94a3b8', font: { size: 10 } },
                        title: { display: true, text: 'Contribution to Similarity', color: '#475569', font: { size: 11 } },
                        min: 0, max: 0.5,
                    }
                }
            }
        });
    }

    // ─── CHART 3: VECTOR HEATMAP ─────────────────────────────
    // Renders a mini heatmap of a 16-dim vector as colored blocks
    function renderVectorHeatmap(containerId, vec, label = '') {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        if (label) {
            const lel = document.createElement('div');
            lel.style.cssText = 'font-size:11px;color:#94a3b8;margin-bottom:6px;font-weight:600;';
            lel.textContent = label;
            container.appendChild(lel);
        }

        const wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:3px;';

        const maxVal = Math.max(...vec.map(Math.abs));
        vec.forEach((v, i) => {
            const cell = document.createElement('div');
            const intensity = Math.abs(v) / (maxVal || 1);
            const hue = v >= 0 ? 142 : 0; // green for positive, red for negative
            cell.style.cssText = `
        width:28px;height:28px;border-radius:4px;
        background:hsla(${hue},80%,${30 + intensity * 35}%,${0.4 + intensity * 0.6});
        display:flex;align-items:center;justify-content:center;
        font-size:7px;color:rgba(255,255,255,0.7);
        font-family:'JetBrains Mono',monospace;font-weight:600;
        border:1px solid rgba(255,255,255,0.06);
        cursor:default;
        title:${v.toFixed(3)};
      `;
            cell.title = `dim[${i}] = ${v.toFixed(4)}`;
            cell.textContent = v.toFixed(2);
            wrap.appendChild(cell);
        });
        container.appendChild(wrap);
    }

    // ─── ANIMATION: MEAN POOLING CANVAS ─────────────────────
    // Animates the mean pooling process step-by-step on a canvas
    let poolAnimFrame = null;
    let poolAnimStep = 0;
    let poolTokenVecs = [];
    let pooledResult = [];

    function startPoolingAnimation(canvasId, tokens, tokenVectors, pooledVec) {
        poolTokenVecs = tokenVectors;
        pooledResult = pooledVec;
        poolAnimStep = 0;

        if (poolAnimFrame) cancelAnimationFrame(poolAnimFrame);
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        const W = canvas.width = canvas.offsetWidth;
        const H = canvas.height = 280;

        function drawFrame() {
            ctx.clearRect(0, 0, W, H);

            const N = tokens.length;
            if (N === 0) return;

            const rowH = 36;
            const cols = 16; // EMBED_DIM
            const cellW = Math.min(32, (W * 0.55) / cols);
            const cellH = 28;
            const startX = 20;
            const startY = 30;

            // Title
            ctx.font = 'bold 13px Inter';
            ctx.fillStyle = '#94a3b8';
            ctx.fillText('Token Vectors  (each row = one token)', startX, 18);

            // Draw token rows
            tokens.forEach((tok, ri) => {
                const y = startY + ri * (cellH + 6);
                const highlight = poolAnimStep >= ri + 1;

                // Token label
                ctx.font = `${highlight ? 'bold' : ''} 11px JetBrains Mono`;
                ctx.fillStyle = highlight ? '#4f8ef7' : '#94a3b8';
                ctx.fillText(tok, startX, y + cellH / 2 + 4);

                tokenVectors[ri].forEach((val, ci) => {
                    const x = startX + 60 + ci * (cellW + 2);
                    const maxV = 0.8;
                    const norm = Math.abs(val) / maxV;
                    const hue = val >= 0 ? 142 : 0;
                    ctx.fillStyle = highlight
                        ? `hsla(${hue},70%,${30 + norm * 30}%,${0.5 + norm * 0.5})`
                        : `rgba(255,255,255,0.05)`;
                    ctx.strokeStyle = highlight ? `hsla(${hue},70%,50%,0.4)` : 'rgba(255,255,255,0.06)';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.roundRect(x, y, cellW, cellH, 3);
                    ctx.fill();
                    ctx.stroke();
                });
            });

            // Pooling arrow
            const arrowY = startY + N * (cellH + 6) + 10;
            if (poolAnimStep >= N + 1) {
                ctx.font = 'bold 12px Inter';
                ctx.fillStyle = '#fbbf24';
                const arrowLabel = '⬇ Mean Pool: average each column ÷ ' + N;
                ctx.fillText(arrowLabel, startX, arrowY);
            }

            // Pooled result row
            if (poolAnimStep >= N + 2) {
                const poolY = arrowY + 20;
                ctx.font = 'bold 11px Inter';
                ctx.fillStyle = '#fbbf24';
                ctx.fillText('Pooled ▶', startX, poolY + cellH / 2 + 4);
                pooledVec.forEach((val, ci) => {
                    const x = startX + 60 + ci * (cellW + 2);
                    const norm = Math.abs(val) / 0.8;
                    const hue = val >= 0 ? 142 : 0;
                    ctx.fillStyle = `hsla(${hue},80%,${35 + norm * 30}%,0.9)`;
                    ctx.strokeStyle = '#fbbf24';
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.roundRect(x, poolY, cellW, cellH, 3);
                    ctx.fill();
                    ctx.stroke();
                });
            }

            poolAnimStep++;
            if (poolAnimStep <= N + 2) {
                poolAnimFrame = setTimeout(drawFrame, 600);
            }
        }
        drawFrame();
    }

    // ─── RENDER: VECTOR AS COLORED SPANS ────────────────────
    function renderVectorSpans(containerId, vec, maxShow = 16) {
        const el = document.getElementById(containerId);
        if (!el) return;
        const shown = vec.slice(0, maxShow);
        const remaining = vec.length - maxShow;
        el.innerHTML = '[  ' + shown.map(v => {
            const cls = v >= 0 ? 'vec-pos' : 'vec-neg';
            return `<span class="vector-value ${cls}">${v.toFixed(4)}</span>`;
        }).join('') + (remaining > 0 ? `<span style="color:#475569"> ...+${remaining} more</span>` : '') + '  ]';
    }

    // ─── RENDER: TOKEN TABLE ANIMATION ──────────────────────
    function renderTokenTable(containerId, tokens, ids, delay = 120) {
        const el = document.getElementById(containerId);
        if (!el) return;
        el.innerHTML = '';

        const table = document.createElement('table');
        table.style.cssText = 'width:100%;border-collapse:collapse;font-size:13px;';
        table.innerHTML = `
      <thead>
        <tr>
          <th style="text-align:left;padding:8px 12px;color:#475569;font-size:11px;letter-spacing:.06em;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,0.08);">Index</th>
          <th style="text-align:left;padding:8px 12px;color:#475569;font-size:11px;letter-spacing:.06em;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,0.08);">Token (string)</th>
          <th style="text-align:left;padding:8px 12px;color:#475569;font-size:11px;letter-spacing:.06em;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,0.08);">Token ID (integer)</th>
          <th style="text-align:left;padding:8px 12px;color:#475569;font-size:11px;letter-spacing:.06em;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,0.08);">Known?</th>
          <th style="text-align:left;padding:8px 12px;color:#475569;font-size:11px;letter-spacing:.06em;text-transform:uppercase;border-bottom:1px solid rgba(255,255,255,0.08);">Vector (first 4 dims)</th>
        </tr>
      </thead>
    `;
        const tbody = document.createElement('tbody');
        const E = window.EmbeddingEngine;

        tokens.forEach((tok, i) => {
            const id = ids[i];
            const known = E.VOCABULARY[tok] !== undefined;
            const vec = E.lookupEmbeddings([tok])[0];
            const tr = document.createElement('tr');
            tr.style.cssText = 'opacity:0;transition:opacity 0.3s ease;border-bottom:1px solid rgba(255,255,255,0.04);';
            tr.innerHTML = `
        <td style="padding:10px 12px;color:#475569;font-family:'JetBrains Mono',monospace;">${i}</td>
        <td style="padding:10px 12px;">
          <span style="
            background:${known ? 'rgba(79,142,247,0.15)' : 'rgba(249,115,22,0.15)'};
            color:${known ? '#4f8ef7' : '#f97316'};
            border:1px solid ${known ? 'rgba(79,142,247,0.3)' : 'rgba(249,115,22,0.3)'};
            padding:4px 10px;border-radius:6px;
            font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;
          ">${tok}</span>
        </td>
        <td style="padding:10px 12px;font-family:'JetBrains Mono',monospace;color:#fbbf24;font-weight:700;font-size:15px;">${id}</td>
        <td style="padding:10px 12px;">
          <span style="
            font-size:11px;font-weight:700;padding:3px 8px;border-radius:999px;
            ${known
                    ? 'background:rgba(16,185,129,0.15);color:#10b981;border:1px solid rgba(16,185,129,0.3);'
                    : 'background:rgba(249,115,22,0.15);color:#f97316;border:1px solid rgba(249,115,22,0.3);'
                }
          ">${known ? '✓ In Vocab' : '? [UNK]'}</span>
        </td>
        <td style="padding:10px 12px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#7dd3fc;">
          [${vec.slice(0, 4).map(v => `<span style="color:${v >= 0 ? '#4ade80' : '#f87171'}">${v.toFixed(3)}</span>`).join(', ')} ...]
        </td>
      `;
            tbody.appendChild(tr);
            setTimeout(() => { tr.style.opacity = '1'; }, i * delay);
        });

        table.appendChild(tbody);
        el.appendChild(table);
    }

    return {
        renderEmbeddingSpace,
        renderSimilarityBar,
        renderVectorHeatmap,
        startPoolingAnimation,
        renderVectorSpans,
        renderTokenTable,
        CLUSTER_COLORS,
    };
})();

window.Viz = Viz;
