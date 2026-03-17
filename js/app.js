// ============================================================
//  app.js — App Controller: Wires UI to EmbeddingEngine + Viz
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    const E = window.EmbeddingEngine;
    const V = window.Viz;

    // ─── SECTION ENTRANCE ANIMATION ─────────────────────────
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.classList.add('visible');
                sectionObserver.unobserve(e.target);
            }
        });
    }, { threshold: 0.08 });
    document.querySelectorAll('.section').forEach(s => sectionObserver.observe(s));

    // ─── SECTION 3: LIVE TOKENIZER DEMO ─────────────────────
    const tokInput = document.getElementById('tok-input');
    const tokDisplay = document.getElementById('tok-display');
    const tokCount = document.getElementById('tok-count');
    const tokKnown = document.getElementById('tok-known');
    const tokTable = document.getElementById('tok-table');
    const tokVecWrap = document.getElementById('tok-vec-wrap');

    const TOKEN_COLORS = [
        '#4f8ef7', '#9c6ef7', '#22d3ee', '#10b981', '#f97316', '#f472b6', '#fbbf24',
        '#34d399', '#60a5fa', '#a78bfa', '#f59e0b', '#6ee7b7',
    ];

    function updateTokenizerDemo() {
        const text = tokInput.value;
        const { tokens, ids } = E.encodeText(text);

        // Update chips
        tokDisplay.innerHTML = '';
        if (tokens.length === 0) {
            tokDisplay.innerHTML = '<span style="color:#475569;font-size:13px;padding:4px 8px;">Start typing above…</span>';
        } else {
            tokens.forEach((tok, i) => {
                const known = E.VOCABULARY[tok] !== undefined;
                const chip = document.createElement('div');
                const color = TOKEN_COLORS[i % TOKEN_COLORS.length];
                chip.className = `token-chip ${known ? 'token-known' : 'token-unknown'}`;
                chip.style.animationDelay = `${i * 50}ms`;
                chip.innerHTML = `
          <span class="token-chip-text">${tok}</span>
          <span class="token-chip-id">id: ${ids[i]}</span>
          <div class="token-tooltip">
            Token: "${tok}" → ID: ${ids[i]} → ${known ? 'In Vocab ✓' : 'Unknown [UNK]'}
          </div>
        `;
                tokDisplay.appendChild(chip);
            });
        }

        // Update counters
        if (tokCount) tokCount.textContent = tokens.length;
        if (tokKnown) tokKnown.textContent = tokens.filter(t => E.VOCABULARY[t] !== undefined).length;

        // Update table and vector display
        if (tokTable) V.renderTokenTable('tok-table', tokens, ids);
        if (tokVecWrap && tokens.length > 0) {
            const vecs = E.lookupEmbeddings(tokens);
            V.renderVectorSpans('tok-vec-wrap', vecs[0]);
        }
    }

    if (tokInput) {
        tokInput.addEventListener('input', updateTokenizerDemo);
        updateTokenizerDemo(); // initial run with placeholder
    }

    // ─── SECTION 5: MATH SLIDERS ─────────────────────────────
    const sliderDeg = document.getElementById('angle-slider');
    const mathCosOut = document.getElementById('cos-output');
    const mathDotOut = document.getElementById('dot-output');
    const mathCtx = document.getElementById('angle-canvas');

    function drawAngleCanvas(deg) {
        if (!mathCtx) return;
        const canvas = mathCtx;
        const ctx = canvas.getContext('2d');
        const W = canvas.width = canvas.offsetWidth || 400;
        const H = canvas.height = 200;
        ctx.clearRect(0, 0, W, H);

        const cx = W / 2, cy = H - 30;
        const r = Math.min(cx - 30, cy - 20);
        const rad = (deg * Math.PI) / 180;

        // Reference vector A (always pointing right)
        const ax = cx + r, ay = cy;
        // Vector B at angle
        const bx = cx + r * Math.cos(rad);
        const by = cy - r * Math.sin(rad);

        // Arc
        ctx.beginPath();
        ctx.arc(cx, cy, 40, -rad, 0, true);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Angle label
        ctx.fillStyle = '#94a3b8';
        ctx.font = '12px Inter';
        ctx.fillText(`θ = ${deg}°`, cx + 14, cy - 14);

        // Vector A
        drawArrow(ctx, cx, cy, ax, ay, '#4f8ef7', 'A');
        // Vector B
        drawArrow(ctx, cx, cy, bx, by, '#f472b6', 'B');

        // Origin dot
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff33';
        ctx.fill();
    }

    function drawArrow(ctx, x1, y1, x2, y2, color, label) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Arrowhead
        const angle = Math.atan2(y2 - y1, x2 - x1);
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - 12 * Math.cos(angle - 0.4), y2 - 12 * Math.sin(angle - 0.4));
        ctx.lineTo(x2 - 12 * Math.cos(angle + 0.4), y2 - 12 * Math.sin(angle + 0.4));
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        // Label
        ctx.fillStyle = color;
        ctx.font = 'bold 13px Inter';
        ctx.fillText(label, x2 + (x2 > x1 ? 6 : -20), y2 - 6);
    }

    function updateMath() {
        if (!sliderDeg) return;
        const deg = parseInt(sliderDeg.value);
        const cos = Math.cos(deg * Math.PI / 180);
        if (mathCosOut) mathCosOut.textContent = cos.toFixed(4);
        if (mathDotOut) {
            const dot = cos; // assuming unit vectors
            mathDotOut.textContent = dot.toFixed(4);
            mathDotOut.style.color = cos > 0.7 ? '#10b981' : cos > 0 ? '#fbbf24' : '#f87171';
        }
        drawAngleCanvas(deg);
        // Update the angle label next to slider
        const degLabel = document.getElementById('angle-deg-label');
        if (degLabel) degLabel.textContent = deg + '°';
    }

    if (sliderDeg) {
        sliderDeg.addEventListener('input', updateMath);
        // Delay initial draw until layout is ready
        setTimeout(updateMath, 100);
    }

    // ─── SECTION 6: EMBEDDING SPACE ──────────────────────────
    function initEmbeddingSpace() {
        const canvas = document.getElementById('embed-space-canvas');
        if (!canvas) return;
        setTimeout(() => V.renderEmbeddingSpace('embed-space-canvas'), 200);
    }
    initEmbeddingSpace();

    // ─── SECTION 7: SIMILARITY EXPLORER ─────────────────────
    const simInputA = document.getElementById('sim-input-a');
    const simInputB = document.getElementById('sim-input-b');
    const simScoreNum = document.getElementById('sim-score-num');
    const simScoreBar = document.getElementById('sim-score-bar');
    const simScoreLbl = document.getElementById('sim-score-label');
    const simVecA = document.getElementById('sim-vec-a');
    const simVecB = document.getElementById('sim-vec-b');

    function updateSimilarity() {
        const textA = simInputA ? simInputA.value : '';
        const textB = simInputB ? simInputB.value : '';
        if (!textA.trim() || !textB.trim()) return;

        const resA = E.embedText(textA);
        const resB = E.embedText(textB);
        const sim = E.cosineSimilarity(resA.normalized, resB.normalized);
        const { label, color } = E.similarityLabel(sim);
        const pct = Math.round(((sim + 1) / 2) * 100); // map -1..1 to 0..100

        if (simScoreNum) simScoreNum.textContent = sim.toFixed(4);
        if (simScoreBar) simScoreBar.style.width = pct + '%';
        if (simScoreLbl) { simScoreLbl.textContent = label; simScoreLbl.style.color = color; }

        // Vector heatmaps
        if (document.getElementById('sim-heat-a')) V.renderVectorHeatmap('sim-heat-a', resA.normalized, 'Sentence A vector');
        if (document.getElementById('sim-heat-b')) V.renderVectorHeatmap('sim-heat-b', resB.normalized, 'Sentence B vector');

        // Dimension breakdown
        V.renderSimilarityBar('sim-bar-canvas', resA.normalized, resB.normalized);

        // Update scatter with current words
        const wordsA = resA.tokens.filter(t => t in E.EMBEDDING_TABLE);
        const wordsB = resB.tokens.filter(t => t in E.EMBEDDING_TABLE);
        V.renderEmbeddingSpace('embed-space-canvas', [...new Set([...wordsA, ...wordsB])]);
    }

    if (simInputA) simInputA.addEventListener('input', updateSimilarity);
    if (simInputB) simInputB.addEventListener('input', updateSimilarity);
    // Initial demo values
    if (simInputA) { simInputA.value = 'I love my cat'; }
    if (simInputB) { simInputB.value = 'My dog is happy'; }
    setTimeout(updateSimilarity, 300);

    // ─── SECTION 8: MEAN POOLING DEMO ───────────────────────
    const poolInput = document.getElementById('pool-input');
    const poolBtn = document.getElementById('pool-btn');

    function runPoolingDemo() {
        const text = poolInput ? poolInput.value : 'the cat sat';
        const { tokens, ids } = E.encodeText(text);
        const vecs = E.lookupEmbeddings(tokens);
        const pooled = E.meanPool(vecs);
        V.startPoolingAnimation('pool-canvas', tokens, vecs, pooled);

        // Update result
        const poolResult = document.getElementById('pool-result');
        if (poolResult) {
            V.renderVectorSpans('pool-result', pooled);
        }
        // Normalized
        const poolNorm = document.getElementById('pool-normalized');
        if (poolNorm) {
            V.renderVectorSpans('pool-normalized', E.normalize(pooled));
        }
    }

    if (poolBtn) poolBtn.addEventListener('click', runPoolingDemo);
    if (poolInput) {
        poolInput.addEventListener('keydown', e => { if (e.key === 'Enter') runPoolingDemo(); });
        setTimeout(runPoolingDemo, 500);
    }

    // ─── TYPE FLOW STEP HOVER ───────────────────────────────
    // Animate the type flow steps on hover
    document.querySelectorAll('.type-step').forEach((step, i) => {
        step.style.animationDelay = `${i * 0.1}s`;
    });

    console.log('%c Text Embedding Learning App', 'color:#4f8ef7;font-weight:bold;font-size:18px;');
    console.log('%c EmbeddingEngine available at window.EmbeddingEngine', 'color:#22d3ee;');
    console.log('%c Try: EmbeddingEngine.embedText("hello world")', 'color:#10b981;');
});
