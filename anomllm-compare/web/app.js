const state = {
  activeSample: 0,
  mode: "compare",
  lastResult: null,
};

const TASK_LABELS = {
  point_anomaly: "point",
  range_anomaly: "range",
  trend_anomaly: "trend",
  frequency_anomaly: "freq",
  general_anomaly: "general",
};

const SAMPLE_DEFS = [
  {
    title: "趋势异常",
    hint: "后半段开始持续上升",
    request: "这段 CSV 后半段是不是出现了趋势异常？请先判断任务类型，再返回异常区间。",
    kind: "trend",
  },
  {
    title: "尖峰/孤立点",
    hint: "只找突刺，不要报长区间",
    request: "只检测孤立尖峰或突然下探的点异常，输出 JSON 区间即可。",
    kind: "point",
  },
  {
    title: "持续越界",
    hint: "均值漂移形成长区间",
    request: "这段数据有没有持续越界或均值漂移？请给出异常开始和结束位置。",
    kind: "range",
  },
  {
    title: "频率变化",
    hint: "中段振荡变快",
    request: "帮我判断是否有频率异常，也就是振荡突然变快或粗糙度改变。",
    kind: "freq",
  },
  {
    title: "不说明类型",
    hint: "先让系统自己路由",
    request: "我不确定这是什么异常。请先理解我的需求，然后找出不正常的时间段。",
    kind: "range",
  },
  {
    title: "解释型问题",
    hint: "输出区间并说明原因",
    request: "请解释这段序列后面为什么不正常，并给出异常区间。",
    kind: "trend",
  },
];

function seededNoise(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function makeSeries(kind, n = 220) {
  const rand = seededNoise(1207 + kind.length * 19);
  const values = [];
  for (let i = 0; i < n; i += 1) {
    const base = Math.sin(i / 8) * 0.55 + Math.sin(i / 25) * 0.35;
    const noise = (rand() - 0.5) * 0.18;
    values.push(base + noise);
  }
  if (kind === "trend") {
    const start = Math.floor(n * 0.8);
    for (let i = start; i < n; i += 1) {
      values[i] += (i - start) * 0.055;
    }
  } else if (kind === "point") {
    [39, 118, 171].forEach((idx, j) => {
      values[idx] += j === 1 ? -3.1 : 3.0;
    });
  } else if (kind === "range") {
    for (let i = 88; i < 146; i += 1) {
      values[i] += 2.25;
    }
  } else if (kind === "freq") {
    for (let i = 94; i < 162; i += 1) {
      values[i] += Math.sin(i * 1.85) * 0.72;
    }
  }
  return values;
}

function toCsv(values) {
  const lines = ["idx,value"];
  values.forEach((v, i) => lines.push(`${i},${v.toFixed(4)}`));
  return lines.join("\n");
}

function parseCsv(text) {
  const rows = text.trim().split(/\r?\n/);
  const values = [];
  rows.forEach((line, lineNo) => {
    const parts = line.split(/[,\t;]/).map((s) => s.trim()).filter(Boolean);
    if (!parts.length) return;
    if (lineNo === 0 && /[a-zA-Z_\u4e00-\u9fa5]/.test(parts.join(""))) return;
    const value = Number(parts.length > 1 ? parts[1] : parts[0]);
    if (Number.isFinite(value)) values.push(value);
  });
  return values;
}

function median(arr) {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function quantile(arr, q) {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  const pos = (a.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return a[lo];
  return a[lo] * (hi - pos) + a[hi] * (pos - lo);
}

function robustStats(values) {
  const med = median(values);
  const mad = median(values.map((v) => Math.abs(v - med))) || 1e-6;
  return { med, mad };
}

function movingAverage(values, width) {
  const out = [];
  const half = Math.floor(width / 2);
  for (let i = 0; i < values.length; i += 1) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(values.length - 1, i + half); j += 1) {
      sum += values[j];
      count += 1;
    }
    out.push(sum / count);
  }
  return out;
}

function slope(values, start, end) {
  const n = Math.max(0, end - start);
  if (n < 3) return 0;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i += 1) {
    const x = i;
    const y = values[start + i];
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const den = n * sxx - sx * sx;
  return den === 0 ? 0 : (n * sxy - sx * sy) / den;
}

function intervalsFromMask(mask, gap = 0, minLen = 1) {
  const raw = [];
  let start = null;
  mask.forEach((v, i) => {
    if (v && start === null) start = i;
    if ((!v || i === mask.length - 1) && start !== null) {
      const end = v && i === mask.length - 1 ? i + 1 : i;
      raw.push([start, end]);
      start = null;
    }
  });
  if (!raw.length) return [];
  const merged = [raw[0]];
  for (const [s, e] of raw.slice(1)) {
    const last = merged[merged.length - 1];
    if (s <= last[1] + gap) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }
  return merged.filter(([s, e]) => e - s >= minLen).map(([s, e]) => ({ start: s, end: e }));
}

function inferTokens(request, values) {
  const text = request.toLowerCase();
  const scores = {
    point_anomaly: 0,
    range_anomaly: 0,
    trend_anomaly: 0,
    frequency_anomaly: 0,
    general_anomaly: 0.2,
  };
  if (/趋势|斜率|上升|下降|后半段|trend|slope|drift/.test(text)) scores.trend_anomaly += 1.1;
  if (/尖峰|突刺|孤立|点异常|spike|point|dip/.test(text)) scores.point_anomaly += 1.1;
  if (/越界|均值|漂移|持续|level|range|shift/.test(text)) scores.range_anomaly += 1.0;
  if (/频率|振荡|粗糙|周期|frequency|oscillat|rough/.test(text)) scores.frequency_anomaly += 1.0;

  if (!request.trim() || scores.general_anomaly >= Math.max(...Object.values(scores))) {
    const dataToken = inferTokenFromData(values);
    scores[dataToken] += 0.7;
  }

  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const task = ranked[0][0];
  const confidence = Math.max(0.35, Math.min(0.98, ranked[0][1] / (ranked[0][1] + ranked[1][1] + 0.3)));
  let output = "intervals";
  if (/解释|为什么|原因|explain|why/.test(text)) output = "explanation";
  if (/总结|summary/.test(text)) output = "summary";
  if (/画图|plot|图/.test(text)) output = "plot_request";
  return { task_token: task, output_token: output, confidence };
}

function inferTokenFromData(values) {
  if (values.length < 12) return "general_anomaly";
  const { med, mad } = robustStats(values);
  const maxZ = Math.max(...values.map((v) => Math.abs(v - med) / (1.4826 * mad + 1e-6)));
  if (maxZ > 5) return "point_anomaly";
  const n = values.length;
  const tailStart = Math.floor(n * 0.8);
  const tailSlope = Math.abs(slope(movingAverage(values, 9), tailStart, n));
  if (tailSlope > 0.018) return "trend_anomaly";
  const diffs = values.slice(1).map((v, i) => v - values[i]);
  const diffMad = robustStats(diffs).mad;
  const midDiff = diffs.slice(Math.floor(n * 0.4), Math.floor(n * 0.72));
  if (robustStats(midDiff).mad > diffMad * 1.8) return "frequency_anomaly";
  return "range_anomaly";
}

function oldConstrained(values) {
  const { med, mad } = robustStats(values);
  const diffs = values.map((v, i) => (i ? Math.abs(v - values[i - 1]) : 0));
  const score = values.map((v, i) => Math.abs(v - med) / (1.4826 * mad + 1e-6) + diffs[i] * 0.18);
  const thr = quantile(score, 0.965);
  const mask = score.map((s) => s > thr);
  return intervalsFromMask(mask, 3, 1).slice(0, 8).map((iv) => ({ ...iv, source: "candidate_score" }));
}

function tokenPolicy(values, tokens) {
  const task = tokens.task_token;
  if (task === "trend_anomaly") return trendPolicy(values);
  if (task === "point_anomaly") return pointPolicy(values);
  if (task === "range_anomaly") return rangePolicy(values);
  if (task === "frequency_anomaly") return freqPolicy(values);
  return oldConstrained(values).map((iv) => ({ ...iv, source: "general_fallback" }));
}

function pointPolicy(values) {
  const { med, mad } = robustStats(values);
  const z = values.map((v) => Math.abs(v - med) / (1.4826 * mad + 1e-6));
  const threshold = Math.max(3.5, quantile(z, 0.985));
  const mask = z.map((s) => s > threshold);
  return intervalsFromMask(mask, 2, 1).map((iv) => ({
    start: Math.max(0, iv.start - 1),
    end: Math.min(values.length, iv.end + 1),
    source: "point_token_policy",
  }));
}

function rangePolicy(values) {
  const smooth = movingAverage(values, 17);
  const { med, mad } = robustStats(smooth);
  const z = smooth.map((v) => Math.abs(v - med) / (1.4826 * mad + 1e-6));
  const mask = z.map((s) => s > Math.max(2.2, quantile(z, 0.93)));
  return intervalsFromMask(mask, 8, 8).map((iv) => ({ ...iv, source: "range_token_policy" }));
}

function trendPolicy(values) {
  const n = values.length;
  const tailStart = Math.floor(n * 0.8);
  const smoothed = movingAverage(values, Math.max(9, Math.floor(n * 0.08) | 1));
  const preSlope = slope(smoothed, Math.max(0, tailStart - Math.floor(n * 0.25)), tailStart);
  const tailSlope = slope(smoothed, tailStart, n);
  const diffs = values.slice(1).map((v, i) => Math.abs(v - values[i]));
  const noise = median(diffs) || 1e-6;
  const contrast = Math.abs(tailSlope - preSlope);
  const active = contrast > noise / 14 || Math.abs(tailSlope) > noise / 10;
  if (!active) return [];
  let start = tailStart;
  for (let i = tailStart; i < n - 6; i += 1) {
    const local = Math.abs(slope(smoothed, i, Math.min(n, i + 18)));
    if (local > noise / 13) {
      start = i;
      break;
    }
  }
  return [{ start, end: n, source: "trend_token_policy" }];
}

function freqPolicy(values) {
  const diffs = values.slice(1).map((v, i) => Math.abs(v - values[i]));
  const rough = movingAverage(diffs, 11);
  const med = median(rough);
  const mask = [false].concat(rough.map((v) => v > med * 1.55 && v > quantile(rough, 0.82)));
  return intervalsFromMask(mask, 8, 8).map((iv) => ({ ...iv, source: "frequency_token_policy" }));
}

function setBusy(isBusy) {
  const runBtn = document.querySelector("#runBtn");
  runBtn.disabled = isBusy;
  runBtn.textContent = isBusy ? "分析中..." : "分析";
}

async function analyze() {
  const request = document.querySelector("#requestText").value;
  const csv = document.querySelector("#csvText").value;
  setBusy(true);
  try {
    const resp = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request, csv }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }
    state.lastResult = {
      tokens: data.tokens,
      oldIntervals: data.oldIntervals,
      newIntervals: data.newIntervals,
      trace: data.trace,
      values: data.values,
      router: data.router,
    };
    renderResult();
  } catch (err) {
    const values = parseCsv(csv);
    const tokens = inferTokens(request, values);
    tokens.router_source = "browser_fallback";
    tokens.router_note = String(err.message || err);
    const oldIntervals = oldConstrained(values);
    const newIntervals = tokenPolicy(values, tokens);
    const trace = [
      {
        step: "USER_REQUEST_ROUTER",
        router_source: "browser_fallback",
        task_token: tokens.task_token,
        output_token: tokens.output_token,
        confidence: Number(tokens.confidence.toFixed(3)),
        note: tokens.router_note,
      },
      { step: "OLD_CONSTRAINED_SELECTION", intervals: oldIntervals.map(({ start, end }) => ({ start, end })) },
      { step: "TOKEN_POLICY", route: TASK_LABELS[tokens.task_token], intervals: newIntervals.map(({ start, end }) => ({ start, end })) },
    ];
    state.lastResult = { tokens, oldIntervals, newIntervals, trace, values, router: null };
    renderResult();
  } finally {
    setBusy(false);
  }
}

function analyzeLocalOnly() {
  const request = document.querySelector("#requestText").value;
  const csv = document.querySelector("#csvText").value;
  const values = parseCsv(csv);
  const tokens = inferTokens(request, values);
  tokens.router_source = "browser_local";
  const oldIntervals = oldConstrained(values);
  const newIntervals = tokenPolicy(values, tokens);
  const trace = [
    { step: "USER_REQUEST_ROUTER", router_source: "browser_local", task_token: tokens.task_token, output_token: tokens.output_token, confidence: Number(tokens.confidence.toFixed(3)) },
    { step: "OLD_CONSTRAINED_SELECTION", intervals: oldIntervals.map(({ start, end }) => ({ start, end })) },
    { step: "TOKEN_POLICY", route: TASK_LABELS[tokens.task_token], intervals: newIntervals.map(({ start, end }) => ({ start, end })) },
  ];
  state.lastResult = { tokens, oldIntervals, newIntervals, trace, values };
  renderResult();
}

function renderSamples() {
  const wrap = document.querySelector("#sampleList");
  wrap.innerHTML = "";
  SAMPLE_DEFS.forEach((sample, idx) => {
    const btn = document.createElement("button");
    btn.className = `sample-button${idx === state.activeSample ? " is-active" : ""}`;
    btn.innerHTML = `<strong>${sample.title}</strong><span>${sample.hint}</span>`;
    btn.addEventListener("click", () => loadSample(idx));
    wrap.appendChild(btn);
  });
}

function loadSample(idx) {
  state.activeSample = idx;
  const sample = SAMPLE_DEFS[idx];
  document.querySelector("#requestText").value = sample.request;
  document.querySelector("#csvText").value = toCsv(makeSeries(sample.kind));
  renderSamples();
  analyzeLocalOnly();
}

function renderResult() {
  const result = state.lastResult;
  if (!result) return;
  const { tokens, oldIntervals, newIntervals, trace, values } = result;
  document.querySelector("#tokenStrip").innerHTML = [
    `<span class="token-chip">task_token=${tokens.task_token}</span>`,
    `<span class="token-chip output">output_token=${tokens.output_token}</span>`,
    `<span class="token-chip confidence">confidence=${tokens.confidence.toFixed(2)}</span>`,
    `<span class="token-chip router">router=${tokens.router_source || "unknown"}</span>`,
  ].join("");
  renderRows("#oldRows", oldIntervals, "source-old");
  renderRows("#newRows", newIntervals, "source-new");
  document.querySelector("#traceBox").textContent = JSON.stringify(trace, null, 2);
  drawSeries(values, oldIntervals, newIntervals);
}

function renderRows(selector, intervals, cls) {
  const body = document.querySelector(selector);
  if (!intervals.length) {
    body.innerHTML = `<tr><td colspan="3">[]</td></tr>`;
    return;
  }
  body.innerHTML = intervals.map((iv) => (
    `<tr><td>${iv.start}</td><td>${iv.end}</td><td class="${cls}">${iv.source}</td></tr>`
  )).join("");
}

function drawSeries(values, oldIntervals, newIntervals) {
  const canvas = document.querySelector("#seriesCanvas");
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(640, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(320 * ratio);
  ctx.scale(ratio, ratio);
  const w = canvas.width / ratio;
  const h = canvas.height / ratio;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fbfcfe";
  ctx.fillRect(0, 0, w, h);
  if (!values.length) return;
  const pad = { l: 42, r: 16, t: 18, b: 32 };
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const span = Math.max(1e-6, maxV - minV);
  const xOf = (i) => pad.l + (i / Math.max(1, values.length - 1)) * (w - pad.l - pad.r);
  const yOf = (v) => pad.t + (1 - (v - minV) / span) * (h - pad.t - pad.b);

  shadeIntervals(ctx, oldIntervals, values.length, xOf, pad.t, h - pad.b, "rgba(163, 93, 0, 0.16)");
  shadeIntervals(ctx, newIntervals, values.length, xOf, pad.t, h - pad.b, "rgba(194, 59, 53, 0.18)");

  ctx.strokeStyle = "#d9dee7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();

  ctx.strokeStyle = "#1d2a3a";
  ctx.lineWidth = 1.7;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = xOf(i);
    const y = yOf(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  drawLegend(ctx, w);
}

function shadeIntervals(ctx, intervals, n, xOf, top, bottom, color) {
  ctx.fillStyle = color;
  intervals.forEach((iv) => {
    const x1 = xOf(Math.max(0, Math.min(n - 1, iv.start)));
    const x2 = xOf(Math.max(0, Math.min(n - 1, iv.end)));
    ctx.fillRect(x1, top, Math.max(2, x2 - x1), bottom - top);
  });
}

function drawLegend(ctx, w) {
  ctx.font = "12px Segoe UI, Microsoft YaHei, Arial";
  ctx.fillStyle = "#1d2a3a";
  ctx.fillText("series", 52, 20);
  ctx.fillStyle = "rgba(163, 93, 0, 0.9)";
  ctx.fillRect(w - 210, 12, 12, 8);
  ctx.fillStyle = "#5c3900";
  ctx.fillText("old constrained", w - 192, 20);
  ctx.fillStyle = "rgba(194, 59, 53, 0.9)";
  ctx.fillRect(w - 92, 12, 12, 8);
  ctx.fillStyle = "#7f201c";
  ctx.fillText("token policy", w - 74, 20);
}

function copyJson() {
  if (!state.lastResult) return;
  const payload = {
    internal_tokens: state.lastResult.tokens,
    old_lafr_llm: state.lastResult.oldIntervals,
    token_policy: state.lastResult.newIntervals,
    trace: state.lastResult.trace,
  };
  navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
}

function bindUi() {
  document.querySelector("#runBtn").addEventListener("click", analyze);
  document.querySelector("#copyBtn").addEventListener("click", copyJson);
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("is-active", b === btn));
      document.querySelector("#traceBox").style.maxHeight = state.mode === "json" ? "520px" : "260px";
    });
  });
}

window.addEventListener("resize", () => {
  if (state.lastResult) drawSeries(state.lastResult.values, state.lastResult.oldIntervals, state.lastResult.newIntervals);
});

bindUi();
renderSamples();
loadSample(0);
