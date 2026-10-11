/* Decode / prefill speed against context depth.
   Data: #depth-data (JSON, from tools/extract_bench.py). Draws into #depth-chart. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var host = document.getElementById("depth-chart");
  var dataEl = document.getElementById("depth-data");
  if (!host || !dataEl) return;

  var data;
  try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }

  var SERIES = [
    { key: "q2", color: "var(--d1)", short: "Q2_0" },
    { key: "iq3", color: "var(--d2)", short: "IQ3_XXS" },
    { key: "ds", color: "var(--d3)", short: "DeepSeek V4 Flash" }
  ].filter(function (s) { return data[s.key]; });

  var METRICS = {
    decode: { label: "tok/s", max: 50, step: 10, digits: 1 },
    prefill: { label: "tok/s", max: 400, step: 100, digits: 0 }
  };
  var W = 900, H = 340, M = { l: 44, r: 58, t: 14, b: 34 };
  var X_MAX = 420000, TRAINED = 262144;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var state = { metric: "decode", hidden: {} };
  var svg, tip, plot;

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function sx(v) { return M.l + (v / X_MAX) * (W - M.l - M.r); }
  function sy(v) { var m = METRICS[state.metric].max; return H - M.b - (v / m) * (H - M.t - M.b); }
  function fmt(v) { return v.toFixed(METRICS[state.metric].digits); }
  function kfmt(v) { return Math.round(v / 1000) + "K"; }

  function draw() {
    host.textContent = "";
    var m = METRICS[state.metric];
    svg = el("svg", { viewBox: "0 0 " + W + " " + H, "class": "chart", role: "img",
      "aria-label": "Line chart of " + state.metric + " speed in tokens per second against context length, one line per model" }, host);

    for (var v = 0; v <= m.max; v += m.step) {
      el("line", { x1: M.l, x2: W - M.r, y1: sy(v), y2: sy(v), "class": v === 0 ? "axis" : "grid" }, svg);
      var t = el("text", { x: M.l - 8, y: sy(v) + 4, "text-anchor": "end" }, svg);
      t.textContent = v;
    }
    for (var x = 0; x <= 400000; x += 100000) {
      var lab = el("text", { x: sx(x), y: H - M.b + 20, "text-anchor": "middle" }, svg);
      lab.textContent = x === 0 ? "0" : kfmt(x);
    }
    var axisTitle = el("text", { x: W - M.r, y: H - 4, "text-anchor": "end" }, svg);
    axisTitle.textContent = "tokens already in the context";

    el("line", { x1: sx(TRAINED), x2: sx(TRAINED), y1: M.t, y2: H - M.b, "class": "mark" }, svg);
    var tr = el("text", { x: sx(TRAINED) + 6, y: M.t + 10 }, svg);
    tr.textContent = "trained window 262K";

    plot = el("g", {}, svg);
    SERIES.forEach(function (s, i) {
      if (state.hidden[s.key]) return;
      var pts = data[s.key].points;
      var d = pts.map(function (p, j) {
        return (j ? "L" : "M") + sx(p.ctx).toFixed(1) + " " + sy(p[state.metric]).toFixed(1);
      }).join("");
      var path = el("path", { d: d, "class": "line", stroke: s.color }, plot);
      if (!reduce) {
        var len = path.getTotalLength();
        path.style.strokeDasharray = len;
        path.style.strokeDashoffset = len;
        path.getBoundingClientRect();
        path.style.transition = "stroke-dashoffset 1.1s cubic-bezier(0.22,1,0.36,1) " + (i * 0.12) + "s";
        path.style.strokeDashoffset = 0;
      }
      pts.forEach(function (p) {
        el("circle", { cx: sx(p.ctx), cy: sy(p[state.metric]), r: 3.2, fill: s.color, "class": "dot" }, plot);
      });
      var last = pts[pts.length - 1];
      var end = el("text", { x: sx(last.ctx) + 9, y: sy(last[state.metric]) + 4, fill: s.color, "class": "endlabel" }, plot);
      end.style.fill = s.color;
      end.textContent = fmt(last[state.metric]);
    });

    var cross = el("line", { y1: M.t, y2: H - M.b, "class": "cross", opacity: 0 }, svg);
    var hit = el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", function (e) { onMove(e, cross); });
    hit.addEventListener("pointerleave", function () { cross.setAttribute("opacity", 0); tip.classList.remove("on"); });
  }

  function nearest(pts, ctx) {
    var best = null, dist = Infinity;
    pts.forEach(function (p) {
      var d = Math.abs(p.ctx - ctx);
      if (d < dist) { dist = d; best = p; }
    });
    return { p: best, d: dist };
  }

  function onMove(e, cross) {
    var box = svg.getBoundingClientRect();
    var px = ((e.clientX - box.left) / box.width) * W;
    var ctx = ((px - M.l) / (W - M.l - M.r)) * X_MAX;
    var rows = [];
    var anchor = null;
    SERIES.forEach(function (s) {
      if (state.hidden[s.key]) return;
      var n = nearest(data[s.key].points, ctx);
      if (n.d > 28000) return;
      rows.push({ s: s, p: n.p });
      if (!anchor || n.d < anchor.d) anchor = { d: n.d, p: n.p };
    });
    if (!rows.length) { cross.setAttribute("opacity", 0); tip.classList.remove("on"); return; }
    var cx = sx(anchor.p.ctx);
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("opacity", 1);

    tip.textContent = "";
    var head = document.createElement("div");
    var b = document.createElement("b");
    b.textContent = Math.round(anchor.p.ctx / 1000) + "K tokens in context";
    head.appendChild(b);
    tip.appendChild(head);
    rows.forEach(function (r) {
      var row = document.createElement("div");
      row.className = "row";
      var l = document.createElement("span");
      var sw = document.createElement("i");
      sw.className = "sw"; sw.style.background = r.s.color;
      l.appendChild(sw); l.appendChild(document.createTextNode(r.s.short));
      var v = document.createElement("span");
      v.textContent = fmt(r.p[state.metric]) + " tok/s";
      row.appendChild(l); row.appendChild(v);
      tip.appendChild(row);
    });
    var left = (cx / W) * box.width + 14;
    if (left > box.width - 190) left -= 190 + 28;
    tip.style.left = left + "px";
    tip.style.top = "14px";
    tip.classList.add("on");
  }

  function build() {
    tip = document.createElement("div");
    tip.className = "tip";
    tip.setAttribute("aria-hidden", "true");
    host.parentNode.insertBefore(tip, host.nextSibling);

    document.querySelectorAll("[data-metric]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.metric = btn.getAttribute("data-metric");
        document.querySelectorAll("[data-metric]").forEach(function (b) {
          b.setAttribute("aria-pressed", String(b === btn));
        });
        draw();
      });
    });

    var legend = document.getElementById("depth-legend");
    if (legend) {
      SERIES.forEach(function (s) {
        var li = document.createElement("li");
        var btn = document.createElement("button");
        btn.type = "button";
        btn.setAttribute("aria-pressed", "true");
        var sw = document.createElement("i");
        sw.className = "sw"; sw.style.background = s.color;
        btn.appendChild(sw);
        btn.appendChild(document.createTextNode(data[s.key].label));
        btn.addEventListener("click", function () {
          state.hidden[s.key] = !state.hidden[s.key];
          btn.setAttribute("aria-pressed", String(!state.hidden[s.key]));
          draw();
        });
        li.appendChild(btn);
        legend.appendChild(li);
      });
    }
    draw();
  }

  build();
})();
