(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const log = $("#log"), form = $("#composer"), input = $("#msg"), sendBtn = $("#sendBtn");
  const topicsEl = $("#topics"), filter = $("#topicFilter"), rail = $("#rail"), toggle = $("#topicsToggle");
  const startersEl = $("#starters");
  const META = window.SUNNY || {};
  const SIGNAL_NAMES = { question: "Question match", char: "Spelling-tolerant", answer: "Answer match",
                         coverage: "Words covered", semantic: "Meaning match" };
  const STARTERS = ["Where is my order?", "How do I return an item?", "When will I get my refund?",
                    "Is Cash on Delivery available?", "I forgot my password"];

  let contextId = null;
  let busy = false;

  // ------------------------------------------------------------- helpers
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|\W)\*(.+?)\*(?=\W|$)/g, "$1<em>$2</em>");

  /** Tiny, safe markdown: paragraphs, **bold**, *italic*, "- " and "1. " lists. */
  function md(text) {
    const out = [];
    let list = null;
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      const ul = line.match(/^[-•]\s+(.*)/), ol = line.match(/^\d+[.)]\s+(.*)/);
      const kind = ul ? "ul" : ol ? "ol" : null;
      if (kind) {
        if (!list || list.kind !== kind) { if (list) out.push(`</${list.kind}>`); list = { kind }; out.push(`<${kind}>`); }
        out.push(`<li>${inline((ul || ol)[1])}</li>`);
        continue;
      }
      if (list) { out.push(`</${list.kind}>`); list = null; }
      if (line) out.push(`<p>${inline(line)}</p>`);
    }
    if (list) out.push(`</${list.kind}>`);
    return out.join("");
  }

  const icon = {
    up: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 11v9H4v-9h3Zm0 0 4-7c1.5 0 2.5 1 2.2 2.6L12.6 10H19a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 17.8 20H7" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
    down: '<svg viewBox="0 0 24 24" aria-hidden="true" style="transform:rotate(180deg)"><path d="M7 11v9H4v-9h3Zm0 0 4-7c1.5 0 2.5 1 2.2 2.6L12.6 10H19a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 17.8 20H7" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
    copy: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
    check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 5 5 9-10" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };

  function scrollDown() { log.scrollTop = log.scrollHeight; }

  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  }

  // ------------------------------------------------------------- persistence (per tab)
  const history = [];
  function save() { try { sessionStorage.setItem("sunny-chat", JSON.stringify({ history, contextId })); } catch (_) {} }
  function restore() {
    try {
      const data = JSON.parse(sessionStorage.getItem("sunny-chat") || "null");
      if (!data || !data.history.length) return false;
      contextId = data.contextId;
      for (const h of data.history) h.role === "user" ? addUser(h.text, false) : addBot(h.res, h.question, false);
      return true;
    } catch (_) { return false; }
  }

  // ------------------------------------------------------------- rendering
  function addUser(text, record = true) {
    const row = el("div", "row user enter");
    row.append(el("div", "bubble", esc(text).replace(/\n/g, "<br>")));
    log.append(row);
    if (record) { history.push({ role: "user", text }); save(); }
    scrollDown();
  }

  function chipRow(label, items) {
    const wrap = el("div", "chips");
    wrap.append(el("p", "chips-label", esc(label)));
    for (const it of items) {
      const b = el("button", "chip", esc(it.question));
      b.type = "button";
      b.addEventListener("click", () => openFaq(it.id, it.question));
      wrap.append(b);
    }
    return wrap;
  }

  function whyPanel(res) {
    const d = res.debug || {};
    const panel = el("div", "why");
    const corr = Object.entries(d.corrections || {});
    let html = "<dl>";
    if (res.matched_question) html += `<dt>Matched</dt><dd>${esc(res.matched_question)}</dd>`;
    if (res.category) html += `<dt>Topic</dt><dd>${esc(res.category)}</dd>`;
    if (d.tokens) html += `<dt>Key words</dt><dd>${d.tokens.map((t) => `<span class="tok">${esc(t)}</span>`).join("") || "none"}</dd>`;
    if (corr.length) html += `<dt>Spelling</dt><dd>${corr.map(([a, b]) => `${esc(a)} → ${esc(b)}`).join(", ")}</dd>`;
    if (d.context_used) html += `<dt>Context</dt><dd>Read together with your previous question</dd>`;
    html += "</dl>";
    if (d.signals) {
      html += '<div class="bars">';
      for (const [k, v] of Object.entries(d.signals)) {
        html += `<span>${SIGNAL_NAMES[k] || k}</span><span class="bar"><i style="width:${Math.round(Math.min(1, v) * 100)}%"></i></span><b>${Math.round(v * 100)}%</b>`;
      }
      html += `<span><strong>Overall</strong></span><span class="bar"><i style="width:${Math.round(Math.min(1, res.confidence) * 100)}%;background:var(--pink)"></i></span><b>${Math.round(res.confidence * 100)}%</b></div>`;
    }
    panel.innerHTML = html;
    return panel;
  }

  function addBot(res, question, record = true) {
    const row = el("div", "row bot enter");
    row.append(el("div", "avatar"));
    const stack = el("div", "stack");
    row.append(stack);

    if (res.warning) stack.append(el("div", "warn", md(res.warning)));

    let body = res.answer;
    if (res.type === "multi") {
      // "**Question**\nanswer" blocks -> headed sections
      body = body.split("\n\n").map((blk) => blk.replace(/^\*\*(.+?)\*\*\n/, "§§$1\n")).join("\n");
    }
    let html = md(body).replace(/<p>§§(.+?)<\/p>/g, '<span class="q-title">$1</span>');
    if (res.type === "clarify") html = `<p><strong>I think you're asking:</strong> ${esc(res.matched_question)}</p>` + html;
    stack.append(el("div", "bubble", html));

    if (res.type !== "smalltalk") {
      const meta = el("div", "meta");
      const label = { high: "Sure match", medium: "Best guess", low: "No match" }[res.confidence_label] || "";
      meta.append(el("span", `conf ${res.confidence_label}`, `<span>${label}</span>`));

      if (res.faq_id) {
        const up = el("button", "icon-btn up", icon.up), down = el("button", "icon-btn down", icon.down);
        up.title = "Helpful"; down.title = "Not helpful";
        up.setAttribute("aria-pressed", "false"); down.setAttribute("aria-pressed", "false");
        const note = el("span", "thanks-note");
        const vote = (helpful) => {
          up.setAttribute("aria-pressed", String(helpful)); down.setAttribute("aria-pressed", String(!helpful));
          note.textContent = helpful ? "Thanks for the feedback" : "Thanks. We'll use this to improve.";
          fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: question, faq_id: res.faq_id, helpful }) }).catch(() => {});
        };
        up.onclick = () => vote(true); down.onclick = () => vote(false);
        const copy = el("button", "icon-btn", icon.copy);
        copy.title = "Copy answer";
        copy.onclick = async () => {
          try { await navigator.clipboard.writeText(res.answer.replace(/\*\*/g, "")); copy.innerHTML = icon.check; setTimeout(() => (copy.innerHTML = icon.copy), 1400); } catch (_) {}
        };
        meta.append(up, down, copy);
        if (res.debug && res.debug.signals) {
          const whyBtn = el("button", "link-btn", "Why this answer?");
          whyBtn.type = "button";
          const panel = whyPanel(res);
          whyBtn.setAttribute("aria-expanded", "false");
          whyBtn.onclick = () => {
            const open = panel.classList.toggle("open");
            whyBtn.setAttribute("aria-expanded", String(open));
            whyBtn.textContent = open ? "Hide details" : "Why this answer?";
            if (open) panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
          };
          meta.append(whyBtn);
          stack.append(meta, panel);
        } else stack.append(meta);
        meta.append(note);
      } else stack.append(meta);
    }

    if (res.type === "clarify" && res.alternatives.length) stack.append(chipRow("Not it? Maybe one of these:", res.alternatives));
    else if (res.type === "fallback" && res.alternatives.length) stack.append(chipRow("Closest topics I know:", res.alternatives));
    else if (res.related && res.related.length) stack.append(chipRow("People also ask", res.related.slice(0, 3)));

    log.append(row);
    if (res.faq_id) contextId = res.faq_id;
    if (record) { history.push({ role: "bot", res, question }); save(); }
    scrollDown();
  }

  function typing() {
    const row = el("div", "row bot typing");
    row.append(el("div", "avatar"));
    row.append(el("div", "bubble", "<i></i><i></i><i></i>"));
    log.append(row);
    scrollDown();
    return row;
  }

  // ------------------------------------------------------------- talking to the server
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  async function respond(question, request) {
    if (busy) return;
    busy = true; sendBtn.disabled = true; startersEl.innerHTML = "";
    const dots = typing();
    try {
      const [res] = await Promise.all([request(), wait(420 + Math.min(600, question.length * 8))]);
      dots.remove();
      addBot(res, question);
    } catch (e) {
      dots.remove();
      addBot({ type: "fallback", answer: "I can't reach the help server right now. Check that `python app.py` is still running, then try again.",
               confidence: 0, confidence_label: "low", alternatives: [], related: [] }, question, false);
    } finally {
      busy = false; sendBtn.disabled = false; input.focus();
    }
  }

  async function postJSON(url, data) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    if (!r.ok) throw new Error(r.status);
    return r.json();
  }

  function ask(text) {
    text = text.trim();
    if (!text || busy) return;
    addUser(text);
    respond(text, () => postJSON("/api/chat", { message: text, context_id: contextId }));
  }

  function openFaq(id, question) {
    if (busy) return;
    addUser(question);
    if (rail.classList.contains("open")) setRail(false);
    respond(question, async () => { const r = await fetch(`/api/faq/${encodeURIComponent(id)}`); if (!r.ok) throw new Error(); return r.json(); });
  }

  // ------------------------------------------------------------- topics rail
  let TOPICS = [];
  function renderTopics(q = "") {
    q = q.trim().toLowerCase();
    topicsEl.innerHTML = "";
    let shown = 0;
    for (const { category: cat, items } of TOPICS) {
      const hits = q ? items.filter((i) => i.question.toLowerCase().includes(q) || cat.toLowerCase().includes(q)) : items;
      if (!hits.length) continue;
      shown++;
      const box = el("div", "topic" + (q ? " open" : ""));
      const btn = el("button", "", `<span>${esc(cat)}</span><span class="count">${hits.length}</span>`);
      btn.type = "button";
      btn.setAttribute("aria-expanded", q ? "true" : "false");
      btn.onclick = () => { const o = box.classList.toggle("open"); btn.setAttribute("aria-expanded", String(o)); };
      const ul = el("ul");
      for (const it of hits) {
        const li = el("li"), b = el("button", "", esc(it.question));
        b.type = "button";
        b.onclick = () => openFaq(it.id, it.question);
        li.append(b); ul.append(li);
      }
      box.append(btn, ul);
      topicsEl.append(box);
    }
    if (!shown) topicsEl.append(el("p", "empty", "No questions match that. Try asking Sunny directly."));
  }

  function renderStarters() {
    startersEl.innerHTML = "";
    for (const s of STARTERS) {
      const b = el("button", "chip", esc(s));
      b.type = "button";
      b.onclick = () => ask(s);
      startersEl.append(b);
    }
  }

  function greet() {
    addBot({ type: "smalltalk", answer: META.greeting || "Hi! How can I help?", confidence: 1, confidence_label: "high",
             alternatives: [], related: [] }, "", true);
    renderStarters();
  }

  // ------------------------------------------------------------- events
  form.addEventListener("submit", (e) => { e.preventDefault(); const t = input.value; input.value = ""; autosize(); ask(t); });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(140, input.scrollHeight) + "px"; }
  input.addEventListener("input", autosize);
  filter.addEventListener("input", () => renderTopics(filter.value));
  const scrim = $("#scrim");
  function setRail(open) {
    rail.classList.toggle("open", open); scrim.classList.toggle("show", open);
    toggle.setAttribute("aria-expanded", String(open));
    if (open) filter.focus();
  }
  toggle.addEventListener("click", () => setRail(!rail.classList.contains("open")));
  scrim.addEventListener("click", () => setRail(false));
  $("#railClose").addEventListener("click", () => { setRail(false); toggle.focus(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && rail.classList.contains("open")) { setRail(false); toggle.focus(); } });
  $("#clearBtn").addEventListener("click", () => {
    log.innerHTML = ""; history.length = 0; contextId = null;
    try { sessionStorage.removeItem("sunny-chat"); } catch (_) {}
    greet(); input.focus();
  });

  fetch("/api/meta").then((r) => r.json()).then((m) => { TOPICS = m.topics; renderTopics(); }).catch(() => {
    topicsEl.append(el("p", "empty", "Topics couldn't load. Is the server running?"));
  });
  if (!restore()) greet();
  else if (history.length === 1) renderStarters();
  input.focus();
})();
