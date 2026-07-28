/* Querio front end.
 *
 * Talks to the FastAPI backend in server.py. The session id lives in an
 * HttpOnly cookie, so nothing here reads or stores credentials: the connect
 * form posts them once and they are never sent back to the browser.
 */

(() => {
  "use strict";

  // Engine badge colours, matching the design mock.
  const ENGINE_STYLE = {
    postgresql: { initials: "Pg", color: "oklch(0.5 0.14 255)" },
    mysql:      { initials: "My", color: "oklch(0.6 0.14 55)" },
    sqlite:     { initials: "Sq", color: "oklch(0.5 0.01 250)" },
    mssql:      { initials: "MS", color: "oklch(0.55 0.18 25)" },
    oracle:     { initials: "Or", color: "oklch(0.55 0.14 30)" },
  };

  const state = {
    connected: false,
    engines: [],
    conversations: [],
    activeId: null,
    messages: [],
    suggestions: [],
    pendingEngine: "postgresql",
    usingUrl: false,
    busy: false,
    // The live connection, kept so history threads can be compared against it.
    connectionName: "",
    connectionEngine: "",
    connectionEngineLabel: "",
    connectionFullName: "",
  };

  const $ = (id) => document.getElementById(id);

  const el = {
    connCard: $("conn-card"),
    connName: $("conn-name"),
    connMeta: $("conn-meta"),
    connEmpty: $("conn-empty"),
    sidebarConversations: $("sidebar-conversations"),
    historyList: $("history-list"),
    schemaPanel: $("schema-panel"),
    schemaCount: $("schema-count"),
    schemaTables: $("schema-tables"),
    headerTitle: $("header-title"),
    headerBadge: $("header-badge"),
    badgeText: $("badge-text"),
    viewConnect: $("view-connect"),
    viewChat: $("view-chat"),
    engineGrid: $("engine-grid"),
    chatScroll: $("chat-scroll"),
    suggestions: $("suggestions"),
    suggestionList: $("suggestion-list"),
    messages: $("messages"),
    input: $("input"),
    send: $("btn-send"),
    modal: $("modal"),
    modalMark: $("modal-mark"),
    modalTitle: $("modal-title"),
    modalError: $("modal-error"),
    form: $("connect-form"),
    fieldsStandard: $("fields-standard"),
    fieldsSqlite: $("fields-sqlite"),
    fieldsUrl: $("fields-url"),
    btnConnect: $("btn-connect"),
    btnCancel: $("btn-cancel"),
    btnToggleUrl: $("btn-toggle-url"),
    btnUseUrl: $("btn-use-url"),
    btnChange: $("btn-change-connection"),
    btnNew: $("btn-new-question"),
    btnToggleSchema: $("btn-toggle-schema"),
    fHost: $("f-host"),
    fPort: $("f-port"),
    fDatabase: $("f-database"),
    fUsername: $("f-username"),
    fPassword: $("f-password"),
    fSqlitePath: $("f-sqlite-path"),
    fUrl: $("f-url"),
  };

  // ---------- utilities ----------

  async function api(path, options = {}) {
    const response = await fetch(path, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`);
    }
    return response.json();
  }

  /** Escape text before it goes anywhere near innerHTML. */
  function esc(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function relativeTime(seconds) {
    const diff = Date.now() / 1000 - seconds;
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
    const when = new Date(seconds * 1000);
    const time = when.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    if (diff < 86400) return `Today · ${time}`;
    if (diff < 172800) return `Yesterday · ${time}`;
    return when.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  /** A deliberately small Markdown subset: what the agent actually emits. */
  function renderMarkdown(text) {
    const source = String(text ?? "");
    const lines = source.split("\n");
    const out = [];
    let paragraph = [];
    let list = null;
    let table = null;

    const flushParagraph = () => {
      if (paragraph.length) {
        out.push(`<p>${inline(paragraph.join(" "))}</p>`);
        paragraph = [];
      }
    };
    const flushList = () => {
      if (list) {
        out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`);
        list = null;
      }
    };
    const flushTable = () => {
      if (table && table.rows.length) {
        const head = table.head.map((h) => `<th>${inline(h)}</th>`).join("");
        const body = table.rows
          .map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`)
          .join("");
        out.push(`<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`);
      }
      table = null;
    };
    const flushAll = () => { flushParagraph(); flushList(); flushTable(); };

    const splitRow = (line) =>
      line.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

    for (const raw of lines) {
      const line = raw.trimEnd();

      if (!line.trim()) { flushAll(); continue; }

      // Table rows
      if (/^\s*\|.*\|\s*$/.test(line)) {
        const cells = splitRow(line.trim());
        if (/^[\s|:-]+$/.test(line)) continue;      // separator row
        flushParagraph(); flushList();
        if (!table) table = { head: cells, rows: [] };
        else table.rows.push(cells);
        continue;
      }
      flushTable();

      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        flushAll();
        const level = Math.min(heading[1].length + 2, 6);
        out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
        continue;
      }

      const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
      if (bullet) {
        flushParagraph();
        if (!list || list.tag !== "ul") { flushList(); list = { tag: "ul", items: [] }; }
        list.items.push(bullet[1]);
        continue;
      }

      const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (numbered) {
        flushParagraph();
        if (!list || list.tag !== "ol") { flushList(); list = { tag: "ol", items: [] }; }
        list.items.push(numbered[1]);
        continue;
      }

      flushList();
      paragraph.push(line.trim());
    }

    flushAll();
    return out.join("");
  }

  function inline(text) {
    // Escape first: everything below only re-introduces tags we control.
    return esc(text)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
  }

  // ---------- rendering ----------

  function renderConnection(data) {
    state.connected = Boolean(data.connected);

    el.connCard.classList.toggle("is-hidden", !state.connected);
    el.connEmpty.classList.toggle("is-hidden", state.connected);
    el.sidebarConversations.classList.toggle("is-hidden", !state.connected);
    el.schemaPanel.classList.toggle("is-hidden", !state.connected);
    el.headerBadge.classList.toggle("is-hidden", !state.connected);
    el.viewConnect.classList.toggle("is-hidden", state.connected);
    el.viewChat.classList.toggle("is-hidden", !state.connected);

    if (state.connected) {
      state.connectionName = data.name || "";
      state.connectionEngine = data.engine || "";
      state.connectionEngineLabel = data.engineLabel || "";
      state.connectionFullName = data.fullName || data.name || "";
      el.connName.textContent = state.connectionName;
      // The card shows the short name; the full path stays reachable on hover.
      el.connName.title = state.connectionFullName;
      el.connMeta.textContent = `${state.connectionEngineLabel} · connected`;
      setHeaderDatabase(state.connectionName, state.connectionEngine, state.connectionEngineLabel);
      loadSchema();
    } else {
      state.connectionName = "";
      state.connectionEngine = "";
      state.connectionEngineLabel = "";
      state.connectionFullName = "";
      el.headerTitle.textContent = "Querio";
    }
  }

  function renderEngines() {
    el.engineGrid.innerHTML = "";
    for (const engine of state.engines) {
      const style = ENGINE_STYLE[engine.id] || { initials: "DB", color: "oklch(0.5 0.02 250)" };
      const button = document.createElement("button");
      button.type = "button";
      button.className = "engine-card";
      button.innerHTML =
        `<div class="engine-initials" style="background:${style.color}">${esc(style.initials)}</div>` +
        `<div class="engine-name">${esc(engine.label)}</div>`;
      button.addEventListener("click", () => openModal(engine.id));
      el.engineGrid.appendChild(button);
    }
  }

  function renderHistory() {
    el.historyList.innerHTML = "";
    for (const conversation of state.conversations) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-item" + (conversation.id === state.activeId ? " is-active" : "");

      // Which database this thread was asked against. The server stamps it when
      // the conversation is created, so reconnecting elsewhere does not relabel
      // older threads.
      let meta = `<div class="history-time">${esc(relativeTime(conversation.updated_at))}</div>`;
      if (conversation.database) {
        const style = ENGINE_STYLE[conversation.engine] || { color: "oklch(0.5 0.02 250)" };
        const title = `${conversation.engineLabel || "Database"} · ${conversation.database}`;
        meta +=
          `<span class="history-sep">·</span>` +
          `<span class="history-db" title="${esc(title)}">` +
          `<span class="history-db-dot" style="background:${style.color}"></span>` +
          `<span class="history-db-name">${esc(conversation.database)}</span>` +
          `</span>`;
      }

      button.innerHTML =
        `<div class="history-title">${esc(conversation.title)}</div>` +
        `<div class="history-meta">${meta}</div>`;
      button.addEventListener("click", () => openConversation(conversation.id));
      el.historyList.appendChild(button);
    }
  }

  function renderMessages() {
    el.messages.innerHTML = "";
    const hasMessages = state.messages.length > 0;
    el.suggestions.classList.toggle("is-hidden", hasMessages || !state.suggestions.length);

    for (const message of state.messages) {
      el.messages.appendChild(
        message.role === "user" ? userBubble(message) : assistantBubble(message)
      );
    }
    scrollToBottom();
  }

  function userBubble(message) {
    const row = document.createElement("div");
    row.className = "row row-user";
    const bubble = document.createElement("div");
    bubble.className = "bubble-user";
    bubble.textContent = message.text;
    row.appendChild(bubble);
    return row;
  }

  function assistantBubble(message) {
    const row = document.createElement("div");
    row.className = "row row-assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble-assistant" + (message.error ? " is-error" : "");

    const answer = document.createElement("div");
    answer.className = "answer";
    answer.innerHTML = renderMarkdown(message.text);
    bubble.appendChild(answer);

    const chart = buildChart(message);
    if (chart) bubble.appendChild(chart);

    if (message.rows && message.rows.length) {
      bubble.appendChild(buildTable(message));
    }

    if (message.corrections && message.corrections.length) {
      const note = document.createElement("div");
      note.className = "corrections";
      note.textContent =
        "Names corrected automatically: " + message.corrections.join("; ");
      bubble.appendChild(note);
    }

    if (message.sql) {
      bubble.appendChild(buildSqlDisclosure(message.sql));
    }

    row.appendChild(bubble);
    return row;
  }

  function buildTable(message) {
    const wrap = document.createElement("div");

    const scroller = document.createElement("div");
    scroller.className = "result-table-wrap";

    const table = document.createElement("table");
    table.className = "result-table";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const column of message.columns) {
      const th = document.createElement("th");
      th.textContent = column;
      headRow.appendChild(th);
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const row of message.rows) {
      const tr = document.createElement("tr");
      for (const cell of row) {
        const td = document.createElement("td");
        td.textContent = cell === null || cell === undefined ? "—" : String(cell);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    scroller.appendChild(table);
    wrap.appendChild(scroller);

    const note = document.createElement("div");
    note.className = "result-note";
    note.textContent =
      `${message.rows.length} row${message.rows.length === 1 ? "" : "s"}` +
      (message.truncated ? " (truncated)" : "");
    wrap.appendChild(note);

    return wrap;
  }

  /**
   * Draw a bar chart when the result looks like a label/number series.
   * Anything else renders as a table only - a chart of unrelated numbers
   * would be worse than none.
   */
  function buildChart(message) {
    const rows = message.rows || [];
    const columns = message.columns || [];
    if (rows.length < 2 || rows.length > 12 || columns.length < 2) return null;

    const labelIndex = 0;
    let valueIndex = -1;
    for (let i = 1; i < columns.length; i += 1) {
      if (rows.every((r) => typeof r[i] === "number")) { valueIndex = i; break; }
    }
    if (valueIndex === -1) return null;
    if (rows.some((r) => typeof r[labelIndex] === "number")) return null;

    const values = rows.map((r) => r[valueIndex]);
    const max = Math.max(...values);
    if (!(max > 0)) return null;

    const chart = document.createElement("div");
    chart.className = "chart";

    for (const row of rows) {
      const value = row[valueIndex];
      const col = document.createElement("div");
      col.className = "chart-col";

      const valueLabel = document.createElement("div");
      valueLabel.className = "chart-value";
      valueLabel.textContent = formatNumber(value);

      const bar = document.createElement("div");
      bar.className = "chart-bar";
      bar.style.height = `${Math.max(2, (value / max) * 78)}px`;

      const label = document.createElement("div");
      label.className = "chart-label";
      label.textContent = String(row[labelIndex] ?? "");
      label.title = String(row[labelIndex] ?? "");

      col.append(valueLabel, bar, label);
      chart.appendChild(col);
    }
    return chart;
  }

  function formatNumber(value) {
    if (typeof value !== "number") return String(value);
    if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
    if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }

  function buildSqlDisclosure(sql) {
    const wrap = document.createElement("div");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "sql-toggle";
    toggle.textContent = "▸ View SQL";

    const block = document.createElement("pre");
    block.className = "sql-block is-hidden";
    block.textContent = sql;

    toggle.addEventListener("click", () => {
      const hidden = block.classList.toggle("is-hidden");
      toggle.textContent = hidden ? "▸ View SQL" : "▾ Hide SQL";
    });

    wrap.append(toggle, block);
    return wrap;
  }

  function renderSuggestions() {
    el.suggestionList.innerHTML = "";
    for (const text of state.suggestions) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "suggestion";
      button.textContent = text;
      button.addEventListener("click", () => {
        el.input.value = text;
        syncSendButton();
        submitQuestion();
      });
      el.suggestionList.appendChild(button);
    }
    el.suggestions.classList.toggle(
      "is-hidden",
      state.messages.length > 0 || !state.suggestions.length
    );
  }

  function showThinking() {
    const row = document.createElement("div");
    row.className = "row row-assistant";
    row.id = "thinking-row";
    row.innerHTML = `<div class="thinking"><i></i><i></i><i></i></div>`;
    el.messages.appendChild(row);
    scrollToBottom();
  }

  function hideThinking() {
    const row = $("thinking-row");
    if (row) row.remove();
  }

  function scrollToBottom() {
    // Two frames: the first lets the browser lay the new nodes out, the second
    // scrolls once their real heights are known. Scrolling in the same frame
    // measures a stale scrollHeight and lands short.
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        el.chatScroll.scrollTop = el.chatScroll.scrollHeight;
      });
    });
  }

  // ---------- data flow ----------

  async function loadState() {
    const data = await api("/api/state");
    state.engines = data.engines || [];
    state.conversations = data.conversations || [];
    state.suggestions = data.suggestions || [];
    renderEngines();
    renderConnection(data);
    renderHistory();
    renderSuggestions();
    renderMessages();
  }

  async function loadSchema() {
    try {
      const data = await api("/api/schema");
      const tables = data.tables || [];
      el.schemaCount.textContent = `${tables.length} TABLE${tables.length === 1 ? "" : "S"}`;
      el.schemaTables.innerHTML = "";
      for (const table of tables) {
        const row = document.createElement("div");
        row.className = "schema-row";
        row.innerHTML =
          `<span title="${esc(table.columns.join(", "))}">${esc(table.name)}</span>` +
          `<span>${table.rows ? table.rows.toLocaleString() : "—"}</span>`;
        el.schemaTables.appendChild(row);
      }
    } catch (error) {
      el.schemaCount.textContent = "TABLES";
    }
  }

  /** Re-read the conversation list from the server and repaint the sidebar. */
  async function refreshConversations() {
    try {
      const data = await api("/api/state");
      state.conversations = data.conversations || [];
      renderHistory();
    } catch (error) {
      /* Leave the current list in place if the refresh fails. */
    }
  }

  async function openConversation(id) {
    const data = await api(`/api/conversations/${encodeURIComponent(id)}`);
    if (data.error) return;
    state.activeId = data.id;
    state.messages = data.messages || [];
    el.headerTitle.textContent = data.title || "Querio";
    // Reflect the database this thread belongs to, which may differ from the
    // one currently connected.
    setHeaderDatabase(data.database, data.engine, data.engineLabel);
    renderHistory();
    renderMessages();
  }

  /**
   * Point the topbar badge at a specific database. Called with a conversation's
   * stored database when browsing history, and with the live connection
   * otherwise. Falls back to the live connection when a conversation has no
   * stamp (one created before this was recorded).
   */
  function setHeaderDatabase(database, engine, engineLabel) {
    const name = database || state.connectionName;
    const label = engineLabel || state.connectionEngineLabel;
    if (!name) {
      el.headerBadge.classList.add("is-hidden");
      return;
    }
    el.headerBadge.classList.remove("is-hidden");
    el.badgeText.textContent = `${label || ""} · ${name}`;

    // A thread asked against a database other than the one attached now is
    // worth flagging, since its answers describe data that is no longer live.
    const stale = Boolean(database) && Boolean(state.connectionName) &&
      database !== state.connectionName;
    el.headerBadge.classList.toggle("is-stale", stale);
    el.headerBadge.title = stale
      ? `This conversation used ${name}. You are now connected to ${state.connectionName}.`
      : "";
  }

  function startNewConversation() {
    state.activeId = null;
    state.messages = [];
    el.headerTitle.textContent = "New question";
    // A fresh question runs against whatever is connected now.
    setHeaderDatabase(state.connectionName, state.connectionEngine, state.connectionEngineLabel);
    renderHistory();
    renderMessages();
    renderSuggestions();
    el.input.focus();
  }

  async function submitQuestion() {
    const question = el.input.value.trim();
    if (!question || state.busy || !state.connected) return;

    state.busy = true;
    syncSendButton();

    state.messages.push({ role: "user", text: question });
    el.input.value = "";
    autoGrow();
    renderMessages();
    showThinking();

    try {
      const data = await api("/api/ask", {
        method: "POST",
        body: JSON.stringify({ question, conversationId: state.activeId }),
      });
      hideThinking();

      state.activeId = data.conversationId || state.activeId;
      state.conversations = data.conversations || state.conversations;
      if (data.message) state.messages.push(data.message);
      if (data.title) el.headerTitle.textContent = data.title;

      renderHistory();
      renderMessages();
    } catch (error) {
      hideThinking();
      state.messages.push({
        role: "assistant",
        text: `Could not reach the server. ${error.message}`,
        error: true,
      });
      renderMessages();
    } finally {
      state.busy = false;
      syncSendButton();
      el.input.focus();
    }
  }

  // ---------- connect modal ----------

  function openModal(engineId) {
    state.pendingEngine = engineId || "postgresql";
    state.usingUrl = false;
    const engine = state.engines.find((e) => e.id === state.pendingEngine);
    const style = ENGINE_STYLE[state.pendingEngine] || { initials: "DB", color: "var(--accent)" };

    el.modalMark.textContent = style.initials;
    el.modalMark.style.background = style.color;
    el.modalTitle.textContent = `Connect to ${engine ? engine.label : "database"}`;
    if (engine && engine.defaultPort) el.fPort.value = engine.defaultPort;

    el.modalError.classList.add("is-hidden");
    applyModalMode();
    el.modal.classList.remove("is-hidden");

    const first = state.pendingEngine === "sqlite" ? el.fSqlitePath : el.fHost;
    setTimeout(() => first.focus(), 40);
  }

  function applyModalMode() {
    const isSqlite = state.pendingEngine === "sqlite";
    el.fieldsUrl.classList.toggle("is-hidden", !state.usingUrl);
    el.fieldsStandard.classList.toggle("is-hidden", state.usingUrl || isSqlite);
    el.fieldsSqlite.classList.toggle("is-hidden", state.usingUrl || !isSqlite);
    el.btnToggleUrl.textContent = state.usingUrl ? "Use fields" : "Use connection string";
  }

  function closeModal() {
    el.modal.classList.add("is-hidden");
    // Never leave a password sitting in the DOM after the modal closes.
    el.fPassword.value = "";
    el.fUrl.value = "";
  }

  async function submitConnect(event) {
    event.preventDefault();
    if (state.busy) return;

    const payload = state.usingUrl
      ? { url: el.fUrl.value }
      : state.pendingEngine === "sqlite"
      ? { engine: "sqlite", database: el.fSqlitePath.value }
      : {
          engine: state.pendingEngine,
          host: el.fHost.value,
          port: el.fPort.value,
          database: el.fDatabase.value,
          username: el.fUsername.value,
          password: el.fPassword.value,
        };

    state.busy = true;
    el.btnConnect.disabled = true;
    el.btnConnect.innerHTML = `<span class="spinner"></span>Connecting…`;
    el.modalError.classList.add("is-hidden");

    try {
      const data = await api("/api/connect", {
        method: "POST",
        body: JSON.stringify(payload),
      });

      if (!data.ok) {
        el.modalError.textContent = data.message || "Could not connect.";
        el.modalError.classList.remove("is-hidden");
        return;
      }

      // Clear the credentials from the form as soon as they are accepted.
      el.fPassword.value = "";
      el.fUrl.value = "";

      state.suggestions = data.suggestions || [];
      state.messages = [];
      state.activeId = null;
      closeModal();
      renderConnection(data);
      renderSuggestions();
      renderMessages();
      // Earlier threads survive a reconnect on the server, so bring the
      // sidebar back in step with it - they stay readable, each labelled with
      // the database it actually used.
      refreshConversations();
      el.input.focus();
    } catch (error) {
      el.modalError.textContent = error.message;
      el.modalError.classList.remove("is-hidden");
    } finally {
      state.busy = false;
      el.btnConnect.disabled = false;
      el.btnConnect.textContent = "Connect";
    }
  }

  async function disconnect() {
    await api("/api/disconnect", { method: "POST" });
    state.connected = false;
    state.messages = [];
    state.activeId = null;
    state.suggestions = [];
    // state.conversations is deliberately kept: disconnecting drops the
    // database, not the history. The server still holds those threads, and
    // they reappear in the sidebar once a database is attached again.
    renderConnection({ connected: false });
    renderHistory();
    renderMessages();
  }

  // ---------- input behaviour ----------

  function autoGrow() {
    el.input.style.height = "auto";
    el.input.style.height = `${Math.min(el.input.scrollHeight, 180)}px`;
  }

  function syncSendButton() {
    el.send.disabled = state.busy || !el.input.value.trim();
  }

  el.input.addEventListener("input", () => { autoGrow(); syncSendButton(); });
  el.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitQuestion();
    }
  });

  el.send.addEventListener("click", submitQuestion);
  el.form.addEventListener("submit", submitConnect);
  el.btnCancel.addEventListener("click", closeModal);
  el.btnChange.addEventListener("click", disconnect);
  el.btnNew.addEventListener("click", startNewConversation);
  el.btnUseUrl.addEventListener("click", () => {
    openModal(state.pendingEngine);
    state.usingUrl = true;
    applyModalMode();
    setTimeout(() => el.fUrl.focus(), 40);
  });
  el.btnToggleUrl.addEventListener("click", () => {
    state.usingUrl = !state.usingUrl;
    applyModalMode();
  });
  el.btnToggleSchema.addEventListener("click", () => {
    el.schemaTables.classList.toggle("is-hidden");
  });

  el.modal.addEventListener("mousedown", (event) => {
    if (event.target === el.modal) closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !el.modal.classList.contains("is-hidden")) closeModal();
  });

  loadState().catch((error) => {
    document.body.innerHTML =
      `<div style="padding:40px;font-family:system-ui">Could not load Querio: ${esc(error.message)}</div>`;
  });
})();
