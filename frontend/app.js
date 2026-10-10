(() => {
  "use strict";

  const apiBaseUrl = (window.LEGAL_RAG_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
  const form = document.querySelector("#query-form");
  const queryInput = document.querySelector("#query");
  const characterCount = document.querySelector("#query-count");
  const submitButton = document.querySelector("#submit-button");
  const buttonLabel = document.querySelector(".button-label");
  const requestStatus = document.querySelector("#request-status");
  const results = document.querySelector("#results");
  const answerText = document.querySelector("#answer-text");
  const metrics = document.querySelector("#metrics");
  const sources = document.querySelector("#sources");
  const sourceCount = document.querySelector("#source-count");
  const emptySources = document.querySelector("#empty-sources");
  const warning = document.querySelector("#warning");
  const citationWarning = document.querySelector("#citation-warning");
  const disclaimer = document.querySelector("#disclaimer");
  const resultIntent = document.querySelector("#result-intent");
  const diagnostics = document.querySelector("#diagnostics");

  queryInput.addEventListener("input", () => {
    characterCount.textContent = `${queryInput.value.length.toLocaleString()} / 4,000`;
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = queryInput.value.trim();
    if (!query) {
      queryInput.setCustomValidity("Enter a legal research question.");
      queryInput.reportValidity();
      queryInput.setCustomValidity("");
      return;
    }
    if (query.length < 10) {
      queryInput.setCustomValidity("Enter at least 10 characters.");
      queryInput.reportValidity();
      queryInput.setCustomValidity("");
      return;
    }

    setLoading(true);
    setStatus("Searching the judgment collection…", "loading");
    results.hidden = true;
    clearResultContent();

    try {
      const response = await fetch(`${apiBaseUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      let payload;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail = readableError(payload);
        throw new Error(detail || `The API returned HTTP ${response.status}.`);
      }
      if (!payload || typeof payload !== "object") {
        throw new Error("The API returned an unreadable response.");
      }

      renderResponse(payload);
      results.hidden = false;
      setStatus("Research response received.", "success");
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      const message = error instanceof TypeError
        ? `Could not connect to the API at ${apiBaseUrl}. Check that the backend is running and the configured URL is correct.`
        : error.message || "The request could not be completed.";
      setStatus(message, "error");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(loading) {
    submitButton.disabled = loading;
    submitButton.setAttribute("aria-busy", String(loading));
    buttonLabel.textContent = loading ? "Searching…" : "Search judgments";
  }

  function setStatus(message, state) {
    requestStatus.textContent = message;
    requestStatus.dataset.state = state;
    requestStatus.setAttribute("role", state === "error" ? "alert" : "status");
    requestStatus.setAttribute("aria-live", state === "error" ? "assertive" : "polite");
  }

  function clearResultContent() {
    answerText.replaceChildren();
    metrics.replaceChildren();
    sources.replaceChildren();
    sourceCount.textContent = "";
    emptySources.hidden = true;
    warning.hidden = true;
    warning.textContent = "";
    citationWarning.hidden = true;
    citationWarning.textContent = "";
    disclaimer.textContent = "";
    resultIntent.hidden = true;
    resultIntent.textContent = "";
    diagnostics.hidden = true;
    diagnostics.open = false;
  }

  function renderResponse(data) {
    const answer = typeof data.answer === "string" ? data.answer.trim() : "";
    const sourceItems = Array.isArray(data.sources) ? data.sources : [];
    const sourceIds = new Set(sourceItems
      .map((item) => item && item.source_id)
      .filter((id) => typeof id === "string" && /^\d+$/.test(id)));
    const citationValidation = data.citation_validation && typeof data.citation_validation === "object"
      ? data.citation_validation
      : {};
    const unknownIds = Array.isArray(citationValidation.unknown_source_ids)
      ? citationValidation.unknown_source_ids.filter((id) => /^\d+$/.test(String(id)))
      : [];

    renderAnswer(answer, sourceIds, new Set(unknownIds.map(String)));
    renderSources(sourceItems);
    renderDiagnostics(data.scores, data.metadata, sourceItems);

    if (unknownIds.length) {
      citationWarning.textContent = `The answer refers to source ID${unknownIds.length === 1 ? "" : "s"} that could not be matched to a retrieved judgment: ${unknownIds.map((id) => `[Source ${id}]`).join(", ")}. These references are not linked.`;
      citationWarning.hidden = false;
    }

    if (typeof data.warning === "string" && data.warning.trim()) {
      const otherWarnings = data.warning.split(";")
        .filter((part) => !part.includes("unknown source ID(s)"))
        .join(";").trim();
      if (otherWarnings) {
        warning.textContent = otherWarnings;
        warning.hidden = false;
      }
    }
    if (typeof data.disclaimer === "string" && data.disclaimer.trim()) {
      disclaimer.textContent = data.disclaimer;
    }
    const intent = data.metadata && data.metadata.detected_intent;
    if (typeof intent === "string" && intent.trim()) {
      resultIntent.textContent = intent;
      resultIntent.hidden = false;
    }
  }

  function renderAnswer(answer, sourceIds, unknownIds) {
    answerText.replaceChildren();
    if (!answer) {
      const empty = document.createElement("p");
      empty.className = "answer-empty";
      empty.textContent = "No answer text was returned.";
      answerText.append(empty);
      return;
    }

    let paragraph = [];
    let list = null;
    const flushParagraph = () => {
      if (!paragraph.length) return;
      const block = document.createElement("p");
      appendInline(block, paragraph.join(" "), sourceIds, unknownIds);
      answerText.append(block);
      paragraph = [];
    };
    const flushList = () => { list = null; };

    for (const line of answer.split(/\r?\n/)) {
      const itemMatch = line.match(/^\s*(?:[-*]\s+|\d+[.)]\s+)(.*)$/);
      if (itemMatch) {
        flushParagraph();
        const ordered = /^\s*\d+[.)]\s+/.test(line);
        const tagName = ordered ? "OL" : "UL";
        if (!list || list.tagName !== tagName) {
          list = document.createElement(ordered ? "ol" : "ul");
          answerText.append(list);
        }
        const item = document.createElement("li");
        appendInline(item, itemMatch[1], sourceIds, unknownIds);
        list.append(item);
      } else if (!line.trim()) {
        flushParagraph();
        flushList();
      } else {
        flushList();
        paragraph.push(line.trim());
      }
    }
    flushParagraph();
  }

  function appendInline(parent, text, sourceIds, unknownIds) {
    const tokens = /\[\s*Source\s+(\d+)(?:\s*:[^\]]*)?\s*\]|\*\*([^*]+)\*\*/gi;
    let previous = 0;
    for (const match of text.matchAll(tokens)) {
      parent.append(document.createTextNode(text.slice(previous, match.index)));
      if (match[1]) {
        const sourceId = match[1];
        if (sourceIds.has(sourceId) && !unknownIds.has(sourceId)) {
          const link = document.createElement("a");
          link.className = "inline-citation";
          link.href = `#source-${sourceId}`;
          link.textContent = match[0];
          link.setAttribute("aria-label", `Jump to supporting judgment ${sourceId}`);
          parent.append(link);
        } else {
          const unresolved = document.createElement("span");
          unresolved.className = "unresolved-citation";
          unresolved.textContent = match[0];
          parent.append(unresolved);
        }
      } else {
        const strong = document.createElement("strong");
        strong.textContent = match[2];
        parent.append(strong);
      }
      previous = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(previous)));
  }

  function renderDiagnostics(scores, metadata, sourceItems) {
    const items = [];
    if (scores && hasValue(scores.relevance)) {
      items.push(["Retrieval reranker score (not a probability)", formatScore(scores.relevance)]);
    }
    if (scores && hasValue(scores.faithfulness)) {
      items.push(["Rule-based faithfulness (not legal verification)", formatScore(scores.faithfulness)]);
    }
    if (metadata && hasValue(metadata.latency_ms)) {
      const latency = Number(metadata.latency_ms);
      if (Number.isFinite(latency)) items.push(["Latency", `${latency.toLocaleString()} ms`]);
    }
    if (metadata && hasValue(metadata.chunks_retrieved)) {
      items.push(["Chunks retrieved", String(metadata.chunks_retrieved)]);
    }
    if (metadata && typeof metadata.self_healed === "boolean") {
      items.push(["Retrieval recovery", metadata.self_healed ? "Recovered" : "Not triggered"]);
    }
    if (metadata && hasValue(metadata.model)) items.push(["Model", String(metadata.model)]);
    if (metadata && typeof metadata.rewrite_attempted === "boolean") {
      items.push(["Query rewrite attempted", metadata.rewrite_attempted ? "Yes" : "No"]);
    }
    if (metadata && typeof metadata.retry_attempted === "boolean") {
      items.push(["Retrieval retry attempted", metadata.retry_attempted ? "Yes" : "No"]);
    }
    if (metadata && typeof metadata.reference_recovered === "boolean") {
      items.push(["Statutory reference recovered", metadata.reference_recovered ? "Yes" : "No"]);
    }
    for (const source of sourceItems) {
      if (!source || !hasValue(source.relevance_score) || !/^\d+$/.test(String(source.source_id || ""))) continue;
      items.push([`Source ${source.source_id} reranker score`, formatScore(source.relevance_score)]);
    }

    for (const [label, value] of items) {
      const wrapper = document.createElement("div");
      wrapper.className = "metric";
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      description.textContent = value;
      wrapper.append(term, description);
      metrics.append(wrapper);
    }
    diagnostics.hidden = items.length === 0;
  }

  function renderSources(items) {
    const unique = [];
    const seen = new Set();
    for (const item of items) {
      if (!item || typeof item !== "object") continue;
      const id = typeof item.source_id === "string" && /^\d+$/.test(item.source_id)
        ? `source:${item.source_id}`
        : validatedHttpUrl(item.url) || `record:${unique.length}`;
      if (seen.has(id)) continue;
      seen.add(id);
      unique.push(item);
    }
    sourceCount.textContent = `${unique.length} ${unique.length === 1 ? "judgment" : "judgments"}`;
    emptySources.hidden = unique.length > 0;

    for (const item of unique) {
      const card = document.createElement("article");
      card.className = "source-card";
      const sourceId = typeof item.source_id === "string" && /^\d+$/.test(item.source_id)
        ? item.source_id
        : "";
      if (sourceId) {
        card.id = `source-${sourceId}`;
        card.tabIndex = -1;
        const sourceLabel = document.createElement("p");
        sourceLabel.className = "source-label";
        sourceLabel.textContent = `[Source ${item.source_id}]`;
        card.append(sourceLabel);
      }
      const titleText = typeof item.title === "string" ? item.title.trim() : "";
      if (titleText) {
        const title = document.createElement("h4");
        title.className = "source-title";
        title.textContent = titleText;
        card.append(title);
      }

      const detailValues = [
        ["Court", item.court],
        ["Date", item.date || item.judgment_date],
        ["Year", item.year],
        ["Case type", item.case_type],
      ];
      const excerpt = typeof item.excerpt === "string" ? item.excerpt.trim() : "";
      if (excerpt) {
        const quote = document.createElement("blockquote");
        quote.className = "source-excerpt";
        quote.textContent = excerpt;
        card.append(quote);
      }
      const safeUrl = validatedHttpUrl(item.url);
      if (safeUrl) {
        const originalLink = document.createElement("a");
        originalLink.className = "original-link";
        originalLink.href = safeUrl;
        originalLink.target = "_blank";
        originalLink.rel = "noopener noreferrer";
        originalLink.textContent = "Open original judgment";
        card.append(originalLink);
      }

      const details = document.createElement("div");
      details.className = "source-details";
      for (const [key, value] of detailValues) {
        if (!hasValue(value) || typeof value !== "string" || !value.trim()) continue;
        const detail = document.createElement("span");
        detail.textContent = `${key}: ${value.trim()}`;
        details.append(detail);
      }
      if (details.childElementCount) card.append(details);
      sources.append(card);
    }
    if (sources.childElementCount === 0) emptySources.hidden = false;
  }

  function validatedHttpUrl(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try {
      const url = new URL(value);
      if ((url.protocol !== "http:" && url.protocol !== "https:") || !url.hostname) return null;
      return url.href;
    } catch {
      return null;
    }
  }

  function hasValue(value) {
    return value !== null && value !== undefined && value !== "";
  }

  function formatScore(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(3) : String(value);
  }

  function readableError(payload) {
    if (!payload || !hasValue(payload.detail)) return "";
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item && item.msg).filter(Boolean).join("; ");
    }
    return "";
  }
})();
