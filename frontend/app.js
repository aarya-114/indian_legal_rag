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
    const sourcesById = new Map();
    for (const source of sourceItems) {
      if (!source || typeof source !== "object") continue;
      const id = typeof source.source_id === "string" ? source.source_id : "";
      if (/^\d+$/.test(id) && !sourcesById.has(id)) {
        sourcesById.set(id, {
          ...source,
          safeUrl: validatedHttpUrl(source.url),
        });
      }
    }
    const citationValidation = data.citation_validation && typeof data.citation_validation === "object"
      ? data.citation_validation
      : {};
    const unknownIds = new Set(
      Array.isArray(citationValidation.unknown_source_ids)
        ? citationValidation.unknown_source_ids
          .map(String).filter((id) => /^\d+$/.test(id))
        : [],
    );
    for (const match of answer.matchAll(/\[\s*Source\s+(\d+)(?:\s*:[^\]]*)?\s*\]/gi)) {
      if (!sourcesById.has(match[1])) unknownIds.add(match[1]);
    }
    for (const match of answer.matchAll(/\bSources?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*/gi)) {
      for (const id of match[0].match(/\d+/g) || []) {
        if (!sourcesById.has(id)) unknownIds.add(id);
      }
    }

    renderAnswer(answer, sourcesById, unknownIds);
    renderSources(sourceItems);
    renderDiagnostics(data.scores, data.metadata, sourceItems);

    if (unknownIds.size) {
      citationWarning.textContent = `The answer refers to source ID${unknownIds.size === 1 ? "" : "s"} that could not be matched to a retrieved judgment: ${Array.from(unknownIds, (id) => `[Source ${id}]`).join(", ")}. These references are not linked.`;
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

  function renderAnswer(answer, sourcesById, unknownIds) {
    answerText.replaceChildren();
    if (!answer) {
      const empty = document.createElement("p");
      empty.className = "answer-empty";
      empty.textContent = "No answer text was returned.";
      answerText.append(empty);
      return;
    }

    const lines = answer.split(/\r?\n/);
    let index = 0;
    let paragraph = [];
    let list = null;
    const flushParagraph = () => {
      if (!paragraph.length) return;
      const text = paragraph.join(" ");
      const block = document.createElement("p");
      appendInline(block, text, sourcesById, unknownIds);
      answerText.append(block);
      paragraph = [];
    };
    const flushList = () => { list = null; };

    while (index < lines.length) {
      const line = lines[index];
      const separatorIndex = findTableSeparator(lines, index);
      if (separatorIndex >= 0) {
        flushParagraph();
        flushList();
        const tableLines = [lines[index], lines[separatorIndex]];
        let next = separatorIndex + 1;
        while (next < lines.length) {
          if (lines[next].includes("|")) {
            tableLines.push(lines[next]);
            next += 1;
          } else if (!lines[next].trim() && next + 1 < lines.length
            && lines[next + 1].includes("|")) {
            next += 1;
          } else {
            break;
          }
        }
        renderStackedTable(tableLines, sourcesById, unknownIds);
        index = next;
        continue;
      }
      const headingMatch = line.match(/^\s*(#{1,3})\s+(.+)$/);
      if (headingMatch) {
        flushParagraph();
        flushList();
        const heading = document.createElement(`h${headingMatch[1].length + 1}`);
        heading.className = "answer-title";
        appendInline(heading, headingMatch[2], sourcesById, unknownIds);
        answerText.append(heading);
        index += 1;
        continue;
      }
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
        appendInline(item, itemMatch[1], sourcesById, unknownIds);
        list.append(item);
      } else if (!line.trim()) {
        flushParagraph();
        flushList();
      } else {
        flushList();
        paragraph.push(line.trim());
      }
      index += 1;
    }
    flushParagraph();
  }

  function findTableSeparator(lines, index) {
    if (!lines[index].includes("|")) return -1;
    let separator = index + 1;
    while (separator < lines.length && !lines[separator].trim()) separator += 1;
    if (separator >= lines.length || !lines[separator].includes("|")) return -1;
    const cells = splitTableRow(lines[separator]);
    return cells.length > 0 && cells.every((cell) => /^\s*:?-{3,}:?\s*$/.test(cell))
      ? separator
      : -1;
  }

  function splitTableRow(line) {
    const escapedPipe = "\u0000";
    return line.trim().replace(/\\\|/g, escapedPipe).replace(/^\|/, "").replace(/\|$/, "")
      .split("|").map((cell) => cell.replaceAll(escapedPipe, "|").trim());
  }

  function renderStackedTable(tableLines, sourcesById, unknownIds) {
    const headers = splitTableRow(tableLines[0]);
    const normalizedHeaders = headers.map((header) => header.replace(/\*\*/g, "").trim());
    const principleIndex = normalizedHeaders.findIndex((header) => /^(?:legal\s+)?(?:principle|factor|consideration|issue|topic|ground|circumstance|category)s?$/i.test(header));
    const sourceIndex = normalizedHeaders.findIndex((header) => /^(?:sources?|citations?|references?)(?:\s*\(.*\))?$/i.test(header));
    const headingIndex = principleIndex >= 0 ? principleIndex : 0;
    const explanationIndices = headers.map((_, i) => i)
      .filter((i) => i !== headingIndex && i !== sourceIndex);

    let rowNumber = 0;
    for (const rowLine of tableLines.slice(2)) {
      const cells = splitTableRow(rowLine);
      if (!cells.some(Boolean)) continue;
      const section = document.createElement("section");
      section.className = "principle-section";
      rowNumber += 1;
      const headingText = cells[headingIndex] || `Principle ${rowNumber}`;
      const heading = document.createElement("h3");
      heading.className = "principle-heading";
      appendInline(heading, headingText, sourcesById, unknownIds);
      section.append(heading);

      for (const cellIndex of explanationIndices) {
        const value = cells[cellIndex] || "";
        if (!value) continue;
        const header = normalizedHeaders[cellIndex] || "";
        const explanation = document.createElement("p");
        explanation.className = "principle-explanation";
        if (header && !/^(explanation|description|details?|reason)$/i.test(header)) {
          const label = document.createElement("strong");
          appendInline(label, `${headers[cellIndex]}: `, sourcesById, unknownIds);
          explanation.append(label);
        }
        appendInline(explanation, value, sourcesById, unknownIds);
        section.append(explanation);
      }

      if (sourceIndex >= 0 && cells[sourceIndex]) {
        const sourceLine = document.createElement("p");
        sourceLine.className = "principle-source";
        const label = document.createElement("strong");
        label.textContent = "Source: ";
        sourceLine.append(label);
        appendTableSource(sourceLine, cells[sourceIndex], sourcesById, unknownIds);
        section.append(sourceLine);
      }
      answerText.append(section);
    }
  }

  function appendTableSource(parent, text, sourcesById, unknownIds) {
    const sourceToken = /\[?\s*Source\s+(\d+)(?:\s*:[^\]]*)?\s*\]?/gi;
    let previous = 0;
    for (const match of text.matchAll(sourceToken)) {
      parent.append(document.createTextNode(text.slice(previous, match.index)));
      appendCitation(parent, match[0], match[1], sourcesById, unknownIds, true);
      previous = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(previous)));
  }

  function appendInline(parent, text, sourcesById, unknownIds) {
    const tokens = /(\[\s*Source\s+\d+(?:\s*:[^\]]*)?\s*\]|\bSources?\s+\d+(?:\s*(?:,|and|&)\s*\d+)*)|\*\*([^*]+)\*\*/gi;
    let previous = 0;
    for (const match of text.matchAll(tokens)) {
      parent.append(document.createTextNode(text.slice(previous, match.index)));
      if (match[1]) {
        if (match[1].startsWith("[")) {
          const sourceId = match[1].match(/\d+/)[0];
          appendCitation(parent, match[1], sourceId, sourcesById, unknownIds, false);
        } else {
          appendCitationGroup(parent, match[1], sourcesById, unknownIds);
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

  function appendCitationGroup(parent, text, sourcesById, unknownIds) {
    let previous = 0;
    let index = 0;
    for (const match of text.matchAll(/\d+/g)) {
      let before = text.slice(previous, match.index);
      if (index === 0) before = before.replace(/^Sources?\s+/i, "");
      parent.append(document.createTextNode(before));
      const sourceId = match[0];
      appendCitation(parent, `Source ${sourceId}`, sourceId, sourcesById, unknownIds, false);
      previous = match.index + match[0].length;
      index += 1;
    }
    parent.append(document.createTextNode(text.slice(previous)));
  }

  function appendCitation(parent, originalText, sourceId, sourcesById, unknownIds, tableSource) {
    const source = sourcesById.get(sourceId);
    if (!source || unknownIds.has(sourceId)) {
      if (!source) unknownIds.add(sourceId);
      const unresolved = document.createElement("span");
      unresolved.className = "unresolved-citation";
      unresolved.textContent = originalText;
      parent.append(unresolved);
      return;
    }
    if (!source.safeUrl) {
      const unavailable = document.createElement("span");
      unavailable.className = "citation-unavailable";
      unavailable.textContent = `${originalText} (original link unavailable)`;
      parent.append(unavailable);
      return;
    }
    const link = document.createElement("a");
    link.className = "inline-citation";
    link.href = source.safeUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = tableSource
      ? `[Source ${sourceId} — Open original judgment ↗]`
      : `${originalText} ↗`;
    link.setAttribute("aria-label", `Open original judgment for source ${sourceId}`);
    parent.append(link);
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
    if (metadata && typeof metadata.retry_attempted === "boolean") {
      const retryStatus = !metadata.retry_attempted
        ? "Not attempted"
        : metadata.retry_succeeded
          ? (metadata.self_healed ? "Recovered" : "Attempted; recovery threshold not met")
          : "Attempted; retrieval failed";
      items.push(["Retrieval retry", retryStatus]);
    } else if (metadata && typeof metadata.self_healed === "boolean") {
      items.push(["Retrieval recovery", metadata.self_healed ? "Recovered" : "Not reported"]);
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
    const timingLabels = {
      initial_retrieval_ms: "Initial retrieval",
      initial_dense_retrieval_ms: "Initial dense retrieval",
      initial_sparse_retrieval_ms: "Initial BM25 retrieval",
      initial_rrf_fusion_ms: "Initial RRF fusion",
      initial_cross_encoder_reranking_ms: "Initial reranking",
      query_rewriting_ms: "Query rewriting",
      retry_retrieval_ms: "Retry retrieval",
      retry_dense_retrieval_ms: "Retry dense retrieval",
      retry_sparse_retrieval_ms: "Retry BM25 retrieval",
      retry_rrf_fusion_ms: "Retry RRF fusion",
      retry_cross_encoder_reranking_ms: "Retry reranking",
      answer_generation_ms: "Answer generation",
      faithfulness_check_ms: "Faithfulness check",
      total_pipeline_ms: "RAG pipeline total",
    };
    if (metadata && metadata.timings_ms && typeof metadata.timings_ms === "object") {
      for (const [key, label] of Object.entries(timingLabels)) {
        const value = metadata.timings_ms[key];
        if (hasValue(value) && Number.isFinite(Number(value))) {
          items.push([label, `${Number(value).toLocaleString()} ms`]);
        }
      }
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
