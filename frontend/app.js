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
  const disclaimer = document.querySelector("#disclaimer");
  const resultIntent = document.querySelector("#result-intent");

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
    answerText.textContent = "";
    metrics.replaceChildren();
    sources.replaceChildren();
    sourceCount.textContent = "";
    emptySources.hidden = true;
    warning.hidden = true;
    warning.textContent = "";
    disclaimer.textContent = "";
    resultIntent.hidden = true;
    resultIntent.textContent = "";
  }

  function renderResponse(data) {
    const answer = typeof data.answer === "string" ? data.answer.trim() : "";
    answerText.textContent = answer || "No answer text was returned.";
    renderMetrics(data.scores, data.metadata);
    renderSources(Array.isArray(data.sources) ? data.sources : []);

    if (typeof data.warning === "string" && data.warning.trim()) {
      warning.textContent = data.warning;
      warning.hidden = false;
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

  function renderMetrics(scores, metadata) {
    const items = [];
    if (scores && hasValue(scores.relevance)) {
      items.push(["Retrieval reranker score", formatScore(scores.relevance)]);
    }
    if (scores && hasValue(scores.faithfulness)) {
      items.push(["Rule-based faithfulness", formatScore(scores.faithfulness)]);
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
    metrics.parentElement.hidden = items.length === 0;
  }

  function renderSources(items) {
    sourceCount.textContent = `${items.length} ${items.length === 1 ? "source" : "sources"}`;
    emptySources.hidden = items.length > 0;

    for (const item of items) {
      if (!item || typeof item !== "object") continue;
      const card = document.createElement("article");
      card.className = "source-card";
      const title = document.createElement("h4");
      title.className = "source-title";
      const titleText = typeof item.title === "string" && item.title.trim()
        ? item.title
        : "Untitled judgment";
      const safeUrl = validatedHttpUrl(item.url);
      if (safeUrl) {
        const link = document.createElement("a");
        link.href = safeUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = titleText;
        title.append(link);
      } else {
        title.textContent = titleText;
      }
      card.append(title);

      const details = document.createElement("div");
      details.className = "source-details";
      for (const [key, value] of [
        ["Court", item.court],
        ["Year", item.year],
        ["Case type", item.case_type],
        ["Reranker score", item.relevance_score],
      ]) {
        if (!hasValue(value)) continue;
        const detail = document.createElement("span");
        detail.textContent = key === "Reranker score"
          ? `${key}: ${formatScore(value)}`
          : `${key}: ${String(value)}`;
        details.append(detail);
      }
      card.append(details);
      if (!safeUrl) {
        const note = document.createElement("p");
        note.className = "source-link-note";
        note.textContent = "Source link unavailable";
        card.append(note);
      }
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
