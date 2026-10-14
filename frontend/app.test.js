const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.hidden = false;
    this.value = "A sufficiently long legal question";
    this.open = false;
  }

  addEventListener(name, callback) { this[`on${name}`] = callback; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this._text = undefined; }
  setAttribute(name, value) { this.attributes[name] = value; }
  setCustomValidity() {}
  reportValidity() {}
  scrollIntoView() {}
  get childElementCount() { return this.children.filter((child) => child instanceof FakeElement).length; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() {
    return this._text !== undefined
      ? this._text
      : this.children.map((child) => child.textContent || "").join("");
  }
}

function makeResponse(answer, sources, unknownIds = []) {
  const selectors = [
    "#query-form", "#query", "#query-count", "#submit-button", ".button-label",
    "#request-status", "#results", "#answer-text", "#metrics", "#sources",
    "#source-count", "#empty-sources", "#warning", "#citation-warning",
    "#disclaimer", "#result-intent", "#diagnostics",
  ];
  const elements = new Map(selectors.map((selector) => [selector, new FakeElement()]));
  elements.get("#submit-button").setAttribute = function (name, value) {
    this.attributes[name] = value;
  };
  const document = {
    querySelector: (selector) => elements.get(selector),
    createElement: (tag) => new FakeElement(tag),
    createTextNode: (text) => ({ textContent: String(text) }),
  };
  const response = {
    answer,
    sources,
    citation_validation: { unknown_source_ids: unknownIds },
    scores: {},
    metadata: {},
    disclaimer: "Legal disclaimer",
  };
  const context = {
    window: {}, document, URL,
    fetch: async () => ({ ok: true, json: async () => response }),
  };
  const appPath = path.join(__dirname, "app.js");
  vm.runInNewContext(fs.readFileSync(appPath, "utf8"), context);
  return { elements, submit: elements.get("#query-form").onsubmit };
}

async function render(answer, sources, unknownIds) {
  const harness = makeResponse(answer, sources, unknownIds);
  await harness.submit({ preventDefault() {} });
  return harness.elements;
}

function descendants(element, tagName) {
  return element.children.flatMap((child) => {
    if (!(child instanceof FakeElement)) return [];
    return [
      ...(child.tagName === tagName.toUpperCase() ? [child] : []),
      ...descendants(child, tagName),
    ];
  });
}

test("Markdown principle table renders as stacked sections with direct judgment links", async () => {
  const elements = await render(
    "# Anticipatory bail principles\n\n| Principle | Explanation | Source |\n| --- | --- | --- |\n| Risk of flight | The court may consider whether the accused may abscond. | Source 1 |\n| Severity of offence | The seriousness of the alleged offence may matter. | Source 2 |",
    [
      { source_id: "1", url: "https://court.example/judgment/one" },
      { source_id: "2", url: "https://court.example/judgment/two" },
    ],
  );
  const answer = elements.get("#answer-text");
  assert.equal(answer.children[0].tagName, "H2");
  assert.equal(descendants(answer, "section").length, 2);
  assert.equal(descendants(answer, "table").length, 0);
  assert.match(answer.textContent, /Risk of flight/);
  assert.match(answer.textContent, /whether the accused may abscond/);
  const links = descendants(answer, "a");
  assert.deepEqual(links.map((link) => link.href), [
    "https://court.example/judgment/one",
    "https://court.example/judgment/two",
  ]);
  assert.ok(links.every((link) => link.target === "_blank" && link.rel === "noopener noreferrer"));
});

test("bold category table and separator row render as labeled stacked content", async () => {
  const elements = await render(
    "Penalty categories\n\n| **Category** | **Minimum Sentence** | **Maximum Sentence** | **Additional Penalties** | **Source** |\n\n| :--- | ---: | :---: | :--- | --- |\n| **Category A** | At least three years | Up to ten years | Fine and forfeiture | Source 1 |\n| **Category B** | Five years | Life imprisonment | Fine [Source 2] | Source 2 |",
    [
      { source_id: "1", url: "https://court.example/category-a" },
      { source_id: "2", url: "https://court.example/category-b" },
    ],
  );
  const answer = elements.get("#answer-text");
  assert.equal(descendants(answer, "section").length, 2);
  assert.equal(descendants(answer, "table").length, 0);
  assert.doesNotMatch(answer.textContent, /\|\s*:?-{3,}/);
  assert.match(answer.textContent, /Minimum Sentence: At least three years/);
  assert.match(answer.textContent, /Maximum Sentence: Up to ten years/);
  assert.match(answer.textContent, /Additional Penalties: Fine and forfeiture/);
  assert.equal(descendants(answer, "strong").some((node) => node.textContent === "Category A"), true);
  assert.deepEqual(descendants(answer, "a").map((link) => link.href), [
    "https://court.example/category-a",
    "https://court.example/category-b",
    "https://court.example/category-b",
  ]);
});

test("existing headings and paragraphs stay ordinary answer content", async () => {
  const elements = await render(
    "## Answer title\n\nThe court considered the available evidence Source 1, Section 13.\n\nFurther explanation remains a paragraph.",
    [{ source_id: "1", url: "https://court.example/judgment/one" }],
  );
  const answer = elements.get("#answer-text");
  assert.equal(answer.children[0].tagName, "H3");
  assert.equal(descendants(answer, "section").length, 0);
  assert.equal(descendants(answer, "p").length, 2);
  assert.equal(descendants(answer, "a")[0].href, "https://court.example/judgment/one");
  assert.match(answer.textContent, /Section 13/);
});

test("plain source references link without consuming associated statutory provisions", async () => {
  const elements = await render(
    "Source 1, Section 376(1); Source 1, Section 376(2)(f) & (g). Sources 1 and 3.",
    [
      { source_id: "1", url: "https://court.example/one" },
      { source_id: "3", url: "https://court.example/three" },
    ],
  );
  const answer = elements.get("#answer-text");
  const links = descendants(answer, "a");
  assert.deepEqual(links.map((link) => link.href), [
    "https://court.example/one",
    "https://court.example/one",
    "https://court.example/one",
    "https://court.example/three",
  ]);
  assert.match(answer.textContent, /Section 376\(1\)/);
  assert.match(answer.textContent, /Section 376\(2\)\(f\) & \(g\)/);
  assert.match(answer.textContent, /Source 1 .* and Source 3/);
});

test("multiple source IDs resolve independently to their judgment URLs", async () => {
  const elements = await render(
    "| Principle | Explanation | Sources |\n| --- | --- | --- |\n| Procedure | More than one judgment is relevant. | Source 1, Source 2 |",
    [
      { source_id: "1", url: "https://court.example/one" },
      { source_id: "2", url: "https://court.example/two" },
    ],
  );
  assert.deepEqual(descendants(elements.get("#answer-text"), "a").map((link) => link.href), [
    "https://court.example/one",
    "https://court.example/two",
  ]);
});

test("unknown and linkless references are not fabricated or linked", async () => {
  const elements = await render(
    "| Principle | Explanation | Source |\n| --- | --- | --- |\n| Evidence | Explanation Source 9. | Source 1, Source 9 |",
    [{ source_id: "1", url: "javascript:alert(1)" }],
  );
  const answer = elements.get("#answer-text");
  assert.equal(descendants(answer, "a").length, 0);
  assert.match(answer.textContent, /original link unavailable/);
  assert.match(answer.textContent, /Source 9/);
  assert.equal(elements.get("#citation-warning").hidden, false);
  assert.match(elements.get("#citation-warning").textContent, /\[Source 9\]/);
});

test("answer without inline source references keeps ordinary text unlinked", async () => {
  const elements = await render(
    "The answer contains no source references or citations.",
    [{ source_id: "1", url: "https://court.example/one" }],
  );
  assert.equal(descendants(elements.get("#answer-text"), "a").length, 0);
  assert.equal(elements.get("#citation-warning").hidden, true);
});

test("long explanations are preserved and responsive principle styles remain defined", async () => {
  const longText = "Legal explanation ".repeat(300).trimEnd();
  const elements = await render(
    `| Principle | Explanation | Source |\n| --- | --- | --- |\n| Long explanation | ${longText} | Source 1 |`,
    [{ source_id: "1", url: "https://court.example/one" }],
  );
  assert.ok(elements.get("#answer-text").textContent.includes(longText));
  const styles = fs.readFileSync(path.join(__dirname, "styles.css"), "utf8");
  assert.match(styles, /\.principle-section\s*\{/);
  assert.match(styles, /@media\s*\(max-width:\s*560px\)/);
});
