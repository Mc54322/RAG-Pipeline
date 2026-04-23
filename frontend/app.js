/**
 * RAG Pipeline frontend — vanilla JS, no dependencies.
 *
 * Handles PDF upload (drag-and-drop or browse, multiple files), query
 * submission, and rendering of answers with collapsible source passages.
 */

"use strict";

// ── DOM references ────────────────────────────────────────────────────────────

const dropzone       = document.getElementById("dropzone");
const fileInput      = document.getElementById("file-input");
const browseBtn      = document.getElementById("browse-btn");
const clearBtn       = document.getElementById("clear-btn");
const uploadStatus   = document.getElementById("upload-status");

const questionInput  = document.getElementById("question-input");
const kInput         = document.getElementById("k-input");
const minScoreInput  = document.getElementById("min-score-input");
const queryBtn       = document.getElementById("query-btn");
const queryStatus    = document.getElementById("query-status");

const answerSection  = document.getElementById("answer-section");
const answerText     = document.getElementById("answer-text");
const sourcesList    = document.getElementById("sources-list");

// ── State ─────────────────────────────────────────────────────────────────────

let documentIngested = false;

// ── Upload ────────────────────────────────────────────────────────────────────

browseBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) uploadPdfs(fileInput.files);
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) uploadPdfs(e.dataTransfer.files);
});

/**
 * Upload one or more PDF files sequentially to POST /ingest.
 * @param {FileList} files
 */
async function uploadPdfs(files) {
  const pdfs = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
  const invalid = files.length - pdfs.length;

  if (pdfs.length === 0) {
    setStatus(uploadStatus, "Only .pdf files are supported.", "error");
    return;
  }

  setStatus(uploadStatus, `Uploading ${pdfs.length} file(s)…`, "loading");
  queryBtn.disabled = true;
  documentIngested = false;
  clearBtn.classList.add("hidden");

  const errors = [];
  let lastData = null;  // last successful ingest response (has backend totals)

  for (const file of pdfs) {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res  = await fetch("/ingest", { method: "POST", body: formData });
      const data = await res.json();

      if (res.ok) {
        lastData = data;
      } else {
        errors.push(`${file.name}: ${data.detail}`);
      }
    } catch {
      errors.push(`${file.name}: upload failed — is the server running?`);
    }
  }

  // Use the backend's running totals so the status always reflects the full index,
  // not just the files uploaded in this batch.
  if (lastData) {
    const skippedNote = invalid > 0 ? `  (${invalid} non-PDF skipped)` : "";
    const errNote = errors.length > 0 ? `  Errors: ${errors.join("; ")}` : "";
    setStatus(
      uploadStatus,
      `✓ ${lastData.total_documents} document(s) in index — ${lastData.total_chunks} chunks total${skippedNote}${errNote}`,
      errors.length > 0 ? "error" : "success"
    );
    documentIngested = true;
    queryBtn.disabled = false;
  } else {
    setStatus(uploadStatus, `Upload failed: ${errors.join("; ")}`, "error");
  }

  clearBtn.classList.remove("hidden");
}

// ── Clear ─────────────────────────────────────────────────────────────────────

clearBtn.addEventListener("click", async () => {
  clearBtn.disabled = true;
  setStatus(uploadStatus, "Clearing index…", "loading");

  try {
    await fetch("/reset", { method: "POST" });
  } catch {
    // Server unreachable — reset frontend anyway
  }

  fileInput.value = "";
  documentIngested = false;
  queryBtn.disabled = true;
  uploadStatus.className = "status hidden";
  uploadStatus.textContent = "";
  clearBtn.classList.add("hidden");
  clearBtn.disabled = false;
  answerSection.classList.add("hidden");
  queryStatus.className = "status hidden";
});

// ── Query ─────────────────────────────────────────────────────────────────────

// Cmd/Ctrl+Enter submits the query.
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submitQuery();
});
queryBtn.addEventListener("click", submitQuery);

/**
 * Send the current question to POST /query and render the response.
 */
async function submitQuery() {
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }

  const k        = Math.max(1, Math.min(20, parseInt(kInput.value, 10) || 5));
  const minScore = Math.max(0, Math.min(1, parseFloat(minScoreInput.value) || 0.0));

  setStatus(queryStatus, "Retrieving passages and generating answer…", "loading");
  queryBtn.disabled = true;
  answerSection.classList.add("hidden");

  try {
    const res  = await fetch("/query", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ question, k, min_score: minScore }),
    });
    const data = await res.json();

    if (res.ok) {
      queryStatus.classList.add("hidden");
      renderAnswer(data);
    } else {
      setStatus(queryStatus, `Error: ${data.detail}`, "error");
    }
  } catch {
    setStatus(queryStatus, "Query failed — is the server running?", "error");
  } finally {
    queryBtn.disabled = !documentIngested;
  }
}

// ── Rendering ─────────────────────────────────────────────────────────────────

/**
 * Render the answer and source passages into the answer section.
 *
 * Score thresholds are calibrated for all-MiniLM-L6-v2 cosine similarity.
 * Real retrieval scores for this model typically land in 0.2–0.6; a score
 * of 0.5+ indicates a strong match.
 *
 * @param {{ answer: string, sources: Array<{text: string, score: number, source: string}> }} data
 */
function renderAnswer(data) {
  answerText.textContent = data.answer;

  sourcesList.innerHTML = "";
  data.sources.forEach((source, i) => {
    const scoreClass =
      source.score >= 0.5 ? "score-high" :
      source.score >= 0.25 ? "score-mid" : "score-low";

    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="source-header">
        <span>Passage ${i + 1} &mdash; <em>${escapeHtml(source.source)}</em></span>
        <div class="source-meta">
          <span class="score-badge ${scoreClass}">${source.score.toFixed(3)}</span>
          <span class="toggle-arrow">▾</span>
        </div>
      </div>
      <div class="source-body">${escapeHtml(source.text)}</div>
    `;

    item.querySelector(".source-header").addEventListener("click", () => {
      item.classList.toggle("open");
    });

    sourcesList.appendChild(item);
  });

  answerSection.classList.remove("hidden");
  answerSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Show a status message in a target element.
 * @param {HTMLElement} el
 * @param {string} message
 * @param {"success"|"error"|"loading"} type
 */
function setStatus(el, message, type) {
  el.textContent = message;
  el.className   = `status ${type}`;
  el.classList.remove("hidden");
}

/**
 * Escape HTML special characters to prevent XSS when inserting user-
 * supplied text (chunk content) into innerHTML.
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
