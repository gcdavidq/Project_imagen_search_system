import './style.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

(() => {
  "use strict";

  const TOP_K = 6;

  // ---------------------------------------------------------------------- //
  // Tiny DOM helpers
  // ---------------------------------------------------------------------- //
  const $ = (sel, root = document) => root.querySelector(sel);

  const ICONS = {
    image:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.6-3.6a2 2 0 0 0-2.8 0L6 20"/></svg>',
    search:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>',
    alert:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    empty:
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/><path d="M8 11h6"/></svg>',
  };

  // ---------------------------------------------------------------------- //
  // Shared result rendering
  // ---------------------------------------------------------------------- //
  const els = {
    grid: $("#results-grid"),
    head: $("#results-head"),
    title: $("#results-title"),
    meta: $("#results-meta"),
    empty: $("#state-empty"),
    error: $("#state-error"),
    errorText: $("#state-error-text"),
  };

  function hideStates() {
    els.empty && els.empty.classList.remove("is-visible");
    els.error && els.error.classList.remove("is-visible");
  }

  function clearGrid() {
    if (els.grid) els.grid.innerHTML = "";
  }

  function showSkeletons(n = TOP_K) {
    hideStates();
    clearGrid();
    if (els.head) els.head.style.visibility = "hidden";
    const frag = document.createDocumentFragment();
    for (let i = 0; i < n; i += 1) {
      const sk = document.createElement("div");
      sk.className = "skeleton";
      frag.appendChild(sk);
    }
    els.grid.appendChild(frag);
  }

  function showError(message) {
    clearGrid();
    if (els.head) els.head.style.visibility = "hidden";
    if (els.errorText) els.errorText.textContent = message;
    if (els.error) els.error.classList.add("is-visible");
  }

  function showEmpty() {
    clearGrid();
    if (els.head) els.head.style.visibility = "hidden";
    if (els.empty) els.empty.classList.add("is-visible");
  }

  function pct(score) {
    // CLIP cosine similarity is in [-1, 1]; clamp to [0, 1] for display.
    const clamped = Math.max(0, Math.min(1, score));
    return `${(clamped * 100).toFixed(1)}%`;
  }

  function renderResults(payload, label) {
    hideStates();
    clearGrid();

    const results = (payload && payload.results) || [];
    if (results.length === 0) {
      showEmpty();
      return;
    }

    if (els.head) els.head.style.visibility = "visible";
    if (els.title) els.title.textContent = label;
    if (els.meta) {
      const took = payload.took_ms != null ? ` · ${payload.took_ms} ms` : "";
      els.meta.textContent = `${results.length} matches${took}`;
    }

    const frag = document.createDocumentFragment();
    results.forEach((item, i) => {
      const card = document.createElement("figure");
      card.className = "card";
      card.style.animationDelay = `${i * 45}ms`;
      card.tabIndex = 0;

      const img = document.createElement("img");
      img.loading = "lazy";
      img.src = item.image_url.startsWith('http') ? item.image_url : API_BASE_URL + item.image_url;
      img.alt = `Result ${i + 1} — similarity ${pct(item.score)}`;
      img.addEventListener("error", () => {
        card.style.display = "none"; // hide broken images gracefully
      });

      const overlay = document.createElement("figcaption");
      overlay.className = "card__overlay";
      overlay.innerHTML = `<span class="score">${pct(item.score)}</span>`;

      card.append(img, overlay);
      frag.appendChild(card);
    });
    els.grid.appendChild(frag);
  }

  // Convert an HTTP error response into a readable message.
  async function describeError(response) {
    try {
      const data = await response.json();
      if (data && data.detail) {
        return typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail);
      }
    } catch (_) {
      /* not JSON */
    }
    if (response.status === 503) {
      return "The search index isn't loaded yet. Build it with scripts/build_index.py and restart the server.";
    }
    return `Request failed (HTTP ${response.status}).`;
  }


  function initTextSearch() {
    const input = $("#text-query");
    const button = $("#text-search-btn");
    if (!input || !button) return;

    async function run() {
      const query = input.value.trim();
      if (!query) {
        input.focus();
        return;
      }

      button.disabled = true;
      showSkeletons();

      try {
        const response = await fetch(`${API_BASE_URL}/search/text`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: TOP_K }),
        });

        if (!response.ok) {
          showError(await describeError(response));
          return;
        }

        const data = await response.json();
        renderResults(data, `Results for “${query}”`);
      } catch (err) {
        showError("Couldn't reach the server. Is it running on this address?");
      } finally {
        button.disabled = false;
      }
    }

    button.addEventListener("click", run);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") run();
    });

    // Example chips fill the box and search immediately.
    document.querySelectorAll("[data-example]").forEach((chip) => {
      chip.addEventListener("click", () => {
        input.value = chip.getAttribute("data-example") || chip.textContent.trim();
        run();
      });
    });

    input.focus();
  }

  // ---------------------------------------------------------------------- //
  // Page: IMAGE SEARCH
  // ---------------------------------------------------------------------- //
  function initImageSearch() {
    const dropzone = $("#dropzone");
    const fileInput = $("#file-input");
    const preview = $("#preview");
    const previewImg = $("#preview-img");
    const previewName = $("#preview-name");
    const previewSize = $("#preview-size");
    const searchBtn = $("#image-search-btn");
    const resetBtn = $("#reset-btn");
    if (!dropzone || !fileInput) return;

    let currentFile = null;

    function humanSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function setFile(file) {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        showError("That file isn't an image. Choose a JPG, PNG or WebP.");
        return;
      }
      currentFile = file;

      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
      };
      reader.readAsDataURL(file);

      previewName.textContent = file.name;
      previewSize.textContent = humanSize(file.size);
      preview.classList.add("is-visible");
      dropzone.style.display = "none";
      hideStates();
    }

    function reset() {
      currentFile = null;
      fileInput.value = "";
      previewImg.removeAttribute("src");
      preview.classList.remove("is-visible");
      dropzone.style.display = "";
      clearGrid();
      hideStates();
      if (els.head) els.head.style.visibility = "hidden";
    }

    async function run() {
      if (!currentFile) return;

      searchBtn.disabled = true;
      showSkeletons();

      const form = new FormData();
      form.append("file", currentFile);
      form.append("top_k", String(TOP_K));

      try {
        const response = await fetch(`${API_BASE_URL}/search/image`, { method: "POST", body: form });

        if (!response.ok) {
          showError(await describeError(response));
          return;
        }

        const data = await response.json();
        renderResults(data, "Visually similar images");
      } catch (err) {
        showError("Couldn't reach the server. Is it running on this address?");
      } finally {
        searchBtn.disabled = false;
      }
    }

    // Click-to-open and file picker.
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });
    fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

    // Drag & drop.
    ["dragenter", "dragover"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("is-dragging");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("is-dragging");
      })
    );
    dropzone.addEventListener("drop", (e) => {
      const file = e.dataTransfer && e.dataTransfer.files[0];
      if (file) setFile(file);
    });

    // Allow pasting an image straight from the clipboard.
    window.addEventListener("paste", (e) => {
      const item = [...(e.clipboardData?.items || [])].find((it) =>
        it.type.startsWith("image/")
      );
      if (item) setFile(item.getAsFile());
    });

    searchBtn.addEventListener("click", run);
    resetBtn.addEventListener("click", reset);
  }

  // ---------------------------------------------------------------------- //
  // Boot
  // ---------------------------------------------------------------------- //
  document.addEventListener("DOMContentLoaded", () => {
    const page = document.body.getAttribute("data-page");
    if (page === "text") initTextSearch();
    if (page === "image") initImageSearch();
  });
})();
