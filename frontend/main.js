import "./style.css";

// URL base de la API. En desarrollo apunta al backend local; en producción se
// define VITE_API_URL en el build (ver frontend/.env.example).
const API_BASE_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);

  // Formato de números en español (7 843 → "7843" con separador local).
  const nf = new Intl.NumberFormat("es");

  // ---------------------------------------------------------------------- //
  // Estado compartido
  // ---------------------------------------------------------------------- //
  const state = {
    mode: "text",
    topK: 6,
    backendReady: false,
  };

  const els = {
    status: $("#status"),
    statusText: $("#status-text"),
    topK: $("#top-k"),
    tabs: [...document.querySelectorAll('.tab[role="tab"]')],
    modes: { text: $("#mode-text"), image: $("#mode-image") },
    grid: $("#results-grid"),
    head: $("#results-head"),
    title: $("#results-title"),
    meta: $("#results-meta"),
    empty: $("#state-empty"),
    error: $("#state-error"),
    errorText: $("#state-error-text"),
    footerStats: $("#footer-stats"),
    docsLink: $("#docs-link"),
  };

  // ---------------------------------------------------------------------- //
  // Estado del backend (/health)
  // ---------------------------------------------------------------------- //
  function setStatus(kind, text) {
    els.status.dataset.state = kind;
    els.statusText.textContent = text;
  }

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const health = await response.json();

      if (health.model_loaded && health.index_loaded && health.index_size > 0) {
        state.backendReady = true;
        setStatus("ok", `En línea · ${nf.format(health.index_size)} imágenes · ${health.backend}`);
        return true;
      }
      if (!health.index_loaded) {
        setStatus("warn", "Servidor en línea, pero el índice no está construido");
        return false;
      }
      setStatus("loading", "Cargando el modelo CLIP…");
      return false;
    } catch (_err) {
      // En Hugging Face Spaces el contenedor puede tardar 1-2 min en despertar.
      setStatus("loading", "Despertando el servidor… puede tardar un minuto");
      return false;
    }
  }

  async function loadStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (!response.ok) return;
      const stats = await response.json();
      if (stats.total_images > 0) {
        els.footerStats.textContent =
          `${nf.format(stats.total_images)} imágenes de MS COCO en ${nf.format(stats.categories.length)} categorías`;
      }
    } catch (_err) {
      /* opcional */
    }
  }

  function startHealthPolling() {
    let attempts = 0;
    const tick = async () => {
      const ready = await checkHealth();
      if (ready) {
        loadStats();
        return;
      }
      attempts += 1;
      // Reintenta con espera creciente hasta ~5 minutos (arranque en frío).
      const delay = Math.min(15000, 2000 + attempts * 1000);
      if (attempts < 40) setTimeout(tick, delay);
    };
    tick();
  }

  // ---------------------------------------------------------------------- //
  // Pestañas de modo
  // ---------------------------------------------------------------------- //
  function setMode(mode) {
    state.mode = mode;
    els.tabs.forEach((tab) => {
      const active = tab.dataset.mode === mode;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    Object.entries(els.modes).forEach(([key, panel]) => {
      panel.hidden = key !== mode;
    });
    if (mode === "text") $("#text-query").focus();
  }

  els.tabs.forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));

  els.topK.addEventListener("change", () => {
    state.topK = Number(els.topK.value) || 6;
  });

  // ---------------------------------------------------------------------- //
  // Render de resultados
  // ---------------------------------------------------------------------- //
  function hideStates() {
    els.empty.classList.remove("is-visible");
    els.error.classList.remove("is-visible");
  }

  function clearGrid() {
    els.grid.innerHTML = "";
  }

  function showSkeletons(n = state.topK) {
    hideStates();
    clearGrid();
    els.head.style.visibility = "hidden";
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
    els.head.style.visibility = "hidden";
    els.errorText.textContent = message;
    els.error.classList.add("is-visible");
  }

  function showEmpty() {
    clearGrid();
    els.head.style.visibility = "hidden";
    els.empty.classList.add("is-visible");
  }

  function pct(score) {
    // La similitud coseno de CLIP cae en [-1, 1]; se recorta a [0, 1] para mostrar.
    const clamped = Math.max(0, Math.min(1, score));
    return `${(clamped * 100).toFixed(1)}%`;
  }

  function resolveImageUrl(url) {
    return /^https?:\/\//i.test(url) ? url : API_BASE_URL + url;
  }

  function renderResults(payload, label) {
    hideStates();
    clearGrid();

    const results = (payload && payload.results) || [];
    if (results.length === 0) {
      showEmpty();
      return;
    }

    els.head.style.visibility = "visible";
    els.title.textContent = label;
    const took = payload.took_ms != null ? ` · ${nf.format(payload.took_ms)} ms` : "";
    els.meta.textContent = `${results.length} resultados${took}`;

    const frag = document.createDocumentFragment();
    results.forEach((item, i) => {
      const card = document.createElement("figure");
      card.className = "card";
      card.style.animationDelay = `${i * 45}ms`;
      card.tabIndex = 0;

      const img = document.createElement("img");
      img.loading = "lazy";
      img.decoding = "async";
      img.src = resolveImageUrl(item.image_url);
      const cats = (item.categories || []).join(", ");
      img.alt = cats ? `${cats} — similitud ${pct(item.score)}` : `Resultado ${i + 1} — similitud ${pct(item.score)}`;
      img.addEventListener("error", () => {
        card.style.display = "none"; // oculta imágenes rotas sin dejar huecos
      });

      const overlay = document.createElement("figcaption");
      overlay.className = "card__overlay";

      const score = document.createElement("span");
      score.className = "score";
      score.textContent = pct(item.score);
      overlay.appendChild(score);

      if (item.categories && item.categories.length) {
        const list = document.createElement("span");
        list.className = "card__cats";
        item.categories.slice(0, 3).forEach((name) => {
          const tag = document.createElement("span");
          tag.className = "cat";
          tag.textContent = name;
          list.appendChild(tag);
        });
        if (item.categories.length > 3) {
          const more = document.createElement("span");
          more.className = "cat cat--more";
          more.textContent = `+${item.categories.length - 3}`;
          list.appendChild(more);
        }
        overlay.appendChild(list);
      }

      card.append(img, overlay);
      frag.appendChild(card);
    });
    els.grid.appendChild(frag);
  }

  async function describeError(response) {
    try {
      const data = await response.json();
      if (data && data.detail) {
        return typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch (_) {
      /* no es JSON */
    }
    if (response.status === 503) {
      return "El índice de búsqueda todavía no está listo. Inténtalo en unos segundos.";
    }
    return `La petición falló (HTTP ${response.status}).`;
  }

  const NETWORK_ERROR =
    "No se pudo conectar con el servidor. Si el demo lleva un rato inactivo, espera a que despierte y vuelve a intentar.";

  // ---------------------------------------------------------------------- //
  // Modo: texto
  // ---------------------------------------------------------------------- //
  function initTextSearch() {
    const input = $("#text-query");
    const button = $("#text-search-btn");

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
          body: JSON.stringify({ query, top_k: state.topK }),
        });

        if (!response.ok) {
          showError(await describeError(response));
          return;
        }
        renderResults(await response.json(), `Resultados para “${query}”`);
      } catch (_err) {
        showError(NETWORK_ERROR);
        checkHealth();
      } finally {
        button.disabled = false;
      }
    }

    button.addEventListener("click", run);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") run();
    });

    document.querySelectorAll("[data-example]").forEach((chip) => {
      chip.addEventListener("click", () => {
        input.value = chip.getAttribute("data-example") || chip.textContent.trim();
        run();
      });
    });
  }

  // ---------------------------------------------------------------------- //
  // Modo: imagen
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

    const MAX_BYTES = 10 * 1024 * 1024;
    let currentFile = null;

    function humanSize(bytes) {
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function setFile(file) {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        showError("Ese archivo no es una imagen. Elige un JPG, PNG o WebP.");
        return;
      }
      if (file.size > MAX_BYTES) {
        showError("La imagen supera los 10 MB. Elige una más ligera.");
        return;
      }
      currentFile = file;

      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
      };
      reader.readAsDataURL(file);

      previewName.textContent = file.name || "imagen del portapapeles";
      previewSize.textContent = humanSize(file.size);
      preview.classList.add("is-visible");
      dropzone.hidden = true;
      hideStates();
    }

    function reset() {
      currentFile = null;
      fileInput.value = "";
      previewImg.removeAttribute("src");
      preview.classList.remove("is-visible");
      dropzone.hidden = false;
      clearGrid();
      hideStates();
      els.head.style.visibility = "hidden";
    }

    async function run() {
      if (!currentFile) return;

      searchBtn.disabled = true;
      showSkeletons();

      const form = new FormData();
      form.append("file", currentFile, currentFile.name || "consulta.png");
      form.append("top_k", String(state.topK));

      try {
        const response = await fetch(`${API_BASE_URL}/search/image`, { method: "POST", body: form });
        if (!response.ok) {
          showError(await describeError(response));
          return;
        }
        renderResults(await response.json(), "Imágenes visualmente similares");
      } catch (_err) {
        showError(NETWORK_ERROR);
        checkHealth();
      } finally {
        searchBtn.disabled = false;
      }
    }

    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });
    fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

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
      if (file) {
        setMode("image");
        setFile(file);
      }
    });

    // Pegar una imagen desde el portapapeles activa el modo imagen.
    window.addEventListener("paste", (e) => {
      const item = [...(e.clipboardData?.items || [])].find((it) => it.type.startsWith("image/"));
      if (item) {
        setMode("image");
        setFile(item.getAsFile());
      }
    });

    searchBtn.addEventListener("click", run);
    resetBtn.addEventListener("click", reset);
  }

  // ---------------------------------------------------------------------- //
  // Arranque
  // ---------------------------------------------------------------------- //
  document.addEventListener("DOMContentLoaded", () => {
    els.docsLink.href = `${API_BASE_URL}/docs`;
    initTextSearch();
    initImageSearch();
    setMode("text");
    startHealthPolling();
  });
})();
