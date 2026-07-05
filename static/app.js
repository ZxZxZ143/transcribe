const API = {
    upload: "/api/upload",
    results: "/api/results",
    csv: "/api/csv",
    runClustering: "/api/clustering/run",
    clusteringResults: "/api/clustering/results",
};

const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const refreshButton = document.getElementById("refreshButton");
const clearButton = document.getElementById("clearButton");
const downloadCsvButton = document.getElementById("downloadCsvButton");
const clusterButton = document.getElementById("clusterButton");
const refreshClustersButton = document.getElementById("refreshClustersButton");

const statusMessage = document.getElementById("statusMessage");
const resultsTableBody = document.getElementById("resultsTableBody");
const resultsInfo = document.getElementById("resultsInfo");

const tabButtons = document.querySelectorAll(".tab-button");
const transcriptsTab = document.getElementById("transcriptsTab");
const clustersTab = document.getElementById("clustersTab");

const clustersBoard = document.getElementById("clustersBoard");
const clustersInfo = document.getElementById("clustersInfo");

const clusterModal = document.getElementById("clusterModal");
const clusterModalTitle = document.getElementById("clusterModalTitle");
const clusterModalSubtitle = document.getElementById("clusterModalSubtitle");
const clusterModalBody = document.getElementById("clusterModalBody");
const closeClusterModalButton = document.getElementById("closeClusterModalButton");

let lastClusterItems = [];

function setStatus(message, type = "") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`.trim();
}

function setLoading(isLoading) {
    uploadButton.disabled = isLoading;
    refreshButton.disabled = isLoading;
    clearButton.disabled = isLoading;
    clusterButton.disabled = isLoading;
    refreshClustersButton.disabled = isLoading;
    uploadButton.textContent = isLoading ? "Обработка..." : "Загрузить и обработать";
}

function setClusteringLoading(isLoading) {
    clusterButton.disabled = isLoading;
    refreshClustersButton.disabled = isLoading;
    clusterButton.textContent = isLoading ? "Кластеризация..." : "Сделать кластеризацию";
}

function normalizeResultsResponse(data) {
    if (Array.isArray(data)) return data;
    if (Array.isArray(data?.results)) return data.results;
    if (Array.isArray(data?.items)) return data.items;
    if (Array.isArray(data?.data)) return data.data;
    return [];
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function truncateText(text, maxLength = 220) {
    if (!text) return "";
    if (text.length <= maxLength) return text;
    return `${text.slice(0, maxLength)}...`;
}

function renderBadge(value, type = "neutral") {
    return `<span class="badge ${type}">${escapeHtml(value)}</span>`;
}

function renderBoolean(value) {
    if (value === true || value === "true") return renderBadge("yes", "success");
    if (value === false || value === "false") return renderBadge("no", "neutral");
    return renderBadge("-", "neutral");
}

function getStatusBadge(status) {
    if (status === "processed" || status === "done") return renderBadge(status, "success");
    if (status === "error") return renderBadge(status, "error");
    return renderBadge(status || "-", "neutral");
}

function getClusterId(item) {
    const rawClusterId = item.cluster_id ?? item.cluster ?? item.label ?? -1;
    const numericClusterId = Number(rawClusterId);
    return Number.isNaN(numericClusterId) ? -1 : numericClusterId;
}

function getProblemKeywords(item) {
    const keywords = item.problem_keywords_found ?? item.problem_keywords ?? item.keywords ?? "";
    if (Array.isArray(keywords)) return keywords.filter(Boolean).join(", ");
    return String(keywords || "").trim();
}

function getShortClusterText(item) {
    return item.semantic_text || item.short_text || item.cleaned_transcript || item.raw_transcript || "";
}

function activateTab(tabName) {
    tabButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.tab === tabName);
    });
    transcriptsTab.classList.toggle("active", tabName === "transcripts");
    clustersTab.classList.toggle("active", tabName === "clusters");
    if (tabName === "clusters" && !lastClusterItems.length) {
        loadClusterResults({ silent: true });
    }
}

function renderResults(results) {
    resultsTableBody.innerHTML = "";
    if (!results.length) {
        resultsTableBody.innerHTML = `<tr><td colspan="9" class="empty-cell">Пока нет данных</td></tr>`;
        resultsInfo.textContent = "Последние обработанные файлы";
        return;
    }

    const rows = results.map((item) => {
        const fileName = item.file_name || "-";
        const status = item.status || "processed";
        const language = item.language || item.selected_language || item.language_code || "-";
        const languageConfidence = item.language_confidence || item.confidence || "-";
        const wordCountRaw = item.word_count_raw ?? item.word_count ?? "-";
        const wordCountCleaned = item.word_count_cleaned ?? "-";
        const reviewPriority = item.review_priority || "-";
        const cleanedTranscript = item.cleaned_transcript || "";

        return `
      <tr>
        <td>${escapeHtml(fileName)}</td>
        <td>${getStatusBadge(status)}</td>
        <td>${escapeHtml(language)}</td>
        <td>${escapeHtml(languageConfidence)}</td>
        <td>${escapeHtml(wordCountRaw)}</td>
        <td>${escapeHtml(wordCountCleaned)}</td>
        <td>${escapeHtml(reviewPriority)}</td>
        <td>${renderBoolean(item.ready_for_classification)}</td>
        <td class="transcript-cell">${escapeHtml(truncateText(cleanedTranscript))}</td>
      </tr>`;
    });

    resultsTableBody.innerHTML = rows.join("");
    resultsInfo.textContent = `Записей: ${results.length}`;
}

function groupClusterItems(items) {
    const groups = new Map();
    items.forEach((item) => {
        const clusterId = getClusterId(item);
        if (!groups.has(clusterId)) groups.set(clusterId, []);
        groups.get(clusterId).push(item);
    });
    return [...groups.entries()].sort((a, b) => a[0] - b[0]);
}

function renderClusters(items) {
    lastClusterItems = items;
    clustersBoard.innerHTML = "";

    if (!items.length) {
        clustersBoard.innerHTML = `<div class="empty-clusters">Пока нет данных кластеризации</div>`;
        clustersInfo.textContent = "Кластеризация ещё не запускалась";
        return;
    }

    const groups = groupClusterItems(items);
    const columns = groups.map(([clusterId, clusterItems]) => {
        const isNoise = clusterId === -1;
        const title = isNoise ? "Noise / review" : `Кластер ${clusterId}`;
        const statusClass = isNoise ? "noise" : "clustered";

        const cards = clusterItems.map((item, itemIndex) => {
            const globalIndex = lastClusterItems.indexOf(item);
            const fileName = item.file_name || "-";
            const keywords = getProblemKeywords(item);
            const shortText = getShortClusterText(item);
            const probability = item.cluster_probability ?? item.probability ?? null;
            const reviewPriority = item.review_priority || "-";

            return `
        <button class="cluster-audio-card" data-cluster-item-index="${globalIndex}">
          <div class="cluster-audio-header">
            <strong>${escapeHtml(fileName)}</strong>
            ${probability !== null && probability !== "" ? `<span>${escapeHtml(probability)}</span>` : ""}
          </div>
          ${keywords ? `<div class="cluster-keywords">${escapeHtml(truncateText(keywords, 90))}</div>` : ""}
          <p>${escapeHtml(truncateText(shortText, 150))}</p>
          <div class="cluster-card-footer">
            <span>Review: ${escapeHtml(reviewPriority)}</span>
            <span>#${itemIndex + 1}</span>
          </div>
        </button>`;
        }).join("");

        return `
      <section class="cluster-column ${statusClass}">
        <div class="cluster-column-header">
          <h4>${escapeHtml(title)}</h4>
          <span>${clusterItems.length}</span>
        </div>
        <div class="cluster-items">${cards}</div>
      </section>`;
    });

    clustersBoard.innerHTML = columns.join("");
    clustersInfo.textContent = `Кластеров: ${groups.length}, записей: ${items.length}`;

    clustersBoard.querySelectorAll("[data-cluster-item-index]").forEach((card) => {
        card.addEventListener("click", () => {
            const index = Number(card.dataset.clusterItemIndex);
            const item = lastClusterItems[index];
            if (item) openClusterModal(item);
        });
    });
}

function renderDetailBlock(title, value, className = "") {
    const text = value || "-";
    return `
    <section class="detail-block ${className}">
      <h4>${escapeHtml(title)}</h4>
      <p>${escapeHtml(text)}</p>
    </section>`;
}

function openClusterModal(item) {
    const fileName = item.file_name || "-";
    const clusterId = getClusterId(item);
    const probability = item.cluster_probability ?? item.probability ?? "-";
    const keywords = getProblemKeywords(item);

    clusterModalTitle.textContent = fileName;
    clusterModalSubtitle.textContent = `Кластер: ${clusterId} | вероятность: ${probability}`;
    clusterModalBody.innerHTML = `
    <div class="detail-grid">
      <div><span class="detail-label">Язык</span><strong>${escapeHtml(item.selected_language || item.language || "-")}</strong></div>
      <div><span class="detail-label">Review</span><strong>${escapeHtml(item.review_priority || "-")}</strong></div>
      <div><span class="detail-label">Ready</span><strong>${escapeHtml(item.ready_for_classification ?? "-")}</strong></div>
      <div><span class="detail-label">Слов clean</span><strong>${escapeHtml(item.word_count_cleaned ?? "-")}</strong></div>
    </div>
    ${renderDetailBlock("Ключевые слова", keywords)}
    ${renderDetailBlock("Semantic text", item.semantic_text, "semantic-detail")}
    ${renderDetailBlock("Очищенный текст", item.cleaned_transcript)}
    ${renderDetailBlock("Raw transcript", item.raw_transcript)}
    ${renderDetailBlock("Quality flags", Array.isArray(item.quality_flags) ? item.quality_flags.join(", ") : item.quality_flags)}
  `;
    clusterModal.classList.remove("hidden");
    clusterModal.setAttribute("aria-hidden", "false");
}

function closeClusterModal() {
    clusterModal.classList.add("hidden");
    clusterModal.setAttribute("aria-hidden", "true");
    clusterModalBody.innerHTML = "";
}

async function uploadFiles() {
    const files = Array.from(fileInput.files || []);
    if (!files.length) {
        setStatus("Выбери один или несколько аудиофайлов.", "error");
        return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    setLoading(true);
    setStatus(`Файлы обрабатываются: ${files.length} шт.`);

    try {
        const response = await fetch(API.upload, { method: "POST", body: formData });
        const data = await response.json().catch(() => null);
        if (!response.ok) throw new Error(data?.detail || "Ошибка при загрузке файлов.");

        const results = normalizeResultsResponse(data);
        renderResults(results);
        renderClusters([]);

        const processedCount = data.processed_count ?? results.length;
        const errorCount = data.error_count ?? 0;
        setStatus(`Готово. Успешно: ${processedCount}, ошибок: ${errorCount}.`, errorCount > 0 ? "error" : "success");
        fileInput.value = "";
        activateTab("transcripts");
    } catch (error) {
        setStatus(error.message || "Не удалось обработать файлы.", "error");
    } finally {
        setLoading(false);
    }
}

async function loadResults() {
    setStatus("Загружаю результаты...");
    try {
        const response = await fetch(`${API.results}?limit=50`);
        const data = await response.json().catch(() => null);
        if (!response.ok) throw new Error(data?.detail || "Не удалось получить результаты.");
        renderResults(normalizeResultsResponse(data));
        setStatus("Результаты обновлены.", "success");
    } catch (error) {
        setStatus(error.message || "Ошибка загрузки результатов.", "error");
    }
}

async function runClustering() {
    setClusteringLoading(true);
    setStatus("Запускаю кластеризацию...");
    try {
        const response = await fetch(API.runClustering, { method: "POST" });
        const data = await response.json().catch(() => null);
        if (!response.ok) throw new Error(data?.detail || "Не удалось выполнить кластеризацию.");

        const clusterItems = normalizeResultsResponse(data);
        if (clusterItems.length) renderClusters(clusterItems);
        else await loadClusterResults({ silent: true });

        const clustersCount = data?.clusters_count ?? groupClusterItems(lastClusterItems).length;
        const rowsCount = data?.total_rows ?? lastClusterItems.length;
        activateTab("clusters");
        setStatus(`Кластеризация завершена. Кластеров: ${clustersCount}, записей: ${rowsCount}.`, "success");
    } catch (error) {
        setStatus(error.message || "Ошибка кластеризации.", "error");
    } finally {
        setClusteringLoading(false);
    }
}

async function loadClusterResults(options = {}) {
    if (!options.silent) setStatus("Загружаю результаты кластеризации...");
    try {
        const response = await fetch(API.clusteringResults);
        const data = await response.json().catch(() => null);
        if (!response.ok) throw new Error(data?.detail || "Не удалось получить кластеры.");
        renderClusters(normalizeResultsResponse(data));
        if (!options.silent) setStatus("Кластеры обновлены.", "success");
    } catch (error) {
        renderClusters([]);
        if (!options.silent) setStatus(error.message || "Ошибка загрузки кластеров.", "error");
        else clustersInfo.textContent = "Кластеры пока не найдены";
    }
}

async function clearResults() {
    const confirmed = confirm("Очистить итоговый CSV, временные загруженные файлы и текущие кластеры?");
    if (!confirmed) return;

    setStatus("Очищаю результаты...");
    try {
        const response = await fetch(API.results, { method: "DELETE" });
        const data = await response.json().catch(() => null);
        if (!response.ok) throw new Error(data?.detail || "Не удалось очистить результаты.");
        renderResults([]);
        renderClusters([]);
        setStatus("Результаты очищены.", "success");
        activateTab("transcripts");
    } catch (error) {
        setStatus(error.message || "Ошибка очистки результатов.", "error");
    }
}

function downloadCsv() {
    window.location.href = API.csv;
}

uploadButton.addEventListener("click", uploadFiles);
refreshButton.addEventListener("click", loadResults);
clearButton.addEventListener("click", clearResults);
downloadCsvButton.addEventListener("click", downloadCsv);
clusterButton.addEventListener("click", runClustering);
refreshClustersButton.addEventListener("click", () => loadClusterResults());

tabButtons.forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.tab)));
closeClusterModalButton.addEventListener("click", closeClusterModal);
clusterModal.querySelectorAll("[data-modal-close]").forEach((element) => element.addEventListener("click", closeClusterModal));

document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !clusterModal.classList.contains("hidden")) closeClusterModal();
});

document.addEventListener("DOMContentLoaded", () => {
    loadResults();
    loadClusterResults({ silent: true });
});
