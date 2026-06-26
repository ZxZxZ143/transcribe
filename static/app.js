const API = {
  upload: "/api/upload",
  results: "/api/results",
  csv: "/api/csv",
};

const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const refreshButton = document.getElementById("refreshButton");
const clearButton = document.getElementById("clearButton");
const downloadCsvButton = document.getElementById("downloadCsvButton");
const statusMessage = document.getElementById("statusMessage");
const resultsTableBody = document.getElementById("resultsTableBody");
const resultsInfo = document.getElementById("resultsInfo");

function setStatus(message, type = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status-message ${type}`.trim();
}

function setLoading(isLoading) {
  uploadButton.disabled = isLoading;
  refreshButton.disabled = isLoading;
  clearButton.disabled = isLoading;

  uploadButton.textContent = isLoading
    ? "Обработка..."
    : "Загрузить и обработать";
}

function normalizeResultsResponse(data) {
  if (Array.isArray(data)) {
    return data;
  }

  if (Array.isArray(data.results)) {
    return data.results;
  }

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
  if (!text) {
    return "";
  }

  if (text.length <= maxLength) {
    return text;
  }

  return `${text.slice(0, maxLength)}...`;
}

function renderBadge(value, type = "neutral") {
  return `<span class="badge ${type}">${escapeHtml(value)}</span>`;
}

function renderBoolean(value) {
  if (value === true || value === "true") {
    return renderBadge("yes", "success");
  }

  if (value === false || value === "false") {
    return renderBadge("no", "neutral");
  }

  return renderBadge("-", "neutral");
}

function getStatusBadge(status) {
  if (status === "processed" || status === "done") {
    return renderBadge(status, "success");
  }

  if (status === "error") {
    return renderBadge(status, "error");
  }

  return renderBadge(status || "-", "neutral");
}

function renderResults(results) {
  resultsTableBody.innerHTML = "";

  if (!results.length) {
    resultsTableBody.innerHTML = `
      <tr>
        <td colspan="9" class="empty-cell">
          Пока нет данных
        </td>
      </tr>
    `;
    resultsInfo.textContent = "Последние обработанные файлы";
    return;
  }

  const rows = results.map((item) => {
    const fileName = item.file_name || "-";
    const status = item.status || "processed";

    const language =
      item.language ||
      item.selected_language ||
      item.language_code ||
      "-";

    const languageConfidence =
      item.language_confidence ||
      item.confidence ||
      "-";

    const wordCountRaw =
      item.word_count_raw ??
      item.word_count ??
      "-";

    const wordCountCleaned =
      item.word_count_cleaned ??
      "-";

    const reviewPriority =
      item.review_priority ||
      "-";

    const readyForClassification =
      item.ready_for_classification;

    const cleanedTranscript =
      item.cleaned_transcript ||
      "";

    return `
      <tr>
        <td>${escapeHtml(fileName)}</td>
        <td>${getStatusBadge(status)}</td>
        <td>${escapeHtml(language)}</td>
        <td>${escapeHtml(languageConfidence)}</td>
        <td>${escapeHtml(wordCountRaw)}</td>
        <td>${escapeHtml(wordCountCleaned)}</td>
        <td>${escapeHtml(reviewPriority)}</td>
        <td>${renderBoolean(readyForClassification)}</td>
        <td class="transcript-cell">${escapeHtml(truncateText(cleanedTranscript))}</td>
      </tr>
    `;
  });

  resultsTableBody.innerHTML = rows.join("");
  resultsInfo.textContent = `Записей: ${results.length}`;
}

async function uploadFiles() {
  const files = Array.from(fileInput.files || []);

  if (!files.length) {
    setStatus("Выбери один или несколько аудиофайлов.", "error");
    return;
  }

  const formData = new FormData();

  files.forEach((file) => {
    formData.append("files", file);
  });

  setLoading(true);
  setStatus(`Файлы обрабатываются: ${files.length} шт.`);

  try {
    const response = await fetch(API.upload, {
      method: "POST",
      body: formData,
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = data?.detail || "Ошибка при загрузке файлов.";
      throw new Error(detail);
    }

    const results = normalizeResultsResponse(data);
    renderResults(results);

    const processedCount = data.processed_count ?? results.length;
    const errorCount = data.error_count ?? 0;

    setStatus(
      `Готово. Успешно: ${processedCount}, ошибок: ${errorCount}.`,
      errorCount > 0 ? "error" : "success"
    );

    fileInput.value = "";
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

    if (!response.ok) {
      const detail = data?.detail || "Не удалось получить результаты.";
      throw new Error(detail);
    }

    const results = normalizeResultsResponse(data);
    renderResults(results);

    setStatus("Результаты обновлены.", "success");
  } catch (error) {
    setStatus(error.message || "Ошибка загрузки результатов.", "error");
  }
}

async function clearResults() {
  const confirmed = confirm(
    "Очистить итоговый CSV и временные загруженные файлы?"
  );

  if (!confirmed) {
    return;
  }

  setStatus("Очищаю результаты...");

  try {
    const response = await fetch(API.results, {
      method: "DELETE",
    });

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = data?.detail || "Не удалось очистить результаты.";
      throw new Error(detail);
    }

    renderResults([]);
    setStatus("Результаты очищены.", "success");
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

document.addEventListener("DOMContentLoaded", loadResults);