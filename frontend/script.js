/**
 * DrawMe — Frontend Logic (v3 — MS Paint Retro Edition)
 * Canvas drawing, tool management, predictions, and retro UI interactions.
 */

// ─── Configuration ──────────────────────────────────────────────────────────

const API_BASE = window.location.origin;
const PREDICT_URL = `${API_BASE}/api/predict`;
const HEALTH_URL  = `${API_BASE}/api/health`;
const CATEGORIES_URL = `${API_BASE}/api/categories`;

// Emoji mapping for categories (used in history)
const CATEGORY_EMOJIS = {
    cloud:   "☁️",  sun:     "☀️",  tree:    "🌳",  car:     "🚗",
    fish:    "🐟",  cat:     "🐱",  dog:     "🐶",  house:   "🏠",
    star:    "⭐",  flower:  "🌸",  bird:    "🐦",  bicycle: "🚲",
    guitar:  "🎸",  moon:    "🌙",  hat:     "🎩"
};

// ─── DOM Elements ───────────────────────────────────────────────────────────

const canvas             = document.getElementById("drawCanvas");
const ctx                = canvas.getContext("2d");
const canvasFrame        = document.getElementById("canvasFrame");
const canvasPlaceholder  = document.getElementById("canvasPlaceholder");
const canvasGrid         = document.getElementById("canvasGrid");

const btnPredict         = document.getElementById("btnPredict");
const btnClear           = document.getElementById("btnClear");
const btnUndo            = document.getElementById("btnUndo");
const btnRedo            = document.getElementById("btnRedo");
const btnThemeToggle     = document.getElementById("btnThemeToggle");

const modelStatusEl      = document.getElementById("modelStatus");
const statusCoords       = document.getElementById("statusCoords");
const statusBrush        = document.getElementById("statusBrush");
const statusTool         = document.getElementById("statusTool");
const statusModel        = document.getElementById("statusModel");

const stateEmpty         = document.getElementById("stateEmpty");
const stateLoading       = document.getElementById("stateLoading");
const stateResults       = document.getElementById("stateResults");
const stateError         = document.getElementById("stateError");
const errorMessage       = document.getElementById("errorMessage");

const topGuessName       = document.getElementById("topGuessName");
const topConfFill        = document.getElementById("topConfFill");
const topConfPct         = document.getElementById("topConfPct");
const predictionsList    = document.getElementById("predictionsList");
const categoriesGrid     = document.getElementById("categoriesGrid");

const historySection     = document.getElementById("historySection");
const historyList        = document.getElementById("historyList");

// ─── State ──────────────────────────────────────────────────────────────────

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Tool state
let currentTool = "pencil";      // "pencil" | "eraser"
let brushSize = 12;
let drawColor = "#000000";

// Drawing history
let strokeHistory = [];
let redoStack = [];
let currentStroke = [];
let hasContent = false;

// Prediction history
let predictionHistory = [];

// Grid visible
let gridVisible = false;

// History panel visible
let historyVisible = false;

// ─── Sound Effects (subtle clicks) ──────────────────────────────────────────

const AudioCtx = window.AudioContext || window.webkitAudioContext;
let audioCtx = null;

function playClick() {
    try {
        if (!audioCtx) audioCtx = new AudioCtx();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.frequency.value = 800;
        osc.type = "square";
        gain.gain.value = 0.03;
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.06);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.06);
    } catch (e) { /* silent fail */ }
}

function playDrawTick() {
    try {
        if (!audioCtx) audioCtx = new AudioCtx();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.frequency.value = 1200 + Math.random() * 200;
        osc.type = "sine";
        gain.gain.value = 0.008;
        gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.02);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.02);
    } catch (e) { /* silent fail */ }
}

function playSuccess() {
    try {
        if (!audioCtx) audioCtx = new AudioCtx();
        [523, 659, 784].forEach((freq, i) => {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.frequency.value = freq;
            osc.type = "square";
            gain.gain.value = 0.02;
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1 * (i + 1) + 0.1);
            osc.start(audioCtx.currentTime + 0.08 * i);
            osc.stop(audioCtx.currentTime + 0.08 * i + 0.12);
        });
    } catch (e) { /* silent fail */ }
}

// ─── Canvas Setup ───────────────────────────────────────────────────────────

function setupCanvas() {
    const rect = canvasFrame.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    redrawStrokes();
}

function redrawStrokes() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    strokeHistory.forEach(stroke => {
        if (stroke.points.length < 2) return;
        ctx.strokeStyle = stroke.color;
        ctx.lineWidth = stroke.size;
        ctx.beginPath();
        ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
        for (let i = 1; i < stroke.points.length; i++) {
            ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
        }
        ctx.stroke();
    });
}

// ─── Drawing Logic ──────────────────────────────────────────────────────────

function getPointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;

    const pos = getPointerPos(e);
    lastX = pos.x;
    lastY = pos.y;

    const strokeColor = currentTool === "eraser" ? "#ffffff" : drawColor;
    const strokeSize = currentTool === "eraser" ? brushSize * 3 : brushSize;
    currentStroke = [{ x: pos.x, y: pos.y }];

    // Clear redo stack on new drawing
    redoStack = [];

    if (!hasContent && currentTool !== "eraser") {
        hasContent = true;
        canvasPlaceholder.classList.add("hidden");
        updatePredictButton();
    }
}

function draw(e) {
    e.preventDefault();
    if (!isDrawing) return;

    const pos = getPointerPos(e);
    const strokeColor = currentTool === "eraser" ? "#ffffff" : drawColor;
    const strokeWidth = currentTool === "eraser" ? brushSize * 3 : brushSize;

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    currentStroke.push({ x: pos.x, y: pos.y });
    lastX = pos.x;
    lastY = pos.y;

    // Sound feedback (throttled)
    if (currentStroke.length % 8 === 0) {
        playDrawTick();
    }
}

function stopDrawing(e) {
    if (e) e.preventDefault();
    if (!isDrawing) return;
    isDrawing = false;

    if (currentStroke.length > 0) {
        const strokeColor = currentTool === "eraser" ? "#ffffff" : drawColor;
        const strokeWidth = currentTool === "eraser" ? brushSize * 3 : brushSize;
        strokeHistory.push({ points: [...currentStroke], size: strokeWidth, color: strokeColor });
        currentStroke = [];
        updatePredictButton();
    }
}

// ─── Canvas Events ──────────────────────────────────────────────────────────

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing, { passive: false });

// Track mouse position for status bar
canvas.addEventListener("mousemove", (e) => {
    const pos = getPointerPos(e);
    statusCoords.textContent = `Pos: ${Math.round(pos.x)}, ${Math.round(pos.y)}`;
});

canvas.addEventListener("mouseleave", () => {
    statusCoords.textContent = "Pos: —, —";
});

// ─── Tool Selection ─────────────────────────────────────────────────────────

function setTool(tool) {
    currentTool = tool;
    playClick();

    // Update toolbar buttons
    document.querySelectorAll(".tool-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tool === tool);
    });

    // Update canvas cursor
    canvasFrame.classList.toggle("tool-eraser", tool === "eraser");

    // Update status bar
    statusTool.textContent = `Tool: ${tool === "pencil" ? "Pencil" : "Eraser"}`;
}

document.getElementById("toolPencil").addEventListener("click", () => setTool("pencil"));
document.getElementById("toolEraser").addEventListener("click", () => setTool("eraser"));

// ─── Brush Size ─────────────────────────────────────────────────────────────

document.querySelectorAll(".brush-size-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        brushSize = parseInt(btn.dataset.size);
        playClick();

        document.querySelectorAll(".brush-size-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        statusBrush.textContent = `Brush: ${brushSize}px`;
    });
});

// ─── Color Palette ──────────────────────────────────────────────────────────

document.querySelectorAll(".color-swatch").forEach(swatch => {
    swatch.addEventListener("click", () => {
        drawColor = swatch.dataset.color;
        playClick();

        document.querySelectorAll(".color-swatch").forEach(s => s.classList.remove("active"));
        swatch.classList.add("active");

        // Update preview
        document.querySelector(".active-color-inner").style.background = drawColor;

        // Auto-switch to pencil when picking a non-white color
        if (drawColor !== "#ffffff" && currentTool === "eraser") {
            setTool("pencil");
        }
    });
});

// ─── Actions ────────────────────────────────────────────────────────────────

// Clear
btnClear.addEventListener("click", () => {
    playClick();
    clearCanvas();
});

function clearCanvas() {
    strokeHistory = [];
    redoStack = [];
    currentStroke = [];
    hasContent = false;
    canvasPlaceholder.classList.remove("hidden");
    setupCanvas();
    showState("empty");
    updatePredictButton();
}

// Undo
btnUndo.addEventListener("click", () => {
    playClick();
    undoStroke();
});

function undoStroke() {
    if (strokeHistory.length === 0) return;

    const removed = strokeHistory.pop();
    redoStack.push(removed);
    redrawStrokes();

    if (strokeHistory.length === 0) {
        hasContent = false;
        canvasPlaceholder.classList.remove("hidden");
        updatePredictButton();
    }
}

// Redo
btnRedo.addEventListener("click", () => {
    playClick();
    redoStroke();
});

function redoStroke() {
    if (redoStack.length === 0) return;

    const restored = redoStack.pop();
    strokeHistory.push(restored);
    redrawStrokes();

    if (!hasContent) {
        hasContent = true;
        canvasPlaceholder.classList.add("hidden");
        updatePredictButton();
    }
}

// Predict button state
function updatePredictButton() {
    btnPredict.disabled = !hasContent || strokeHistory.length === 0;
}

btnPredict.addEventListener("click", () => {
    playClick();
    predict();
});

// ─── Prediction ─────────────────────────────────────────────────────────────

async function predict() {
    if (!hasContent || strokeHistory.length === 0) {
        showError("Please draw something first!");
        return;
    }

    showState("loading");
    animateProgressBar();

    try {
        const imageData = canvas.toDataURL("image/png");

        const response = await fetch(PREDICT_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageData, debug: true })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            showError(data.error || "Prediction failed");
            return;
        }

        playSuccess();
        displayResults(data.predictions, data.debug_image);
    } catch (err) {
        console.error("Prediction error:", err);
        showError("Could not connect to the server. Is the API running?");
    }
}

// ─── Retro Progress Bar Animation ───────────────────────────────────────────

let progressInterval = null;

function animateProgressBar() {
    const blocks = document.querySelectorAll(".progress-block");
    let idx = 0;

    // Reset
    blocks.forEach(b => b.classList.remove("filled"));

    progressInterval = setInterval(() => {
        if (idx < blocks.length) {
            blocks[idx].classList.add("filled");
            idx++;
        } else {
            // Loop
            blocks.forEach(b => b.classList.remove("filled"));
            idx = 0;
        }
    }, 150);
}

function stopProgressBar() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    // Fill all blocks
    document.querySelectorAll(".progress-block").forEach(b => b.classList.add("filled"));
}

// ─── Display Results ────────────────────────────────────────────────────────

function displayResults(predictions, debugImage) {
    stopProgressBar();
    showState("results");

    const top = predictions[0];

    // Top guess
    topGuessName.textContent = top.class;
    topConfPct.textContent = `${(top.confidence * 100).toFixed(1)}%`;

    // Color the fill based on confidence
    const confColor = getConfidenceColor(top.confidence);
    topConfFill.style.background = confColor;

    // Animate confidence bar
    requestAnimationFrame(() => {
        topConfFill.style.width = `${top.confidence * 100}%`;
    });

    // Debug image
    const debugSection = document.getElementById("debugSection");
    const debugImgWrap = document.getElementById("debugImgWrap");
    if (debugImage && debugSection) {
        debugImgWrap.innerHTML = `<img src="${debugImage}" alt="Model input">`;
        debugSection.classList.remove("hidden");
    }

    // Prediction rows
    predictionsList.innerHTML = "";
    predictions.forEach((pred, idx) => {
        const row = document.createElement("div");
        row.classList.add("pred-row");
        row.style.animationDelay = `${idx * 0.05}s`;

        const pct = (pred.confidence * 100).toFixed(1);
        const barColor = getConfidenceColor(pred.confidence);

        row.innerHTML = `
            <span class="pred-rank">#${idx + 1}</span>
            <span class="pred-name">${pred.class}</span>
            <div class="pred-bar-wrap">
                <div class="pred-bar" style="width:0%; background:${barColor}"></div>
            </div>
            <span class="pred-conf">${pct}%</span>
        `;

        predictionsList.appendChild(row);

        // Animate bars
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const bar = row.querySelector(".pred-bar");
                bar.style.width = `${pred.confidence * 100}%`;
            });
        });
    });

    // Highlight active category
    document.querySelectorAll(".category-chip").forEach(chip => {
        chip.classList.toggle("active", chip.dataset.category === top.class);
    });

    // Add to history
    addToHistory(top.class, top.confidence);
}

function getConfidenceColor(confidence) {
    if (confidence >= 0.6) return "var(--color-conf-high)";
    if (confidence >= 0.3) return "var(--color-conf-med)";
    return "var(--color-conf-low)";
}

// ─── Prediction History ─────────────────────────────────────────────────────

function addToHistory(name, confidence) {
    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}`;

    predictionHistory.unshift({ name, confidence, time: timeStr });
    if (predictionHistory.length > 10) predictionHistory.pop();

    renderHistory();
}

function renderHistory() {
    historyList.innerHTML = "";
    predictionHistory.forEach(item => {
        const el = document.createElement("div");
        el.classList.add("history-item");
        el.innerHTML = `
            <span class="history-item-name">${item.name}</span>
            <span class="history-item-conf">${(item.confidence * 100).toFixed(1)}%</span>
            <span class="history-item-time">${item.time}</span>
        `;
        historyList.appendChild(el);
    });
}

document.getElementById("btnClearHistory").addEventListener("click", () => {
    playClick();
    predictionHistory = [];
    renderHistory();
});

// ─── State Management ───────────────────────────────────────────────────────

function showState(state) {
    stateEmpty.classList.toggle("hidden", state !== "empty");
    stateLoading.classList.toggle("hidden", state !== "loading");
    stateResults.classList.toggle("hidden", state !== "results");
    stateError.classList.toggle("hidden", state !== "error");

    // Reset confidence bar width when showing results
    if (state === "results") {
        topConfFill.style.width = "0%";
    }
}

function showError(message) {
    stopProgressBar();
    showState("error");
    errorMessage.textContent = message;
}

document.getElementById("btnRetry").addEventListener("click", () => {
    playClick();
    predict();
});

// ─── Menu System ────────────────────────────────────────────────────────────

let openMenu = null;

document.querySelectorAll(".menu-item").forEach(item => {
    const label = item.querySelector(".menu-label");

    label.addEventListener("click", (e) => {
        e.stopPropagation();
        playClick();

        if (openMenu === item) {
            closeMenus();
        } else {
            closeMenus();
            item.classList.add("open");
            openMenu = item;
        }
    });

    // Hover to switch menus when one is open
    label.addEventListener("mouseenter", () => {
        if (openMenu && openMenu !== item) {
            closeMenus();
            item.classList.add("open");
            openMenu = item;
        }
    });
});

function closeMenus() {
    document.querySelectorAll(".menu-item").forEach(i => i.classList.remove("open"));
    openMenu = null;
}

document.addEventListener("click", (e) => {
    if (!e.target.closest(".menu-item")) {
        closeMenus();
    }
});

// Menu actions
document.getElementById("menuNew").addEventListener("click", () => {
    closeMenus();
    clearCanvas();
});

document.getElementById("menuSave").addEventListener("click", () => {
    closeMenus();
    saveAsPNG();
});

document.getElementById("menuPredict").addEventListener("click", () => {
    closeMenus();
    predict();
});

document.getElementById("menuUndo").addEventListener("click", () => {
    closeMenus();
    undoStroke();
});

document.getElementById("menuRedo").addEventListener("click", () => {
    closeMenus();
    redoStroke();
});

document.getElementById("menuClear").addEventListener("click", () => {
    closeMenus();
    clearCanvas();
});

document.getElementById("menuGrid").addEventListener("click", () => {
    closeMenus();
    toggleGrid();
});

document.getElementById("menuHistory").addEventListener("click", () => {
    closeMenus();
    toggleHistory();
});

document.getElementById("menuAbout").addEventListener("click", () => {
    closeMenus();
    document.getElementById("aboutDialog").classList.remove("hidden");
});

document.getElementById("menuShortcuts").addEventListener("click", () => {
    closeMenus();
    document.getElementById("shortcutsDialog").classList.remove("hidden");
});

// ─── Dialog Management ──────────────────────────────────────────────────────

document.getElementById("aboutClose").addEventListener("click", () => {
    playClick();
    document.getElementById("aboutDialog").classList.add("hidden");
});

document.getElementById("aboutOk").addEventListener("click", () => {
    playClick();
    document.getElementById("aboutDialog").classList.add("hidden");
});

document.getElementById("shortcutsClose").addEventListener("click", () => {
    playClick();
    document.getElementById("shortcutsDialog").classList.add("hidden");
});

document.getElementById("shortcutsOk").addEventListener("click", () => {
    playClick();
    document.getElementById("shortcutsDialog").classList.add("hidden");
});

// Close dialog on overlay click
document.querySelectorAll(".dialog-overlay").forEach(overlay => {
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) {
            overlay.classList.add("hidden");
        }
    });
});

// ─── Save as PNG ────────────────────────────────────────────────────────────

function saveAsPNG() {
    playClick();
    const link = document.createElement("a");
    link.download = `drawme-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
}

// ─── Grid Toggle ────────────────────────────────────────────────────────────

function toggleGrid() {
    gridVisible = !gridVisible;
    canvasGrid.classList.toggle("hidden", !gridVisible);
    document.getElementById("gridCheck").textContent = gridVisible ? "☑" : "☐";
    playClick();
}

// ─── History Toggle ─────────────────────────────────────────────────────────

function toggleHistory() {
    historyVisible = !historyVisible;
    historySection.classList.toggle("hidden", !historyVisible);
    document.getElementById("historyCheck").textContent = historyVisible ? "☑" : "☐";
    playClick();
}

// ─── Theme Toggle ───────────────────────────────────────────────────────────

btnThemeToggle.addEventListener("click", () => {
    playClick();
    const body = document.body;
    const isLight = body.classList.contains("theme-light");

    body.classList.toggle("theme-light", !isLight);
    body.classList.toggle("theme-dark", isLight);

    // Toggle icon visibility
    document.querySelector(".theme-icon-light").style.display = isLight ? "none" : "block";
    document.querySelector(".theme-icon-dark").style.display = isLight ? "block" : "none";
});

// ─── Health Check & Categories ──────────────────────────────────────────────

async function checkHealth() {
    const statusDot = document.querySelector(".status-model-dot");

    try {
        const res = await fetch(HEALTH_URL);
        const data = await res.json();

        if (data.model_loaded) {
            modelStatusEl.classList.add("online");
            modelStatusEl.classList.remove("offline");
            modelStatusEl.querySelector(".status-text").textContent = "Ready";
            statusModel.innerHTML = '<span class="status-model-dot online"></span> Model: Ready';
        } else {
            modelStatusEl.classList.add("offline");
            modelStatusEl.classList.remove("online");
            modelStatusEl.querySelector(".status-text").textContent = "No Model";
            statusModel.innerHTML = '<span class="status-model-dot offline"></span> Model: No Model';
        }
    } catch {
        modelStatusEl.classList.add("offline");
        modelStatusEl.classList.remove("online");
        modelStatusEl.querySelector(".status-text").textContent = "Offline";
        statusModel.innerHTML = '<span class="status-model-dot offline"></span> Model: Offline';
    }
}

async function loadCategories() {
    let categories = Object.keys(CATEGORY_EMOJIS);

    try {
        const res = await fetch(CATEGORIES_URL);
        const data = await res.json();
        if (data.success) categories = data.categories;
    } catch {
        // Use defaults
    }

    categoriesGrid.innerHTML = "";
    categories.forEach(cat => {
        const chip = document.createElement("span");
        chip.classList.add("category-chip");
        chip.dataset.category = cat;
        chip.textContent = cat;
        categoriesGrid.appendChild(chip);
    });
}

// ─── Keyboard Shortcuts ─────────────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
    // Don't handle if a dialog is open
    if (document.querySelector(".dialog-overlay:not(.hidden)")) return;

    const ctrl = e.ctrlKey || e.metaKey;

    // Ctrl+Z = Undo
    if (ctrl && e.key === "z" && !e.shiftKey) {
        e.preventDefault();
        undoStroke();
    }
    // Ctrl+Y or Ctrl+Shift+Z = Redo
    if ((ctrl && e.key === "y") || (ctrl && e.shiftKey && e.key === "z") || (ctrl && e.shiftKey && e.key === "Z")) {
        e.preventDefault();
        redoStroke();
    }
    // Ctrl+N = New/Clear
    if (ctrl && e.key === "n") {
        e.preventDefault();
        clearCanvas();
    }
    // Ctrl+S = Save
    if (ctrl && e.key === "s") {
        e.preventDefault();
        saveAsPNG();
    }
    // Enter = Predict
    if (e.key === "Enter" && !ctrl) {
        e.preventDefault();
        predict();
    }
    // Escape = Clear
    if (e.key === "Escape") {
        // Close any open dialogs first
        const dialogs = document.querySelectorAll(".dialog-overlay:not(.hidden)");
        if (dialogs.length > 0) {
            dialogs.forEach(d => d.classList.add("hidden"));
        } else {
            closeMenus();
            clearCanvas();
        }
    }
    // P = Pencil
    if (e.key === "p" && !ctrl && !e.target.closest("input")) {
        setTool("pencil");
    }
    // E = Eraser
    if (e.key === "e" && !ctrl && !e.target.closest("input")) {
        setTool("eraser");
    }
    // G = Grid toggle
    if (e.key === "g" && !ctrl && !e.target.closest("input")) {
        toggleGrid();
    }
});

// ─── Window Resize ──────────────────────────────────────────────────────────

let resizeTimer;
window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(setupCanvas, 150);
});

// ─── Initialization ─────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
    setupCanvas();
    checkHealth();
    loadCategories();
    showState("empty");
    updatePredictButton();

    // Periodic health check
    setInterval(checkHealth, 30000);
});
