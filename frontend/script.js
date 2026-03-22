/**
 * DrawMe — Frontend Logic
 * Canvas drawing, image preprocessing, and API communication.
 */

// ─── Configuration ──────────────────────────────────────────────────────────

const API_BASE = window.location.origin;
const PREDICT_URL = `${API_BASE}/api/predict`;
const HEALTH_URL  = `${API_BASE}/api/health`;
const CATEGORIES_URL = `${API_BASE}/api/categories`;

// Emoji mapping for categories
const CATEGORY_EMOJIS = {
    cloud:   "☁️",  sun:     "☀️",  tree:    "🌳",  car:     "🚗",
    fish:    "🐟",  cat:     "🐱",  dog:     "🐶",  house:   "🏠",
    star:    "⭐",  flower:  "🌸",  bird:    "🐦",  bicycle: "🚲",
    guitar:  "🎸",  moon:    "🌙",  hat:     "🎩"
};

// ─── DOM Elements ───────────────────────────────────────────────────────────

const canvas           = document.getElementById("drawCanvas");
const ctx              = canvas.getContext("2d");
const canvasWrapper    = document.getElementById("canvasWrapper");
const canvasPlaceholder= document.getElementById("canvasPlaceholder");
const brushSlider      = document.getElementById("brushSize");
const brushValueLabel  = document.getElementById("brushValue");

const btnPredict       = document.getElementById("btnPredict");
const btnClear         = document.getElementById("btnClear");
const btnUndo          = document.getElementById("btnUndo");

const modelBadge       = document.getElementById("modelBadge");
const emptyState       = document.getElementById("emptyState");
const loadingState     = document.getElementById("loadingState");
const resultsState     = document.getElementById("resultsState");
const errorState       = document.getElementById("errorState");
const errorMessage     = document.getElementById("errorMessage");

const topEmoji         = document.getElementById("topEmoji");
const topClass         = document.getElementById("topClass");
const topConfidenceFill= document.getElementById("topConfidenceFill");
const topConfidenceText= document.getElementById("topConfidenceText");
const predictionsList  = document.getElementById("predictionsList");
const categoriesGrid   = document.getElementById("categoriesGrid");

// ─── Canvas State ───────────────────────────────────────────────────────────

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 4;
let hasContent = false;

// Stroke history for undo
let strokeHistory = [];
let currentStroke = [];

// ─── Canvas Setup ───────────────────────────────────────────────────────────

function setupCanvas() {
    // Set internal resolution to match display
    const rect = canvasWrapper.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Default drawing settings
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = brushSize;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Redraw existing strokes
    redrawStrokes();
}

function redrawStrokes() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#000000";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    strokeHistory.forEach(stroke => {
        if (stroke.points.length < 2) return;
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
    currentStroke = [{ x: pos.x, y: pos.y }];

    if (!hasContent) {
        hasContent = true;
        canvasPlaceholder.classList.add("hidden");
    }
}

function draw(e) {
    e.preventDefault();
    if (!isDrawing) return;

    const pos = getPointerPos(e);

    ctx.lineWidth = brushSize;
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();

    currentStroke.push({ x: pos.x, y: pos.y });
    lastX = pos.x;
    lastY = pos.y;
}

function stopDrawing(e) {
    if (e) e.preventDefault();
    if (!isDrawing) return;
    isDrawing = false;

    if (currentStroke.length > 0) {
        strokeHistory.push({ points: [...currentStroke], size: brushSize });
        currentStroke = [];
    }
}

// ─── Canvas Event Listeners ────────────────────────────────────────────────

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

// Touch events
canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing, { passive: false });

// Brush size
brushSlider.addEventListener("input", (e) => {
    brushSize = parseInt(e.target.value);
    brushValueLabel.textContent = brushSize;
});

// Resize handler
let resizeTimer;
window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(setupCanvas, 150);
});

// ─── Actions ────────────────────────────────────────────────────────────────

btnClear.addEventListener("click", () => {
    strokeHistory = [];
    currentStroke = [];
    hasContent = false;
    canvasPlaceholder.classList.remove("hidden");
    setupCanvas();
    showState("empty");
});

btnUndo.addEventListener("click", () => {
    if (strokeHistory.length === 0) return;
    strokeHistory.pop();
    redrawStrokes();

    if (strokeHistory.length === 0) {
        hasContent = false;
        canvasPlaceholder.classList.remove("hidden");
    }
});

btnPredict.addEventListener("click", predict);

// ─── Prediction ─────────────────────────────────────────────────────────────

async function predict() {
    if (!hasContent || strokeHistory.length === 0) {
        showError("Please draw something first!");
        return;
    }

    showState("loading");

    try {
        // Get canvas as Base64 PNG
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

        displayResults(data.predictions, data.debug_image);
    } catch (err) {
        console.error("Prediction error:", err);
        showError("Could not connect to the server. Is the API running?");
    }
}

// ─── Display Results ────────────────────────────────────────────────────────

function displayResults(predictions, debugImage) {
    showState("results");

    // Top prediction
    const top = predictions[0];
    topEmoji.textContent = CATEGORY_EMOJIS[top.class] || "❓";
    topClass.textContent = top.class;
    topConfidenceText.textContent = `${(top.confidence * 100).toFixed(1)}%`;

    // Animate confidence bar
    requestAnimationFrame(() => {
        topConfidenceFill.style.width = `${top.confidence * 100}%`;
    });

    // Show debug image (what the model sees)
    const debugContainer = document.getElementById("debugImageContainer");
    if (debugImage && debugContainer) {
        debugContainer.innerHTML = `<img src="${debugImage}" alt="Model input" style="width: 56px; height: 56px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.1); image-rendering: pixelated;">`;
        debugContainer.classList.remove("hidden");
    }

    // Build prediction rows (show all)
    predictionsList.innerHTML = "";
    predictions.forEach((pred, idx) => {
        const row = document.createElement("div");
        row.classList.add("prediction-row");
        row.style.animationDelay = `${idx * 0.04}s`;

        const emoji = CATEGORY_EMOJIS[pred.class] || "❓";
        const pct = (pred.confidence * 100).toFixed(1);

        row.innerHTML = `
            <span class="prediction-rank">#${idx + 1}</span>
            <span class="prediction-emoji">${emoji}</span>
            <span class="prediction-name">${pred.class}</span>
            <div class="prediction-bar-wrapper">
                <div class="prediction-bar" style="width: 0%"></div>
            </div>
            <span class="prediction-confidence">${pct}%</span>
        `;

        predictionsList.appendChild(row);

        // Animate bars
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const bar = row.querySelector(".prediction-bar");
                bar.style.width = `${pred.confidence * 100}%`;
            });
        });
    });

    // Highlight active category chip
    document.querySelectorAll(".category-chip").forEach(chip => {
        chip.classList.toggle("active", chip.dataset.category === top.class);
    });
}

// ─── State Management ───────────────────────────────────────────────────────

function showState(state) {
    emptyState.classList.toggle("hidden", state !== "empty");
    loadingState.classList.toggle("hidden", state !== "loading");
    resultsState.classList.toggle("hidden", state !== "results");
    errorState.classList.toggle("hidden", state !== "error");
}

function showError(message) {
    showState("error");
    errorMessage.textContent = message;
}

// ─── Health Check & Categories ──────────────────────────────────────────────

async function checkHealth() {
    try {
        const res = await fetch(HEALTH_URL);
        const data = await res.json();

        if (data.model_loaded) {
            modelBadge.classList.add("online");
            modelBadge.classList.remove("offline");
            modelBadge.querySelector("span:last-child").textContent = "Model Ready";
        } else {
            modelBadge.classList.add("offline");
            modelBadge.classList.remove("online");
            modelBadge.querySelector("span:last-child").textContent = "No Model";
        }
    } catch {
        modelBadge.classList.add("offline");
        modelBadge.classList.remove("online");
        modelBadge.querySelector("span:last-child").textContent = "Offline";
    }
}

async function loadCategories() {
    // Default categories (used if API unavailable)
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
        chip.textContent = `${CATEGORY_EMOJIS[cat] || ""} ${cat}`;
        categoriesGrid.appendChild(chip);
    });
}

// ─── Background Particles ───────────────────────────────────────────────────

function createParticles() {
    const container = document.getElementById("bgParticles");
    const colors = ["#818cf8", "#a78bfa", "#c084fc", "#f472b6"];

    for (let i = 0; i < 20; i++) {
        const particle = document.createElement("div");
        particle.classList.add("particle");

        const size = Math.random() * 4 + 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const left = Math.random() * 100;
        const duration = Math.random() * 15 + 10;
        const delay = Math.random() * 10;

        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            left: ${left}%;
            animation-duration: ${duration}s;
            animation-delay: ${delay}s;
            box-shadow: 0 0 ${size * 3}px ${color};
        `;

        container.appendChild(particle);
    }
}

// ─── Keyboard Shortcuts ─────────────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
    // Ctrl/Cmd + Z = Undo
    if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        btnUndo.click();
    }
    // Enter = Predict
    if (e.key === "Enter" && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        btnPredict.click();
    }
    // Escape = Clear
    if (e.key === "Escape") {
        btnClear.click();
    }
});

// ─── Initialization ─────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
    setupCanvas();
    createParticles();
    checkHealth();
    loadCategories();
    showState("empty");

    // Periodic health check
    setInterval(checkHealth, 30000);
});
