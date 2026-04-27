/**
 * DrawMe — Multiplayer Game Client
 * Socket.IO-based real-time 1v1 drawing game.
 */

// ─── Configuration ──────────────────────────────────────────────────────────

const SOCKET_URL = window.location.origin;
const PREDICTION_INTERVAL = 500; // ms between predictions

// ─── Socket.IO Connection ───────────────────────────────────────────────────

let socket = null;

// ─── Game State ─────────────────────────────────────────────────────────────

let gameState = {
    screen: "lobby",        // lobby | waiting | playing | results
    roomId: null,
    playerId: null,
    playerName: "",
    opponentId: null,
    opponentName: "",
    words: [],
    numRounds: 5,
    timerDuration: 60,
    timeRemaining: 60,
    currentWordIndex: 0,
    completed: 0,
    opponentCompleted: 0,
};

// ─── Canvas State ───────────────────────────────────────────────────────────

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let brushSize = 5;
let strokeHistory = [];
let currentStroke = [];
let predictionTimer = null;
let hasCanvasContent = false;

// ─── DOM Elements ───────────────────────────────────────────────────────────

const screens = {
    lobby: document.getElementById("lobbyScreen"),
    waiting: document.getElementById("waitingScreen"),
    game: document.getElementById("gameScreen"),
    results: document.getElementById("resultsScreen"),
};

// Lobby
const playerNameInput = document.getElementById("playerNameInput");
const roomCodeInput = document.getElementById("roomCodeInput");
const btnCreateRoom = document.getElementById("btnCreateRoom");
const btnJoinRoom = document.getElementById("btnJoinRoom");
const lobbyStatus = document.getElementById("lobbyStatus");
const lobbyStatusText = document.getElementById("lobbyStatusText");

// Waiting
const roomCodeDisplay = document.getElementById("roomCodeDisplay");
const waitingP1Name = document.getElementById("waitingP1Name");
const waitingP2Name = document.getElementById("waitingP2Name");
const waitingP2Status = document.getElementById("waitingP2Status");
const btnCopyCode = document.getElementById("btnCopyCode");
const btnReady = document.getElementById("btnReady");
const btnLeaveRoom = document.getElementById("btnLeaveRoom");

// Game
const gameCanvas = document.getElementById("gameCanvas");
const gameCtx = gameCanvas.getContext("2d");
const targetWord = document.getElementById("targetWord");
const roundInfo = document.getElementById("roundInfo");
const timerFill = document.getElementById("timerFill");
const timerText = document.getElementById("timerText");
const livePredName = document.getElementById("livePredName");
const livePredFill = document.getElementById("livePredFill");
const livePredPct = document.getElementById("livePredPct");
const livePredList = document.getElementById("livePredList");
const predMatchIndicator = document.getElementById("predMatchIndicator");
const yourProgressDots = document.getElementById("yourProgressDots");
const yourProgressCount = document.getElementById("yourProgressCount");
const oppProgressDots = document.getElementById("oppProgressDots");
const oppProgressCount = document.getElementById("oppProgressCount");
const opponentNameEl = document.getElementById("opponentName");

// Game Controls
const gameBtnClear = document.getElementById("gameBtnClear");
const gameBtnUndo = document.getElementById("gameBtnUndo");

// Results
const resultsTrophy = document.getElementById("resultsTrophy");
const resultsTitle = document.getElementById("resultsTitle");
const resultsSubtitle = document.getElementById("resultsSubtitle");
const resultsStats = document.getElementById("resultsStats");
const resultsWordList = document.getElementById("resultsWordList");
const btnPlayAgain = document.getElementById("btnPlayAgain");

// ─── Audio ──────────────────────────────────────────────────────────────────

const AudioCtx = window.AudioContext || window.webkitAudioContext;
let audioCtx = null;

function playSound(freq, duration = 0.06, type = "square", vol = 0.03) {
    try {
        if (!audioCtx) audioCtx = new AudioCtx();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.frequency.value = freq;
        osc.type = type;
        gain.gain.value = vol;
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) { /* silent */ }
}

function playClick() { playSound(800); }
function playSuccess() {
    [523, 659, 784].forEach((f, i) => {
        setTimeout(() => playSound(f, 0.12, "square", 0.02), i * 80);
    });
}
function playRoundComplete() {
    [660, 880, 1100].forEach((f, i) => {
        setTimeout(() => playSound(f, 0.15, "sine", 0.04), i * 100);
    });
}
function playGameWin() {
    [523, 659, 784, 1047].forEach((f, i) => {
        setTimeout(() => playSound(f, 0.2, "square", 0.03), i * 150);
    });
}
function playGameLose() {
    [400, 350, 300].forEach((f, i) => {
        setTimeout(() => playSound(f, 0.25, "sawtooth", 0.02), i * 200);
    });
}
function playTick() { playSound(1000, 0.03, "sine", 0.015); }

// ─── Screen Management ──────────────────────────────────────────────────────

function showScreen(name) {
    gameState.screen = name;
    Object.entries(screens).forEach(([key, el]) => {
        el.classList.toggle("hidden", key !== name);
    });
}

// ─── Connection ─────────────────────────────────────────────────────────────

function connectSocket() {
    if (socket && socket.connected) return;

    socket = io(SOCKET_URL, {
        transports: ["websocket", "polling"],
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
    });

    socket.on("connect", () => {
        console.log("✓ Connected:", socket.id);
        gameState.playerId = socket.id;
        showLobbyStatus("Connected!", "success");
    });

    socket.on("disconnect", () => {
        console.log("✗ Disconnected");
        showLobbyStatus("Disconnected. Reconnecting...", "error");
    });

    socket.on("error", (data) => {
        console.error("Server error:", data.message);
        showLobbyStatus(data.message, "error");
    });

    // ─── Room Events ──────────────────────────────────────────────────

    socket.on("room_created", (data) => {
        gameState.roomId = data.room_id;
        roomCodeDisplay.textContent = data.room_id;
        waitingP1Name.textContent = data.player_name;
        showScreen("waiting");
    });

    socket.on("player_joined", (data) => {
        gameState.roomId = data.room_id;

        // Update waiting screen
        const players = data.players;
        const playerIds = Object.keys(players);

        playerIds.forEach((pid) => {
            if (pid !== gameState.playerId) {
                gameState.opponentId = pid;
                gameState.opponentName = players[pid].player_name;
                waitingP2Name.textContent = players[pid].player_name;
                document.getElementById("waitingP2").querySelector(".waiting-player-icon").textContent = "🎨";
            }
        });

        if (data.is_full) {
            btnReady.disabled = false;
            btnReady.textContent = "✓ I'm Ready!";
        }

        // If we just joined (we're not the creator)
        if (!roomCodeDisplay.textContent || roomCodeDisplay.textContent === "------") {
            roomCodeDisplay.textContent = data.room_id;
            const myData = players[gameState.playerId];
            if (myData) {
                waitingP1Name.textContent = myData.player_name;
            }
            showScreen("waiting");
        }

        playClick();
    });

    socket.on("ready_update", (data) => {
        const players = data.players;
        Object.keys(players).forEach((pid) => {
            if (pid !== gameState.playerId && players[pid].ready) {
                waitingP2Status.textContent = "✓ Ready";
                waitingP2Status.classList.add("ready");
            }
        });
    });

    socket.on("player_left", (data) => {
        if (gameState.screen === "waiting") {
            waitingP2Name.textContent = "Waiting...";
            waitingP2Status.textContent = "";
            waitingP2Status.classList.remove("ready");
            document.getElementById("waitingP2").querySelector(".waiting-player-icon").textContent = "⏳";
            btnReady.disabled = true;
        }
    });

    // ─── Game Events ──────────────────────────────────────────────────

    socket.on("game_start", (data) => {
        gameState.words = data.words;
        gameState.numRounds = data.num_rounds;
        gameState.timerDuration = data.timer_duration;
        gameState.timeRemaining = data.timer_duration;
        gameState.currentWordIndex = 0;
        gameState.completed = 0;
        gameState.opponentCompleted = 0;

        // Find opponent name
        Object.keys(data.players).forEach((pid) => {
            if (pid !== gameState.playerId) {
                gameState.opponentId = pid;
                gameState.opponentName = data.players[pid].player_name;
            }
        });

        showScreen("game");
        initGameUI();
        startPredictionLoop();
        playSuccess();
    });

    socket.on("prediction_result", (data) => {
        updatePredictionUI(data);
    });

    socket.on("round_complete", (data) => {
        gameState.completed = data.completed;
        gameState.currentWordIndex = data.completed;

        // Show match indicator briefly
        predMatchIndicator.classList.remove("hidden");
        setTimeout(() => {
            predMatchIndicator.classList.add("hidden");
        }, 1500);

        // Clear canvas for next word
        clearGameCanvas();

        // Update target word
        if (data.next_word) {
            targetWord.textContent = data.next_word;
            roundInfo.textContent = `Round ${gameState.currentWordIndex + 1}/${gameState.numRounds}`;
        } else {
            targetWord.textContent = "All done! 🎉";
            roundInfo.textContent = `Complete!`;
        }

        updateProgressUI();
        playRoundComplete();
    });

    socket.on("progress_update", (data) => {
        const players = data.players;
        Object.keys(players).forEach((pid) => {
            if (pid !== gameState.playerId) {
                gameState.opponentCompleted = players[pid].completed;
            }
        });
        updateProgressUI();
    });

    socket.on("timer_update", (data) => {
        gameState.timeRemaining = data.time_remaining;
        updateTimerUI();

        // Tick sound for last 10 seconds
        if (data.time_remaining <= 10 && data.time_remaining > 0) {
            playTick();
        }
    });

    socket.on("game_end", (data) => {
        stopPredictionLoop();
        showResults(data);
    });

    socket.on("player_disconnected", (data) => {
        // Opponent left
        showLobbyStatus(data.message, "success");
    });
}

// ─── Lobby Logic ────────────────────────────────────────────────────────────

function showLobbyStatus(text, type = "info") {
    lobbyStatus.classList.remove("hidden");
    lobbyStatusText.textContent = text;
    lobbyStatus.className = `lobby-status lobby-status-${type}`;

    if (type !== "error") {
        setTimeout(() => {
            lobbyStatus.classList.add("hidden");
        }, 3000);
    }
}

// Round selector
document.querySelectorAll("#roundSelector .round-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll("#roundSelector .round-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        gameState.numRounds = parseInt(btn.dataset.rounds);
        playClick();
    });
});

// Timer selector
document.querySelectorAll("#timerSelector .round-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll("#timerSelector .round-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        gameState.timerDuration = parseInt(btn.dataset.timer);
        playClick();
    });
});

// Room code input — auto-uppercase
roomCodeInput.addEventListener("input", () => {
    roomCodeInput.value = roomCodeInput.value.toUpperCase().replace(/[^A-Z0-9]/g, "");
});

// Create Room
btnCreateRoom.addEventListener("click", () => {
    const name = playerNameInput.value.trim() || "Player";
    gameState.playerName = name;
    connectSocket();

    // Wait a bit for connection then emit
    const tryCreate = () => {
        if (socket && socket.connected) {
            socket.emit("create_room", {
                player_name: name,
                num_rounds: gameState.numRounds,
                timer_duration: gameState.timerDuration,
            });
        } else {
            setTimeout(tryCreate, 200);
        }
    };
    setTimeout(tryCreate, 300);
    playClick();
});

// Join Room
btnJoinRoom.addEventListener("click", () => {
    const code = roomCodeInput.value.trim().toUpperCase();
    if (code.length < 4) {
        showLobbyStatus("Enter a valid room code", "error");
        return;
    }

    const name = playerNameInput.value.trim() || "Player";
    gameState.playerName = name;
    connectSocket();

    const tryJoin = () => {
        if (socket && socket.connected) {
            socket.emit("join_room", {
                room_id: code,
                player_name: name,
            });
        } else {
            setTimeout(tryJoin, 200);
        }
    };
    setTimeout(tryJoin, 300);
    playClick();
});

// Copy room code
btnCopyCode.addEventListener("click", () => {
    const code = roomCodeDisplay.textContent;
    navigator.clipboard.writeText(code).then(() => {
        btnCopyCode.textContent = "✓ Copied!";
        setTimeout(() => {
            btnCopyCode.textContent = "📋 Copy";
        }, 2000);
    });
    playClick();
});

// Ready
btnReady.addEventListener("click", () => {
    if (socket && socket.connected) {
        socket.emit("player_ready", {});
        btnReady.disabled = true;
        btnReady.textContent = "Waiting for opponent...";
        playClick();
    }
});

// Leave Room
btnLeaveRoom.addEventListener("click", () => {
    if (socket && socket.connected) {
        socket.emit("leave_room", {});
    }
    showScreen("lobby");
    playClick();
});

// ─── Game Canvas Setup ──────────────────────────────────────────────────────

function setupGameCanvas() {
    const frame = document.getElementById("gameCanvasFrame");
    const rect = frame.getBoundingClientRect();

    // Use the frame's CSS dimensions
    gameCanvas.width = rect.width;
    gameCanvas.height = rect.height;

    gameCtx.fillStyle = "#ffffff";
    gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);
    gameCtx.lineCap = "round";
    gameCtx.lineJoin = "round";
}

function clearGameCanvas() {
    strokeHistory = [];
    currentStroke = [];
    hasCanvasContent = false;
    gameCtx.fillStyle = "#ffffff";
    gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);
}

function redrawGameStrokes() {
    gameCtx.fillStyle = "#ffffff";
    gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);
    gameCtx.lineCap = "round";
    gameCtx.lineJoin = "round";

    strokeHistory.forEach((stroke) => {
        if (stroke.points.length < 2) return;
        gameCtx.strokeStyle = "#000000";
        gameCtx.lineWidth = stroke.size;
        gameCtx.beginPath();
        gameCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
        for (let i = 1; i < stroke.points.length; i++) {
            gameCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
        }
        gameCtx.stroke();
    });
}

// Canvas event helpers
function getCanvasPos(e) {
    const rect = gameCanvas.getBoundingClientRect();
    const scaleX = gameCanvas.width / rect.width;
    const scaleY = gameCanvas.height / rect.height;

    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY,
        };
    }
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

function startGameDraw(e) {
    e.preventDefault();
    if (gameState.screen !== "playing") return;

    isDrawing = true;
    const pos = getCanvasPos(e);
    lastX = pos.x;
    lastY = pos.y;
    currentStroke = [{ x: pos.x, y: pos.y }];
    hasCanvasContent = true;
}

function gameDraw(e) {
    e.preventDefault();
    if (!isDrawing) return;

    const pos = getCanvasPos(e);
    gameCtx.strokeStyle = "#000000";
    gameCtx.lineWidth = brushSize;
    gameCtx.beginPath();
    gameCtx.moveTo(lastX, lastY);
    gameCtx.lineTo(pos.x, pos.y);
    gameCtx.stroke();

    currentStroke.push({ x: pos.x, y: pos.y });
    lastX = pos.x;
    lastY = pos.y;
}

function stopGameDraw(e) {
    if (e) e.preventDefault();
    if (!isDrawing) return;
    isDrawing = false;

    if (currentStroke.length > 0) {
        strokeHistory.push({ points: [...currentStroke], size: brushSize });
        currentStroke = [];
    }
}

// Canvas event listeners
gameCanvas.addEventListener("mousedown", startGameDraw);
gameCanvas.addEventListener("mousemove", gameDraw);
gameCanvas.addEventListener("mouseup", stopGameDraw);
gameCanvas.addEventListener("mouseleave", stopGameDraw);
gameCanvas.addEventListener("touchstart", startGameDraw, { passive: false });
gameCanvas.addEventListener("touchmove", gameDraw, { passive: false });
gameCanvas.addEventListener("touchend", stopGameDraw, { passive: false });

// Canvas controls
gameBtnClear.addEventListener("click", () => {
    clearGameCanvas();
    playClick();
});

gameBtnUndo.addEventListener("click", () => {
    if (strokeHistory.length > 0) {
        strokeHistory.pop();
        redrawGameStrokes();
        if (strokeHistory.length === 0) hasCanvasContent = false;
        playClick();
    }
});

// Brush sizes
document.querySelectorAll(".game-brush-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".game-brush-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        brushSize = parseInt(btn.dataset.size);
        playClick();
    });
});

// ─── Game UI Initialization ─────────────────────────────────────────────────

function initGameUI() {
    gameState.screen = "playing";
    setupGameCanvas();
    clearGameCanvas();

    // Set target word
    targetWord.textContent = gameState.words[0];
    roundInfo.textContent = `Round 1/${gameState.numRounds}`;

    // Set opponent name
    opponentNameEl.textContent = gameState.opponentName || "Opponent";

    // Timer
    updateTimerUI();

    // Progress dots
    updateProgressUI();

    // Reset prediction
    livePredName.textContent = "—";
    livePredFill.style.width = "0%";
    livePredPct.textContent = "0%";
    livePredList.innerHTML = "";
    predMatchIndicator.classList.add("hidden");
}

// ─── Timer UI ───────────────────────────────────────────────────────────────

function updateTimerUI() {
    const t = gameState.timeRemaining;
    const pct = (t / gameState.timerDuration) * 100;

    timerText.textContent = t;
    timerFill.style.width = `${pct}%`;

    // Color based on time remaining
    if (t <= 10) {
        timerFill.style.background = "var(--color-danger)";
        timerFill.classList.add("timer-urgent");
    } else if (t <= 20) {
        timerFill.style.background = "var(--color-warning)";
        timerFill.classList.remove("timer-urgent");
    } else {
        timerFill.style.background = "var(--accent)";
        timerFill.classList.remove("timer-urgent");
    }
}

// ─── Progress UI ────────────────────────────────────────────────────────────

function updateProgressUI() {
    const n = gameState.numRounds;

    // Your progress
    yourProgressCount.textContent = `${gameState.completed}/${n}`;
    yourProgressDots.innerHTML = "";
    for (let i = 0; i < n; i++) {
        const dot = document.createElement("span");
        dot.className = "progress-dot";
        if (i < gameState.completed) {
            dot.classList.add("done");
            dot.textContent = "✓";
        } else if (i === gameState.currentWordIndex) {
            dot.classList.add("current");
            dot.textContent = i + 1;
        } else {
            dot.textContent = i + 1;
        }
        yourProgressDots.appendChild(dot);
    }

    // Opponent progress
    oppProgressCount.textContent = `${gameState.opponentCompleted}/${n}`;
    oppProgressDots.innerHTML = "";
    for (let i = 0; i < n; i++) {
        const dot = document.createElement("span");
        dot.className = "progress-dot";
        if (i < gameState.opponentCompleted) {
            dot.classList.add("done");
            dot.textContent = "✓";
        } else {
            dot.textContent = i + 1;
        }
        oppProgressDots.appendChild(dot);
    }
}

// ─── Prediction Loop ────────────────────────────────────────────────────────

function startPredictionLoop() {
    stopPredictionLoop();
    predictionTimer = setInterval(() => {
        if (gameState.screen !== "playing") return;
        if (!socket || !socket.connected) return;
        if (!hasCanvasContent) return;

        // Capture canvas and send prediction
        const imageData = gameCanvas.toDataURL("image/png");
        socket.emit("predict_frame", { image: imageData });
    }, PREDICTION_INTERVAL);
}

function stopPredictionLoop() {
    if (predictionTimer) {
        clearInterval(predictionTimer);
        predictionTimer = null;
    }
}

// ─── Prediction UI Update ───────────────────────────────────────────────────

function updatePredictionUI(data) {
    const top = data.top_prediction;
    const conf = data.confidence;
    const pct = (conf * 100).toFixed(1);

    livePredName.textContent = top;
    livePredPct.textContent = `${pct}%`;
    livePredFill.style.width = `${conf * 100}%`;

    // Color based on confidence
    if (conf >= 0.6) {
        livePredFill.style.background = "var(--color-conf-high)";
    } else if (conf >= 0.3) {
        livePredFill.style.background = "var(--color-conf-med)";
    } else {
        livePredFill.style.background = "var(--color-conf-low)";
    }

    // Highlight if matches target
    const isMatch = data.matched;
    livePredName.classList.toggle("pred-match", isMatch);

    // Top 3
    if (data.top3) {
        livePredList.innerHTML = "";
        data.top3.forEach((pred, idx) => {
            const row = document.createElement("div");
            row.className = "game-pred-row";

            const rowPct = (pred.confidence * 100).toFixed(1);
            const isTarget = pred.class === gameState.words[gameState.currentWordIndex];

            row.innerHTML = `
                <span class="game-pred-row-name ${isTarget ? 'is-target' : ''}">${pred.class}</span>
                <div class="game-pred-row-bar">
                    <div class="game-pred-row-fill" style="width:${pred.confidence * 100}%"></div>
                </div>
                <span class="game-pred-row-pct">${rowPct}%</span>
            `;
            livePredList.appendChild(row);
        });
    }
}

// ─── Results ────────────────────────────────────────────────────────────────

function showResults(data) {
    showScreen("results");

    const isTie = data.is_tie;
    const myStats = data.players.find((p) => p.player_id === gameState.playerId);
    const oppStats = data.players.find((p) => p.player_id !== gameState.playerId);
    const iWon = myStats && myStats.is_winner;

    // Trophy & Title
    if (isTie) {
        resultsTrophy.textContent = "🤝";
        resultsTitle.textContent = "It's a Tie!";
        resultsSubtitle.textContent = "Great match!";
        playClick();
    } else if (iWon) {
        resultsTrophy.textContent = "🏆";
        resultsTitle.textContent = "You Win!";
        resultsSubtitle.textContent = "Congratulations, champion!";
        resultsTitle.classList.add("winner-bounce");
        playGameWin();
    } else {
        resultsTrophy.textContent = "💪";
        resultsTitle.textContent = "You Lost";
        resultsSubtitle.textContent = "Better luck next time!";
        playGameLose();
    }

    // Stats table
    resultsStats.innerHTML = "";
    if (myStats && oppStats) {
        resultsStats.innerHTML = `
            <table class="results-table">
                <thead>
                    <tr>
                        <th></th>
                        <th>🎨 ${myStats.player_name}</th>
                        <th>⚔️ ${oppStats.player_name}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Completed</td>
                        <td class="${myStats.completed > oppStats.completed ? 'stat-winner' : ''}">${myStats.completed}/${data.words.length}</td>
                        <td class="${oppStats.completed > myStats.completed ? 'stat-winner' : ''}">${oppStats.completed}/${data.words.length}</td>
                    </tr>
                    <tr>
                        <td>Avg. Time</td>
                        <td>${myStats.avg_time != null ? myStats.avg_time + 's' : '—'}</td>
                        <td>${oppStats.avg_time != null ? oppStats.avg_time + 's' : '—'}</td>
                    </tr>
                    <tr>
                        <td>Avg. Confidence</td>
                        <td>${myStats.avg_confidence != null ? myStats.avg_confidence + '%' : '—'}</td>
                        <td>${oppStats.avg_confidence != null ? oppStats.avg_confidence + '%' : '—'}</td>
                    </tr>
                </tbody>
            </table>
        `;
    }

    // Words
    resultsWordList.innerHTML = "";
    data.words.forEach((word, idx) => {
        const chip = document.createElement("span");
        chip.className = "results-word-chip";

        const myDone = myStats && idx < myStats.completed;
        const oppDone = oppStats && idx < oppStats.completed;

        if (myDone && oppDone) chip.classList.add("both-done");
        else if (myDone) chip.classList.add("you-done");
        else if (oppDone) chip.classList.add("opp-done");

        chip.textContent = word;
        resultsWordList.appendChild(chip);
    });
}

// Play Again
btnPlayAgain.addEventListener("click", () => {
    gameState.roomId = null;
    gameState.words = [];
    gameState.currentWordIndex = 0;
    gameState.completed = 0;
    gameState.opponentCompleted = 0;
    showScreen("lobby");
    playClick();
});

// ─── Keyboard Shortcuts ─────────────────────────────────────────────────────

document.addEventListener("keydown", (e) => {
    if (gameState.screen !== "playing") return;

    if (e.key === "Escape") {
        clearGameCanvas();
    }
    if (e.ctrlKey && e.key === "z") {
        e.preventDefault();
        if (strokeHistory.length > 0) {
            strokeHistory.pop();
            redrawGameStrokes();
            if (strokeHistory.length === 0) hasCanvasContent = false;
        }
    }
});

// ─── Handle Enter Key in Lobby ──────────────────────────────────────────────

roomCodeInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        btnJoinRoom.click();
    }
});

playerNameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        btnCreateRoom.click();
    }
});

// ─── Init ───────────────────────────────────────────────────────────────────

showScreen("lobby");
