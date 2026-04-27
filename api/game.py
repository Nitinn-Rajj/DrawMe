"""
DrawMe — Multiplayer Game Engine
Room management, game state machine, and scoring logic
for the real-time 1v1 drawing game.
"""

import time
import random
import string
from dataclasses import dataclass, field
from typing import Optional


# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TIMER = 60          # seconds
DEFAULT_ROUNDS = 5
CONFIDENCE_THRESHOLD = 0.85
AFK_TIMEOUT = 15            # seconds without a frame
DISCONNECT_GRACE = 10       # seconds before auto-loss on disconnect
ROOM_CODE_LENGTH = 6


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PlayerProgress:
    """Track a single player's game progress."""
    player_id: str
    player_name: str
    current_index: int = 0          # which word they're on
    completed: int = 0              # how many words finished
    timestamps: list = field(default_factory=list)   # time taken per round
    confidences: list = field(default_factory=list)   # confidence per round
    last_frame_time: float = 0.0    # for AFK detection
    connected: bool = True
    ready: bool = False

    def avg_time(self) -> float:
        if not self.timestamps:
            return float("inf")
        return sum(self.timestamps) / len(self.timestamps)

    def avg_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "current_index": self.current_index,
            "completed": self.completed,
            "timestamps": self.timestamps,
            "confidences": self.confidences,
            "connected": self.connected,
            "ready": self.ready,
        }


class GameRoom:
    """
    Manages the state of a single game room.

    State machine: waiting → playing → finished
    """

    def __init__(self, room_id: str, num_rounds: int = DEFAULT_ROUNDS,
                 timer_duration: int = DEFAULT_TIMER, categories: list = None):
        self.room_id = room_id
        self.state = "waiting"          # waiting | playing | finished
        self.num_rounds = num_rounds
        self.timer_duration = timer_duration
        self.categories = categories or []

        # Players (max 2)
        self.players: dict[str, PlayerProgress] = {}
        self.player_order: list[str] = []  # to maintain join order

        # Game data
        self.words: list[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.winner: Optional[str] = None

        # Timer
        self.time_remaining: int = timer_duration

    @property
    def is_full(self) -> bool:
        return len(self.players) >= 2

    @property
    def player_count(self) -> int:
        return len(self.players)

    def add_player(self, player_id: str, player_name: str) -> bool:
        """Add a player to the room. Returns False if room is full."""
        if self.is_full:
            return False
        if player_id in self.players:
            return True  # already in room

        self.players[player_id] = PlayerProgress(
            player_id=player_id,
            player_name=player_name,
        )
        self.player_order.append(player_id)
        return True

    def remove_player(self, player_id: str):
        """Remove a player from the room."""
        if player_id in self.players:
            self.players[player_id].connected = False
            # Don't remove from dict during game — keep stats

    def set_ready(self, player_id: str) -> bool:
        """Mark a player as ready. Returns True if all players are ready."""
        if player_id in self.players:
            self.players[player_id].ready = True
        return self.all_ready()

    def all_ready(self) -> bool:
        """Check if all players are ready (and room is full)."""
        if not self.is_full:
            return False
        return all(p.ready for p in self.players.values())

    def start_game(self) -> list[str]:
        """Initialize and start the game. Returns the word list."""
        if self.state != "waiting" or not self.all_ready():
            return []

        # Pick random words from categories
        available = list(self.categories)
        if len(available) < self.num_rounds:
            # Repeat if needed
            self.words = random.choices(available, k=self.num_rounds)
        else:
            self.words = random.sample(available, self.num_rounds)

        self.state = "playing"
        self.start_time = time.time()
        self.time_remaining = self.timer_duration

        # Reset player progress
        for p in self.players.values():
            p.current_index = 0
            p.completed = 0
            p.timestamps = []
            p.confidences = []
            p.last_frame_time = time.time()

        return self.words

    def check_prediction(self, player_id: str, predicted_class: str,
                         confidence: float) -> dict:
        """
        Check if a prediction matches the player's current target word.

        Returns a dict with:
            - matched: bool
            - round_index: int
            - time_taken: float (if matched)
            - all_done: bool (if player finished all rounds)
        """
        if self.state != "playing" or player_id not in self.players:
            return {"matched": False}

        player = self.players[player_id]

        # Already finished all rounds
        if player.current_index >= self.num_rounds:
            return {"matched": False, "all_done": True}

        target_word = self.words[player.current_index]

        # Update last frame time (AFK detection)
        player.last_frame_time = time.time()

        if predicted_class == target_word and confidence >= CONFIDENCE_THRESHOLD:
            # Round complete!
            elapsed = time.time() - self.start_time
            round_time = elapsed - sum(player.timestamps) if player.timestamps else elapsed

            player.timestamps.append(round(round_time, 2))
            player.confidences.append(round(confidence, 4))
            player.completed += 1
            player.current_index += 1

            all_done = player.current_index >= self.num_rounds

            # Check if this player won (completed all rounds first)
            if all_done and self.winner is None:
                self.winner = player_id
                self.end_game()

            return {
                "matched": True,
                "round_index": player.current_index - 1,
                "time_taken": round_time,
                "confidence": confidence,
                "all_done": all_done,
            }

        return {"matched": False}

    def tick_timer(self) -> int:
        """Decrement timer by 1 second. Returns remaining time."""
        if self.state != "playing":
            return self.time_remaining

        self.time_remaining -= 1

        if self.time_remaining <= 0:
            self.time_remaining = 0
            self.end_game()

        return self.time_remaining

    def end_game(self):
        """End the game and determine the winner."""
        if self.state == "finished":
            return

        self.state = "finished"
        self.end_time = time.time()

        if self.winner is not None:
            return  # already set (someone finished all rounds)

        # Determine winner by tiebreaker rules
        players = list(self.players.values())
        if len(players) < 2:
            self.winner = players[0].player_id if players else None
            return

        p1, p2 = players[0], players[1]

        # 1. More completed drawings wins
        if p1.completed != p2.completed:
            self.winner = p1.player_id if p1.completed > p2.completed else p2.player_id
        # 2. Faster average completion time
        elif p1.avg_time() != p2.avg_time():
            self.winner = p1.player_id if p1.avg_time() < p2.avg_time() else p2.player_id
        # 3. Higher average confidence
        elif p1.avg_confidence() != p2.avg_confidence():
            self.winner = p1.player_id if p1.avg_confidence() > p2.avg_confidence() else p2.player_id
        else:
            self.winner = "tie"

    def get_opponent_id(self, player_id: str) -> Optional[str]:
        """Get the opponent's player ID."""
        for pid in self.players:
            if pid != player_id:
                return pid
        return None

    def get_state(self) -> dict:
        """Get the full room state for broadcasting."""
        return {
            "room_id": self.room_id,
            "state": self.state,
            "words": self.words if self.state != "waiting" else [],
            "num_rounds": self.num_rounds,
            "timer_duration": self.timer_duration,
            "time_remaining": self.time_remaining,
            "players": {pid: p.to_dict() for pid, p in self.players.items()},
            "player_order": self.player_order,
            "winner": self.winner,
        }

    def get_results(self) -> dict:
        """Get final game results for the results screen."""
        players_stats = []
        for pid in self.player_order:
            p = self.players[pid]
            players_stats.append({
                "player_id": p.player_id,
                "player_name": p.player_name,
                "completed": p.completed,
                "avg_time": round(p.avg_time(), 2) if p.timestamps else None,
                "avg_confidence": round(p.avg_confidence() * 100, 1) if p.confidences else None,
                "timestamps": p.timestamps,
                "confidences": [round(c * 100, 1) for c in p.confidences],
                "is_winner": p.player_id == self.winner,
            })

        return {
            "room_id": self.room_id,
            "winner": self.winner,
            "is_tie": self.winner == "tie",
            "words": self.words,
            "players": players_stats,
            "total_time": round(self.end_time - self.start_time, 1) if self.end_time and self.start_time else None,
        }


class RoomManager:
    """Manages all active game rooms."""

    def __init__(self, categories: list = None):
        self.rooms: dict[str, GameRoom] = {}
        self.player_rooms: dict[str, str] = {}  # player_id → room_id
        self.categories = categories or []

    def _generate_room_id(self) -> str:
        """Generate a unique 6-character room code."""
        while True:
            code = "".join(random.choices(string.ascii_uppercase + string.digits, k=ROOM_CODE_LENGTH))
            if code not in self.rooms:
                return code

    def create_room(self, player_id: str, player_name: str,
                    num_rounds: int = DEFAULT_ROUNDS,
                    timer_duration: int = DEFAULT_TIMER) -> GameRoom:
        """Create a new room and add the creator to it."""
        # Leave any existing room first
        self.leave_room(player_id)

        room_id = self._generate_room_id()
        room = GameRoom(
            room_id=room_id,
            num_rounds=num_rounds,
            timer_duration=timer_duration,
            categories=self.categories,
        )
        room.add_player(player_id, player_name)

        self.rooms[room_id] = room
        self.player_rooms[player_id] = room_id
        return room

    def join_room(self, player_id: str, player_name: str,
                  room_id: str) -> Optional[GameRoom]:
        """Join an existing room. Returns None if room not found or full."""
        room = self.rooms.get(room_id)
        if room is None:
            return None
        if room.state != "waiting":
            return None

        # Leave any existing room first
        self.leave_room(player_id)

        if not room.add_player(player_id, player_name):
            return None  # room full

        self.player_rooms[player_id] = room_id
        return room

    def leave_room(self, player_id: str) -> Optional[GameRoom]:
        """Remove a player from their current room."""
        room_id = self.player_rooms.pop(player_id, None)
        if room_id is None:
            return None

        room = self.rooms.get(room_id)
        if room is None:
            return None

        room.remove_player(player_id)

        # Clean up empty rooms in waiting state
        connected_count = sum(1 for p in room.players.values() if p.connected)
        if connected_count == 0 and room.state == "waiting":
            del self.rooms[room_id]

        return room

    def get_room(self, room_id: str) -> Optional[GameRoom]:
        """Get a room by ID."""
        return self.rooms.get(room_id)

    def get_player_room(self, player_id: str) -> Optional[GameRoom]:
        """Get the room a player is currently in."""
        room_id = self.player_rooms.get(player_id)
        if room_id:
            return self.rooms.get(room_id)
        return None

    def cleanup_finished_rooms(self, max_age: int = 300):
        """Remove finished rooms older than max_age seconds."""
        now = time.time()
        to_remove = []
        for room_id, room in self.rooms.items():
            if room.state == "finished" and room.end_time:
                if now - room.end_time > max_age:
                    to_remove.append(room_id)

        for room_id in to_remove:
            # Remove player mappings
            for pid in list(self.player_rooms.keys()):
                if self.player_rooms.get(pid) == room_id:
                    del self.player_rooms[pid]
            del self.rooms[room_id]
