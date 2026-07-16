import os
import sys
import uuid
import time
import datetime
from typing import Optional

# Ensure the qila directory is in sys.path to allow clean imports of qila_core
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from qila_core import QILA, QILAResponse, QILAFastResponse
from persistence import save_memories, load_memories, save_ghosts, load_ghosts, save_turn, load_history, clear_session
from logger import get_logger

log = get_logger("qila.session")


class QILASession:
    def __init__(self,
                 nim_api_key: Optional[str] = None,
                 token_capacity: int = 4096,
                 theta: float = 0.1,
                 maintenance_interval: int = 3,
                 run_kairos: bool = True,
                 session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.qila = QILA(
            nim_api_key=nim_api_key,
            token_capacity=token_capacity,
            theta=theta,
            maintenance_interval=maintenance_interval,
            run_kairos=run_kairos
        )
        self.conversation_history = []
        self.session_start_time = time.time()
        self.total_reconsolidations = 0
        self.total_uncertain_claims = 0
        # Stores the pending fast response for Phase 2 streaming
        self._pending_fast_response = None
        self._pending_turn_index = None

        # Load existing data from SQLite if session exists
        self._load_from_db()
        log.info(f"Session initialized: {self.session_id}")

    def _load_from_db(self):
        """Loads persisted memories, ghosts, and history from SQLite."""
        try:
            saved_memories = load_memories(self.session_id)
            if saved_memories:
                self.qila.memory_store.update(saved_memories)
                log.info(f"Loaded {len(saved_memories)} memories from disk")

            saved_ghosts = load_ghosts(self.session_id)
            if saved_ghosts:
                self.qila.ghost_store.store.update(saved_ghosts)
                log.info(f"Loaded {len(saved_ghosts)} ghosts from disk")

            saved_history = load_history(self.session_id)
            if saved_history:
                self.conversation_history = saved_history
                self.qila.turn_counter = len(saved_history)
                log.info(f"Loaded {len(saved_history)} turns from disk")
        except Exception as e:
            log.warning(f"Could not load from DB: {e}")

    def _persist(self):
        """Saves current state to SQLite."""
        try:
            save_memories(self.qila.memory_store, self.session_id)
            save_ghosts(self.qila.ghost_store.store, self.session_id)
        except Exception as e:
            log.warning(f"Could not persist to DB: {e}")

    def chat_fast(self, user_input: str) -> dict:
        """
        Phase 1 — Returns the answer instantly (~3-5 seconds).
        Stores internal state for Phase 2 streaming via stream_analysis().
        """
        fast = self.qila.query_fast(user_input)

        # Track reconsolidations
        if fast.reconsolidated_ids:
            self.total_reconsolidations += len(fast.reconsolidated_ids)

        turn_history = {
            "turn": fast.turn,
            "user_input": user_input,
            "answer": fast.answer,
            "kairos_field": [],  # Will be populated by stream_analysis()
            "uncertain_claims": [],
            "reconsolidated": fast.reconsolidated_ids,
            "memory_count": len(fast.memory_state),
            "ghost_count": fast.ghost_count,
            "pressure": fast.peak_pressure
        }

        self.conversation_history.append(turn_history)
        self._pending_fast_response = fast
        self._pending_turn_index = len(self.conversation_history) - 1

        return turn_history

    def stream_analysis(self):
        """
        Phase 2 — Generator that yields KAIROS claim events one at a time.
        Must be called after chat_fast(). Updates the turn history as claims arrive.
        """
        if self._pending_fast_response is None:
            yield {"type": "complete", "total_claims": 0, "kairos_field": []}
            return

        fast = self._pending_fast_response
        turn_idx = self._pending_turn_index
        self._pending_fast_response = None
        self._pending_turn_index = None

        for event in self.qila.analyze_claims_streaming(fast):
            if event["type"] == "claim":
                # Update the turn history with each claim as it arrives
                claim_data = {
                    "claim": event["claim"],
                    "H_e_norm": event["H_e_norm"],
                    "G_norm": event["G_norm"],
                    "Cons": event["Cons"],
                    "U": event["U"],
                    "zone": event["zone"]
                }
                if turn_idx is not None and turn_idx < len(self.conversation_history):
                    self.conversation_history[turn_idx]["kairos_field"].append(claim_data)
                    if event["zone"] != "SOLID":
                        self.conversation_history[turn_idx]["uncertain_claims"].append(claim_data)
                        self.total_uncertain_claims += 1

            yield event

        # Persist after analysis is complete
        self._persist()
        if turn_idx is not None and turn_idx < len(self.conversation_history):
            try:
                save_turn(self.conversation_history[turn_idx], self.session_id)
            except Exception as e:
                log.warning(f"Could not persist turn: {e}")

    def chat(self, user_input: str) -> dict:
        """Backwards-compatible synchronous chat. Runs both phases."""
        result = self.chat_fast(user_input)
        
        for event in self.stream_analysis():
            pass  # Drain the generator to completion
        
        # Return the now-populated turn history
        return self.conversation_history[-1]

    def status(self) -> dict:
        import time
        from survival_engine import compute_survival
        
        active_list = []
        for unit in self.qila.memory_store.values():
            compute_survival(unit, time.time())
            active_list.append({
                "id": unit.id,
                "fidelity": unit.fidelity,
                "survival_score": round(unit.cached_survival, 4),
                "content_preview": unit.original_content[:60] if unit.original_content else unit.content[:60],
                "last_accessed": round(time.time() - unit.last_accessed_at, 1) if unit.last_accessed_at else 0.0
            })

        ghost_list = []
        for unit in self.qila.ghost_store.store.values():
            compute_survival(unit, time.time())
            ghost_list.append({
                "id": unit.id,
                "fidelity": "GHOST",
                "survival_score": round(unit.cached_survival, 4),
                "content_preview": unit.original_content[:60] if unit.original_content else unit.content[:60],
                "last_accessed": round(time.time() - unit.last_accessed_at, 1) if unit.last_accessed_at else 0.0
            })

        return {
            "session_id": self.session_id,
            "turn_count": len(self.conversation_history),
            "memory_count": len(self.qila.memory_store),
            "ghost_count": self.qila.ghost_store.status()["ghost_count"],
            "pressure": self.qila.window_manager.pressure_level(),
            "uptime_seconds": round(time.time() - self.session_start_time, 1),
            "total_reconsolidations": self.total_reconsolidations,
            "total_uncertain_claims": self.total_uncertain_claims,
            "active_memories": active_list,
            "ghost_memories": ghost_list
        }

    def summary(self) -> str:
        lines = [
            "QILA Session Summary",
            f"Session ID: {self.session_id}",
            f"Turns: {len(self.conversation_history)}",
            f"Total memories: {len(self.qila.memory_store)}",
            f"Ghost memories: {self.qila.ghost_store.status()['ghost_count']}",
            f"Total reconsolidations: {self.total_reconsolidations}",
            f"Total uncertain claims flagged: {self.total_uncertain_claims}",
            f"Uptime: {round(time.time() - self.session_start_time, 1)} seconds"
        ]

        has_uncertain = False
        for turn in self.conversation_history:
            if turn["uncertain_claims"]:
                if not has_uncertain:
                    lines.append("\nUncertain claims by turn:")
                    has_uncertain = True
                for claim in turn["uncertain_claims"]:
                    lines.append(f"Turn {turn['turn']}: {claim['claim']} — {claim['zone']} (U={claim['U']:.4f})")

        return "\n".join(lines)

    def reset(self):
        self.qila.memory_store.clear()
        self.qila.ghost_store.store.clear()
        self.qila.turn_counter = 0
        self.conversation_history.clear()
        self.total_reconsolidations = 0
        self.total_uncertain_claims = 0
        try:
            clear_session(self.session_id)
        except Exception as e:
            log.warning(f"Could not clear DB: {e}")
        log.info(f"Session reset. ID retained: {self.session_id}")


if __name__ == "__main__":
    print("=== QILA SESSION TEST ===")

    session = QILASession(run_kairos=False, token_capacity=100, theta=0.1, maintenance_interval=5)

    queries = [
        "I work as a doctor in a hospital and my patient has a high fever",
        "What are common causes of high fever in adults?",
        "The patient also has a stiff neck and sensitivity to light",
        "What should I do first?",
        "What was the first thing I told you about myself?"
    ]

    for idx, q in enumerate(queries):
        result = session.chat(q)
        print(f"\nTurn {idx + 1}: {result['answer'][:120]}...")
        print(f"Memory: {result['memory_count']} | Ghost: {result['ghost_count']} | Pressure: {result['pressure']}")
        if result["reconsolidated"]:
            print(f"*** RECONSOLIDATED: {result['reconsolidated']}")

    print("\n--- Session Status ---")
    print(session.status())

    print("\n--- Session Summary ---")
    print(session.summary())
