import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Ensure that the KAIROS and MAKS directories are in the system path for clean imports
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(current_dir, ".."))

kairos_path = os.path.join(workspace_root, "KAIROS")
maks_path = os.path.join(workspace_root, "MAKS")

if kairos_path not in sys.path:
    sys.path.insert(0, kairos_path)
if maks_path not in sys.path:
    sys.path.insert(0, maks_path)

# Import KAIROS modules
import config
from sampler import sample, SampleSet
from claim_extractor import extract_claims
from pressure_engine import compute_all_pressures
from zone_classifier import classify_all

# Import MAKS modules
from memory_unit import MemoryUnit, create_memory_unit
from survival_engine import get_survival, compute_survival
import fidelity
from sahel.context.ghost_store import GhostStore
from sahel.context.window_manager import WindowManager
from sahel.context.eviction import EvictionManager
from sahel.core.maintenance import MaintenanceLoop

# Set the single source of truth model to meta/llama-3.2-3b-instruct
config.NIM_MODEL = "meta/llama-3.2-3b-instruct"
os.environ["NIM_MODEL"] = "meta/llama-3.2-3b-instruct"

# Map MAKS summarization model to use the single source of truth from config
fidelity.SUMMARIZE_MODEL = config.NIM_MODEL


@dataclass
class QILAFastResponse:
    """Returned instantly from query_fast() — contains the answer but no KAIROS analysis."""
    answer: str
    memory_state: list[dict]
    reconsolidated_ids: list[str]
    ghost_count: int
    peak_pressure: str
    turn: int
    # Internal references for Phase 2 streaming — not serialized to JSON
    _sample_set: object = field(default=None, repr=False)
    _full_prompt: str = field(default="", repr=False)
    _response_text: str = field(default="", repr=False)
    _mem_id: str = field(default="", repr=False)


@dataclass
class QILAResponse:
    answer: str
    memory_state: list[dict]
    kairos_field: list[dict]
    reconsolidated_ids: list[str]
    ghost_count: int
    peak_pressure: str
    turn: int


class QILA:
    def __init__(self,
                 nim_api_key: Optional[str] = None,
                 token_capacity: int = 4096,
                 theta: float = 0.1,
                 maintenance_interval: int = 3,
                 run_kairos: bool = True):
        # Configure API key priority
        if nim_api_key:
            config.NIM_API_KEY = nim_api_key
            os.environ["NIM_API_KEY"] = nim_api_key
            os.environ["NVIDIA_API_KEY"] = nim_api_key
        else:
            env_key = os.environ.get("NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
            if env_key:
                nim_api_key = env_key
            else:
                nim_api_key = config.NIM_API_KEY
            config.NIM_API_KEY = nim_api_key
            os.environ["NIM_API_KEY"] = nim_api_key
            os.environ["NVIDIA_API_KEY"] = nim_api_key

        self.token_capacity = token_capacity
        self.theta = theta
        self.maintenance_interval = maintenance_interval
        self.run_kairos = run_kairos

        # Initialize MAKS states
        self.memory_store = {}
        self.ghost_store = GhostStore(use_embeddings=False)
        self.window_manager = WindowManager(self.memory_store, self.token_capacity)
        self.eviction_manager = EvictionManager(self.memory_store, self.window_manager, theta=self.theta)
        
        self.turn_counter = 0

    def query_fast(self, user_input: str) -> QILAFastResponse:
        """
        Phase 1 — Instant answer. Runs MAKS maintenance, reconsolidation,
        LLM sampling, and memory storage. Returns in ~3-5 seconds.
        No KAIROS analysis.
        """
        # STEP 1 — Maintenance and eviction
        self.turn_counter += 1
        if self.turn_counter % self.maintenance_interval == 0:
            loop = MaintenanceLoop(memory_store=self.memory_store, theta=self.theta, interval_seconds=0.0)
            loop.start(cycles=1, verbose=False)
            
            ghost_candidates = [u for u in self.memory_store.values() if u.fidelity == "GHOST"]
            for unit in ghost_candidates:
                self.ghost_store.archive(unit, self.memory_store)
                
            self.eviction_manager.run_eviction()

        # STEP 2 — Ghost Store search and reconsolidation
        first_5_words = user_input.split()[:5]
        stop_words = {"the", "is", "a", "of", "and", "to", "in", "that", "it", "you", "am", "are", "what", "who", "which"}
        filtered_words = [w for w in first_5_words if w.lower().strip(",.?!:;") not in stop_words]
        if not filtered_words:
            filtered_words = first_5_words
        keywords = " ".join(filtered_words)
        
        ghost_results = self.ghost_store.search(keywords, top_k=3)
        reconsolidated_ids = []
        for unit in ghost_results:
            if self.ghost_store.reconsolidate(unit.id, self.memory_store):
                reconsolidated_ids.append(unit.id)

        # STEP 3 — Build context and call LLM
        current_time = time.time()
        sorted_units = sorted(
            self.memory_store.values(),
            key=lambda u: get_survival(u, current_time),
            reverse=True
        )
        top_memories = sorted_units[:8]
        
        if top_memories:
            memory_context = "MEMORY CONTEXT:\n" + "".join(f"- [{u.id}]: {u.content}\n" for u in top_memories)
            full_prompt = memory_context + "\n\nUSER: " + user_input
        else:
            memory_context = ""
            full_prompt = user_input

        # Sample — only 1 call if KAIROS disabled, N calls if enabled
        N = config.DEFAULT_N if self.run_kairos else 1
        sample_set = sample(full_prompt, N=N, tau=config.DEFAULT_TAU)
        response_text = sample_set.samples[0]

        # STEP 4 — Store the new memory
        mem_id = "mem_" + str(int(time.time()))
        content = "Q: " + user_input + " A: " + response_text[:300]
        new_unit = create_memory_unit(mem_id, content)
        new_unit.original_content = content
        self.memory_store[mem_id] = new_unit

        # STEP 5 — Snapshot memory state
        memory_state = []
        current_time = time.time()
        for unit in self.memory_store.values():
            score = get_survival(unit, current_time)
            memory_state.append({
                "id": unit.id,
                "fidelity": unit.fidelity,
                "survival_score": score,
                "content_preview": unit.content[:80]
            })

        return QILAFastResponse(
            answer=response_text,
            memory_state=memory_state,
            reconsolidated_ids=reconsolidated_ids,
            ghost_count=len(self.ghost_store.store),
            peak_pressure=self.window_manager.pressure_level(),
            turn=self.turn_counter,
            _sample_set=sample_set,
            _full_prompt=full_prompt,
            _response_text=response_text,
            _mem_id=mem_id,
        )

    def analyze_claims_streaming(self, fast_response: QILAFastResponse):
        """
        Phase 2 — Streaming KAIROS analysis. A Python generator that yields
        one claim dict at a time as each is scored and classified.
        
        Yields:
            dict with keys: type, claim, H_e_norm, G_norm, Cons, U, zone, index, total
            Final yield: type="complete", summary stats
        """
        if not self.run_kairos:
            yield {"type": "complete", "total_claims": 0, "kairos_field": []}
            return

        response_text = fast_response._response_text
        sample_set = fast_response._sample_set
        full_prompt = fast_response._full_prompt
        mem_id = fast_response._mem_id

        # Extract claims
        claims = extract_claims(response_text)
        if not claims:
            yield {"type": "complete", "total_claims": 0, "kairos_field": []}
            return

        # Yield a "started" event so the frontend knows how many claims to expect
        yield {"type": "analysis_started", "total_claims": len(claims)}

        # Batch gradient computation (the slowest step — but must be done in bulk)
        from gradient_engine import compute_gradients
        from entropy_engine import claim_entropy
        from consistency_engine import compute_consistency

        gradients = compute_gradients(claims, response_text, full_prompt)

        # Yield a "gradients_ready" event
        yield {"type": "gradients_ready"}

        # Score each claim and yield immediately
        all_records = []
        for idx, claim in enumerate(claims):
            H = claim_entropy(claim, response_text, sample_set, config.VOCAB_SIZE)
            G = gradients.get(claim, 0.0)
            C = compute_consistency(claim, sample_set)

            # Compute U
            inconsistency = 1.0 - C
            if H == 0.0:
                U = G * inconsistency * 0.5
            else:
                U = H * G * inconsistency

            U = round(U, 4)
            H = round(H, 4)
            G = round(G, 4)
            C = round(C, 4)

            # Classify zone
            if U <= config.PHI_S:
                zone = "SOLID"
            elif U >= config.PHI_F:
                zone = "FAULT LINE"
            else:
                zone = "GRADIENT"

            claim_dict = {
                "type": "claim",
                "index": idx,
                "total": len(claims),
                "claim": claim,
                "H_e_norm": H,
                "G_norm": G,
                "Cons": C,
                "U": U,
                "zone": zone
            }
            all_records.append(claim_dict)
            yield claim_dict

        # MAKS integration: penalize memory if avg U is high
        if all_records and mem_id in self.memory_store:
            avg_U = sum(r["U"] for r in all_records) / len(all_records)
            new_unit = self.memory_store[mem_id]
            compute_survival(new_unit, time.time())
            if avg_U > 0.08:
                new_unit.cached_survival *= 0.6
            elif avg_U > 0.06:
                new_unit.cached_survival *= 0.8

        # Final summary event
        kairos_field = [
            {k: v for k, v in r.items() if k not in ("type", "index", "total")}
            for r in all_records
        ]
        yield {
            "type": "complete",
            "total_claims": len(all_records),
            "kairos_field": kairos_field
        }

    def query(self, user_input: str) -> QILAResponse:
        """Backwards-compatible synchronous query. Runs both phases."""
        fast = self.query_fast(user_input)
        
        kairos_field = []
        for event in self.analyze_claims_streaming(fast):
            if event["type"] == "claim":
                kairos_field.append({
                    "claim": event["claim"],
                    "H_e_norm": event["H_e_norm"],
                    "G_norm": event["G_norm"],
                    "Cons": event["Cons"],
                    "U": event["U"],
                    "zone": event["zone"]
                })

        return QILAResponse(
            answer=fast.answer,
            memory_state=fast.memory_state,
            kairos_field=kairos_field,
            reconsolidated_ids=fast.reconsolidated_ids,
            ghost_count=fast.ghost_count,
            peak_pressure=fast.peak_pressure,
            turn=fast.turn
        )


if __name__ == "__main__":
    print("=== QILA INTEGRATION TEST ===")

    # Initialize QILA with run_kairos=True
    qila_fast = QILA(run_kairos=True, token_capacity=100, theta=0.1, maintenance_interval=2)

    queries = [
        "My name is Sahil and I am building an AI product called QILA that combines memory and uncertainty",
        "Explain the difference between epistemology and memory management in AI systems",
        "What is the name of the AI product I am building and what does it combine?"
    ]

    response = None
    for idx, q in enumerate(queries):
        print(f"\n--- Turn {idx + 1} query: '{q}' ---")
        response = qila_fast.query(q)
        print("Response object:")
        print(f"  Answer: {response.answer}")
        print(f"  Memory State: {response.memory_state}")
        print(f"  Ghost Count: {response.ghost_count}")
        print(f"  Peak Pressure: {response.peak_pressure}")
        print(f"  Turn: {response.turn}")

    print("\n=== RECONSOLIDATION CHECK ===")
    if response:
        print("Reconsolidated IDs: " + str(response.reconsolidated_ids))
        print("Expected: mem from Turn 1 should have been found and revived")

    print("\n--- Running KAIROS Verification ---")
    # Initialize a second QILA instance with run_kairos=True
    qila_kairos = QILA(run_kairos=True)
    response_k = qila_kairos.query("Who composed the Italian Symphony?")
    print("\nResponse object:")
    print(f"  Answer: {response_k.answer}")
    print(f"  Kairos Field:")
    for claim_dict in response_k.kairos_field:
        print(f"    - Claim: {claim_dict['claim']}")
        print(f"      U: {claim_dict['U']} | Zone: {claim_dict['zone']}")
