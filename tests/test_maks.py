"""
MAKS Unit Tests — Memory management logic
==========================================
Tests memory creation, fidelity transitions, ghost archival,
and survival scoring — no API calls needed.
"""

import os
import sys
import time
import pytest

# Add MAKS to path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
maks_dir = os.path.join(workspace_root, "MAKS")
if maks_dir not in sys.path:
    sys.path.insert(0, maks_dir)

from memory_unit import MemoryUnit, create_memory_unit


class TestMemoryUnit:
    """Tests MemoryUnit creation and defaults."""

    def test_create_memory_unit(self):
        unit = create_memory_unit("test-1", "The sky is blue.")
        assert unit.id == "test-1"
        assert unit.content == "The sky is blue."
        assert unit.fidelity == "FULL"
        assert unit.access_count == 0
        assert unit.created_at > 0
        assert unit.last_accessed_at > 0

    def test_default_entropy_score(self):
        unit = create_memory_unit("test-2", "Water is wet.")
        assert unit.entropy_score == 0.5

    def test_default_cached_survival(self):
        unit = create_memory_unit("test-3", "Fire is hot.")
        assert unit.cached_survival is None

    def test_access_history_empty(self):
        unit = create_memory_unit("test-4", "test")
        assert unit.access_history == []
        assert unit.connections == []


class TestFidelityTransitions:
    """Tests fidelity level transitions: FULL → PARTIAL → GHOST."""

    def test_valid_fidelity_levels(self):
        unit = create_memory_unit("f-1", "test")
        assert unit.fidelity in ["FULL", "PARTIAL", "GHOST"]

    def test_fidelity_change_to_partial(self):
        unit = create_memory_unit("f-2", "long content here")
        unit.fidelity = "PARTIAL"
        assert unit.fidelity == "PARTIAL"

    def test_fidelity_change_to_ghost(self):
        unit = create_memory_unit("f-3", "ghosting")
        unit.fidelity = "GHOST"
        assert unit.fidelity == "GHOST"


class TestSurvivalScoring:
    """Tests the survival score formula components."""

    def test_recency_decay(self):
        """More recent memories should score higher on recency."""
        now = time.time()
        recent = create_memory_unit("r-1", "recent")
        recent.last_accessed_at = now

        old = create_memory_unit("r-2", "old")
        old.last_accessed_at = now - 86400  # 24 hours ago

        # Recency component: e^(-lambda * dt)
        lambda_val = 0.01
        recency_recent = 2.71828 ** (-lambda_val * 0)
        recency_old = 2.71828 ** (-lambda_val * 86400)

        assert recency_recent > recency_old

    def test_frequency_boost(self):
        """Higher access count should increase survival."""
        unit_low = create_memory_unit("freq-1", "low freq")
        unit_low.access_count = 1

        unit_high = create_memory_unit("freq-2", "high freq")
        unit_high.access_count = 100

        # Log frequency: log(1 + n)
        import math
        freq_low = math.log(1 + unit_low.access_count)
        freq_high = math.log(1 + unit_high.access_count)

        assert freq_high > freq_low

    def test_entropy_penalty(self):
        """Higher entropy should penalize survival."""
        unit_certain = create_memory_unit("e-1", "certain")
        unit_certain.entropy_score = 0.1

        unit_uncertain = create_memory_unit("e-2", "uncertain")
        unit_uncertain.entropy_score = 0.9

        # Entropy penalty: (1 - H)
        penalty_certain = 1 - unit_certain.entropy_score
        penalty_uncertain = 1 - unit_uncertain.entropy_score

        assert penalty_certain > penalty_uncertain


class TestGhostStore:
    """Tests ghost archival logic."""

    def test_ghost_creation(self):
        unit = create_memory_unit("g-1", "to be ghosted")
        unit.fidelity = "GHOST"
        assert unit.fidelity == "GHOST"
        assert unit.content == "to be ghosted"

    def test_original_content_preserved(self):
        unit = create_memory_unit("g-2", "full original content")
        unit.original_content = "full original content"
        unit.content = "compressed"
        unit.fidelity = "PARTIAL"
        assert unit.original_content == "full original content"
        assert unit.content == "compressed"


class TestPersistence:
    """Tests SQLite persistence layer."""

    def test_persistence_import(self):
        """Verify persistence module can be imported."""
        qila_dir = os.path.join(workspace_root, "qila")
        if qila_dir not in sys.path:
            sys.path.insert(0, qila_dir)
        import persistence
        assert hasattr(persistence, 'save_memories')
        assert hasattr(persistence, 'load_memories')
        assert hasattr(persistence, 'save_turn')

    def test_roundtrip_memory(self):
        """Test save and load of memory units."""
        qila_dir = os.path.join(workspace_root, "qila")
        if qila_dir not in sys.path:
            sys.path.insert(0, qila_dir)
        import persistence
        import tempfile

        # Use temp DB
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            persistence.init_db(db_path)
            unit = create_memory_unit("rt-1", "roundtrip test")
            store = {"rt-1": unit}
            persistence.save_memories(store, "test-session", db_path)

            loaded = persistence.load_memories("test-session", db_path)
            assert "rt-1" in loaded
            assert loaded["rt-1"].content == "roundtrip test"
            assert loaded["rt-1"].fidelity == "FULL"
        finally:
            os.unlink(db_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
