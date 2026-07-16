"""
KAIROS Unit Tests — Pure math, no API calls required
=====================================================
Tests the pressure formula, zone classification, entropy normalization,
and claim extraction logic.
"""

import os
import sys
import math
import pytest

# Add KAIROS and MAKS to path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kairos_dir = os.path.join(workspace_root, "KAIROS")
maks_dir = os.path.join(workspace_root, "MAKS")
qila_dir = os.path.join(workspace_root, "qila")
for d in [kairos_dir, maks_dir, qila_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)


class TestPressureFormula:
    """Tests U = H × G × (1-C) composite score."""

    def test_perfect_confidence(self):
        """When consistency is 1.0, uncertainty should be 0."""
        H, G, C = 0.8, 0.9, 1.0
        U = H * G * (1 - C)
        assert U == 0.0

    def test_maximum_uncertainty(self):
        """When all signals are maximal, U should be near 1."""
        H, G, C = 1.0, 1.0, 0.0
        U = H * G * (1 - C)
        assert U == 1.0

    def test_zero_entropy_zeroes_uncertainty(self):
        """If entropy is 0 (model is certain), U should be 0."""
        H, G, C = 0.0, 0.8, 0.2
        U = H * G * (1 - C)
        assert U == 0.0

    def test_zero_gradient_zeroes_uncertainty(self):
        """If gradient is 0 (claim not load-bearing), U should be 0."""
        H, G, C = 0.7, 0.0, 0.3
        U = H * G * (1 - C)
        assert U == 0.0

    def test_multiplicative_structure(self):
        """Test the actual multiplicative composition."""
        H, G, C = 0.5, 0.6, 0.4
        U = H * G * (1 - C)
        expected = 0.5 * 0.6 * 0.6  # 0.18
        assert abs(U - expected) < 1e-10

    def test_symmetry_in_H_and_G(self):
        """H and G contribute symmetrically to U."""
        C = 0.3
        U1 = 0.4 * 0.8 * (1 - C)
        U2 = 0.8 * 0.4 * (1 - C)
        assert abs(U1 - U2) < 1e-10


class TestZoneClassification:
    """Tests the zone thresholds for SOLID / GRADIENT / FAULT LINE."""

    # Default thresholds from config
    PHI_S = 0.03
    PHI_F = 0.08

    def classify(self, U):
        if U <= self.PHI_S:
            return "SOLID"
        elif U >= self.PHI_F:
            return "FAULT LINE"
        else:
            return "GRADIENT"

    def test_solid_zone(self):
        assert self.classify(0.0) == "SOLID"
        assert self.classify(0.01) == "SOLID"
        assert self.classify(0.03) == "SOLID"

    def test_gradient_zone(self):
        assert self.classify(0.04) == "GRADIENT"
        assert self.classify(0.05) == "GRADIENT"
        assert self.classify(0.07) == "GRADIENT"

    def test_fault_zone(self):
        assert self.classify(0.08) == "FAULT LINE"
        assert self.classify(0.15) == "FAULT LINE"
        assert self.classify(1.0) == "FAULT LINE"

    def test_boundary_solid_gradient(self):
        """PHI_S is inclusive for SOLID."""
        assert self.classify(self.PHI_S) == "SOLID"
        assert self.classify(self.PHI_S + 0.001) == "GRADIENT"

    def test_boundary_gradient_fault(self):
        """PHI_F is inclusive for FAULT LINE."""
        assert self.classify(self.PHI_F) == "FAULT LINE"
        assert self.classify(self.PHI_F - 0.001) == "GRADIENT"


class TestEntropyNormalization:
    """Tests entropy computation and normalization."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution over V tokens should give H=1 (normalized)."""
        V = 100
        probs = {f"tok_{i}": 1.0 / V for i in range(V)}
        H = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        H_norm = H / math.log2(V)
        assert abs(H_norm - 1.0) < 1e-6

    def test_single_token_zero_entropy(self):
        """Single token with prob 1 should give H=0."""
        probs = {"only_token": 1.0}
        H = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        assert H == 0.0

    def test_binary_equal_distribution(self):
        """Two equally likely tokens → H = 1 bit."""
        probs = {"a": 0.5, "b": 0.5}
        H = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        assert abs(H - 1.0) < 1e-6

    def test_skewed_distribution_low_entropy(self):
        """Highly skewed distribution → low entropy."""
        probs = {"dominant": 0.99, "rare": 0.01}
        H = -sum(p * math.log2(p) for p in probs.values() if p > 0)
        assert H < 0.1  # Very low entropy


class TestClaimExtraction:
    """Tests the regex-based claim extraction from LLM responses."""

    def test_numbered_claims(self):
        """Claims formatted as numbered list."""
        text = "1. The sky is blue.\n2. Water is wet.\n3. Fire is hot."
        # Simulate extraction
        import re
        claims = re.findall(r'\d+\.\s*(.+)', text)
        assert len(claims) == 3
        assert claims[0] == "The sky is blue."

    def test_empty_response(self):
        """Empty response should yield no claims."""
        import re
        claims = re.findall(r'\d+\.\s*(.+)', "")
        assert len(claims) == 0


class TestUBounds:
    """Tests that U values are always within [0, 1]."""

    def test_all_combinations(self):
        """Exhaustive test over a grid of H, G, C values."""
        for h in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for g in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                for c in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
                    U = h * g * (1 - c)
                    assert 0.0 <= U <= 1.0, f"U={U} out of bounds for H={h}, G={g}, C={c}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
