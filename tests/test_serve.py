"""
P053 — FastAPI Endpoint Tests
==============================
Tests for API endpoints using httpx async test client.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_die_input():
    """Realistic single die input matching WaferDieInput schema."""
    return {
        "test_temp_c": 45.0,
        "cell_leakage_fa": 15.2,
        "retention_time_ms": 68.5,
        "row_hammer_threshold": 0.6,
        "disturb_margin_mv": 120.0,
        "adjacent_row_activations": 5000.0,
        "rh_susceptibility": 0.08,
        "bit_error_rate": 0.0001,
        "correctable_errors_per_1m": 3.0,
        "ecc_syndrome_entropy": 1.5,
        "uncorrectable_in_extended": 0.0,
        "trcd_ns": 13.5,
        "trp_ns": 13.5,
        "tras_ns": 36.0,
        "rw_latency_ns": 14.5,
        "idd4_active_ma": 320.0,
        "idd2p_standby_ma": 85.0,
        "idd5_refresh_ma": 120.0,
        "gate_oxide_thickness_a": 54.0,
        "channel_length_nm": 18.0,
        "vt_shift_mv": 5.0,
        "block_erase_count": 1200.0,
        "tester_id": "T001",
        "probe_card_id": "PC001",
        "chamber_id": "CH001",
        "recipe_version": "R1.0",
        "die_x": 10.0,
        "die_y": 12.0,
        "edge_distance": 45.0,
    }


class TestServeImportable:
    """Smoke tests — verify serve module can be imported and has expected components."""

    def test_import_serve(self):
        from serve import app  # noqa: F401

    def test_import_schemas(self):
        from serve import WaferDieInput, PredictionResponse  # noqa: F401

    def test_wafer_die_input_validation(self, sample_die_input):
        """Pydantic model should accept valid input."""
        from serve import WaferDieInput
        die = WaferDieInput(**sample_die_input)
        assert die.test_temp_c == 45.0
        assert die.die_x == 10.0

    def test_wafer_die_input_rejects_bad_values(self):
        """Pydantic model should reject out-of-range values."""
        from serve import WaferDieInput
        with pytest.raises(Exception):  # ValidationError
            WaferDieInput(
                test_temp_c=-999.0,  # Invalid: below -50
                cell_leakage_fa=-1.0,  # Invalid: negative
                retention_time_ms=-100.0,  # Invalid: negative
            )
