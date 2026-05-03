import pytest
from unittest.mock import patch

from applications.mixture_predictor.results import (
    display_mixture_summary,
    predict_dcn,
    save_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _components():
    return [
        {"smiles": "CCCCCC", "formula": "C6H14", "mole_fraction": 0.5},
        {"smiles": "CCCCCCC", "formula": "C7H16", "mole_fraction": 0.5},
    ]


# ---------------------------------------------------------------------------
# display_mixture_summary
# ---------------------------------------------------------------------------

class TestDisplayMixtureSummary:
    def test_header_present(self, capsys):
        display_mixture_summary(_components())
        out = capsys.readouterr().out
        assert "MIXTURE COMPOSITION SUMMARY" in out

    def test_total_row_present(self, capsys):
        display_mixture_summary(_components())
        out = capsys.readouterr().out
        assert "TOTAL" in out

    def test_smiles_printed(self, capsys):
        display_mixture_summary(_components())
        out = capsys.readouterr().out
        assert "CCCCCC" in out

    def test_fractions_sum_shown(self, capsys):
        display_mixture_summary(_components())
        out = capsys.readouterr().out
        assert "1.0000" in out

    def test_single_component(self, capsys):
        components = [{"smiles": "CCCC", "formula": "C4H10", "mole_fraction": 1.0}]
        display_mixture_summary(components)
        out = capsys.readouterr().out
        assert "MIXTURE COMPOSITION SUMMARY" in out


# ---------------------------------------------------------------------------
# predict_dcn
# ---------------------------------------------------------------------------

class TestPredictDcn:
    def test_returns_positive_float(self):
        dcn = predict_dcn(_components())
        assert dcn is not None
        assert isinstance(dcn, float)
        assert dcn > 0

    def test_prints_result(self, capsys):
        predict_dcn(_components())
        out = capsys.readouterr().out
        assert "DCN" in out

    def test_exception_returns_none(self, capsys):
        """When the predictor raises, the function should return None gracefully."""
        with patch(
            "applications.mixture_predictor.results.MixtureDCNPredictor",
            side_effect=RuntimeError("model load failed"),
        ):
            result = predict_dcn(_components())
        assert result is None
        out = capsys.readouterr().out
        assert "FAILED" in out


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_skips_on_n(self, tmp_path):
        with patch("builtins.input", return_value="n"):
            save_results(_components(), 52.0)
        # No file should have been written

    def test_writes_file_on_y(self, tmp_path):
        out_file = str(tmp_path / "output.txt")
        with patch("builtins.input", side_effect=["y", out_file]):
            save_results(_components(), 52.0)
        assert (tmp_path / "output.txt").exists()
        content = (tmp_path / "output.txt").read_text()
        assert "52.00" in content

    def test_default_filename_used_when_empty_input(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("builtins.input", side_effect=["y", ""]):
            save_results(_components(), 48.5)
        assert (tmp_path / "mixture_prediction.txt").exists()

    def test_txt_extension_appended(self, tmp_path):
        out_file = str(tmp_path / "result")
        with patch("builtins.input", side_effect=["y", out_file]):
            save_results(_components(), 50.0)
        assert (tmp_path / "result.txt").exists()

    def test_file_contains_smiles(self, tmp_path):
        out_file = str(tmp_path / "out.txt")
        with patch("builtins.input", side_effect=["y", out_file]):
            save_results(_components(), 52.0)
        content = (tmp_path / "out.txt").read_text()
        assert "CCCCCC" in content
