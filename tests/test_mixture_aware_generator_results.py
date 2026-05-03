import matplotlib
matplotlib.use("Agg")  # prevent GUI windows during tests

import pytest
import pandas as pd

from applications.mixture_aware_generator.results import (
    display_results,
    plot_pareto_front,
    save_results,
)
from core.config import EvolutionConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dfs(with_mixture_ysi=True):
    cols = {
        "rank": [1, 2, 3],
        "smiles": ["CCCC", "CCCCC", "CCCCCC"],
        "cn": [50.0, 48.0, 46.0],
        "cn_error": [1.0, 3.0, 5.0],
        "bp": [150.0, 160.0, 170.0],
        "density": [750.0, 760.0, 770.0],
        "lhv": [42.0, 43.0, 44.0],
        "dynamic_viscosity": [2.5, 2.8, 3.0],
    }
    if with_mixture_ysi:
        cols["mixture_ysi"] = [25.0, 28.0, 31.0]

    final_df = pd.DataFrame(cols)
    pareto_df = pd.DataFrame({
        "rank": [1, 2],
        "smiles": ["CCCC", "CCCCC"],
        "cn": [50.0, 48.0],
        "cn_error": [1.0, 3.0],
        "mixture_ysi": [25.0, 28.0],
    })
    return final_df, pareto_df, final_df.copy()


# ---------------------------------------------------------------------------
# display_results
# ---------------------------------------------------------------------------

class TestDisplayResults:
    def test_shows_best_candidates_header(self, capsys):
        final_df, pareto_df, unfiltered_df = _make_dfs()
        config = EvolutionConfig(minimize_ysi=True, maximize_cn=False)
        display_results(final_df, pareto_df, unfiltered_df, config)
        out = capsys.readouterr().out
        assert "BEST CANDIDATES" in out

    def test_shows_pareto_section_when_minimize_ysi(self, capsys):
        final_df, pareto_df, unfiltered_df = _make_dfs()
        config = EvolutionConfig(minimize_ysi=True, maximize_cn=False)
        display_results(final_df, pareto_df, unfiltered_df, config)
        out = capsys.readouterr().out
        assert "PARETO FRONT" in out

    def test_no_pareto_section_when_empty_pareto(self, capsys):
        final_df, _, unfiltered_df = _make_dfs()
        config = EvolutionConfig(minimize_ysi=True, maximize_cn=False)
        display_results(final_df, pd.DataFrame(), unfiltered_df, config)
        out = capsys.readouterr().out
        assert "BEST CANDIDATES" in out
        assert "PARETO FRONT" not in out

    def test_no_pareto_section_when_minimize_ysi_false(self, capsys):
        final_df, pareto_df, unfiltered_df = _make_dfs()
        config = EvolutionConfig(minimize_ysi=False, maximize_cn=False)
        display_results(final_df, pareto_df, unfiltered_df, config)
        out = capsys.readouterr().out
        assert "PARETO FRONT" not in out

    def test_maximize_cn_drops_cn_error_col(self, capsys):
        final_df, pareto_df, unfiltered_df = _make_dfs()
        config = EvolutionConfig(minimize_ysi=True, maximize_cn=True)
        display_results(final_df, pareto_df, unfiltered_df, config)
        out = capsys.readouterr().out
        assert "cn_error" not in out


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_creates_final_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        final_df, pareto_df, unfiltered_df = _make_dfs()
        save_results(final_df, pareto_df, unfiltered_df, minimize_ysi=False)
        assert (tmp_path / "results" / "final_population.csv").exists()

    def test_creates_unfiltered_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        final_df, pareto_df, unfiltered_df = _make_dfs()
        save_results(final_df, pareto_df, unfiltered_df, minimize_ysi=False)
        assert (tmp_path / "results" / "final_population_unfiltered.csv").exists()

    def test_creates_pareto_csv_when_minimize_ysi(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        final_df, pareto_df, unfiltered_df = _make_dfs()
        save_results(final_df, pareto_df, unfiltered_df, minimize_ysi=True)
        assert (tmp_path / "results" / "pareto_front.csv").exists()

    def test_no_pareto_csv_when_minimize_ysi_false(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        final_df, pareto_df, unfiltered_df = _make_dfs()
        save_results(final_df, pareto_df, unfiltered_df, minimize_ysi=False)
        assert not (tmp_path / "results" / "pareto_front.csv").exists()

    def test_no_pareto_csv_when_pareto_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        final_df, _, unfiltered_df = _make_dfs()
        save_results(final_df, pd.DataFrame(), unfiltered_df, minimize_ysi=True)
        assert not (tmp_path / "results" / "pareto_front.csv").exists()

    def test_prints_confirmation(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        final_df, pareto_df, unfiltered_df = _make_dfs()
        save_results(final_df, pareto_df, unfiltered_df, minimize_ysi=False)
        assert "Results saved" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# plot_pareto_front
# ---------------------------------------------------------------------------

class TestPlotParetoFront:
    def test_creates_png(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        final_df, pareto_df, unfiltered_df = _make_dfs()
        plot_pareto_front(pareto_df, final_df, unfiltered_df)
        assert (tmp_path / "results" / "mixture_pareto_front.png").exists()

    def test_maximize_cn_mode(self, tmp_path, monkeypatch):
        """cn_error column absent triggers maximize_cn label path."""
        monkeypatch.chdir(tmp_path)
        final_df, _, unfiltered_df = _make_dfs()
        pareto_df = pd.DataFrame({
            "cn": [50.0, 48.0],
            "mixture_ysi": [25.0, 28.0],
        })
        plot_pareto_front(pareto_df, final_df, unfiltered_df)
        assert (tmp_path / "results" / "mixture_pareto_front.png").exists()

    def test_ysi_col_fallback(self, tmp_path, monkeypatch):
        """Falls back to 'ysi' column when 'mixture_ysi' is absent."""
        monkeypatch.chdir(tmp_path)
        final_df_no_mysi, pareto_df_no_mysi, unfiltered_df_no_mysi = _make_dfs(with_mixture_ysi=False)
        final_df_no_mysi["ysi"] = [20.0, 22.0, 24.0]
        pareto_df_no_mysi = pd.DataFrame({
            "cn": [50.0, 48.0],
            "cn_error": [1.0, 3.0],
            "ysi": [20.0, 22.0],
        })
        unfiltered_df_no_mysi["ysi"] = [20.0, 22.0, 24.0]
        plot_pareto_front(pareto_df_no_mysi, final_df_no_mysi, unfiltered_df_no_mysi)
        assert (tmp_path / "results" / "mixture_pareto_front.png").exists()
