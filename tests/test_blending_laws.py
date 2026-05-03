import pytest
from core.blending.blending_law import (
    blend_ysi_mass_weighted,
    count_carbon_types,
    ysi_from_carbon_types,
    blend_ysi_carbon_type,
)


class TestBlendYsiMassWeighted:
    def test_basic(self):
        result = blend_ysi_mass_weighted(["CCCC", "CCCCC"], [0.5, 0.5], [20.0, 30.0])
        assert result is not None
        assert result > 0

    def test_none_ysi_returns_none(self):
        result = blend_ysi_mass_weighted(["CCCC", "CCCCC"], [0.5, 0.5], [None, 30.0])
        assert result is None

    def test_invalid_smiles_returns_none(self):
        result = blend_ysi_mass_weighted(["NOT_VALID", "CCCCC"], [0.5, 0.5], [20.0, 30.0])
        assert result is None

    def test_single_component(self):
        result = blend_ysi_mass_weighted(["CCCC"], [1.0], [25.0])
        assert result == pytest.approx(25.0)


class TestCountCarbonTypes:
    def test_n_pentane(self):
        counts = count_carbon_types("CCCCC")
        assert counts is not None
        assert counts["ct1"] == 2
        assert counts["ct2"] == 3

    def test_n_butane(self):
        counts = count_carbon_types("CCCC")
        assert counts is not None
        assert counts["ct1"] == 2
        assert counts["ct2"] == 2

    def test_benzene_aromatic(self):
        counts = count_carbon_types("c1ccccc1")
        assert counts is not None
        assert counts["ct7"] == 6

    def test_invalid_smiles_returns_none(self):
        assert count_carbon_types("NOT_VALID") is None

    def test_total_carbon_count(self):
        counts = count_carbon_types("CCCCC")
        assert sum(counts.values()) == 5

    def test_isobutane_has_ct3(self):
        counts = count_carbon_types("CC(C)C")
        assert counts is not None
        assert counts["ct3"] >= 1

    def test_cyclohexane_ct4(self):
        counts = count_carbon_types("C1CCCCC1")
        assert counts is not None
        assert counts["ct4"] == 6

    def test_heteroatom_skipped(self):
        counts = count_carbon_types("CCO")
        assert counts is not None
        assert sum(counts.values()) == 2

    def test_methylcyclohexane_ct5(self):
        counts = count_carbon_types("CC1CCCCC1")
        assert counts is not None
        assert counts["ct5"] >= 1


class TestYsiFromCarbonTypes:
    def test_returns_float_for_alkane(self):
        result = ysi_from_carbon_types("CCCC")
        assert result is not None
        assert isinstance(result, float)

    def test_invalid_smiles_returns_none(self):
        assert ysi_from_carbon_types("INVALID") is None

    def test_longer_chain_higher_ysi(self):
        ysi_butane = ysi_from_carbon_types("CCCC")
        ysi_octane = ysi_from_carbon_types("CCCCCCCC")
        assert ysi_octane > ysi_butane


class TestBlendYsiCarbonType:
    def test_basic(self):
        result = blend_ysi_carbon_type(["CCCC", "CCCCC"], [0.5, 0.5])
        assert result is not None
        assert isinstance(result, float)

    def test_invalid_smiles_returns_none(self):
        result = blend_ysi_carbon_type(["NOT_VALID", "CCCCC"], [0.5, 0.5])
        assert result is None

    def test_single_component_matches_pure(self):
        result = blend_ysi_carbon_type(["CCCC"], [1.0])
        pure = ysi_from_carbon_types("CCCC")
        assert result == pytest.approx(pure)
