

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import warnings


@dataclass
class BlendComponent:
    """Represents a single component in a fuel blend."""
    smiles: str
    fraction: float  # Volume or mole fraction
    properties: Dict[str, float] 


class BlendingLaws:
    """
    Implements various blending laws for fuel property prediction.
    
    Different properties require different blending rules:

    """
    
    # Property-specific blending methods
    BLENDING_METHODS = {
        'density': 'linear',  # Volume-weighted average
        'dynamic_viscosity': 'viscosity_blending',  
        'kinematic_viscosity': 'viscosity_blending',
        'lhv': 'mass_weighted',  # Energy content is mass-weighted
        'bp': 'molar_weighted',  # Boiling point uses molar averaging
        'ysi': 'linear',  # YSI is approximately linear
        'flash_point': 'flash_point_blending',  # Non-linear flash point
    }
    
    def __init__(self):
        """Initialize blending laws calculator."""
        pass
    
    # =========================================================================
    # Main Interface
    # =========================================================================
    
    def predict_blend_properties(self,
                                 components: List[BlendComponent],
                                 properties: List[str],
                                 blend_basis: str = 'volume') -> Dict[str, float]:
        """
        Predict multiple blend properties at once.
        
        Args:
            components: List of BlendComponent objects
            properties: List of property names to predict (e.g., ['cn', 'density', 'ysi'])
            blend_basis: 'volume' or 'mole' or 'mass'
        
        Returns:
            Dictionary of predicted blend properties
        
        Example:
            >>> comp1 = BlendComponent(smiles='CCCCCCCC', fraction=0.7, 
            ...                        properties={'cn': 52.0, 'density': 0.75})
            >>> comp2 = BlendComponent(smiles='CC(C)C', fraction=0.3,
            ...                        properties={'cn': 38.0, 'density': 0.69})
            >>> laws = BlendingLaws()
            >>> blend_props = laws.predict_blend_properties([comp1, comp2], ['cn', 'density'])
            >>> print(blend_props)
            {'cn': 47.2, 'density': 0.732}
        """
        # Validate fractions
        self._validate_fractions(components)
        
        # Convert fractions to volume basis if needed
        if blend_basis != 'volume':
            components = self._convert_to_volume_basis(components, blend_basis)
        
        # Predict each property
        blend_properties = {}
        
        for prop in properties:
            method_name = self.BLENDING_METHODS.get(prop, 'linear')
            method = getattr(self, method_name, self.linear)
            
            # Extract component fractions and property values
            fractions = [c.fraction for c in components]
            values = [c.properties.get(prop) for c in components]
            
            # Skip if any component missing this property
            if None in values:
                warnings.warn(f"Property '{prop}' missing for some components, skipping")
                continue
            
            # Calculate blend property
            blend_properties[prop] = method(fractions, values)
        
        return blend_properties
    
    # =========================================================================
    # Blending Law Implementations
    # =========================================================================
    
    def linear(self, fractions: List[float], values: List[float]) -> float:
        """
        Simple linear (volume-weighted) blending.
        
        Blend_Property = Σ(x_i * Property_i)
        
        Used for: YSI, approximate CN, density
        """
        return sum(f * v for f, v in zip(fractions, values))
    
    def mass_weighted(self, fractions: List[float], values: List[float],
                     densities: Optional[List[float]] = None) -> float:
        """
        Mass-weighted blending.
        
        Used for: Lower Heating Value (LHV), energy content
        
        Args:
            fractions: Volume fractions
            values: Property values
            densities: Densities for mass conversion (if None, assumes equal density)
        """
        if densities is None:
            # If no densities provided, equivalent to linear
            return self.linear(fractions, values)
        
        # Convert volume fractions to mass fractions
        masses = [f * d for f, d in zip(fractions, densities)]
        total_mass = sum(masses)
        mass_fractions = [m / total_mass for m in masses]
        
        # Mass-weighted average
        return sum(mf * v for mf, v in zip(mass_fractions, values))
    
    def molar_weighted(self, fractions: List[float], values: List[float],
                      molecular_weights: Optional[List[float]] = None) -> float:
        """
        Molar-weighted blending.
        
        Used for: Boiling point, some physical properties
        
        Args:
            fractions: Volume fractions
            values: Property values
            molecular_weights: MWs for molar conversion (if None, assumes equal MW)
        """
        if molecular_weights is None:
            # If no MWs provided, equivalent to linear
            return self.linear(fractions, values)
        
        # Convert volume fractions to mole fractions (simplified, assumes ideal)
        moles = [f / mw for f, mw in zip(fractions, molecular_weights)]
        total_moles = sum(moles)
        mole_fractions = [m / total_moles for m in moles]
        
        # Molar-weighted average
        return sum(xf * v for xf, v in zip(mole_fractions, values))
    
    
    
    def viscosity_blending(self, fractions: List[float], 
                          viscosity_values: List[float]) -> float:
        """
        Viscosity blending using ASTM D7152 method.
        
        VBN (Viscosity Blending Number) method:
        VBN = 14.534 * ln(ln(viscosity + 0.8)) + 10.975
        
        Blend VBN, then convert back to viscosity.
        
        Reference: ASTM D7152 - Standard Practice for Calculating Viscosity Blending Number
        """
        # Convert viscosities to VBN
        vbns = []
        for visc in viscosity_values:
            if visc <= 0:
                warnings.warn(f"Invalid viscosity value: {visc}, using 1.0")
                visc = 1.0
            
            vbn = 14.534 * np.log(np.log(visc + 0.8)) + 10.975
            vbns.append(vbn)
        
        # Blend VBNs linearly
        blend_vbn = self.linear(fractions, vbns)
        
        # Convert back to viscosity
        blend_viscosity = np.exp(np.exp((blend_vbn - 10.975) / 14.534)) - 0.8
        
        return blend_viscosity
    
    def flash_point_blending(self, fractions: List[float],
                            flash_points: List[float]) -> float:
        """
        Flash point blending (approximation).
        
        Flash point is dominated by the most volatile (lowest FP) component.
        Uses weighted harmonic mean as approximation.
        
        Note: This is an approximation. True flash point depends on vapor pressure.
        """
        # Weighted harmonic mean
        if any(fp <= 0 for fp in flash_points):
            warnings.warn("Invalid flash point values, using linear blend")
            return self.linear(fractions, flash_points)
        
        reciprocal_sum = sum(f / fp for f, fp in zip(fractions, flash_points))
        
        if reciprocal_sum == 0:
            return self.linear(fractions, flash_points)
        
        return 1.0 / reciprocal_sum
    
    # =========================================================================
    # Advanced Blending Methods
    # =========================================================================
    
    def kay_mixing_rule(self, fractions: List[float], values: List[float],
                       critical_values: Optional[List[float]] = None) -> float:
        """
        Kay's mixing rule for pseudo-critical properties.
        
        Used for: Critical temperature, critical pressure
        Not typically used for cetane, but included for completeness.
        """
        return self.linear(fractions, values)
    
    def ideal_solution_blending(self, fractions: List[float], 
                               pure_properties: List[float],
                               activity_coefficients: Optional[List[float]] = None) -> float:
        """
        Ideal solution model with optional activity coefficients.
        
        For non-ideal mixtures, can incorporate activity coefficients.
        """
        if activity_coefficients is None:
            activity_coefficients = [1.0] * len(fractions)
        
        # Modified blending with activity coefficients
        effective_values = [p * gamma for p, gamma in zip(pure_properties, activity_coefficients)]
        
        return self.linear(fractions, effective_values)
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _validate_fractions(self, components: List[BlendComponent]):
        """Ensure fractions sum to 1.0 (within tolerance)."""
        total = sum(c.fraction for c in components)
        
        if not np.isclose(total, 1.0, atol=0.01):
            # Normalize fractions
            warnings.warn(f"Fractions sum to {total:.4f}, normalizing to 1.0")
            for component in components:
                component.fraction /= total
    
    def _convert_to_volume_basis(self, components: List[BlendComponent],
                                 current_basis: str) -> List[BlendComponent]:
        """Convert mass or mole fractions to volume fractions."""
        if current_basis == 'volume':
            return components
        
        # This is a simplified conversion
        # Real conversion requires densities and molecular weights
        warnings.warn("Fraction conversion not fully implemented, using as-is")
        return components
    
    # =========================================================================
    # Batch Prediction Interface
    # =========================================================================
    
    def predict_blend_property_batch(self,
                                     component_sets: List[List[BlendComponent]],
                                     property_name: str) -> List[float]:
        """
        Predict a single property for multiple blends.
        
        Useful for optimization where you're testing many blend compositions.
        
        Args:
            component_sets: List of component lists (each is one blend)
            property_name: Property to predict
        
        Returns:
            List of predicted values for each blend
        """
        results = []
        
        for components in component_sets:
            blend_props = self.predict_blend_properties(components, [property_name])
            results.append(blend_props.get(property_name))
        
        return results


class BlendingLawsWithConstraints(BlendingLaws):
    """
    Extended blending laws with property constraint checking.
    
    Useful for molecular evolution where you want to filter blends
    that don't meet spec requirements.
    """
    
    def __init__(self, constraints: Optional[Dict[str, tuple]] = None):
        """
        Args:
            constraints: Dict of {property: (min, max)} constraints
                        e.g., {'cn': (40, 55), 'density': (0.82, 0.85)}
        """
        super().__init__()
        self.constraints = constraints or {}
    
    def check_constraints(self, blend_properties: Dict[str, float]) -> tuple[bool, List[str]]:
        """
        Check if blend properties meet all constraints.
        
        Returns:
            (passes: bool, violations: List[str])
        """
        violations = []
        
        for prop, (min_val, max_val) in self.constraints.items():
            if prop not in blend_properties:
                continue
            
            value = blend_properties[prop]
            
            if min_val is not None and value < min_val:
                violations.append(f"{prop}={value:.2f} < {min_val} (min)")
            
            if max_val is not None and value > max_val:
                violations.append(f"{prop}={value:.2f} > {max_val} (max)")
        
        return len(violations) == 0, violations
    
    def predict_and_validate(self,
                           components: List[BlendComponent],
                           properties: List[str]) -> tuple[Dict[str, float], bool, List[str]]:
        """
        Predict blend properties and check constraints in one call.
        
        Returns:
            (blend_properties, passes_constraints, violations)
        """
        blend_props = self.predict_blend_properties(components, properties)
        passes, violations = self.check_constraints(blend_props)
        
        return blend_props, passes, violations


# =============================================================================
# Utility Functions
# =============================================================================

def create_blend_from_smiles(smiles_list: List[str],
                            fractions: List[float],
                            pure_predictor) -> List[BlendComponent]:
    """
    Helper to create BlendComponent list from SMILES and fractions.
    
    Args:
        smiles_list: List of SMILES strings
        fractions: List of volume fractions (must sum to 1.0)
        pure_predictor: PropertyPredictor instance to get pure properties
    
    Returns:
        List of BlendComponent objects
    """
    # Predict pure properties for all components
    predictions = pure_predictor.predict_all_properties(smiles_list)
    
    components = []
    for i, smiles in enumerate(smiles_list):
        properties = {k: v[i] for k, v in predictions.items() if v[i] is not None}
        
        components.append(BlendComponent(
            smiles=smiles,
            fraction=fractions[i],
            properties=properties
        ))
    
    return components


def optimize_blend_composition(target_properties: Dict[str, float],
                              available_components: List[BlendComponent],
                              blending_laws: BlendingLaws,
                              max_iterations: int = 100) -> List[float]:
    """
    Simple optimization to find blend fractions that match target properties.
    
    This is a placeholder - for real optimization use scipy.optimize or genetic algorithms.
    
    Args:
        target_properties: Dict of target values (e.g., {'cn': 50, 'density': 0.84})
        available_components: Components available for blending
        blending_laws: BlendingLaws instance
        max_iterations: Max optimization iterations
    
    Returns:
        Optimized fractions
    """
    # Placeholder - implement with scipy.optimize.minimize or similar
    raise NotImplementedError("Use scipy.optimize for real blend optimization")


if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("BLENDING LAWS - Example Usage")
    print("="*70)
    
    # Create example components
    component1 = BlendComponent(
        smiles='CCCCCCCCCCCCCCCC',  # Hexadecane (cetane)
        fraction=0.7,
        properties={
            'cn': 100.0,
            'density': 0.773,
            'ysi': 15.0,
            'dynamic_viscosity': 3.5
        }
    )
    
    component2 = BlendComponent(
        smiles='CC(C)CCCCC',  # Isoheptane
        fraction=0.3,
        properties={
            'cn': 30.0,
            'density': 0.684,
            'ysi': 60.0,
            'dynamic_viscosity': 0.8
        }
    )
    
    # Initialize blending laws
    laws = BlendingLaws()
    
    # Predict blend properties
    components = [component1, component2]
    properties_to_predict = ['cn', 'density', 'ysi', 'dynamic_viscosity']
    
    blend_props = laws.predict_blend_properties(components, properties_to_predict)
    
    print(f"\nComponent 1 (70%): {component1.smiles}")
    print(f"  CN: {component1.properties['cn']:.1f}")
    print(f"  Density: {component1.properties['density']:.3f}")
    print(f"  YSI: {component1.properties['ysi']:.1f}")
    
    print(f"\nComponent 2 (30%): {component2.smiles}")
    print(f"  CN: {component2.properties['cn']:.1f}")
    print(f"  Density: {component2.properties['density']:.3f}")
    print(f"  YSI: {component2.properties['ysi']:.1f}")
    
    print(f"\n{'='*70}")
    print("BLEND PROPERTIES (70/30 mix):")
    print(f"{'='*70}")
    for prop, value in blend_props.items():
        print(f"  {prop.upper()}: {value:.3f}")
    
    # Example with constraints
    print(f"\n{'='*70}")
    print("BLEND WITH CONSTRAINTS:")
    print(f"{'='*70}")
    
    constraints = {
        'cn': (40, 55),
        'density': (0.75, 0.85),
        'ysi': (0, 50)
    }
    
    laws_with_constraints = BlendingLawsWithConstraints(constraints)
    blend_props, passes, violations = laws_with_constraints.predict_and_validate(
        components, properties_to_predict
    )
    
    print(f"\nConstraints: {constraints}")
    print(f"Passes constraints: {passes}")
    if violations:
        print(f"Violations:")
        for v in violations:
            print(f"  - {v}")
    else:
        print("✓ All constraints satisfied!")