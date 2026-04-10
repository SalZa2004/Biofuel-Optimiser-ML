# Base fuel library
class BaseFuelLibrary:
    """Library of base fuels."""
    @staticmethod
    def get_fossil_diesel():
        """Get fossil diesel composition."""
        smiles = [
            "CCCCCCC",
        ]
        
        fractions = [1.0]
        total = sum(fractions)
        fractions = [f / total for f in fractions]
        
        return smiles, fractions
    
    @staticmethod
    def get_biodiesel():
        """Get biodiesel composition."""
        smiles = [
            "CCCCCCCCCCCCCCCCCC(=O)OC",
            "CCCCCCCCC/C=C/CCCCCCCC(=O)OC",
            "CCCCCC/C=C/C/C=C/CCCCCCC(=O)OC",
            "CCCCCCCCCCCCCCCC(=O)OC",
        ]
        
        fractions = [0.10, 0.50, 0.35, 0.05]
        total = sum(fractions)
        fractions = [f / total for f in fractions]
        
        return smiles, fractions
    
    @staticmethod
    def get_base_fuel(fuel_type: str):
        """Get base fuel by type."""
        if fuel_type == "fossil_diesel":
            return BaseFuelLibrary.get_fossil_diesel()
        elif fuel_type == "biodiesel":
            return BaseFuelLibrary.get_biodiesel()
        else:
            raise ValueError(f"Unknown fuel type: {fuel_type}")