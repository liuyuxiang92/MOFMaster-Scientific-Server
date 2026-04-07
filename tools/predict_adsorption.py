"""
Adsorption Predictor Tool

Predicts the adsorption amount of specific gas molecules in a MOF structure
given pressure, temperature and gas type.
"""

from .base import (
    BaseModel, Field, ValidationError,
    Optional, Atoms, DeepProperty, os, DATA_DIR,
    field_validator
)
import numpy as np
import json

# Load gas type to 128-dimensional vector mapping from data/gas_embeddings.json
GAS_MAPPINGS = {}
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "gas_embeddings.json")
if os.path.exists(EMBEDDINGS_PATH):
    try:
        with open(EMBEDDINGS_PATH, 'r') as f:
            GAS_MAPPINGS = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load gas embeddings from {EMBEDDINGS_PATH}: {e}")


class PredictAdsorptionInput(BaseModel):
    """Input model for gas adsorption prediction."""
    atoms_dict: dict = Field(
        ...,
        description="ASE Atoms object as dictionary (from parse_structure output)"
    )
    pressure: float = Field(..., description="Pressure in Bar")
    temperature: float = Field(..., description="Temperature in Kelvin")
    gas_type: str = Field(
        ..., 
        description="Gas molecule type (CO2, H2, Ar, N2, CH4, Xe)"
    )

    @field_validator('gas_type')
    @classmethod
    def normalize_gas_type(cls, v: str) -> str:
        """Normalize gas type to uppercase for case-insensitive matching."""
        upper_v = v.upper()
        if upper_v not in GAS_MAPPINGS:
            supported = ", ".join(GAS_MAPPINGS.keys())
            raise ValueError(f"Unsupported gas type: {v}. Supported types: {supported}")
        return upper_v


class PredictAdsorptionOutput(BaseModel):
    """Output model for gas adsorption prediction results."""
    success: bool = Field(..., description="Whether the prediction was successful")
    adsorption: Optional[float] = Field(None, description="Predicted adsorption amount, mol/kg (scalar value).")
    error: Optional[str] = Field(None, description="Error message if prediction failed")
    message: str = Field(..., description="Human-readable result message")


def predict_adsorption(atoms_dict: dict, pressure: float, temperature: float, gas_type: str) -> dict:
    """
    Predict the gas adsorption amount of a MOF structure using a DPA-based property model.
    
    Args:
        atoms_dict: ASE Atoms object as dictionary (from parse_structure output)
        pressure: Pressure in Bar (float)
        temperature: Temperature in Kelvin (float)
        gas_type: Gas molecule type (str: CO2, H2, Ar, N2, CH4, Xe)
        
    Returns:
        Dictionary containing adsorption prediction result
        
    Raises:
        ValidationError: If input validation fails
    """
    try:
        # Validate inputs using Pydantic (includes gas type normalization)
        validated_input = PredictAdsorptionInput(
            atoms_dict=atoms_dict,
            pressure=pressure,
            temperature=temperature,
            gas_type=gas_type
        )
        
        # Use normalized gas type from validated input
        normalized_gas = validated_input.gas_type
        
        try:
            # Reconstruct Atoms object from dictionary
            atoms = Atoms(
                numbers=validated_input.atoms_dict["numbers"],
                positions=validated_input.atoms_dict["positions"],
                cell=validated_input.atoms_dict.get("cell"),
                pbc=validated_input.atoms_dict.get("pbc", [False, False, False])
            )
            
            coords = atoms.get_positions()
            cells = atoms.get_cell()
            atom_numbers = atoms.get_atomic_numbers()               
            atom_types = [x - 1 for x in atom_numbers]
            
            # Prepare fparam (130 dimensions)
            # 1. Get 128D gas vector
            gas_vector = GAS_MAPPINGS[normalized_gas]
            # 2. Append pressure and temperature
            fparam = gas_vector + [float(pressure), float(temperature)]
            
            # Load the adsorption model
            # We assume the model file is named 'adsorption.ckpt.pt' in the DATA_DIR
            model_path = os.path.join(DATA_DIR, "adsorption.ckpt.pt")
            
            # Initialize DeepProperty model
            model = DeepProperty(model_path, head="property")
            
            # Predict adsorption value using eval with fparam
            # fparam should be passed as a list of lists (n_frames, n_fparams)
            # coords, cells, atom_types are also handled as single frame by DeepProperty.eval
            result = model.eval(
                coords=coords, 
                cells=cells, 
                atom_types=atom_types,
                fparam=np.array([fparam])
            )
            
            # Extract scalar value from result [n_frames][n_outputs][...]
            adsorption_val = float(result[0][0][0])
            
            output = PredictAdsorptionOutput(
                success=True,
                adsorption=adsorption_val,
                error=None,
                message=f"Adsorption prediction successful for {normalized_gas} at {pressure} Bar and {temperature} K. Result: {adsorption_val:.6f}"
            )
            return output.model_dump()
            
        except Exception as calc_error:
            output = PredictAdsorptionOutput(
                success=False,
                adsorption=None,
                error=str(calc_error),
                message=f"Prediction calculation error: {str(calc_error)}"
            )
            return output.model_dump()
            
    except ValidationError as e:
        error_output = PredictAdsorptionOutput(
            success=False,
            adsorption=None,
            error="Input validation error",
            message=f"Input validation error: {str(e)}"
        )
        return error_output.model_dump()
    except Exception as e:
        error_output = PredictAdsorptionOutput(
            success=False,
            adsorption=None,
            error="Unexpected error",
            message=f"Unexpected error: {str(e)}"
        )
        return error_output.model_dump()
