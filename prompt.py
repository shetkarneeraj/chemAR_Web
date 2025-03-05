import os
import json
import google.generativeai as genai
import re
from typing import Optional, Dict

# Set up Gemini
genai.configure(api_key="AIzaSyAb4TTvJNOcSeZe4BgwvUrBgUQeAoYvNXI")


def geminiGenerate(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text


def safe_json_extract(response: str) -> Optional[Dict]:
    """Robust JSON extraction with parsing"""
    try:
        # Try to find JSON between ``` markers
        json_match = re.search(r'```json(.*?)```', response, re.DOTALL) or re.search(r'```(.*?)```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Fallback to finding first complete JSON object
            json_str = response[response.find('{'):response.rfind('}')+1]

        # Clean JSON string
        json_str = json_str.replace('\\"', '"')
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        return json_str
    
    except (AttributeError, json.JSONDecodeError, KeyError) as e:
        print(f"JSON extraction error: {str(e)}")
        return None
    
# Structured prompt template
def get_compound_data(description: str) -> dict:

    prompt = '''
        Analyze this chemical compound description and return structural data in JSON format. Whatever be the strength of data
        show full representation of the compound in 3D space with all the atoms and bonds. Get all the data do not skip any element.
        Follow this EXACT structure:
        {
        "name": "IUPAC name",
        "description": "Brief chemical description",
        "formula": "Molecular formula",
        "atoms": {
            "C1": {
            "element": "C",
            "atomic_number": 6,
            "position": [x,y,z],
            "valence_electrons": 4,
            "hybridization": "sp3"
            },
            "O2": {
            "element": "O",
            "atomic_number": 8,
            "position": [x,y,z],
            "valence_electrons": 6,
            "hybridization": "sp2"
            },
            ...
        },
        "bonds": [
            {
            "atom1": "C1",
            "atom2": "C2",
            "bond_type": "single|double|triple",
            "plane": "horizontal|vertical",
            "angle": radians,
            "length": angstroms
            },
            ...
        ],
        "functional_groups": ["carboxylic acid", ...],
        "molecular_geometry": {
            "shape": "tetrahedral|trigonal-planar|etc",
            "bond_angles": [
            {
                "atoms": ["C1", "C2", "O1"],
                "degrees": 120.0
            },
            ...
            ]
        },
        "description": "Detailed description of the compound how its synthesized and what are its uses?"
        }

        Important rules:
        1. Give unique IDs to atoms (e.g., C1, C2, O1, H1, H2)
        2. Specify exact atom positions in 3D space
        3. Include bond lengths in angstroms
        4. List all relevant bond angles
        5. Add a clear chemical description
        6. Include all the relevant functional groups
        8. Show all the elements and their positions
        9. Do not truncate any data
        10. Do not return anything other than JSON

        Compound description:
    '''
    
    try:
        response = geminiGenerate(prompt + description)
        json_str = safe_json_extract(response)
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        return None
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    compound_info = get_compound_data("Sugar")
    print(compound_info)