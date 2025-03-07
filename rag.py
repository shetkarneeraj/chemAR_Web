import os
import re
from pymongo import MongoClient
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import json

# Initialize embedding model (choose one)
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings

# Initialize MongoDB
client = MongoClient("mongodb+srv://neerajshetkar:29gx0gMglCCyhdff@cluster0.qfkfv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client[os.getenv("DB_NAME", "chemar")]
collection = db["docs"]

# Create search indexes (run once)
def create_indexes():
    # Text search index
    collection.create_index([("text", "text")])
    
    # Vector search index
    collection.create_index(
        [("embedding", "vector")],
        name="compound_vectors",
        vectorOptions={
            "type": "knnVector",
            "dimensions": 384,  # Match MiniLM-L6 dimensions
            "similarity": "cosine"
        }
    )

# PDF Processing with local embeddings
def process_and_index_pdf(pdf_path, chunk_size=1000):
    try:
        # Load embedding model
        embedding_model = get_embedding_model()
        
        # Extract text
        reader = PdfReader(pdf_path)
        text = " ".join([page.extract_text() for page in reader.pages])
        
        # Clean and chunk text
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Create documents with embeddings
        documents = []
        for idx, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()
            documents.append({
                "text": chunk,
                "embedding": embedding,
                "chunk_number": idx,
                "source": pdf_path
            })
        
        collection.insert_many(documents)
        return True
        
    except Exception as e:
        print(f"PDF processing error: {e}")
        return False


# Hybrid Search (Text + Vector)
def search_documents(query, organisation_id, top_k=5):
    # Get embedding model
    embedding_model = get_embedding_model()
    
    # Text search
    text_results = collection.find(
        {"$text": {"$search": query}, "organisation_id": organisation_id},
        {"score": {"$meta": "textScore"}}
    ).limit(top_k)
    
    # Vector search
    query_embedding = embedding_model.encode(query).tolist()
    vector_results = collection.aggregate([{
        "$vectorSearch": {
            "index": "compound_vectors",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": top_k
        }
    }])
    
    # Combine results
    combined = []
    seen = set()
    
    # Add vector results first
    for doc in vector_results:
        if doc["_id"] not in seen:
            combined.append({
                "text": doc["text"],
                "score": doc.get("vectorSearchScore", 0),
                "type": "vector"
            })
            seen.add(doc["_id"])
    
    # Add text results
    for doc in text_results:
        if doc["_id"] not in seen:
            combined.append({
                "text": doc["text"],
                "score": doc.get("score", 0),
                "type": "text"
            })
            seen.add(doc["_id"])
    
    # Return sorted results
    return sorted(combined, key=lambda x: x["score"], reverse=True)[:top_k]


def safe_json_extract(response: str):
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
    

def generate(description):

    context = " ".join([res["text"] for res in search_documents(description)])

    prompt = '''
        Analyze this chemical compound description with provided context and return structural data in JSON format. Whatever be the strength of data
        show full representation of the compound in 3D space with all the atoms and bonds. Get all the data do not skip any element.
        Think over the resources and find correct data.
        Follow this EXACT structure:
        {
        "name": "IUPAC name",
        "properties": "Brief chemical description",
        "description": "Detailed description of the compound how its synthesized and what are its uses?",
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
        }
        }

        Important rules:
        1. Give unique IDs to atoms (e.g., C1, C2, O1, H1, H2)
        2. Positional co-ordinates be in range 0 to 0.75
        4. List all relevant bonds and bond angles
        5. Add a clear chemical description
        6. Include all the relevant functional groups
        8. Show all the elements and their positions
        9. Do not truncate any data
        10. Do not return anything other than JSON

        Provided desription:
    '''

    answer = ""

    client = genai.Client(
        api_key = "AIzaSyAb4TTvJNOcSeZe4BgwvUrBgUQeAoYvNXI",
    )

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt+description+" Provided context: "+context),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1.5,
        top_p=0.95,
        top_k=40,
        max_output_tokens=16834,
        response_mime_type="application/json",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        answer += chunk.text + ""

    return answer


if __name__ == "__main__":
    # Create search indexes
    create_indexes()
    
    # # Process and index PDF
    # pdf_path = "Quantum Chemistry (Z-Library).pdf"
    # process_and_index_pdf(pdf_path)
