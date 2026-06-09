import json
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="historical_cases"
)

with open("data/historical_cases.json") as f:
    cases = json.load(f)

for case in cases:

    collection.add(
        documents=[
            case["description"]
        ],
        ids=[
            case["id"]
        ],
        metadatas=[
            {
                "resolution":
                    case["resolution"]
            }
        ]
    )

print("Historical cases loaded.")