from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import google.generativeai as genai
import chromadb
import json
import os

# ==================================================
# CONFIG
# ==================================================

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

app = FastAPI()

# ==================================================
# CHROMADB
# ==================================================

client_db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client_db.get_collection(
    "historical_cases"
)

# ==================================================
# MODELS
# ==================================================

class ShipmentRequest(BaseModel):
    shipment_id: str


class InvestigationRequest(BaseModel):
    session_id: str
    message: str

# ==================================================
# HELPERS
# ==================================================

def get_shipment_details(shipment_id):

    with open(
        "data/shipment.json",
        "r"
    ) as file:

        shipments = json.load(file)

    for shipment in shipments:

        if shipment["shipment_id"] == shipment_id:
            return shipment

    return None


def find_similar_cases(query):

    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    return results


def load_conversations():

    path = "data/conversations.json"

    if not os.path.exists(path):

        with open(path, "w") as f:
            json.dump({}, f)

    with open(path, "r") as file:
        return json.load(file)


def save_conversations(conversations):

    with open(
        "data/conversations.json",
        "w"
    ) as file:

        json.dump(
            conversations,
            file,
            indent=4
        )

# ==================================================
# ROUTES
# ==================================================

@app.get("/")
def home():

    return {
        "message":
        "Logistics Exception Resolution Assistant"
    }

# ==================================================
# PHASE 3
# RAG ANALYSIS
# ==================================================

@app.post("/analyze")
def analyze(request: ShipmentRequest):

    shipment = get_shipment_details(
        request.shipment_id
    )

    if shipment is None:

        return {
            "error":
            "Shipment not found"
        }

    query = (
        shipment["customer_complaint"]
        + " "
        + shipment["email_content"]
    )

    similar_cases = find_similar_cases(
        query
    )

    retrieved_docs = (
        similar_cases["documents"][0]
    )

    retrieved_meta = (
        similar_cases["metadatas"][0]
    )

    historical_context = ""

    for doc, meta in zip(
        retrieved_docs,
        retrieved_meta
    ):

        historical_context += f"""
Historical Case:
{doc}

Resolution:
{meta['resolution']}
"""

    prompt = f"""
You are a logistics exception analyst.

Current Shipment:

Shipment ID:
{shipment["shipment_id"]}

Tracking History:
{", ".join(shipment["tracking_history"])}

Email Content:
{shipment["email_content"]}

Customer Complaint:
{shipment["customer_complaint"]}

Relevant Historical Incidents:

{historical_context}

Provide:

1. Root Cause
2. Confidence Score
3. Evidence
4. Recommended Actions
"""

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    response = model.generate_content(
        prompt
    )

    return {
        "shipment_id":
        shipment["shipment_id"],

        "similar_cases":
        retrieved_docs,

        "analysis":
        response.text
    }

# ==================================================
# PHASE 4
# INVESTIGATION CHAT
# ==================================================

@app.post("/investigate")
def investigate(
    request: InvestigationRequest
):

    conversations = load_conversations()

    history = conversations.get(
        request.session_id,
        []
    )

    history.append(
        {
            "role": "user",
            "content": request.message
        }
    )

    conversation_text = ""

    for msg in history:

        conversation_text += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    prompt = f"""
You are a logistics investigation assistant.

Your responsibilities:

- Investigate shipment issues
- Ask follow-up questions
- Gather evidence
- Determine root cause
- Recommend resolution actions

Conversation:

{conversation_text}

Rules:

1. If information is insufficient,
   ask ONE follow-up question.

2. If enough information exists,
   provide:

   - Root Cause
   - Confidence Score
   - Resolution Plan
"""

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    response = model.generate_content(
        prompt
    )

    history.append(
        {
            "role": "assistant",
            "content": response.text
        }
    )

    conversations[
        request.session_id
    ] = history

    save_conversations(
        conversations
    )

    return {
        "session_id":
        request.session_id,

        "response":
        response.text
    }