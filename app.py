"""
Neuro Co-Scientist — Local App
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import json
import uuid
import os
from datetime import datetime
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part, GenerationConfig
from google.cloud import aiplatform, bigquery
from vertexai.language_models import TextEmbeddingModel

app = Flask(__name__)

# ── Library Storage ────────────────────────────────────────────────────────────
LIBRARY_FILE = os.path.join(os.path.dirname(__file__), "library.json")

def _load_library():
    if os.path.exists(LIBRARY_FILE):
        with open(LIBRARY_FILE, "r") as f:
            return json.load(f)
    return {"folders": [], "papers": []}

def _save_library(data):
    with open(LIBRARY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ID        = 'buraydah-1771991853'
LOCATION          = 'us-central1'
ENDPOINT_ID       = '8386487557466095616'
DEPLOYED_INDEX_ID = 'neuro_agent_endpoint_1772132590987'

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)
bq_client         = bigquery.Client(project=PROJECT_ID)
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(ENDPOINT_ID)
_expansion_model  = GenerativeModel("gemini-2.5-pro")

# ── Agent logic (same as notebook) ────────────────────────────────────────────
def expand_query(question):
    prompt = f"""You are a biomedical search expert.
Rewrite this research question into 3 alternative versions using:
- Technical/scientific terminology (MeSH terms, gene names, pathway names)
- Synonyms used in academic papers
- Related biological concepts
Return ONLY a JSON array of 3 strings. No explanation, no markdown.
Question: "{question}"
Example: ["technical version 1", "technical version 2", "technical version 3"]"""
    try:
        response = _expansion_model.generate_content(
            prompt, generation_config=GenerationConfig(temperature=0.2)
        )
        text = response.text.strip().replace("```json","").replace("```","").strip()
        expansions = json.loads(text)
        return [question] + expansions[:3]
    except:
        return [question]

def get_valid_rag_context(question):
    embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    all_queries = expand_query(question)
    seen_ids = {}
    for q in all_queries:
        try:
            qv = embed_model.get_embeddings([q])[0].values
            resp = my_index_endpoint.find_neighbors(
                deployed_index_id=DEPLOYED_INDEX_ID, queries=[qv], num_neighbors=5
            )
            if resp and resp[0]:
                for n in resp[0]:
                    if n.id not in ['0','null',None,'']:
                        if n.id not in seen_ids or n.distance < seen_ids[n.id]:
                            seen_ids[n.id] = n.distance
        except: pass
    if not seen_ids:
        return {"error": "No results found."}
    top_ids    = [k for k,_ in sorted(seen_ids.items(), key=lambda x:x[1])][:5]
    best_score = min(seen_ids.values())
    low_conf   = best_score > 1.2
    sql = """
        SELECT m.pmid, m.title, m.article_text, c.drug_name, c.potency_ic50,
               c.standard_units, c.protein_target
        FROM `buraydah-1771991853.neuro_rag.pubmed_neuro_master` AS m
        LEFT JOIN `buraydah-1771991853.neuro_rag.neurology_compounds_master` AS c
            ON m.pmid = c.pubmed_id
        WHERE m.pmid IN UNNEST(@ids) LIMIT 3
    """
    df = bq_client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("ids","STRING",top_ids)]
    )).to_dataframe()
    if df.empty:
        return {"error": "No records found."}
    docs = []
    for _, row in df.iterrows():
        docs.append({
            "pmid": row["pmid"], "title": str(row["title"] or "Unknown"),
            "article_excerpt": str(row["article_text"])[:15000],
            "drug_name": row["drug_name"], "potency_ic50": row["potency_ic50"],
            "standard_units": row["standard_units"], "protein_target": row["protein_target"],
        })
    return {
        "documents": docs, "num_docs_found": len(docs),
        "best_similarity": round(best_score, 4), "low_confidence": low_conf,
        "confidence_note": (
            "LOW CONFIDENCE: Documents may not be closely relevant." if low_conf
            else "Good confidence: Documents appear relevant."
        )
    }

def search_chembl_drugs(drug_name):
    sql = """
        SELECT drug_name, protein_target, uniprot_id, pubmed_id, potency_ic50, standard_units
        FROM `buraydah-1771991853.neuro_rag.neurology_compounds_master`
        WHERE UPPER(drug_name) LIKE @drug ORDER BY drug_name ASC LIMIT 10
    """
    try:
        df = bq_client.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("drug","STRING",f"%{drug_name.upper()}%")]
        )).to_dataframe()
        if df.empty: return {"error": f"No entries found for '{drug_name}'."}
        return {"results": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

TOOL_REGISTRY = {
    "get_valid_rag_context": get_valid_rag_context,
    "search_chembl_drugs":   search_chembl_drugs,
}

neuro_tools = Tool(function_declarations=[
    FunctionDeclaration(
        name="get_valid_rag_context",
        description="Semantically searches PubMed neurology literature using query expansion. Returns up to 3 relevant papers. Use for mechanism questions, disease concepts, open-ended queries. If low_confidence is true, acknowledge uncertainty.",
        parameters={"type":"object","properties":{"question":{"type":"string","description":"The research question."}},"required":["question"]}
    ),
    FunctionDeclaration(
        name="search_chembl_drugs",
        description="Looks up a specific drug in ChEMBL. Use when the user asks about a drug name, IC50, or binding target.",
        parameters={"type":"object","properties":{"drug_name":{"type":"string","description":"Drug name to search for."}},"required":["drug_name"]}
    ),
])

def run_neuro_agent(user_query, progress_callback=None):
    model = GenerativeModel(
        model_name="gemini-2.5-pro", tools=[neuro_tools],
        system_instruction=(
            "You are a neuroscience research assistant. Answer by calling tools to retrieve real data. "
            "Call multiple tools in sequence. Synthesize ALL documents returned and cite each PMID. "
            "If low_confidence is true or documents don't address the question, say you don't have "
            "sufficient evidence and suggest resources. Never fabricate. Always cite PMIDs."
        )
    )
    chat    = model.start_chat()
    message = user_query
    hops    = 0
    trace   = []
    cited_papers = {}

    while hops < 5:
        response  = chat.send_message(message)
        candidate = response.candidates[0]

        tool_calls = [
            p for p in candidate.content.parts
            if hasattr(p,'function_call') and p.function_call is not None
            and p.function_call.name and p.function_call.name != ""
        ]

        if not tool_calls:
            final_text = "".join(
                p.text for p in candidate.content.parts
                if hasattr(p,'text') and p.text
            )
            idk = any(x in final_text.lower() for x in
                      ["don't have sufficient","insufficient evidence","cannot find","i don't know"])
            return {
                "answer":     final_text,
                "hops":       hops,
                "low_confidence": idk,
                "trace":      trace,
                "cited_papers": list(cited_papers.values())
            }

        tool_results = []
        for p in tool_calls:
            fn_name = p.function_call.name
            fn_args = dict(p.function_call.args)
            result  = TOOL_REGISTRY[fn_name](**fn_args) if fn_name in TOOL_REGISTRY \
                      else {"error": f"Unknown tool: {fn_name}"}
            if fn_name == "get_valid_rag_context" and "documents" in result:
                for doc in result["documents"]:
                    pmid = doc.get("pmid")
                    if pmid and pmid not in cited_papers:
                        cited_papers[pmid] = {"pmid": pmid, "title": doc.get("title", "Untitled")}
            trace.append({"hop": hops+1, "tool": fn_name, "args": fn_args, "result_summary": _summarise(fn_name, result)})
            if progress_callback:
                progress_callback(hops+1, fn_name, fn_args)
            tool_results.append(Part.from_function_response(name=fn_name, response=result))

        message = tool_results
        hops   += 1

    return {"answer": "Agent reached max hops.", "hops": hops, "low_confidence": True, "trace": trace, "cited_papers": list(cited_papers.values())}

def _summarise(fn_name, result):
    if "error" in result:
        return f"Error: {result['error']}"
    if fn_name == "search_chembl_drugs" and "results" in result:
        r = result["results"][0]
        return f"{r.get('drug_name')} → {r.get('protein_target')} IC50={r.get('potency_ic50')} {r.get('standard_units')}"
    if fn_name == "get_valid_rag_context":
        docs = result.get("documents", [])
        pmids = [d["pmid"] for d in docs]
        return f"{len(docs)} docs retrieved: PMIDs {', '.join(pmids)} | conf={result.get('confidence_note','')[:30]}"
    return str(result)[:120]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    data  = request.json
    q     = data.get("question","").strip()
    if not q:
        return jsonify({"error": "No question provided"}), 400
    try:
        result = run_neuro_agent(q)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Library API ────────────────────────────────────────────────────────────────
@app.route("/library", methods=["GET"])
def get_library():
    return jsonify(_load_library())

@app.route("/library/folders", methods=["POST"])
def create_folder():
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Folder name required"}), 400
    lib = _load_library()
    folder = {
        "id": str(uuid.uuid4()),
        "name": name,
        "created_at": datetime.utcnow().isoformat()
    }
    lib["folders"].append(folder)
    _save_library(lib)
    return jsonify(folder), 201

@app.route("/library/folders/<folder_id>", methods=["DELETE"])
def delete_folder(folder_id):
    lib = _load_library()
    lib["folders"] = [f for f in lib["folders"] if f["id"] != folder_id]
    for paper in lib["papers"]:
        if paper.get("folder_id") == folder_id:
            paper["folder_id"] = None
    _save_library(lib)
    return jsonify({"success": True})

@app.route("/library/folders/<folder_id>", methods=["PATCH"])
def update_folder(folder_id):
    data = request.json
    lib = _load_library()
    for folder in lib["folders"]:
        if folder["id"] == folder_id:
            if "name" in data:
                folder["name"] = data["name"]
            _save_library(lib)
            return jsonify(folder)
    return jsonify({"error": "Folder not found"}), 404

@app.route("/library/papers", methods=["POST"])
def save_paper():
    data = request.json
    pmid = data.get("pmid", "").strip()
    title = data.get("title", "").strip()
    if not pmid:
        return jsonify({"error": "PMID required"}), 400
    lib = _load_library()
    for p in lib["papers"]:
        if p["pmid"] == pmid:
            return jsonify({"error": "Paper already in library"}), 409
    paper = {
        "id": str(uuid.uuid4()),
        "pmid": pmid,
        "title": title or "Untitled",
        "folder_id": data.get("folder_id"),
        "saved_at": datetime.utcnow().isoformat(),
        "notes": ""
    }
    lib["papers"].append(paper)
    _save_library(lib)
    return jsonify(paper), 201

@app.route("/library/papers/<paper_id>", methods=["DELETE"])
def delete_paper(paper_id):
    lib = _load_library()
    lib["papers"] = [p for p in lib["papers"] if p["id"] != paper_id]
    _save_library(lib)
    return jsonify({"success": True})

@app.route("/library/papers/<paper_id>", methods=["PATCH"])
def update_paper(paper_id):
    data = request.json
    lib = _load_library()
    for paper in lib["papers"]:
        if paper["id"] == paper_id:
            if "folder_id" in data:
                paper["folder_id"] = data["folder_id"]
            if "notes" in data:
                paper["notes"] = data["notes"]
            _save_library(lib)
            return jsonify(paper)
    return jsonify({"error": "Paper not found"}), 404


if __name__ == "__main__":
    print("\n[Neuro Co-Scientist] Starting...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)