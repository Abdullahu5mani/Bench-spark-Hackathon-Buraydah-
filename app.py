"""
Neuro Co-Scientist — Local App
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import json
import uuid
import os
import time
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


# ── 30 Evaluation Questions ────────────────────────────────────────────────────
EVAL_QUESTIONS = {
    "simple": [
        {"id": "S1",  "question": "What is the mechanism of action of donepezil in Alzheimer's disease?",
         "expected_concepts": ["acetylcholinesterase", "cholinergic", "AChE"]},
        {"id": "S2",  "question": "What protein aggregates are found in the brains of Parkinson's disease patients?",
         "expected_concepts": ["synuclein", "Lewy bodies"]},
        {"id": "S3",  "question": "What is the role of BACE1 in Alzheimer's disease?",
         "expected_concepts": ["amyloid", "APP", "cleavage"]},
        {"id": "S4",  "question": "What neurotransmitter is deficient in Parkinson's disease?",
         "expected_concepts": ["dopamine", "substantia nigra"]},
        {"id": "S5",  "question": "What is the function of myelin in the nervous system?",
         "expected_concepts": ["conduction", "white matter", "demyelination"]},
        {"id": "S6",  "question": "What is the potency of Tramiprosate against Beta amyloid A4 protein?",
         "expected_concepts": ["IC50", "100000", "amyloid"]},
        {"id": "S7",  "question": "What causes excitotoxicity in neurons?",
         "expected_concepts": ["glutamate", "excitatory", "inhibitory"]},
        {"id": "S8",  "question": "What is the blood-brain barrier and why does it matter for drug delivery?",
         "expected_concepts": ["tight junctions", "endothelial", "CNS"]},
        {"id": "S9",  "question": "How does memantine work in treating Alzheimer's disease?",
         "expected_concepts": ["NMDA", "antagonist", "glutamate"]},
        {"id": "S10", "question": "What is the role of microglia in neuroinflammation?",
         "expected_concepts": ["immune", "cytokines", "activation"]},
    ],
    "multi_hop": [
        {"id": "M1",  "question": "Which drugs target Beta amyloid A4 protein and what are their potency values?",
         "expected_concepts": ["Tramiprosate", "Carvedilol", "IC50", "amyloid"]},
        {"id": "M2",  "question": "How does neuroinflammation contribute to amyloid plaque formation in Alzheimer's disease?",
         "expected_concepts": ["microglia", "amyloid", "neuroinflammation", "lysosom"]},
        {"id": "M3",  "question": "What is the relationship between tau phosphorylation and neurodegeneration?",
         "expected_concepts": ["tau", "phosphorylation", "neurofibrillary", "neurodegeneration"]},
        {"id": "M4",  "question": "How do mitochondrial dysfunction and oxidative stress interact in Parkinson's disease?",
         "expected_concepts": ["mitochondria", "oxidative stress", "dopaminergic", "ROS"]},
        {"id": "M5",  "question": "What signaling pathways are involved in both neuroprotection and neurodegeneration?",
         "expected_concepts": ["mTOR", "oxidative stress", "apoptosis", "neuroinflammation"]},
        {"id": "M6",  "question": "How does the gut-brain axis influence neurological disease?",
         "expected_concepts": ["microbiome", "dysbiosis", "inflammation", "gut"]},
        {"id": "M7",  "question": "What is the connection between ALS and TDP-43 protein aggregation?",
         "expected_concepts": ["TDP-43", "TARDBP", "aggregation", "ALS"]},
        {"id": "M8",  "question": "How do CREB and BDNF signaling interact to support memory formation?",
         "expected_concepts": ["CREB", "BDNF", "memory", "neuroprotection"]},
        {"id": "M9",  "question": "What are the common mechanisms between multiple sclerosis and other demyelinating diseases?",
         "expected_concepts": ["myelin", "autoimmune", "demyelination", "neuroinflammation"]},
        {"id": "M10", "question": "How does Carvedilol relate to both cardiovascular disease and neurodegeneration?",
         "expected_concepts": ["Carvedilol", "amyloid", "cardiovascular", "neuroprotect"]},
    ],
    "semantic_gap": [
        {"id": "SG1",  "question": "How do brain cleaning mechanisms fail in Alzheimer's?",
         "expected_concepts": ["clearance", "tau", "accumulation", "protein"],
         "gap": "'cleaning' → 'autophagy / glymphatic clearance'"},
        {"id": "SG2",  "question": "Why do nerve cells stop talking to each other in neurodegeneration?",
         "expected_concepts": ["synaptic", "aggregation", "dopaminergic"],
         "gap": "'stop talking' → 'synaptic dysfunction / transmission failure'"},
        {"id": "SG3",  "question": "How does brain inflammation speed up memory loss?",
         "expected_concepts": ["neuroinflammation", "hippocampal", "TNF"],
         "gap": "'brain inflammation' → 'neuroinflammation', 'memory loss' → 'cognitive decline'"},
        {"id": "SG4",  "question": "What happens when the brain's protein recycling system breaks down?",
         "expected_concepts": ["proteasome", "aggregation", "lysosom", "misfolded"],
         "gap": "'protein recycling' → 'ubiquitin-proteasome pathway'"},
        {"id": "SG5",  "question": "How do sticky protein clumps kill neurons?",
         "expected_concepts": ["aggregates", "synuclein", "neurotoxic", "neuroinflammation"],
         "gap": "'sticky protein clumps' → 'amyloid oligomers / toxic aggregates'"},
        {"id": "SG6",  "question": "What drugs slow down the progression of the shaking disease?",
         "expected_concepts": ["Parkinson", "dopamine", "dopaminergic"],
         "gap": "'shaking disease' → 'Parkinson's disease / tremor'"},
        {"id": "SG7",  "question": "How does the brain's power supply failure contribute to neuron death?",
         "expected_concepts": ["mitochondrial", "oxidative", "ROS", "energy"],
         "gap": "'power supply failure' → 'mitochondrial dysfunction / bioenergetic failure'"},
        {"id": "SG8",  "question": "What causes the protective coating of nerve fibers to deteriorate?",
         "expected_concepts": ["white matter", "myelin", "degeneration"],
         "gap": "'protective coating' → 'myelin sheath / demyelination'"},
        {"id": "SG9",  "question": "How do brain support cells become harmful during disease?",
         "expected_concepts": ["astrocyte", "microglia", "neuroinflammation", "cytokine"],
         "gap": "'support cells' → 'astrocytes / microglia / glia'"},
        {"id": "SG10", "question": "How does faulty electrical signaling in the brain lead to seizures?",
         "expected_concepts": ["epilepsy", "GABA", "inhibitory", "ion channel"],
         "gap": "'faulty electrical signaling' → 'aberrant neuronal firing / GABAergic inhibition'"},
    ]
}

def _score_answer(answer: str, expected_concepts: list) -> dict:
    answer_lower = answer.lower()
    hits   = [c for c in expected_concepts if c.lower() in answer_lower]
    missed = [c for c in expected_concepts if c.lower() not in answer_lower]
    coverage = len(hits) / len(expected_concepts) if expected_concepts else 0
    return {
        "hits":     hits,
        "missed":   missed,
        "coverage": round(coverage, 2),
        "pass":     coverage >= 0.5
    }

@app.route("/eval/questions", methods=["GET"])
def get_eval_questions():
    """Return all 30 evaluation questions"""
    return jsonify(EVAL_QUESTIONS)

@app.route("/eval/run", methods=["POST"])
def run_evaluation():
    """Run evaluation on all or selected questions"""
    data = request.json or {}
    categories = data.get("categories", list(EVAL_QUESTIONS.keys()))
    delay = data.get("delay", 3)
    
    all_rows = []
    category_summary = {}
    
    for category in categories:
        if category not in EVAL_QUESTIONS:
            continue
        questions = EVAL_QUESTIONS[category]
        cat_passed = 0
        
        for q in questions:
            try:
                result = run_neuro_agent(q["question"])
                answer = result.get("answer", "")
                score = _score_answer(answer, q["expected_concepts"])
                if score["pass"]:
                    cat_passed += 1
                all_rows.append({
                    "id": q["id"],
                    "category": category,
                    "question": q["question"],
                    "pass": score["pass"],
                    "coverage": f"{score['coverage']*100:.0f}%",
                    "hits": score["hits"],
                    "missed": score["missed"],
                    "gap": q.get("gap", ""),
                    "answer_preview": answer[:500] if answer else ""
                })
            except Exception as e:
                all_rows.append({
                    "id": q["id"],
                    "category": category,
                    "question": q["question"],
                    "pass": False,
                    "coverage": "0%",
                    "hits": [],
                    "missed": q["expected_concepts"],
                    "error": str(e)
                })
            time.sleep(delay)
        
        pct = (cat_passed / len(questions) * 100) if questions else 0
        category_summary[category] = {
            "passed": cat_passed,
            "total": len(questions),
            "pct": round(pct, 1)
        }
    
    total_passed = sum(s["passed"] for s in category_summary.values())
    total_all = sum(s["total"] for s in category_summary.values())
    overall_pct = (total_passed / total_all * 100) if total_all else 0
    
    return jsonify({
        "results": all_rows,
        "category_summary": category_summary,
        "overall": {
            "passed": total_passed,
            "total": total_all,
            "pct": round(overall_pct, 1),
            "meets_bar": overall_pct >= 70
        }
    })

@app.route("/eval/single", methods=["POST"])
def run_single_eval():
    """Run evaluation on a single question by ID"""
    data = request.json
    question_id = data.get("id", "").strip()
    
    for category, questions in EVAL_QUESTIONS.items():
        for q in questions:
            if q["id"] == question_id:
                try:
                    result = run_neuro_agent(q["question"])
                    answer = result.get("answer", "")
                    score = _score_answer(answer, q["expected_concepts"])
                    return jsonify({
                        "id": q["id"],
                        "category": category,
                        "question": q["question"],
                        "pass": score["pass"],
                        "coverage": f"{score['coverage']*100:.0f}%",
                        "hits": score["hits"],
                        "missed": score["missed"],
                        "gap": q.get("gap", ""),
                        "answer": answer,
                        "trace": result.get("trace", []),
                        "cited_papers": result.get("cited_papers", [])
                    })
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": f"Question ID '{question_id}' not found"}), 404


if __name__ == "__main__":
    print("\n[Neuro Co-Scientist] Starting...")
    print("   Open: http://localhost:5000\n")
    app.run(debug=False, port=5000)