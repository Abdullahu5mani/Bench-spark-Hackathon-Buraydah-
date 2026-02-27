# 🧠 Neuro Co-Scientist

**RAG-Powered Neuroscience Research Assistant on Google Cloud**

Live Demo: https://neuro-co-scientist-900778453257.us-central1.run.app

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                      │
│                   "How do sticky protein clumps kill neurons?"               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOUD RUN (Flask Backend)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌───────────────────────────────┐     ┌───────────────────────────────────────┐
│      QUERY EXPANSION          │     │         TOOL REGISTRY                 │
│      (Gemini 2.5 Pro)         │     │  • get_valid_rag_context              │
│                               │     │  • search_chembl_drugs                │
│  Input: "sticky protein       │     └───────────────────────────────────────┘
│          clumps"              │
│                               │
│  Output:                      │
│  • "toxic protein aggregates" │
│  • "amyloid oligomer          │
│     neurotoxicity"            │
│  • "α-synuclein aggregation"  │
└───────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC EMBEDDING (Text Embedding 004)                   │
│                         Query → 768-dim vector                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  VECTOR SEARCH (Matching Engine)                             │
│                                                                              │
│   • Cosine similarity search (NOT keyword matching)                          │
│   • Returns top-5 similar document IDs                                       │
│   • Confidence score based on similarity distance                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BIGQUERY DATA WAREHOUSE                              │
│                                                                              │
│   ┌─────────────────────────┐    JOIN    ┌────────────────────────────┐     │
│   │  pubmed_neuro_master    │◄──────────►│ neurology_compounds_master │     │
│   │  • pmid                 │            │ • drug_name                │     │
│   │  • title                │            │ • protein_target           │     │
│   │  • article_text         │            │ • potency_ic50             │     │
│   └─────────────────────────┘            │ • uniprot_id               │     │
│                                          └────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEMINI AGENT (Multi-Hop Reasoning)                        │
│                                                                              │
│   Hop 1: get_valid_rag_context("α-synuclein aggregation")                    │
│          → 3 papers retrieved (PMIDs: 12345, 23456, 34567)                   │
│                                                                              │
│   Hop 2: search_chembl_drugs("aggregation inhibitor")                        │
│          → Drug data: Tramiprosate IC50=100000 nM                            │
│                                                                              │
│   Synthesis: Combine all sources with citations                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GROUNDED RESPONSE                                   │
│                                                                              │
│   "Toxic protein aggregates cause neuronal death through several            │
│    mechanisms. Alpha-synuclein oligomers disrupt mitochondrial              │
│    function (PMID: 12345) and trigger neuroinflammation via                 │
│    microglial activation (PMID: 23456)..."                                  │
│                                                                              │
│   📚 Cited Papers: [PMID: 12345] [PMID: 23456] [PMID: 34567]                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Features

### ✅ Minimum Bar

| Requirement | Status |
|-------------|--------|
| RAG indexes biomedical docs with semantic search | ✅ Matching Engine + Text Embedding 004 |
| Grounded answers with source citations | ✅ PMID citations for each claim |
| ≥70% accuracy on 30 evaluation questions | ✅ Run `/eval/run` endpoint |
| Semantic gap handling (3+ questions) | ✅ Query expansion bridges terminology gaps |

### 🏆 Competitive Features

| Feature | Implementation |
|---------|---------------|
| Multi-hop reasoning | Agent chains up to 5 tool calls, trace shows each hop |
| ChEMBL integration | Drug-target data (IC50, protein targets) enriches answers |
| Semantic > keyword search | Query expansion + embeddings outperform exact matching |

---

## Tech Stack

- **Vertex AI Gemini 2.5 Pro** - Query expansion + agentic reasoning
- **Vertex AI Text Embedding 004** - Semantic embeddings
- **Vertex AI Matching Engine** - Vector similarity search
- **BigQuery** - PubMed + ChEMBL data warehouse
- **Cloud Run** - Serverless deployment
- **Flask** - Python backend

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/query` | POST | Ask a question |
| `/eval/questions` | GET | List 30 evaluation questions |
| `/eval/run` | POST | Run full evaluation |
| `/eval/single` | POST | Test single question |
| `/library` | GET | User's saved papers |

---

## Evaluation Questions

30 questions across 3 categories:
- **Simple (10)**: Basic neuroscience facts
- **Multi-hop (10)**: Requires synthesizing multiple sources
- **Semantic Gap (10)**: Casual language → technical terms

Run evaluation:
```bash
curl -X POST https://your-url/eval/run
```

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set GCP credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"

# Run
python app.py
```

---

## Deployment

```bash
gcloud run deploy neuro-co-scientist \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300
```

---

## Limitations

1. **Corpus coverage**: Static snapshot of PubMed neurology articles, not all diseases covered
2. **Real-time updates**: Index is not live-updated
3. **Max 5 hops**: Complex questions requiring more reasoning steps may be incomplete
4. **Low confidence queries**: System correctly reports uncertainty when similarity scores are poor

---

## Team

BenchSpark Hackathon - Buraydah 2026
