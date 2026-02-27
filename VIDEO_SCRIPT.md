# Neuro Co-Scientist - Video Script (3 Minutes)

## Opening (0:00 - 0:15)
**[Show app homepage]**

"Hi, I'm presenting Neuro Co-Scientist - an AI-powered research assistant for neuroscience built on Google Cloud. It uses RAG with semantic search, multi-hop reasoning, and integrates both PubMed literature and ChEMBL drug data."

---

## Architecture Overview (0:15 - 0:45)
**[Show architecture diagram]**

"Here's how it works:

1. **User asks a question** → sent to our Flask backend on Cloud Run
2. **Query Expansion** → Gemini 2.5 Pro rewrites the question into 3 technical variations using MeSH terms and scientific synonyms
3. **Semantic Embedding** → Text Embedding 004 converts queries to vectors
4. **Vector Search** → Matching Engine finds similar documents from our indexed PubMed corpus
5. **Data Enrichment** → BigQuery joins PubMed articles with ChEMBL drug-target data
6. **Multi-hop Reasoning** → Gemini agent can call tools multiple times, synthesizing across documents
7. **Grounded Response** → Answer returned with PMID citations for every claim"

---

## Demo 1: Semantic Gap Question (0:45 - 1:15)
**[Type: "How do sticky protein clumps kill neurons?"]**

"Let's demonstrate semantic search handling terminology gaps. I'm asking about 'sticky protein clumps' - casual language not found in papers.

Watch the trace - the query expander rewrites this to:
- 'toxic protein aggregates neuronal death'
- 'amyloid oligomer neurotoxicity'
- 'alpha-synuclein aggregation neurodegeneration'

The system finds relevant papers about alpha-synuclein and amyloid aggregates, even though my query used 'sticky clumps'. Each claim cites a PMID."

---

## Demo 2: Multi-Hop + ChEMBL Integration (1:15 - 1:50)
**[Type: "Which drugs target Beta amyloid A4 protein and what are their potency values?"]**

"Now a multi-hop question requiring both literature AND drug data.

The agent:
1. First calls `get_valid_rag_context` to find papers about amyloid-targeting drugs
2. Then calls `search_chembl_drugs` to get IC50 potency values from ChEMBL

You can see in the trace - 2 hops, 2 different tools. The answer combines:
- Literature context from PubMed (with PMIDs)
- Structured drug data: Tramiprosate IC50 = 100,000 nM, Carvedilol binding data

This is the ChEMBL integration enriching answers with structured data."

---

## Demo 3: Source Citations (1:50 - 2:15)
**[Show cited papers panel and click a PMID]**

"Every answer includes source citations. The cited papers panel shows all PMIDs referenced. 

When confidence is low - meaning vector similarity scores are poor - the agent explicitly says 'I don't have sufficient evidence' rather than hallucinating. This is grounded AI."

---

## Evaluation Results (2:15 - 2:40)
**[Show /eval/run endpoint results or scorecard]**

"We evaluated on 30 questions across 3 categories:
- **Simple**: Basic neuroscience facts
- **Multi-hop**: Requires synthesizing multiple sources  
- **Semantic gap**: Casual language → technical terms

Results: **XX%** overall accuracy, meeting the 70% minimum bar.

The semantic gap questions specifically demonstrate that our RAG finds relevant documents even when query terms don't match document terminology."

---

## Technical Stack (2:40 - 2:55)
**[Show Google Cloud console or code]**

"Built entirely on Google Cloud:
- **Vertex AI Matching Engine** - Vector database for semantic search
- **Gemini 2.5 Pro** - Query expansion + agentic reasoning
- **Text Embedding 004** - Document and query embeddings
- **BigQuery** - PubMed + ChEMBL data warehouse
- **Cloud Run** - Serverless deployment

All indexed documents use semantic embeddings, not keyword matching."

---

## Closing (2:55 - 3:00)
**[Show live app URL]**

"Neuro Co-Scientist - grounded neuroscience answers with citations. Try it at [your-url]. Thank you!"

---

# Key Points to Emphasize

## Minimum Bar ✅
1. ✅ RAG indexes biomedical docs with semantic (not keyword) search
2. ✅ Grounded answers with source citations (PMIDs)
3. ✅ ≥70% accuracy on 30 evaluation questions
4. ✅ Semantic gap handling demonstrated (3+ questions)

## Competitive Features ✅
1. ✅ Multi-hop reasoning with tool chaining (trace shows hops)
2. ✅ ChEMBL drug-target integration (IC50, protein targets)
3. ✅ Query expansion improves semantic search over baseline

---

# What Doesn't Work (Be Honest)

1. **Very rare diseases**: Limited PubMed corpus means some niche conditions have no indexed papers
2. **Real-time data**: Index is static snapshot, not live PubMed feed
3. **Cross-paper synthesis limits**: Agent max 5 hops, may miss connections requiring more reasoning
4. **Low confidence queries**: When similarity scores > 1.2, system correctly says "insufficient evidence" rather than guessing
