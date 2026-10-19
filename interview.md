# ClinicalRAG — Interview Guide (Architecture, System Design, Hard Questions)

Use this with your Emids resume bullets. Prefer **Emids / clinical CDS platform** — do not name third-party product brands in interviews unless the recruiter already has that context and NDA allows it.

---

## Your resume bullets (anchor answers here)

1. **Context:** AI/ML clinical decision support (CDS) integrating predictive risk models with GraphRAG knowledge graphs for EHR data.
2. **ML + rules:** Python / Pandas / Scikit-learn risk scoring and CDI workflows; classical ML + deterministic rule engines for DRG/coding accuracy.
3. **GraphRAG:** Patient-scoped Neo4j + LangGraph pipeline; grounded, citeable clinical summarization for care providers and payers; data isolation.

---

## 60-second elevator pitch

> I built a clinical CDS platform that separates two jobs. Structured risk, CDI flags, and DRG/coding hints come from classical ML plus deterministic clinical rules — they’re auditable scores. Free-text EHR and multi-resource dependencies need citeable language, so I added a patient-scoped GraphRAG layer on Neo4j with a LangGraph workflow: guard input → enrich with risk/CDI/DRG → retrieve only that patient’s evidence → generate provider or payer summaries → guard output. Prediction stays ML; narration stays retrieval-grounded with citations and patient isolation.

---

## Architecture cheatsheet

```text
Devices / EMR
    → Kafka (raw events)
    → FHIR processing → PostgreSQL (FHIR R4)
                      → Neo4j (patient graph + note chunks)
    → CDS ML engine (risk, CDI, DRG, concurrent priority)
    → LangGraph GraphRAG (patient-scoped retrieve → summarize)
    → FastAPI (secure FHIR + summarize + analyze)
```

| Layer | Tech | Job |
|-------|------|-----|
| Ingest | Kafka, FHIR converter | Clean high-volume device/EMR → FHIR |
| Store | Postgres + Neo4j | Source of truth + graph/chunks |
| Predict | Sklearn GBM + rule lexicons | Risk, CDI flags, ICD candidates, DRG |
| Retrieve | Patient-filtered BM25 + graph neighborhood | Evidence for that patient only |
| Generate | LangGraph + LLM or template fallback | Clinician vs insurance narrative |
| Secure | JWT / bearer, patient_id scoping | Isolation + access control |

**Key modules**

- `clinical_rag.ml.risk_model` — `GradientBoostingClassifier`, feature extraction from vitals/conditions
- `clinical_rag.iodineiq.engine` — cognitive coding lexicon, CDI flags, DRG map, priority score, interact Q&A
- `clinical_rag.retrieval.hybrid` — patient-scoped chunk fetch + BM25
- `clinical_rag.agent.graph` — LangGraph: `guard_input → enrich → retrieve → generate → guard_output`
- `clinical_rag.summaries.service` — patient vs insurance orchestration

---

## A. Architecture & system design questions

### A1. Walk me through the system end-to-end.

**Answer:** Raw device metrics and EMR note chunks land on Kafka. A worker cleans and maps to FHIR R4, persists resources in Postgres, and indexes note text into Neo4j under the patient node. The CDS engine loads the patient’s FHIR corpus, runs risk scoring (sklearn), rule-based ICD candidates, CDI documentation gap flags, and DRG forecast, then computes a concurrent priority score. For summarization, LangGraph runs patient-scoped retrieval (chunks + graph neighborhood + CDS outputs) and generates either a clinician summary or a payer claim narrative with citations. FastAPI exposes secure FHIR reads and the summarize/analyze APIs.

### A2. Why both Postgres and Neo4j?

**Answer:** Postgres holds standards-compliant FHIR resources for API, audit, and interoperability (`Patient`, `Observation`, `Condition`, `DocumentReference`). Neo4j models relationships (patient–encounter–condition–code–chunk) and supports neighborhood queries for GraphRAG. Relational store = system of record; graph = retrieval substrate for multi-hop clinical context.

### A3. Why GraphRAG instead of a pure vector store?

**Answer:** EHR is relational and multi-source (vitals, conditions, notes). A flat vector index can return similar text from the wrong patient or miss structured links. GraphRAG keeps **patient as the root**, links codes and chunks, and retrieves neighborhood + text. Vectors can still help similarity *inside* a patient scope; the graph enforces isolation and dependency modeling.

### A4. Why not predict everything with an LLM?

**Answer:** Risk and DRG need calibrated, auditable scores for clinical and revenue workflows. LLMs hallucinate codes and vitals. We use sklearn + deterministic coding/DRG rules for scores/flags, and LLMs only for grounded summarization over retrieved evidence. Different SLAs, eval methods, and failure modes.

### A5. Explain the LangGraph workflow. Is it “multi-agent”?

**Answer:** It is a **stateful multi-stage agentic workflow**, not a swarm of autonomous agents. Nodes: input guardrails → CDS enrichment → patient-scoped retrieve → role-specific generate (provider vs payer prompts) → output guardrails. Separation of concerns makes each stage testable. Call it multi-agent only when describing distinct roles (retriever vs writer vs guard); don’t claim independent agents negotiating.

### A6. How do you ensure patient data isolation?

**Answer:** Retrieval always takes `patient_id` and fetches only that patient’s chunks/neighborhood (`fetch_patient_chunks`). Summaries never run a global corpus search. Auth is JWT/bearer; production should bind user roles to patient panels and audit every FHIR read. Graph edges are patient-rooted so ANN/BM25 cannot leak across patients if filters are mandatory pre-query.

### A7. How does CDI + DRG accuracy work without claiming a certified grouper?

**Answer:** Dual path: (1) **classical ML** risk from vitals/comorbidity features; (2) **deterministic rules** — ICD lexicon cue matching, CDI gap heuristics (e.g., HF type unspecified), DRG map from code sets with relative weights. Production would plug certified groupers and enterprise terminologies; this architecture shows the integration pattern and CDS loop.

### A8. Provider summary vs payer summary — why two paths?

**Answer:** Different audiences and risk. Clinician summary: presentation, findings, risk, next actions. Insurance narrative: principal/secondary diagnoses, medical necessity, DRG rationale, CDI open items. Separate system prompts and output schemas reduce compliance mistakes (wrong tone, speculative language on claims).

### A9. How does Kafka fit capacity / interventions?

**Answer:** Continuous device/EMR streams update the chart. Risk + CDI severity feed `priority_score` so concurrent review queues focus scarce CDI/clinical time — that is the operational capacity lever (who to review next), not a full bed-occupancy optimizer unless explicitly scoped.

### A10. Failure modes and resilience?

**Answer:** Kafka down → sync ingest or in-memory bus. Neo4j down → in-memory chunk fallback. No LLM key → template summaries built from retrieved snippets + CDS outputs (still evidence-tied). Prefer degraded truthful summary over invented fluent text.

### A11. How would you scale to millions of patients?

**Answer:** Partition Kafka by facility; shard Postgres by tenant; Neo4j or graph service with patient-keyed partitioning; retrieval always patient-first then top-k; cache embeddings per patient version; async summarization with queues; horizontal API on ECS/EKS; rate-limit LLM; store citations/audit asynchronously.

### A12. Sketch AWS production (high level).

**Answer:** Private VPC: API on ECS/EKS behind ALB, MSK/Kafka, RDS Postgres, Neo4j on EC2/Aura or OpenSearch+graph pattern, Secrets Manager, KMS at rest, Cognito/OIDC, CloudWatch + Prometheus metrics, no PHI in logs. See `docs/AWS_DEPLOYMENT.md`.

---

## B. RAG / GraphRAG deep dive

### B1. What is retrieved today?

**Answer:** Patient-scoped note chunks. Hybrid mode: (1) BM25 sparse ranking over Neo4j (or pgvector) texts, (2) **pgvector** cosine ANN over persisted embeddings in Postgres (`chunk_embeddings`, filtered by `patient_id`), (3) RRF fuse to top-k. Embeddings are written at ingest time (OpenAI `text-embedding-3-small` with `dimensions=384` if keyed, else local hashing). If pgvector is unavailable (SQLite / extension down), dense falls back to in-memory cosine. Graph neighborhood + CDS outputs are also injected into the prompt.

### B2. How does hybrid RRF + pgvector work here?

**Answer:** (1) Filter `patient_id`. (2) BM25 top-pool. (3) Embed query → `ORDER BY embedding <=> query` on `chunk_embeddings WHERE patient_id=…`. (4) RRF over chunk_ids. (5) Citations from fused top-k. Next levers: cross-encoder rerank, FTS in Postgres alongside BM25.

### B3. Chunking strategy for clinical notes?

**Answer:** Prefer section-aware or FHIR DocumentReference chunks (HPI, Assessment, Plan) with overlap; keep metadata: patient_id, encounter_id, source, timestamp. Avoid giant single-note chunks that drown BM25/embeddings.

### B4. How do you fight hallucination?

**Answer:** Cite-or-refuse: after retrieve, `check_evidence` refuses if patient-scoped chunks < `GROUNDING_MIN_CHUNKS`. Generate attaches citations; `check_grounding` requires citations, blocks answers with no lexical overlap to evidence, and suppresses speculative/SSN leakage. Eval via `scripts/eval_rag.py` + `data/eval/rag_golden.json`.

### B5. How do you evaluate GraphRAG?

**Answer:** Golden set of (patient, question, must-cite facts). Metrics: context recall, citation precision, faithfulness/groundedness, clinical error rate (human), latency, cost/token. Separate eval for ML (AUC, calibration) vs RAG (groundedness).

### B6. GraphRAG vs Naive RAG tradeoffs?

| | Naive vector RAG | This GraphRAG |
|--|------------------|---------------|
| Isolation | Easy to miss filter | Patient-rooted graph |
| Multi-hop (condition↔code↔note) | Weak | Neighborhood query |
| Ops | Simpler | Graph + text ops |
| Similarity phrasing | Strong with dense | Add dense inside patient scope |

---

## C. ML / CDI / DRG questions

### C1. Risk model features and label?

**Answer:** Features: age, HR mean, SpO2 min, systolic BP, temp max, RR, condition count, LOS, prior admissions, abnormal labs. Label: deterioration (demo uses synthetic labels for bootstrap training). Classifier: GradientBoosting in a sklearn Pipeline with StandardScaler. Production: real labels from outcomes, time-aware splits, calibration.

### C2. Why Gradient Boosting, not a neural net?

**Answer:** Tabular EHR features, strong baselines, interpretability via feature importance, low latency CPU inference, easier monitoring/drift. NNs reserved for imaging (other experience) or embeddings, not for this risk head unless volume/feature complexity justifies it.

### C3. What is a CDI flag in this system?

**Answer:** Documentation integrity heuristic — e.g., pneumonia without organism specificity, sepsis without organ dysfunction linkage, HF without HFrEF/HFpEF. Emits severity + suggested physician query. Feeds priority bump for concurrent review.

### C4. How is DRG forecasted?

**Answer:** Map overlapping ICD candidates to a small DRG table with relative weights and probabilities. Honest framing: architecture demo of Forecast-style automation; production uses certified groupers and CC/MCC logic.

### C5. Concurrent priority score?

**Answer:** `priority = clip(risk_score + severity_weighted_CDI_bumps)`. Ranks charts for live CDI/clinical review — operational capacity of the review team.

### C6. Train vs serve for risk model?

**Answer:** `train_and_persist` synthesizes/fits pipeline, joblib dump. Serve loads bundle, `extract_features` from FHIR Observations/Conditions, `predict_proba` → tier + intervention hints. Endpoint: `/v1/iodineiq/train-risk-model` (demo).

---

## D. Security / healthcare compliance

### D1. How is this HIPAA-aware (design, not certification claim)?

**Answer:** Patient-scoped retrieval; auth on APIs; minimize PHI in logs; encrypt at rest/in transit in prod; BAA with LLM/cloud vendors; audit FHIR access; prefer abstractive summaries that don’t echo SSN-like patterns (output guards). Don’t claim “HIPAA certified” for a demo — say “designed for HIPAA controls.”

### D2. LLM + PHI?

**Answer:** Prefer VPC/Bedrock/private endpoints; strip identifiers when possible; contractual BAA; never use PHI for proprietary model training without legal approval; log prompt IDs not raw notes.

### D3. Multi-tenant isolation?

**Answer:** `patient_id` (and future `tenant_id`) on every chunk and query filter; separate encryption keys per tenant if required; no shared prompt cache across patients for free-text.

---

## E. Hard interview questions (expect pushback)

### E1. “Your resume says GraphRAG — show me hybrid retrieval.”

**Answer:** `HybridRetriever`: patient filter → BM25 + **pgvector ANN** → RRF. Chunks indexed at ingest into Neo4j (graph) and `chunk_embeddings` (Postgres/pgvector). Embeddings use OpenAI (dim=384) or local hashing. Metadata exposes `dense_backend=pgvector`. Contrast with Consero document vector RAG; here EHR GraphRAG keeps Neo4j for relationships and pgvector for dense search beside FHIR.

### E2. “Isn’t calling this multi-agent marketing?”

**Answer:** Agree on terms: LangGraph **orchestrated workflow** with specialized stages and prompts. Multi-agent is justified when roles diverge (CDS enricher, retriever, clinician writer, payer writer, guard). Not AutoGPT-style agents.

### E3. “Why RAG if ML already predicts risk and DRG?”

**Answer:** Scores don’t produce citeable chart narratives or payer medical-necessity language. Hallucinated supporting text is a clinical and claim risk. RAG attaches evidence; ML attaches probabilities.

### E4. “How do you know DRG/coding is accurate?”

**Answer:** Demo lexicon/rules ≠ certified coding. Accuracy story in production: coder review loop, CDI query resolution rates, agreement vs final billed codes, grouper parity tests. Architecture separates deterministic coding layer so it can be swapped for enterprise engines.

### E5. “Synthetic training data — isn’t that invalid?”

**Answer:** Used to bootstrap pipeline and serving path for portfolio/demo. Production needs retrospectively labeled encounters, proper leakage control (no future vitals), and monitoring. Call out the difference confidently.

### E6. “Latency budget for summarization?”

**Answer:** Target: retrieve <100–200ms patient-scoped; LLM 1–3s; total <5s interactive. Async for batch payer packets. Cap top-k and tokens; cache CDS analyze per encounter version.

### E7. “How do you prevent prompt injection on clinical chat?”

**Answer:** Input guardrails (block jailbreak patterns); never execute tool calls from note text; retrieve-only tools scoped to patient; strip instructions inside documents; privilege system prompt; output filters.

### E8. “Graph DB vs relational joins for RAG context?”

**Answer:** Joins work for fixed queries; graphs excel at variable-depth clinical neighborhoods and evolving edge types (HAS_CODE, HAS_CHUNK). We still keep FHIR in Postgres for compliance and FHIR API fidelity.

### E9. “What would you add before calling this production?”

**Answer:** Dense+BM25 hybrid; reranker; golden-set eval; IdP + audit; PHI-safe logging; certified grouper integration; grounding check; red-team prompts; canary deploy on AWS private subnets.

### E10. “Conflict: classical ML features say hypoxia, notes say otherwise — who wins?”

**Answer:** Structured Observations win for risk score; notes win for narrative only if retrieved. Surface conflict explicitly in summary (“vitals indicate X; note language Y”) — never silently average. That’s a CDS safety feature.

### E11. “Cost control for LLM?”

**Answer:** Template fallback; summarize only on chart change; cache by content hash; smaller model for draft + large for final; retrieve fewer better chunks; structured outputs to reduce retries.

### E12. “How is this different from a chatbot over EHR?”

**Answer:** Chatbots optimize conversation. This optimizes CDS decisions: scored risk/CDI/DRG plus **constrained** generation for two regulated audiences with citations and isolation. Workflow is deterministic stages, not open chat.

---

## F. Whiteboard exercises (practice aloud)

1. Draw CDS vs GraphRAG boundary and data contracts (`patient_id`, encounter, citations).
2. Sequence diagram: `POST /v1/summaries/insurance` → LangGraph → response JSON.
3. Design patient-safe hybrid retrieval (filter → sparse → dense → RRF → rerank).
4. Design eval harness (ML metrics vs RAG faithfulness).
5. Multi-tenant AWS network diagram (public ALB, private data plane).

---

## G. API surface to memorize

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/ingest` | Event → FHIR (+ optional Kafka) |
| `POST /v1/iodineiq/analyze` | Risk, codes, CDI, DRG, priority |
| `POST /v1/iodineiq/interact` | Physician-style Q&A over CDS outputs |
| `POST /v1/summaries/patient` | GraphRAG clinician summary |
| `POST /v1/summaries/insurance` | GraphRAG payer narrative |
| `GET /fhir/...` | Secure FHIR read/search/`$everything` |

Auth: `Authorization: Bearer <jwt|demo-token>`.

---

## H. Honest limitations (say these proactively — looks senior)

- Hybrid BM25 + **pgvector** dense RRF is implemented patient-scoped; optional next step is cross-encoder rerank.
- Coding/DRG/CDI rules are demonstrative lexicons, not certified groupers.
- Risk model bootstraps with synthetic labels in the open demo.
- Guardrails are regex/policy starters, not a full safety stack.
- Not a medical device; clinical validation required before real PHI.

---

## I. Cross-resume consistency (Consero vs Emids)

| Topic | Emids / ClinicalRAG | Consero (finance) |
|-------|---------------------|-------------------|
| Corpus | FHIR EHR + graph | Legal/tax documents |
| Store | Neo4j GraphRAG + Postgres | Vector DB indexing |
| Output | Clinical + payer summaries | Compliance / tax workflows |
| Models | Sklearn risk + CDI/DRG rules | Agent orchestration + RAG |

If asked about ChromaDB: Consero vector RAG. Emids: Neo4j GraphRAG. Don’t conflate.

---

## J. Quick quiz (self-test)

1. Name the five LangGraph nodes in order.  
2. What three inputs enter the generate prompt?  
3. How is `priority_score` computed?  
4. Why patient filter before ANN?  
5. When do you refuse to summarize?  
6. Postgres vs Neo4j responsibility split?  
7. One metric for ML vs one for RAG?  
8. Provider vs payer prompt difference?

*(Answers: 1 guard→enrich→retrieve→generate→guard_out; 2 chunks, graph facts, CDS outputs; 3 risk + CDI severity bumps; 4 prevent cross-patient leak; 5 empty/blocked evidence or guardrail fail; 6 FHIR SoR vs graph retrieval; 7 AUC/calibration vs faithfulness/citation precision; 8 clinical actions vs medical necessity/coding narrative.)*
