# Production checklist — ClinicalRAG (hospital / Emids-style)

Use this before treating the stack as more than a portfolio/demo CDS. **AWS deploy is deferred**; items below are product, safety, and quality work that should land first.

Status legend: `[x]` present in demo · `[ ]` not done · `[~]` partial / demo-only

---

## Current baseline (demo)

- [x] Kafka → FHIR cleaning → Postgres FHIR store  
- [x] Neo4j GraphRAG chunks + neighborhood  
- [x] Hybrid BM25 + **pgvector** dense + RRF (patient-scoped)  
- [x] Sklearn risk + CDI/DRG rule lexicon  
- [x] LangGraph: guard → enrich → retrieve → generate → guard  
- [x] JWT / demo bearer auth  
- [x] Docker Compose (Postgres/pgvector, Neo4j, …)  
- [x] Grounding: empty-retrieve refuse + citation requirement *(implemented)*  
- [x] Tiny RAG eval harness (`scripts/eval_rag.py`) *(implemented)*  

---

## P0 — Must implement (pre-production)

### 1. Security & PHI
- [ ] Replace demo token with enterprise IdP (OIDC/SAML)  
- [ ] RBAC (clinician / coder / auditor) + patient-panel scoping  
- [ ] Audit log every FHIR read / summarize / analyze (who, what, when)  
- [ ] Never log free-text clinical notes; secret manager for keys  
- [ ] Encryption in transit (TLS) and at rest; BAA with LLM/embed vendors  

### 2. Grounding & safety
- [x] Refuse summarization when retrieval is empty / insufficient  
- [x] Require citations with generated summaries  
- [ ] Claim-level entailment / NLI check against evidence  
- [ ] Prompt-injection defenses on note text (treat docs as untrusted)  
- [ ] Clear UX: CDS suggestion ≠ clinical order; human sign-off  

### 3. RAG evaluation
- [x] Golden set + offline harness (retrieval term recall, citations, refuse)  
- [ ] Expand to 50–200 labeled cases across specialties  
- [ ] Faithfulness / groundedness scoring (LLM-as-judge or NLI)  
- [ ] Latency and $/query budgets in CI  

### 4. Classical ML (risk)
- [ ] Replace synthetic labels with site outcomes; time-aware splits  
- [ ] Calibration, subgroup fairness, drift monitoring  
- [ ] Model card + clinical owner sign-off  

### 5. CDI / coding / DRG
- [ ] Swap demo lexicon for enterprise terminology services  
- [ ] Certified grouper (or vendor API); human-in-loop CDI accept/reject  
- [ ] Measure agreement vs final billed codes  

### 6. Retrieval quality
- [ ] Section-aware chunking (HPI / Assessment / Plan / meds)  
- [ ] Cross-encoder / vendor rerank (top-20 → top-5)  
- [ ] Prod embeddings only (no hashing in `APP_ENV=production`)  
- [ ] Reindex job when embedding model/dim changes  

### 7. Workflow & data ops
- [ ] Idempotent ingest; chart versioning; Kafka replay  
- [ ] Multi-tenant isolation tests (`patient_id` / `tenant_id`)  
- [ ] Backup/restore for Postgres+pgvector and Neo4j  

### 8. Observability
- [ ] End-to-end trace_id (ingest → retrieve → LLM)  
- [ ] Alerts: empty retrieve rate, guardrail refuses, p95 latency  
- [ ] Prompt/response retention policy (PHI-aware)  

---

## P1 — Update existing components

| Component | Demo today | Production update |
|-----------|------------|-------------------|
| Auth | `demo-token` / HMAC JWT | IdP + fine-grained scopes |
| Guardrails | Regex + grounding refuse | Policy engine + claim checks |
| Embeddings | OpenAI or local hashing | Private/BAA neural only in prod |
| DRG/CDI rules | Small lexicon | Certified engines + review UI |
| LLM fallback templates | Always-on degrade | Mark `degraded=true`; limit use |
| Logging | structlog JSON | PHI redaction + audit sink |

---

## P2 — Nice-to-have (still pre- or with AWS)

- [ ] Streaming summaries  
- [ ] Cost/token budgets and cache-by-content-hash  
- [ ] Active learning from CDI accept/reject  
- [ ] Red-team prompt corpus in CI  

---

## Deferred — AWS (later)

- [ ] ECS/EKS, ALB, private subnets, WAF  
- [ ] RDS Postgres (+ pgvector), managed Neo4j / EC2, MSK  
- [ ] Secrets Manager, KMS, CloudWatch / OpenTelemetry  
- [ ] Multi-AZ, DR runbooks, Terraform prod modules  

---

## Suggested implementation order (this repo)

1. ~~Grounding cite-or-refuse~~  
2. ~~Tiny eval harness~~  
3. Section chunking + rerank  
4. Prod embedding profile (forbid hashing)  
5. Audit logging + IdP stub  
6. Grouper/terminology integration boundary  
7. Then AWS  

---

## How to run current safety/eval pieces

```bash
# After seed_demo.py and API deps are available:
python scripts/seed_demo.py
python scripts/eval_rag.py
```

Env knobs (see `.env.example`):

- `GROUNDING_REQUIRE_RETRIEVAL=true`
- `GROUNDING_MIN_CHUNKS=1`
- `GROUNDING_REQUIRE_CITATIONS=true`

---

## Disclaimer

Not a medical device. Hospital use requires clinical validation, legal/regulatory review, and a BAA covering all subprocessors that touch PHI.
