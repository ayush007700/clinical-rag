# AWS Deployment Guide — ClinicalRAG

Target production pattern:

```text
GitHub Actions → Amazon ECR → ECS Fargate (API + worker)
                         │
         ┌───────────────┼────────────────┐
         ▼               ▼                ▼
   Amazon MSK      Aurora PostgreSQL   Neo4j Aura / EC2
   (Kafka)         (FHIR store)        (GraphRAG)
         │
         ▼
   Secrets Manager / SSM  +  CloudWatch  +  ALB
```

## 1. Local → cloud mapping

| Local Compose | AWS |
|---------------|-----|
| `api` container | ECS Fargate service behind ALB |
| `worker` container | ECS service (same image, different command) |
| Kafka | Amazon MSK or MSK Serverless |
| Postgres | Aurora PostgreSQL |
| Neo4j | Neo4j AuraDS or self-managed on EC2 |
| Prometheus | CloudWatch metrics (+ optional AMP) |
| `.env` secrets | Secrets Manager / SSM SecureString |

## 2. Container image

```bash
aws ecr create-repository --repository-name clinical-rag
docker build -t clinical-rag .
docker tag clinical-rag:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/clinical-rag:latest
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/clinical-rag:latest
```

API command:

```text
uvicorn clinical_rag.api.app:app --host 0.0.0.0 --port 8000
```

Worker command:

```text
python -m clinical_rag.workers.kafka_worker
```

## 3. Terraform

Under `infra/terraform/`:

1. Copy `terraform.tfvars.example` → `terraform.tfvars`
2. Fill `openai_api_key`, Neo4j Aura URI/password, MSK bootstrap if ready
3. `terraform init && terraform plan && terraform apply`
4. Wire GitHub OIDC role ARN into Actions secrets

Outputs include ALB DNS and ECR URL.

### Minimal bootstrap without MSK

You can deploy **API only** first, using synchronous `/v1/ingest` (no Kafka). Add MSK + worker when volume requires it.

## 4. CI/CD

`.github/workflows/ci.yml` runs pytest + ruff.  
`.github/workflows/cd.yml` builds/pushes ECR and forces a new ECS deployment.

## 5. HIPAA-oriented checklist

- [ ] BAA with AWS and LLM vendor
- [ ] Private subnets + NAT for egress only
- [ ] Encryption at rest (RDS, EBS, S3) with CMK
- [ ] No PHI in application logs
- [ ] WAF on ALB; rate limits
- [ ] Audit trail (CloudTrail + app access logs)
- [ ] Clinical validation before any care pathway automation

## 6. Smoke test after deploy

```bash
curl https://<alb-dns>/health
curl -X POST https://<alb-dns>/v1/auth/token -H "Content-Type: application/json" -d "{\"username\":\"ops\"}"
```
