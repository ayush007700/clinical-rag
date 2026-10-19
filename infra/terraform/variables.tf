variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "clinical-rag"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "openai_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

variable "neo4j_uri" {
  type = string
  default = "bolt://localhost:7687"
}

variable "neo4j_password" {
  type      = string
  sensitive = true
  default   = ""
}

variable "database_url" {
  type      = string
  sensitive = true
  default   = ""
}

variable "jwt_secret" {
  type      = string
  sensitive = true
  default   = "replace-me"
}

variable "container_image" {
  type        = string
  description = "ECR image URI (account.dkr.ecr.region.amazonaws.com/clinical-rag:tag)"
  default     = ""
}

variable "desired_count" {
  type    = number
  default = 2
}
