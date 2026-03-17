# Dataset Governance

## PII Handling
- Treat CVs and resumes as personal data.
- Strip names/emails/phone numbers where possible before training.
- Keep raw CVs in a restricted folder; only derived features go into training data.

## Retention
- Raw datasets: retain max 12 months unless contractual obligations require less.
- Derived features: retain until model retraining completes + 90 days.
- Delete all raw datasets on request or contract expiration.

## Legal Basis
- Public datasets: verify license (CC BY, MIT, etc.) and store license files.
- Partner datasets: require written permission and documented usage scope.
- Student/academic use: mark data as “non‑commercial”.

## Access & Audit
- Limit raw data access to project owner/admins.
- Track dataset registration with SHA256 and timestamp.
