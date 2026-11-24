# Q3 Software Spend Audit

## 1. Spend Summary
- **Total Q3 spend**: $68,700 across 6 invoices.
- **By department**:
  - Engineering: $32,100 (`INV-301` $12,600 + `INV-302` $19,500)
  - Infrastructure: $15,000 (`INV-305` $15,000)
  - Marketing: $10,500 (`INV-304` $10,500)
  - Product: $8,400 (`INV-303` $8,400)
  - Operations: $2,700 (`INV-306` $2,700)

## 2. Policy Violations (After Pre-Approved Exceptions)
- **`INV-302` — VectorStore Cloud (Engineering)**: $19,500 Q3 spend exceeds the $10,000 threshold without VP approval (`vp_approved=no`, `preapproved_exception=no`).
- **`INV-304` — AdPulse Analytics (Marketing)**: $10,500 Q3 spend exceeds the $10,000 threshold without VP approval (`vp_approved=no`, `preapproved_exception=no`).
- *Excluded by policy*: `INV-305` (DataVault Backup, Infrastructure, $15,000) carries `preapproved_exception=yes` under the Annual Continuity Waiver and is therefore compliant; `INV-301` has VP approval; `INV-303` ($8,400) and `INV-306` ($2,700) are <= $10,000.

## 3. Net Non-Compliant Total
- **Net non-compliant spend**: **$30,000** (`INV-302` $19,500 + `INV-304` $10,500).

## 4. Remediation Actions
- **Engineering (`INV-302`, VectorStore Cloud)**: Submit retroactive VP sign-off within 5 business days or downsize cluster tier before Q4 billing.
- **Marketing (`INV-304`, AdPulse Analytics)**: Obtain retroactive VP approval or reduce monthly spend below $3,333 so Q4 total stays at or below $10,000.
