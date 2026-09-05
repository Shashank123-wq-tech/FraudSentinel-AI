# 🛡️ SentinelRisk AI

### Autonomous, Explainable & Adaptive Fraud-Spike Intelligence for Merchant Risk

<p align="center">

**Detect the signal. Verify the spike. Quantify the risk. Explain the decision. Act early.**

</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange?logo=xgboost)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)](https://www.docker.com/)
[![MLOps](https://img.shields.io/badge/MLOps-End--to--End-purple)](#-mlops--deployment)
[![Status](https://img.shields.io/badge/Project-Hackathon%20Ready-success)](#)

</p>

---

## 🚨 The Problem

Traditional fraud detection systems are often optimized around a simple question:

> **"Is this transaction fraudulent?"**

But merchant fraud attacks are rarely isolated events.

A coordinated attack can appear as:

* a sudden increase in suspicious transactions,
* abnormal activity concentrated in a short time interval,
* rapidly increasing fraud probability,
* participation from many cards,
* unusually high financial exposure,
* repeated suspicious activity across consecutive windows.

By the time a traditional transaction-level system recognizes the pattern, the merchant may already have experienced significant financial loss.

### SentinelRisk AI asks a more important question:

> **"Is this merchant entering a fraud attack state, how serious is it, why is it happening, and what should the risk team do next?"**

---

# 🎯 Our Solution

**SentinelRisk AI** is an end-to-end merchant risk intelligence platform designed to detect, verify, explain, prioritize and operationalize emerging fraud spikes.

It combines:

```text
Transaction Intelligence
        +
Temporal Fraud-Spike Intelligence
        +
Explainable AI
        +
Financial Risk Intelligence
        +
Decision Intelligence
        +
Risk Copilot
        +
Real-Time Dashboard
        +
Production MLOps
```

Instead of treating every suspicious transaction independently, SentinelRisk AI converts transaction-level signals into **merchant-level fraud events**.

---

# 🧠 Core Idea

The system works in four levels of intelligence:

| Level                           | Question                                | Output                      |
| ------------------------------- | --------------------------------------- | --------------------------- |
| **1. Transaction Intelligence** | Is this transaction suspicious?         | Fraud probability           |
| **2. Temporal Intelligence**    | Is merchant activity becoming abnormal? | Spike score / TAS           |
| **3. Event Intelligence**       | Is this a meaningful fraud attack?      | Verified fraud event        |
| **4. Decision Intelligence**    | What should the risk team do?           | Risk action + expected loss |

This creates a progression from:

```text
Transaction
    ↓
Behavior
    ↓
Event
    ↓
Decision
```

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │   TRANSACTION STREAM │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │   DATA VALIDATION    │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ ONLINE FEATURE ENGINE │
                         └──────────┬───────────┘
                                    ↓
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
       ┌──────────────────┐                    ┌──────────────────┐
       │ PHASE 1          │                    │ PHASE 2          │
       │ TRANSACTION AI   │                    │ TEMPORAL AI      │
       │                  │                    │                  │
       │ Fraud Probability│                    │ 5m / 15m / 60m   │
       │ Risk Score       │                    │ Baseline         │
       │ Risk Band        │                    │ EWMA / CUSUM     │
       └────────┬─────────┘                    │ Persistence       │
                │                              │ Breadth            │
                │                              │ Acceleration       │
                │                              │ Materiality        │
                │                              └────────┬─────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    ↓
                         ┌──────────────────────┐
                         │ VERIFICATION ENGINE  │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │   EVENT BUILDER      │
                         └──────────┬───────────┘
                                    ↓
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
       ┌──────────────────┐                    ┌──────────────────┐
       │     XAI LAYER    │                    │ DECISION ENGINE  │
       │                  │                    │                  │
       │ Why flagged?     │                    │ Severity         │
       │ What changed?    │                    │ Expected loss    │
       │ Evidence         │                    │ Recommended action│
       └────────┬─────────┘                    └────────┬─────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    ↓
                         ┌──────────────────────┐
                         │   FRAUD EVENT OBJECT │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │     RISK COPILOT     │
                         └──────────┬───────────┘
                                    ↓
                ┌───────────────────┴───────────────────┐
                │                                       │
                ▼                                       ▼
       ┌──────────────────┐                    ┌──────────────────┐
       │   FASTAPI / API   │                    │    DASHBOARD     │
       └────────┬─────────┘                    └────────┬─────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    ↓
                         ┌──────────────────────┐
                         │     MONITORING       │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │       MLOps          │
                         │ CI/CD • Docker • K8s │
                         └──────────────────────┘
```

---

# 🔬 Phase 1 — Transaction Intelligence

### Question

> **How suspicious is this transaction?**

Phase 1 generates a calibrated transaction-level fraud probability and risk score.

### Example output

```json
{
  "transaction_id": "TX-10452",
  "merchant_id": "M-001",
  "timestamp": "2026-09-05T12:10:00Z",
  "amount": 450.00,
  "fraud_probability": 0.982,
  "risk_score": 98.2,
  "risk_band": "HIGH",
  "predicted_fraud": 1,
  "model_version": "fraud_model_v1"
}
```

### Phase 1 responsibilities

* temporal transaction features
* amount behavior
* transaction frequency
* card behavior
* historical spending behavior
* distance behavior
* calibrated fraud probability
* transaction-level evaluation
* business-level fraud capture metrics

---

# 📈 Phase 2 — Temporal Fraud-Spike Intelligence

### Question

> **Is the merchant behaving abnormally over time?**

Phase 2 transforms transaction-level risk into **merchant-level temporal intelligence**.

The detector operates across multiple time scales:

```text
5-minute window   → fast burst detection
15-minute window  → sustained activity
60-minute window  → slow-sprawl detection
```

### Statistical and behavioral signals

SentinelRisk AI combines:

* robust merchant-specific baselines
* EWMA
* CUSUM
* persistence
* acceleration
* transaction breadth
* card breadth
* new-card activity
* expected fraud amount
* financial materiality
* multiscale corroboration

### Temporal reasoning

A spike is not considered meaningful simply because one transaction is suspicious.

Instead:

```text
Abnormality
    +
Persistence
    +
Corroboration
    +
Breadth
    +
Acceleration
    +
Financial Materiality
    ↓
Verified Fraud Spike
```

---

# 🧮 Transaction Expected Fraud Loss

For transaction \(i\):

$$
EFL_i = P(Fraud_i) \times Amount_i
$$

For an event:

$$
EFL_{event} =
\sum_i P(Fraud_i) \times Amount_i
$$

This enables the system to distinguish between:

```text
100 low-value suspicious transactions
```

and:

```text
5 high-value transactions
with very high expected fraud exposure
```

The system can therefore prioritize alerts by **financial impact**, not merely alert count.

---

# ✅ Fraud-Spike Verification

Detection and verification are deliberately separated.

A candidate anomaly must satisfy multiple pieces of evidence before becoming a verified event.

Conceptually:

```text
Candidate
   ↓
Historical confidence
   +
Materiality
   +
Persistence / Corroboration
   +
TAS threshold
   ↓
VERIFIED FRAUD SPIKE
   ↓
Criticality assessment
```

Exceptional high-impact cases can follow a separate single-window path when justified by the configured financial and risk criteria.

---

# 🚨 Event Intelligence

Individual windows are converted into meaningful fraud events.

Instead of showing:

```text
Window 1
Window 2
Window 3
Window 4
```

the system creates:

```text
EVENT EVT-001

Merchant: M001
Severity: CRITICAL
Duration: 30 minutes
Peak: 12:10 UTC

Transactions: 17
Unique cards: 11
New cards: 8

Expected fraud exposure: $630.42
```

This event becomes the central object consumed by:

* XAI
* decision engine
* Copilot
* API
* dashboard
* monitoring
* analyst workflows

---

# 🔍 Explainable AI Layer

SentinelRisk AI is designed to answer:

> **"Why was this flagged?"**

### Transaction explanation

```text
Fraud probability: 98.2%

Primary contributing factors:
• abnormal transaction amount
• elevated recent card activity
• unusual transaction timing
• unusual geographic distance
```

### Spike explanation

```text
Fraud spike detected because:

• fraud intensity increased significantly
• suspicious activity persisted across windows
• multiple cards participated
• new-card participation increased
• financial exposure became material
• multiple time scales corroborated the anomaly
```

The explanation engine operates on actual model and detector outputs rather than allowing an LLM to invent evidence.

---

# ⚖️ Risk Decision Engine

Detection answers:

> **"Is something wrong?"**

Decision intelligence answers:

> **"What should we do?"**

Example policy:

```text
LOW
   ↓
MONITOR

MEDIUM
   ↓
ENHANCED MONITORING

HIGH
   ↓
REVIEW

VERIFIED
   ↓
URGENT REVIEW

CRITICAL + MATERIAL LOSS
   ↓
IMMEDIATE RESPONSE
```

Decision policies are configuration-driven rather than hard-coded into the ML model.

---

# 🤖 Risk Copilot

The Copilot is an **analyst assistant**, not the fraud detector.

The deterministic intelligence layer detects and verifies the event.

The Copilot explains and helps investigate it.

### Example questions

```text
Why was this merchant flagged?

What changed from the normal baseline?

What evidence supports this event?

What transactions contributed most to the risk?

How much fraud exposure is expected?

Why is this event considered critical?

What should I investigate first?
```

### Copilot architecture

```text
Fraud Event Object
       ↓
Context Builder
       ↓
Risk Copilot
       ↓
┌────────────┬────────────┬────────────┐
│ Explain    │ Investigate│ Recommend  │
└────────────┴────────────┴────────────┘
```

The Copilot works from structured system evidence rather than becoming a second hidden fraud detector.

---

# 📊 Risk Dashboard

The dashboard is designed around the questions a risk analyst needs answered immediately.

### Executive view

```text
┌─────────────────────────────────────────────────────────┐
│                  SENTINELRISK AI                        │
├────────────┬────────────┬────────────┬────────────────┤
│ Active     │ Critical   │ Expected   │ Fraud Exposure  │
│ Events     │ Events     │ Loss       │                │
├────────────┴────────────┴────────────┴────────────────┤
│                 Fraud Spike Timeline                    │
├────────────────────────────┬───────────────────────────┤
│ High-Risk Merchants        │ Active Fraud Events       │
├────────────────────────────┴───────────────────────────┤
│ Event Investigation                                     │
│                                                         │
│ Severity │ Evidence │ Financial Impact │ Explanation    │
└─────────────────────────────────────────────────────────┘
```

### Event investigation

The analyst can inspect:

* event timeline
* TAS
* verification score
* component scores
* expected fraud loss
* affected transaction count
* unique cards
* new cards
* spike ratio
* event duration
* explanation
* recommended action

---

# 🚀 Production API

The system will expose its intelligence through FastAPI.

### Core endpoints

```text
POST /v1/transactions/score
POST /v1/transactions/batch-score

GET  /v1/merchants/{merchant_id}/risk

GET  /v1/events
GET  /v1/events/{event_id}
GET  /v1/events/{event_id}/explanation

POST /v1/copilot/chat

GET  /v1/health
GET  /v1/metrics
```

This separates the intelligence layer from the UI and allows the same system to support:

```text
Dashboard
Internal tools
Merchant integrations
Risk operations
Automated workflows
```

---

# 🧪 Validation & Evaluation

SentinelRisk AI uses an independent validation philosophy.

The production detector uses:

```text
Phase 1 fraud probability
+
Phase 2 temporal signals
```

Ground truth is reserved for:

```text
evaluation
validation
backtesting
release gates
```

The validation framework checks:

* transaction-level lineage
* raw fraud labels
* Phase 1 prediction lineage
* probability calibration
* temporal window construction
* persistence
* breadth
* acceleration
* materiality
* event matching
* fraud amount capture
* fraud loss prevented
* false-positive amount
* missed fraud amount

### Core business metrics

#### Fraud Amount Capture Rate

$$
FCR =
\frac{Detected\ Fraud\ Amount}
{Total\ Actual\ Fraud\ Amount}
$$

#### Fraud Loss Prevented

$$
FLP =
\sum Amount_{correctly\ blocked}
$$

#### Missed Fraud Amount

$$
MFA =
\sum Amount_{false\ negatives}
$$

These metrics align the ML system with the actual merchant-risk objective.

---

# 📡 Monitoring

SentinelRisk AI includes monitoring at three levels.

### Data monitoring

```text
Schema drift
Missing values
Duplicate transactions
Timestamp anomalies
Distribution drift
```

### Phase 1 monitoring

```text
Fraud probability distribution
Feature drift
Calibration
Prediction drift
Precision / Recall
PR-AUC
Fraud amount capture
```

### Phase 2 monitoring

```text
Candidate windows
Verified windows
Critical windows
Fraud events
TAS distribution
Persistence distribution
Breadth distribution
Acceleration distribution
Expected loss
Event duration
Alert frequency
```

---

# 🔄 Feedback Loop

Risk analysts can label detected events:

```text
CONFIRMED FRAUD
FALSE POSITIVE
NEEDS INVESTIGATION
```

Additional metadata can capture:

```text
Attack type
Reason
Analyst notes
Resolution
Timestamp
```

This creates a feedback loop:

```text
Detection
   ↓
Investigation
   ↓
Analyst Label
   ↓
Feedback Store
   ↓
Evaluation
   ↓
Future Model Improvement
```

---

# ⚙️ MLOps & Deployment

SentinelRisk AI is designed as a production-oriented ML system rather than a notebook-only project.

### Development lifecycle

```text
Git Commit
    ↓
CI Tests
    ↓
Phase 1 Validation
    ↓
Phase 2 Validation
    ↓
Business Metric Checks
    ↓
XAI Validation
    ↓
Docker Build
    ↓
Container Registry
    ↓
Deployment
    ↓
Monitoring
```

### Release gate

A model/detector release should fail when structural or validation requirements are violated.

```text
VALIDATION PASS
        ↓
   DEPLOY ALLOWED

VALIDATION FAIL
        ↓
   RELEASE BLOCKED
```

---

# 🐳 Deployment Architecture

```text
                       ┌─────────────────┐
                       │    Load Balancer │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │    FastAPI      │
                       └────────┬────────┘
                                ↓
             ┌──────────────────┼──────────────────┐
             ↓                  ↓                  ↓
      Transaction AI      Temporal AI        Event Engine
             │                  │                  │
             └──────────────────┼──────────────────┘
                                ↓
                       ┌─────────────────┐
                       │     XAI         │
                       └────────┬────────┘
                                ↓
                       ┌─────────────────┐
                       │ Risk Decision   │
                       └────────┬────────┘
                                ↓
                  ┌─────────────┴─────────────┐
                  ↓                           ↓
             Dashboard                    Copilot
                  │                           │
                  └─────────────┬─────────────┘
                                ↓
                           Monitoring
                                ↓
                             MLOps
```

---

# 📁 Repository Structure

```text
SentinelRisk-AI/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .gitignore
│
├── config/
│   ├── phase1.yaml
│   ├── phase2.yaml
│   ├── xai.yaml
│   ├── decision.yaml
│   ├── copilot.yaml
│   ├── monitoring.yaml
│   └── deployment.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── artifacts/
│   ├── phase1/
│   ├── phase2/
│   ├── xai/
│   └── evaluation/
│
├── src/
│   ├── ingestion/
│   ├── validation/
│   │
│   ├── phase1/
│   │   ├── features/
│   │   ├── model/
│   │   ├── inference/
│   │   └── evaluation/
│   │
│   ├── phase2/
│   │   ├── windows/
│   │   ├── baselines/
│   │   ├── signals/
│   │   ├── scoring/
│   │   ├── verification/
│   │   ├── events/
│   │   └── evaluation/
│   │
│   ├── xai/
│   ├── decision/
│   ├── copilot/
│   ├── serving/
│   ├── monitoring/
│   └── common/
│
├── dashboard/
│
├── scripts/
│   ├── train_phase1.py
│   ├── run_phase2.py
│   ├── validate_phase2.py
│   ├── build_events.py
│   └── run_api.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── end_to_end/
│
├── deployment/
│   ├── docker/
│   ├── k8s/
│   └── monitoring/
│
└── docs/
    ├── architecture/
    ├── methodology/
    ├── validation/
    └── api/
```

---

# 🛠️ Technology Stack

| Layer                 | Technology                            |
| --------------------- | ------------------------------------- |
| Language              | Python                                |
| Transaction ML        | XGBoost                               |
| Explainability        | SHAP / deterministic evidence engine  |
| Temporal Intelligence | EWMA + CUSUM + robust statistics      |
| Data Processing       | Pandas / DuckDB where appropriate     |
| API                   | FastAPI                               |
| Validation            | DuckDB-first validation framework     |
| Dashboard             | Streamlit / React-based UI            |
| Copilot               | LLM + structured tools                |
| Containerization      | Docker                                |
| CI/CD                 | GitHub Actions                        |
| Monitoring            | Prometheus / Grafana-compatible stack |
| Deployment            | Docker / Kubernetes                   |
| Configuration         | YAML                                  |
| Testing               | Pytest                                |

---

# 🔐 Design Principles

### 1. Detection ≠ Decision

The ML system provides evidence.

The policy engine determines the recommended response.

### 2. Detection ≠ Explanation

The detector calculates the signal.

The XAI layer explains the signal.

### 3. Copilot ≠ Fraud Detector

The Copilot never replaces the deterministic risk engine.

### 4. Ground Truth ≠ Production Feature

Actual fraud labels are used for evaluation and validation, not as hidden production detector inputs.

### 5. Financial Risk Matters

Fraud detection should ultimately be connected to expected financial exposure.

### 6. Explainability Must Be Evidence-Based

Every explanation should trace back to actual model or detector outputs.

### 7. Validation Is a Release Gate

A system should not be deployed simply because code executes successfully.

---

# 🏆 Why SentinelRisk AI?

Traditional approach:

```text
Transaction
   ↓
Fraud Model
   ↓
Alert
```

SentinelRisk AI:

```text
Transaction
   ↓
Fraud Probability
   ↓
Temporal Behavior
   ↓
Multiscale Spike Detection
   ↓
Verification
   ↓
Financial Materiality
   ↓
Fraud Event
   ↓
Explainable Evidence
   ↓
Risk Decision
   ↓
Analyst Copilot
   ↓
Actionable Intelligence
```

This shifts fraud detection from:

> **"Find suspicious transactions."**

to:

> **"Understand emerging merchant-level fraud attacks and help risk teams respond before losses escalate."**

---

# 🎬 Example End-to-End Scenario

```text
12:00
Merchant receives abnormal transactions
        ↓
Phase 1 detects elevated fraud probabilities
        ↓
12:05
5-minute temporal window becomes abnormal
        ↓
12:10
Suspicious activity accelerates
        ↓
12:15
Persistence + breadth + financial materiality increase
        ↓
Verification passes
        ↓
EVENT EVT-001 CREATED
        ↓
Expected fraud exposure calculated
        ↓
XAI identifies strongest evidence
        ↓
Decision Engine assigns:
CRITICAL / IMMEDIATE RESPONSE
        ↓
Dashboard displays the event
        ↓
Analyst asks Copilot:
"Why is this critical?"
        ↓
Copilot explains the verified evidence
        ↓
Analyst investigates and labels event
        ↓
Feedback enters MLOps loop
```

---

# 📈 Current Project Roadmap

```text
✅ Phase 1 — Transaction Intelligence

✅ Phase 2 — Temporal Fraud-Spike Intelligence

🔄 Phase 2 — Final Independent Validation

⬜ Event Contract

⬜ Explainable AI Layer

⬜ Risk Decision Engine

⬜ Expected Loss Intelligence

⬜ Risk Event Intelligence

⬜ FastAPI Serving Layer

⬜ Risk Dashboard

⬜ Risk Copilot

⬜ Monitoring

⬜ Analyst Feedback Loop

⬜ Dockerization

⬜ CI/CD

⬜ Production Deployment

⬜ End-to-End Demonstration
```

---

# 🚀 Getting Started

## 1. Clone

```bash
git clone https://github.com/<your-username>/SentinelRisk-AI.git
cd SentinelRisk-AI
```

## 2. Create environment

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run Phase 1

```bash
python scripts/train_phase1.py
```

## 5. Run Phase 2

```bash
python scripts/run_phase2.py
```

## 6. Validate Phase 2

```bash
python scripts/validate_phase2.py --refresh
```

## 7. Start API

```bash
uvicorn src.serving.app:app --reload
```

## 8. Start Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 🧪 Testing

Run the complete test suite:

```bash
pytest -q
```

Run validation:

```bash
python scripts/validate_phase2.py --refresh
```

Run end-to-end checks:

```bash
pytest tests/end_to_end -q
```

---

# 📌 Project Status

```text
Transaction Intelligence       ████████████████████ 100%
Temporal Intelligence          ████████████████████ 100%
Phase-2 Validation             ███████████████░░░░░  In Progress
Explainable AI                 ░░░░░░░░░░░░░░░░░░░░  Planned
Decision Intelligence          ░░░░░░░░░░░░░░░░░░░░  Planned
Risk Copilot                   ░░░░░░░░░░░░░░░░░░░░  Planned
Dashboard                      ░░░░░░░░░░░░░░░░░░░░  Planned
Monitoring                     ░░░░░░░░░░░░░░░░░░░░  Planned
MLOps Deployment               ░░░░░░░░░░░░░░░░░░░░  Planned
```

---

# 💡 Hackathon Vision

SentinelRisk AI is designed around one principle:

> **Fraud detection should not stop at a probability score.**

A useful risk platform must understand:

```text
WHAT happened
      ↓
WHY it is unusual
      ↓
WHETHER it represents a real attack
      ↓
HOW MUCH money is at risk
      ↓
WHAT action should happen next
```

That is the goal of **SentinelRisk AI**.

---

# 👥 Built For

**Merchant Risk Teams • Payment Platforms • Fraud Analysts • Risk Operations • Financial Intelligence Teams**

---

# 📜 License

This project is developed as a hackathon / research prototype.

Add your preferred license before public production use.

---

<p align="center">

### 🛡️ SentinelRisk AI

**From transaction-level suspicion to explainable, actionable fraud intelligence.**

</p>

