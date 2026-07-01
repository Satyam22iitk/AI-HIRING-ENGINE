# 🚀 AI-Powered Semantic Candidate Ranking & Explainable Hiring Pipeline

> Moving Beyond Keywords Towards Evidence-Based Hiring

An end-to-end AI hiring pipeline that uses **semantic understanding**, **Natural Language Inference (NLI)**, and **explainable AI** to rank candidates more accurately than traditional keyword-based Applicant Tracking Systems (ATS).

Built for the **Redrob AI Hiring Challenge**.

---

## 📌 Problem Statement

Traditional Applicant Tracking Systems (ATS) primarily rely on keyword matching, often rejecting highly qualified candidates simply because they describe their experience differently.

This project introduces an **AI-powered semantic hiring pipeline** that evaluates actual candidate capabilities instead of matching keywords.

---

# ✨ Key Features

- 📄 AI-powered Job Description Understanding using Gemini 2.5 Flash
- 🧠 Automatic Capability (Hypothesis) Generation
- ✍ Human-in-the-loop Editing & Weight Assignment
- 🤖 Semantic Resume Evaluation using DeBERTa-v3 Zero-Shot NLI
- 📊 Behavioral Signal Integration
- 📈 Weighted Multi-Factor Candidate Ranking
- 🛡 Explainable AI Recommendations
- 🚩 Automatic Red Flag Detection
- 📉 Interactive Radar Charts
- 📁 CSV Export of Top Candidates
- 🌐 Browser-based Interactive Interface

---

# 🏗 System Workflow

```text
Job Description
        │
        ▼
 Gemini 2.5 Flash
        │
        ▼
Capability Hypotheses
        │
Recruiter Review & Weight Assignment
        │
        ▼
 Resume Parsing
        │
        ▼
DeBERTa-v3 NLI
        │
        ▼
Behavioral Signals
        │
        ▼
Weighted Ranking Engine
        │
        ▼
Explainable Dashboard
        │
        ▼
CSV Export
```

---

# 🛠 Technologies Used

- Python
- Gradio
- Gemini 2.5 Flash
- Hugging Face Transformers
- DeBERTa-v3 (MNLI)
- PyTorch
- SVG Visualization
- CSV
- JSON / JSONL

---

# 🌐 Live Demo (Hugging Face)

### 🔗 Demo

https://huggingface.co/spaces/satyamiitk/ai-hiring-pipeline

### Why Hugging Face?

The Hugging Face Space provides an **online demo** that anyone can access directly from a web browser without installing any dependencies.

**Note**

The free Hugging Face Space runs on **CPU**, therefore inference can be slower while evaluating large candidate datasets or many hypotheses.

---

# ⚡ Google Colab (Recommended)

### 🔗 Colab Notebook

https://colab.research.google.com/drive/1TqKk2AgWSERKTIcxXqSJQDd7hZUcgSN5?usp=sharing

## Why Colab?

The Google Colab notebook is recommended for benchmarking and evaluating large datasets because it supports **NVIDIA T4 GPU acceleration**, making semantic inference significantly faster than the CPU-based Hugging Face deployment.

This is especially useful when testing with thousands (or even 100,000+) candidate profiles.

### How to Run

1. Open the Colab notebook.
2. Go to **Runtime → Change runtime type**.
3. Select **T4 GPU** as the Hardware Accelerator.
4. Click **Save**.
5. Click **Connect**.
6. Click **Run All**.
7. Enter your Gemini API Key when prompted.
8. Upload the Job Description.
9. Upload the Candidate JSON/JSONL file.
10. View ranked candidates and export the CSV.

---

# 🎥 Video Demonstration

### 🔗 Demo Video

https://drive.google.com/file/d/1wfz-TVCJazaDgS8OQ_LOyU8dfxHJDjQJ/view

The demo video showcases the complete end-to-end workflow:

- Job Description Upload
- AI Hypothesis Generation
- Human Editing & Weight Assignment
- Candidate Evaluation
- Semantic Ranking
- Explainable Recommendations
- Radar Visualization
- CSV Export

---

# 📊 Ranking Methodology

The final candidate score combines semantic understanding and behavioral indicators.

```text
Final Score

=
60% Semantic Match
+
40% Behavioral Signals
```

Behavioral Signals include:

- Recruiter Response Rate
- GitHub Activity Score

Additional ranking optimization:

- Weighted Average
- Min-Max Normalization
- Power Transformation
- Percentile Ranking

---

# 🔍 Explainability

Every ranked candidate includes:

- ✅ Strongest Matched Requirement
- ✅ Weakest Matched Requirement
- ✅ Weighted Semantic Score
- ✅ Behavioral Metrics
- ✅ Automatic Red Flags
- ✅ Final Candidate Score

This makes every recommendation transparent, interpretable, and evidence-based.

---

# 📂 Repository Structure

```text
.
├── app.py
├── requirements.txt
└── README.md
```

---

# 🤖 AI Tools Used During Development

### Google Gemini

Used for:

- Brainstorming
- System Design
- Prompt Engineering
- Architecture Discussion
- Documentation Assistance

### Google AI Studio (Antigravity)

Used for:

- Code Generation
- Code Optimization
- Debugging Assistance
- UI Improvements

---

# 🚀 Future Improvements

- Resume PDF Parsing
- Multi-language Resume Support
- Resume Skill Timeline
- LLM-based Interview Question Generation
- Recruiter Analytics Dashboard
- Vector Database Integration
- Batch Resume Upload
- Cloud GPU Deployment

---

# 👨‍💻 Author

**Satyam Srivastava**

Indian Institute of Technology Kanpur

Built for the **Redrob AI Hiring Challenge**.

---

## ⭐ If you found this project useful, consider giving it a Star!
