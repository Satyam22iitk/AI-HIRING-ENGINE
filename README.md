# 🚀 AI-Powered Semantic Candidate Ranking & Explainable Hiring Pipeline

An end-to-end AI hiring pipeline that moves beyond traditional keyword-based ATS by using **semantic reasoning**, **Natural Language Inference (NLI)**, and **explainable AI** to rank candidates.

Built for the **Redrob AI Hiring Challenge**.

---

## 📌 Problem Statement

Traditional Applicant Tracking Systems (ATS) rely heavily on keyword matching, often rejecting highly qualified candidates simply because they use different wording.

This project introduces an **AI-powered semantic hiring pipeline** that evaluates actual candidate capabilities instead of matching keywords.

---

## ✨ Features

- 📄 AI-based Job Description Understanding using **Gemini 2.5 Flash**
- 🧠 Automatic Capability Hypothesis Generation
- ✍ Human-in-the-loop Editing & Weight Assignment
- 🤖 Semantic Resume Evaluation using **DeBERTa-v3 Zero-Shot NLI**
- 📊 Behavioral Signal Integration
- 📈 Weighted Multi-Factor Candidate Ranking
- 🛡 Explainable AI Recommendations
- 🚩 Automatic Red Flag Detection
- 📉 Interactive Radar Charts
- 📁 CSV Export (Top Candidates)

---

## 🏗 System Workflow

```
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

## 🛠 Technologies Used

- Python
- Gradio
- Gemini 2.5 Flash
- Hugging Face Transformers
- DeBERTa-v3 (MNLI)
- PyTorch
- SVG Visualizations
- CSV

---

# 🌐 Live Demo (Hugging Face)

https://huggingface.co/spaces/satyamiitk/ai-hiring-pipeline

### Why Hugging Face?

The Hugging Face deployment allows anyone to test the application directly in a web browser without any installation.

**Note:**  
The free Hugging Face Space runs on a **CPU**, so inference can be slower when evaluating large candidate datasets or many hypotheses.

---

# ⚡ Google Colab (Recommended)

https://colab.research.google.com/drive/1TqKk2AgWSERKTIcxXqSJQDd7hZUcgSN5?usp=sharing

### Why Colab?

The Google Colab notebook is recommended for benchmarking and evaluating large datasets because it supports **NVIDIA T4 GPU acceleration**, providing significantly faster inference than the CPU-based Hugging Face deployment.

### How to Run

1. Open the Colab notebook.
2. Select **Runtime → Change runtime type**.
3. Choose **T4 GPU** as the hardware accelerator.
4. Click **Save**.
5. Click **Connect**.
6. Run **Run All**.
7. Enter your Gemini API Key when prompted.
8. Upload the Job Description file.
9. Upload the Candidate JSON/JSONL file.
10. View ranked candidates and export the CSV.

---

## 📊 Ranking Methodology

The final candidate score combines semantic understanding and behavioral indicators.

```
Final Score =
60% Semantic Match
+
40% Behavioral Signals
```

Behavioral signals include:

- Recruiter Response Rate
- GitHub Activity Score

Additional score optimization includes:

- Weighted Averaging
- Min-Max Normalization
- Power Transformation
- Percentile Ranking

---

## 🔍 Explainability

Every ranked candidate includes:

- Strongest matched requirement
- Weakest matched requirement
- Weighted semantic score
- Behavioral metrics
- Red flags
- Final ranking score

This makes every recommendation transparent and evidence-based.

---

## 📂 Repository Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## 🎯 AI Tools Used During Development

### Google Gemini

- Brainstorming
- System Design
- Prompt Engineering
- Architecture Discussion
- Documentation Assistance

### Google AI Studio (Antigravity)

- Code Generation
- Code Optimization
- Debugging Assistance
- UI Refinement

---

## 🚀 Future Improvements

- Resume PDF Parsing
- Multi-language Resume Support
- Resume Skill Timeline
- LLM-based Interview Question Generation
- Recruiter Analytics Dashboard
- Vector Database Integration
- Batch Resume Upload
- Cloud GPU Deployment

---

## 👨‍💻 Author

**Satyam Srivastava**

Indian Institute of Technology Kanpur

Built for the **Redrob AI Hiring Challenge**.
