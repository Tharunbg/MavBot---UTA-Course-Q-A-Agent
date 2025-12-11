# 🎓 MavBot UTA Course Analytics & Q&A Agent

An intelligent course discovery and analytics system for the **University of Texas at Arlington (UTA)** that helps students make data-driven academic decisions using historical GPA data, grade distributions, professor analytics, and semantic search.

This project was built as a **Master’s Capstone in Data Science** and goes beyond traditional search by combining **structured analytics with a hybrid Retrieval-Augmented Generation (RAG) architecture**.

---

## 🚀 Key Features
- 📚 Course-level analytics (Average GPA, Pass Rate, DFW Rate)
- 👨‍🏫 Professor analytics & teaching-style classification
- 📊 Grade distribution lookup by course, term, and instructor
- 🔍 Semantic search across courses, professors, and sections
- 🧠 Hybrid RAG architecture (intent-aware routing)
- 💡 Course & professor recommendations (easy courses, best professors)
- 🎯 Interactive Gradio user interface

---

## 🧠 Architecture Overview
This system uses **intent-based query routing**:
- **Factual queries** → direct structured data retrieval (zero hallucination)
- **Interpretive queries** → FAISS vector search + LLM reasoning

