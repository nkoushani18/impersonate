# Full Project Architecture & Workflow: Penta-PersonaAI

## 1. High-Level Concept
Penta-PersonaAI is a **Verifiable Digital Twin System**. Unlike standard chatbots, it includes a **verification layer** where AI responses are real-time tested against the actual human's response to measure authenticity objectively.

---

## 2. System Architecture

### **A. The Core Components**
1.  **The Brain (Backend):** Python & Flask for API routing and orchestration.
2.  **The Creator (Persona LLM):** Google Gemini 1.5 Flash. It mimics the user's personality using few-shot prompting from `data/personas/*.json`.
3.  **The Judge (Verification LLM):** Ollama (Llama 3.2). A locally hosted, objective model that compares two text inputs (AI vs Human) without bias.
4.  **The Interface (Frontend):** Modern, dark-themed Web UI for chat, human input, and live judging visualization.

### **B. The Data Flow**
```mermaid
[User/Observer] -> [Chat Interface] -> (Question) -> [Backend API]
                                                      |
                                          +-----------+-----------+
                                          |                       |
                                    [Gemini LLM]             [Human Queue]
                                    (Generates AI Answer)    (Waits for Real Human)
                                          |                       |
                                          +-----> [Pending] <-----+
                                                    |
                                            (Both Answers Ready)
                                                    |
                                             [Ollama Judge]
                                                    |
                                       (Applies Scoring Algorithm)
                                                    |
                                          [Judgment Result] -> [Live UI]
```

---

## 3. Step-by-Step Workflow (How it Works)

### **Phase 1: The Question**
*   An observer asks a question (e.g., *"Mummy or Sister?"*) to the active Persona.
*   **Technique:** The system pulls the Persona's definition (traits, style, examples) and constructs a strict prompt for **Gemini**.
*   **Result:** The AI generates a response (e.g., *"Mummy"*) attempting to match the user's style (short, no slang).

### **Phase 2: The Verification (Human-in-the-Loop)**
*   The AI's answer is **hidden** or marked as "Pending".
*   The **Real Human** opens the `/human` interface. They see the incoming question.
*   The Human types their authentic answer (e.g., *"Mummy"*).

### **Phase 3: The Judgment (The Core Innovation)**
*   The system sends **both** answers to the **Judge Agent** (Ollama).
*   The Judge follows a **Research-Grade Strict Protocol**:
    1.  **Intent Detection:** specifically checks for "X or Y" choice alignment.
    2.  **Hallucination Check:** Ensures words like "Lion" and "Mummy" are strictly flagged as specific mismatches.
    3.  **Scoring:** Assigns raw scores for **Preference**, **Emotional Tone**, and **Factual Reasoning**.

### **Phase 4: The Dynamic Calculation**
*   The system uses our proprietary **Dynamic Weighted Formula**:
    *   **Scenario A (Complex):** If facts/reasons are present:
        `Final Score = (Preference × 60%) + (Tone × 30%) + (Facts × 10%)`
    *   **Scenario B (Simple Preference):** If no facts are present (e.g., one-word answers):
        `Final Score = (Preference × 70%) + (Tone × 30%)` (Facts are ignored).
*   **Result:** A definitive **Authenticity Score (0-100)**.

---

## 4. Key Technical Innovations (For the Panel)

1.  **Dynamic Scoring Algorithm:** The system is "self-aware" enough to know when to grade facts vs. just preferences. It doesn't penalize short answers.
2.  **Blind Double-Blind Verification:** Neither the Human nor the AI sees the other's answer before submission, ensuring 100% unbiased data.
3.  **Local Strict Judging:** Using a specialized local LLM (Ollama) allows us to enforce strict JSON output rules and avoid the "laziness" often seen in cloud LLMs for judging tasks.

## 5. Technology Stack
*   **Language:** Python 3.9+
*   **Framework:** Flask (Backend)
*   **AI Models:** Google Gemini 1.5 Flash (Generation), Meta Llama 3.2 (Judging)
*   **Database:** SQLite / JSON Flat-file storage (for portability)
*   **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
