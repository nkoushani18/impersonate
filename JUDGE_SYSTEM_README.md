# LLM Judge System Documentation

## Overview
The LLM Judge system compares AI persona responses with actual human responses to evaluate how well the AI mimics the persona. It uses **Ollama with LLaMA 3.2** as an independent judge to score semantic similarity.

## How It Works

### Flow:
1. **User asks a question** → AI persona responds (via `/` or main chat)
2. **Question is saved** → System creates a comparison session
3. **Human answers** → Real persona responds via `/human` interface
4. **Judge evaluates** → Ollama LLaMA 3.2 compares both responses
5. **Results displayed** → View scores and analysis at `/judge`

---

## The 3 Endpoints/Links

### 1️⃣ Main Chat - `/` (AI Agent Interface)
- **Purpose**: Normal chat with AI persona
- **What happens**: 
  - User asks questions
  - AI persona responds using LLM
  - Each Q&A is automatically queued for human comparison
- **Access**: http://localhost:5000/

### 2️⃣ Human Interface - `/human`
- **Purpose**: For real humans (the actual personas) to answer questions
- **What happens**:
  - Select your persona (e.g., Rishit, Koushani, Salil)
  - See pending questions that were asked to your AI persona
  - View what the AI responded
  - Type your own authentic response
  - Submit → Automatically sent to LLM judge for evaluation
- **Access**: http://localhost:5000/human

### 3️⃣ Judge Results - `/judge`
- **Purpose**: View comparison results and history
- **What's displayed**:
  - **Statistics**: Total comparisons, average score, highest/lowest scores
  - **Latest Comparison**: Most recent judgment with detailed breakdown
  - **Score Breakdown**:
    - Overall semantic accuracy (0-100)
    - Factual alignment score
    - Emotional alignment score  
    - Preference alignment score
    - Intent match (YES/NO)
  - **Detailed Analysis**:
    - Core intent extracted from AI response
    - Core intent extracted from human response
    - Judge's reasoning for the score
  - **History**: All past comparisons until server restart
- **Access**: http://localhost:5000/judge
- **Auto-refresh**: Updates every 10 seconds

---

## Judge Scoring System

The LLM Judge (LLaMA 3.2) evaluates based on **semantic similarity**, NOT exact wording.

### What the Judge Ignores:
- Slang vs formal language differences
- Tone (casual vs serious)
- Exact wording/phrasing  
- Message length
- Writing style

### What the Judge Focuses On:
- **Core intent**: What are they actually trying to say?
- **Factual alignment**: Do they convey the same facts/preferences?
- **Emotional alignment**: Same emotional direction?
- **Preference alignment**: Same choices/opinions?

### Scoring Scale:
- **90-100**: Nearly identical core meaning
- **70-89**: Same general direction, minor differences
- **50-69**: Partial alignment
- **Below 50**: Different core intents

### Example:
```
User: "school or clg?"

AI: "clg yaar its way better, more freedom and you actually learn real stuff"

Human (Rishit): "college bro, obviously"

Judge Result:
- Score: 95/100
- Intent Match: YES
- Reasoning: "Both responses clearly prefer college over school. While wording differs (casual 'clg yaar' vs 'college bro'), the core intent is identical: expressing a strong preference for college."
- Factual Alignment: 100
- Emotional Alignment: 95
- Preference Alignment: 100
```

---

## Setup & Requirements

### Prerequisites:
1. **Ollama installed** with LLaMA 3.2 model
   ```bash
   # Install Ollama (if not already installed)
   # Download from: https://ollama.ai
   
   # Pull LLaMA 3.2 model
   ollama pull llama3.2
   ```

2. **Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Starting the System:
1. **Start Ollama** (in a separate terminal)
   ```bash
   ollama serve
   ```

2. **Run the Flask app**
   ```bash
   python app.py
   ```

3. **Access the interfaces**:
   - Main chat: http://localhost:5000/
   - Human interface: http://localhost:5000/human
   - Judge results: http://localhost:5000/judge

---

## API Endpoints

### For Developers:

#### Get pending questions for human
```http
GET /api/human/pending?persona_id=rishit
```

#### Submit human response
```http
POST /api/human/respond
Content-Type: application/json

{
  "question_id": "rishit_1_1234567890",
  "response": "college bro, obviously"
}
```

#### Get judge results
```http
GET /api/judge/results?persona_id=rishit
```

---

## How to Use

### Typical Workflow:

1. **Open main chat** (http://localhost:5000/)
   - Select a persona (e.g., Rishit)
   - Ask questions: "What do you prefer, school or college?"
   - AI responds based on persona traits

2. **Open human interface** in another window (http://localhost:5000/human)
   - The real Rishit selects his persona
   - Sees the same question
   - Sees what AI-Rishit said
   - Types his own authentic answer
   - Submits → Gets instant judge score!

3. **Open judge results** in third window (http://localhost:5000/judge)
   - Select Rishit persona
   - See the comparison score
   - View detailed breakdown of why that score was given
   - Track history of all comparisons

### Real-Time Demo:
- Keep all 3 windows open
- Ask questions in main chat
- Human responds in human interface
- See results update automatically in judge view

---

## Data Persistence

⚠️ **Important**: All comparison history is stored **in memory** and will be **lost on server restart**.

If you need persistent storage, consider:
- Adding a database (SQLite, PostgreSQL)
- Saving to JSON files
- Using session storage

---

## Troubleshooting

### "Failed to connect to Ollama"
- Make sure Ollama is running: `ollama serve`
- Check if running on correct port: http://localhost:11434
- Verify LLaMA 3.2 is installed: `ollama list`

### No pending questions showing
- Make sure you've asked questions in the main chat first
- Check that the correct persona is selected
- Questions only appear if they haven't been answered yet

### Judge gives unusual scores
- LLaMA 3.2 focuses on semantic meaning, not exact wording
- Very different responses will score low even if both are "correct"
- This measures "how well AI mimics the human", not correctness

---

## Architecture

```
┌─────────────────┐
│   User Chat     │  (Main Interface - /)
│  Ask Questions  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   AI Persona    │  
│   Responds      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Create Session  │  (Question + AI Response saved)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Human Interface │  (/human)
│  Real Person    │  
│   Responds      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Ollama Judge   │  (LLaMA 3.2)
│   Evaluates     │
│  Both Answers   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Judge Results   │  (/judge)
│ Display Scores  │
│  & Breakdown    │
└─────────────────┘
```

---

## Future Enhancements

Potential improvements:
- [ ] Persistent database storage
- [ ] Export comparison history to CSV
- [ ] Multiple judge models comparison
- [ ] Real-time notifications when new questions arrive
- [ ] Webhook integration for Slack/Discord notifications
- [ ] Persona accuracy dashboard with charts
- [ ] Automatic retraining suggestions based on low scores
