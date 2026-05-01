# 🧠 PentaAI - Persona-Based Chat Application

A smart chat application that lets you converse with AI personas, each with unique personality traits based on real survey data.

---

## 📋 Table of Contents
- [Quick Start](#-quick-start)
- [Installation Guide](#-installation-guide)
- [How to Use](#-how-to-use)
- [Project Structure](#-project-structure)
- [File Explanations](#-file-explanations)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the application
python app.py

# 3. Open in browser
# Go to: http://127.0.0.1:5000
```

---

## 📥 Installation Guide

### Prerequisites
- **Python 3.8+** installed on your system
- **pip** (Python package manager)
- **Web browser** (Chrome, Firefox, Edge, etc.)

### Step-by-Step Installation

#### Step 1: Download/Clone the Project
Copy the entire `Penta-PersonaAI` folder to your computer.

#### Step 2: Open Terminal/Command Prompt
Navigate to the project folder:
```bash
cd path/to/Penta-PersonaAI
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `flask` - Web server framework
- `flask-cors` - Enables cross-origin requests
- `google-generativeai` - Gemini AI integration
- `python-dotenv` - Environment variable management

#### Step 4: (Optional) Set Up API Key
If you have your own Gemini API key, create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```
*Note: The app has a built-in API key, so this step is optional.*

#### Step 5: Run the Application
```bash
python app.py
```

You should see:
```
[*] Gemini LLM initialized successfully!
 * Running on http://127.0.0.1:5000
```

#### Step 6: Open in Browser
Go to: **http://127.0.0.1:5000**

---

## 💬 How to Use

### 1. Select a Persona
Click on any persona in the left sidebar:
- **Koushani Nath** (Pink theme)
- **Rishit Tandon** (Blue theme)
- **Harshv** (Purple theme)
- **Manya Sgarma** (Coral theme)
- **Salil Shekhar** (Slate theme)

### 2. Start Chatting
Type your message and press Enter or click the send button.

### 3. Example Questions
Try asking:
- "What do you value most in life?"
- "How do you handle stress?"
- "Career or family, what's more important?"
- "What would you do if you got a job offer in another city?"

### 4. Clear Chat
Click the "Clear" button to start a new conversation.

---

## 📁 Project Structure

```
Penta-PersonaAI/
├── app.py                    # Main Flask server
├── requirements.txt          # Python dependencies
├── .env                      # API keys (optional)
│
├── engine/                   # Backend logic
│   ├── __init__.py          # Package initializer
│   ├── chat_handler.py      # Main chat controller
│   ├── llm_handler.py       # Gemini AI integration
│   ├── intent_detector.py   # Understands user questions
│   ├── response_generator.py # Template responses
│   └── persona_loader.py    # Loads persona data
│
├── personas/                 # Persona JSON files
│   ├── Persona_1.json       # Koushani's personality
│   ├── Persona_2.json       # Rishit's personality
│   ├── Persona_3.json       # Harshv's personality
│   ├── Persona_4.json       # Manya's personality
│   └── Persona_5.json       # Salil's personality
│
├── templates/                # HTML templates
│   └── index.html           # Main UI page
│
├── static/                   # Static assets
│   └── css/
│       └── style.css        # All styling
│
└── data/                     # Survey data (source)
```

---

## 📚 File Explanations

### Core Files

#### `app.py` - Main Server
**What it does:** Starts the web server and handles all HTTP requests.

**Key parts:**
- Creates Flask application
- Defines API routes (`/api/personas`, `/api/chat`, etc.)
- Serves the HTML page
- Initializes the ChatHandler

```python
# Routes:
GET  /                    → Serves the main page
GET  /api/personas        → Returns list of all personas
POST /api/persona/select  → Selects a persona to chat with
POST /api/chat            → Sends message, gets response
```

---

#### `engine/chat_handler.py` - Chat Controller
**What it does:** The brain that coordinates everything.

**Key features:**
- Manages which persona is active
- Decides whether to use AI (LLM) or templates
- Handles the conversation flow

```python
# Flow:
1. User sends message
2. ChatHandler receives it
3. Tries LLM (Gemini AI) first
4. If LLM fails → Falls back to templates
5. Returns response to user
```

---

#### `engine/llm_handler.py` - AI Integration
**What it does:** Communicates with Google's Gemini AI.

**Key features:**
- **Model rotation:** If one model hits quota, tries another
- **5 fallback models:** gemini-2.5-flash, gemini-2.0-flash, etc.
- Builds persona-aware prompts

```python
# Model Fallback Chain:
gemini-2.5-flash → gemini-2.0-flash → gemini-2.5-flash-lite → ...
```

---

#### `engine/intent_detector.py` - Question Understanding
**What it does:** Figures out what type of question the user asked.

**Intent categories:**
- `career_move` - Job/career related questions
- `relationship_conflict` - Relationship issues
- `family_decision` - Family-related choices
- `value_comparison` - "X or Y?" questions
- `stress_decision` - How to handle stress
- `future_choice` - Future planning

---

#### `engine/response_generator.py` - Template Responses
**What it does:** Provides fallback responses when AI is unavailable.

**Features:**
- Responses based on persona traits
- Casual Indian youth texting style
- Uses "yaar", "tbh", "lol", etc.

---

#### `engine/persona_loader.py` - Persona Data
**What it does:** Loads persona JSON files from the `/personas` folder.

**Persona traits include:**
- Family Priority (High/Medium/Low)
- Career Orientation (Visionary/Logical)
- Risk Tolerance (High/Low)
- Conflict Handling style
- Core Values
- And more...

---

### Frontend Files

#### `templates/index.html` - Main UI
**What it does:** The entire user interface.

**Contains:**
- Sidebar with persona list
- Chat area with messages
- Landing page with feature cards
- Persona-specific color themes
- All JavaScript for interactivity

---

#### `static/css/style.css` - Styling
**What it does:** All visual design.

**Features:**
- Dark theme with glassmorphism
- macOS-style traffic lights
- Persona color themes
- Responsive design
- Smooth animations

---

### Data Files

#### `personas/Persona_X.json` - Personality Files
Each persona has a JSON file with their traits:

```json
{
  "Name": "Koushani Nath",
  "Traits": {
    "Family_Priority": "High",
    "Career_Orientation": "Visionary",
    "Risk_Tolerance": "Medium",
    "Core_Value": "Emotional Security"
  }
}
```

---

## 🔧 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Port already in use" error
Change the port in `app.py`:
```python
app.run(port=5001)  # Use different port
```

### "429 Quota exceeded" error
The app automatically:
1. Rotates to a different AI model
2. Falls back to template responses
No action needed!

### Responses are cut off
Already fixed! Max tokens set to 500.

### Blank page / Nothing loads
1. Check terminal for errors
2. Make sure Flask is running
3. Try hard refresh: `Ctrl + Shift + R`

---

## 🎨 Customization

### Change Persona Colors
Edit `templates/index.html`, find `personaThemes`:
```javascript
'koushani': {
    primary: '#ec4899',      // Change this hex color
    gradient: 'linear-gradient(135deg, #ec4899, #f472b6)',
    glow: 'rgba(236, 72, 153, 0.3)'
},
```

### Add New Personas
1. Create `personas/Persona_6.json` with traits
2. The app will auto-detect it

### Change AI Model
Edit `engine/llm_handler.py`, modify `MODEL_LIST`:
```python
MODEL_LIST = [
    'models/gemini-2.5-flash',
    # Add or remove models here
]
```

---

## 📜 License
This project is for educational purposes.

---

## 👤 Contact
Created for the Penta-PersonaAI project.

---

**Enjoy chatting with your personas! 🧠💬**
