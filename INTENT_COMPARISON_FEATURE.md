# 📊 INTENT COMPARISON FEATURE

## What You Want

Show **3 intents**:
1. **Question Intent** - What was asked
2. **AI Intent** - What AI understood/what it chose
3. **Human Intent** - What human understood/what they chose

---

## 🎯 Example Display

```
╔════════════════════════════════════════════════╗
║ 🎯 INTENT ANALYSIS                             ║
║                                                ║
║ ❓ Question: "Work or food?"                   ║
║                                                ║
║ 🤖 AI Intent:                                  ║
║    "Prefers work over food, values grinding"  ║
║                                                ║
║ 👤 Human Intent:                               ║
║    "Prefers work, focused on hustle"          ║
║                                                ║
║ ✅ Intent Match: YES                           ║
║    Both understood the choice question and     ║
║    both chose WORK as their preference.        ║
║                                                ║
║ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                ║
║ 📊 SCORE BREAKDOWN                             ║
║                                                ║
║ Final Score: 87/100                            ║
║                                                ║
║ ┌───────────┬───────┬────────┬────────┐       ║
║ │ Category  │ Score │ Weight │ Points │       ║
║ ├───────────┼───────┼────────┼────────┤       ║
║ │ Preference│  95   │  50%   │  47.5  │       ║
║ │ Emotional │  85   │  30%   │  25.5  │       ║
║ │ Factual   │   0   │  20%   │   0    │       ║
║ └───────────┴───────┴────────┴────────┘       ║
║                                                ║
║ Formula: (95×0.5) + (85×0.3) + (0×0.2) = 73   ║
╚════════════════════════════════════════════════╝
```

---

## 🔧 Updated Judge Prompt

```python
prompt = f"""You are judging if AI and Human understood and answered a question the same way.

Question: "{question}"
AI Response: "{ai_resp}"
Human Response: "{human_resp}"

STEP 1: Extract what each understood
- AI_INTENT: What did the AI understand/choose? (one sentence)
- HUMAN_INTENT: What did the human understand/choose? (one sentence)

STEP 2: Do the intents match?
- INTENT_MATCH: YES if both chose same thing, NO if different

STEP 3: Score each category (0-100):

PREFERENCE (50% weight):
- Same choice/preference = 90-100
- Similar direction = 60-80
- Different = 0-50

EMOTIONAL (30% weight):
- Same tone/feeling = 80-100
- Neutral = 50
- Opposite = 0-40

FACTUAL (20% weight):
- Same facts mentioned = 80-100
- Related facts = 50-70
- No facts = 0

OUTPUT FORMAT:
AI_INTENT: [what AI chose/understood]
HUMAN_INTENT: [what human chose/understood]
INTENT_MATCH: YES or NO
PREFERENCE: [0-100]
EMOTIONAL: [0-100]
FACTUAL: [0-100]
REASONING: Brief explanation
"""
```

---

## 📊 Display on Frontend

### **Intent Section:**
```html
<div class="intent-section">
    <h3>🎯 Intent Analysis</h3>
    
    <div class="intent-grid">
        <div class="intent-card ai">
            <div class="intent-label">🤖 AI Intent</div>
            <div class="intent-text">"Prefers work over food"</div>
        </div>
        
        <div class="intent-card human">
            <div class="intent-label">👤 Human Intent</div>
            <div class="intent-text">"Chose work, values hustle"</div>
        </div>
    </div>
    
    <div class="intent-match">
        ✅ Both understood and chose WORK
    </div>
</div>
```

---

## 🎓 For Research Paper

This shows:

### **Table: Intent Alignment Analysis**
```
| Q# | Question    | AI Intent      | Human Intent    | Match | Score |
|----|-------------|----------------|-----------------|-------|-------|
| 1  | work/food   | Chose work     | Chose work      | ✅    | 87    |
| 2  | day/night   | Prefers day    | Prefers night   | ❌    | 35    |
| 3  | coffee/tea  | Likes coffee   | Likes coffee    | ✅    | 92    |
```

### **Metrics:**
- **Intent Match Rate:** 23/25 = 92%
- **Score When Matched:** Average 88.5
- **Score When Mismatched:** Average 32.1

---

## 💡 Benefits

### **Shows:**
1. ✅ Did they understand the question?
2. ✅ What did each choose?
3. ✅ Clear comparison of understanding
4. ✅ Helps diagnose AI errors

### **Example Insights:**
```
Low score + Intent Match = AI answered correctly but differently
Low score + Intent Mismatch = AI misunderstood question
High score + Intent Match = Perfect alignment
High score + Intent Mismatch = Coincidental similarity
```

---

## 🚀 Implementation

### **Step 1: Update `judge_handler.py` prompt**
Add AI_INTENT and HUMAN_INTENT to output format

### **Step 2: Update parser**
```python
elif line.startswith("AI_INTENT:"):
    result["ai_intent"] = line.replace("AI_INTENT:", "").strip()

elif line.startswith("HUMAN_INTENT:"):
    result["human_intent"] = line.replace("HUMAN_INTENT:", "").strip()
```

### **Step 3: Update frontend to display**
Add intent cards showing what each understood

---

## 📋 Example Output

```json
{
  "score": 87,
  "ai_intent": "Prefers work over food, values grinding",
  "human_intent": "Chose work, focused on career",
  "intent_match": true,
  "breakdown": {
    "preference": 95,
    "emotional": 85,
    "factual": 0
  },
  "reasoning": "Both chose work. Similar casual tone. No factual claims."
}
```

---

**This adds crucial research value - showing not just IF they match, but HOW they understood the question!** 🎯✨

Want me to implement this?
