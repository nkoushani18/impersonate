# 📐 PROPER SCORING FORMULA

## Current Problem

**Breakdown doesn't match final score:**
- Factual: N/A
- Emotional: 20
- Preference: 75
- **Final Score: 95** ← How?!

**There's no clear formula connecting them!**

---

## ✅ NEW WEIGHTED SCORING FORMULA

### **Formula:**
```
Final Score = (Preference × 0.5) + (Emotional × 0.3) + (Factual × 0.2)
```

### **Weights:**
- **Preference: 50%** - Most important (what they actually choose)
- **Emotional: 30%** - How they express it (tone, enthusiasm)
- **Factual: 20%** - Accuracy of facts stated

---

## 📊 Example Calculation

### **Scenario 1: Perfect Match**
```
Question: "Work or food?"
AI: "Work, yaar. Gotta grind."
Human: "Work for sure, need to hustle."

Preference: 95 (both chose work strongly)
Emotional: 85 (both casual, motivated tone)
Factual: 0 (no factual claims made)

Final Score = (95 × 0.5) + (85 × 0.3) + (0 × 0.2)
            = 47.5 + 25.5 + 0
            = 73

Display:
✅ Intent Match: YES
📊 Score: 73/100
   - Preference: 95 (50% weight → 47.5 points)
   - Emotional: 85 (30% weight → 25.5 points)
   - Factual: 0 (20% weight → 0 points)
```

### **Scenario 2: Strong Preference + Facts**
```
Question: "College or school?"
AI: "College obviously, more freedom and better opportunities."
Human: "College 100%, way more independence and career growth."

Preference: 95 (both strongly prefer college)
Emotional: 90 (both enthusiastic, same energy)
Factual: 85 (both mention freedom/independence, opportunities/growth)

Final Score = (95 × 0.5) + (90 × 0.3) + (85 × 0.2)
            = 47.5 + 27 + 17
            = 91.5 → 92

Display:
✅ Intent Match: YES
📊 Score: 92/100
   - Preference: 95 (50% weight → 47.5 points)
   - Emotional: 90 (30% weight → 27 points)
   - Factual: 85 (20% weight → 17 points)
```

### **Scenario 3: Mismatch**
```
Question: "Morning or evening person?"
AI: "Morning person, I'm most productive early."
Human: "Evening for sure, I'm a night owl."

Preference: 10 (complete opposite)
Emotional: 60 (both explain their choice similarly)
Factual: 40 (both give reasons, but opposite)

Final Score = (10 × 0.5) + (60 × 0.3) + (40 × 0.2)
            = 5 + 18 + 8
            = 31

Display:
❌ Intent Match: NO
📊 Score: 31/100
   - Preference: 10 (50% weight → 5 points)
   - Emotional: 60 (30% weight → 18 points)
   - Factual: 40 (20% weight → 8 points)
```

---

## 🎯 Scoring Guidelines

### **Preference (0-100):**
- **90-100:** Same choice, same strength
- **70-89:** Same choice, different strength
- **50-69:** Similar direction, not exactly same
- **30-49:** Somewhat related
- **0-29:** Different or opposite choices

### **Emotional (0-100):**
- **90-100:** Same tone, enthusiasm, energy
- **70-89:** Similar emotion, different intensity
- **50-69:** Neutral or mildly similar
- **30-49:** Different emotional expression
- **0-29:** Opposite emotions

### **Factual (0-100):**
- **90-100:** State same facts/reasons
- **70-89:** Related facts, similar logic
- **50-69:** Some overlap in reasoning
- **30-49:** Different facts, same category
- **0-29:** Different or opposite facts
- **0:** No factual claims (N/A becomes 0)

---

## 🔧 Implementation

### **Updated Judge Prompt:**
```python
prompt = f"""
SCORING CRITERIA (0-100 each):

1. PREFERENCE (50% weight):
   Same choice = 90-100
   Similar = 60-80
   Different = 0-50

2. EMOTIONAL (30% weight):
   Same tone/feeling = 80-100
   Neutral = 50
   Opposite = 0-40

3. FACTUAL (20% weight):
   Same facts = 80-100
   Related facts = 50-70
   Different = 0-40
   (Use 0 if no factual claims)

FORMULA:
Score = (Preference × 0.5) + (Emotional × 0.3) + (Factual × 0.2)

OUTPUT:
PREFERENCE: [number]
EMOTIONAL: [number]
FACTUAL: [number]
CALCULATED_SCORE: [use formula]
INTENT_MATCH: YES/NO
REASONING: Explain each score
"""
```

### **Parser Update:**
```python
# Extract scores
pref = int(extract("PREFERENCE:"))
emot = int(extract("EMOTIONAL:"))
fact = int(extract("FACTUAL:"))

# Calculate using formula
calculated_score = int((pref * 0.5) + (emot * 0.3) + (fact * 0.2))

# Use calculated score, not LLM's direct score
result["score"] = calculated_score
result["breakdown"] = {
    "preference_alignment": pref,
    "emotional_alignment": emot,
    "factual_alignment": fact
}
```

---

## 📊 Display on Frontend

### **Show the Formula:**
```
╔════════════════════════════════════════╗
║ 📊 Score Breakdown                     ║
║                                        ║
║ Final Score: 73/100                    ║
║                                        ║
║ Formula:                               ║
║ (Pref × 50%) + (Emot × 30%) + (Fact × 20%) ║
║                                        ║
║ Calculation:                           ║
║ (95 × 0.5) + (85 × 0.3) + (0 × 0.2)   ║
║ = 47.5 + 25.5 + 0                     ║
║ = 73                                   ║
║                                        ║
║ ┌──────────┬───────┬────────┬─────┐   ║
║ │ Category │ Score │ Weight │ Pts │   ║
║ ├──────────┼───────┼────────┼─────┤   ║
║ │ Pref     │ 95    │ 50%    │47.5 │   ║
║ │ Emot     │ 85    │ 30%    │25.5 │   ║
║ │ Fact     │ 0     │ 20%    │ 0   │   ║
║ └──────────┴───────┴────────┴─────┘   ║
╚════════════════════════════════════════╝
```

---

## ✅ Benefits

1. **Transparent** - Clear math, reproducible
2. **Weighted** - Preference matters most
3. **Always Complete** - No more N/A
4. **Logical** - Breakdown adds up to final
5. **Research-Ready** - Can cite the formula

---

## 🚀 Next Steps

1. Update `judge_handler.py` prompt
2. Update parser to calculate score using formula
3. Update frontend to show formula
4. Test with multiple examples
5. Export data showing calculation

---

**This is a proper, mathematical, reproducible grading system!** 📐✅
