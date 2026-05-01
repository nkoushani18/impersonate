# 🎯 INTENT-FIRST SCORING FORMULA

## Your Requirement

**Priority Order:**
1. **Intent Match** - Maximum marks (60%)
2. **Tone/Emotional** - Second priority (30%)
3. **Factual Accuracy** - Lower weight (10%)

---

## 📊 NEW FORMULA

```
Final Score = (Intent × 60%) + (Tone × 30%) + (Factual × 10%)
```

### **Weights Explained:**

**Intent (60%)**
- **What it measures:** Did they understand and answer the SAME way?
- **Why highest:** This is what matters most!
- **Scoring:**
  - Both chose same option = 100
  - Similar understanding = 80
  - Partial overlap = 50
  - Different = 0-30

**Tone/Emotional (30%)**
- **What it measures:** Similar energy, enthusiasm, casualness?
- **Why second:** How they express matters
- **Scoring:**
  - Same vibe = 100
  - Similar tone = 70-90
  - Neutral = 50
  - Opposite = 0-40

**Factual (10%)**
- **What it measures:** Did they mention similar facts?
- **Why lowest:** Often no facts in preference questions
- **Scoring:**
  - Same facts = 100
  - Related = 60
  - None/irrelevant = 0

---

## 📐 Example Calculations

### **Example 1: Perfect Intent Match**
```
Question: "Work or food?"
AI: "Work, yaar. Gotta grind."
Human: "Work for sure, need to hustle."

Intent: 100 (both clearly chose WORK)
Tone: 90 (both casual, motivated)
Factual: 0 (no facts mentioned)

Score = (100 × 0.6) + (90 × 0.3) + (0 × 0.1)
      = 60 + 27 + 0
      = 87/100
```

### **Example 2: Intent Match + Facts**
```
Question: "Morning or evening person?"
AI: "Morning person, I'm most productive at 6 AM."
Human: "Morning definitely, I wake up at 5:30 AM sharp."

Intent: 100 (both chose morning)
Tone: 85 (both certain, similar energy)
Factual: 90 (both mention specific early times)

Score = (100 × 0.6) + (85 × 0.3) + (90 × 0.1)
      = 60 + 25.5 + 9
      = 94.5 → 95/100
```

### **Example 3: Intent Mismatch**
```
Question: "Coffee or tea?"
AI: "Coffee all the way, can't function without it."
Human: "Tea for me, coffee makes me jittery."

Intent: 0 (opposite choices)
Tone: 70 (both explain their preference strongly)
Factual: 60 (both give reasons, but opposite)

Score = (0 × 0.6) + (70 × 0.3) + (60 × 0.1)
      = 0 + 21 + 6
      = 27/100
```

### **Example 4: Similar Intent, Different Strength**
```
Question: "Study or party?"
AI: "Study I guess, gotta pass."
Human: "Study 100%, exams are important."

Intent: 80 (both chose study but different conviction)
Tone: 60 (AI hesitant, human certain)
Factual: 50 (both mention exams/passing)

Score = (80 × 0.6) + (60 × 0.3) + (50 × 0.1)
      = 48 + 18 + 5
      = 71/100
```

---

## 🎯 Scoring Guidelines

### **Intent (0-100):**
```
100: Exact same choice/preference, same strength
90:  Same choice, very similar reasoning
80:  Same choice, different emphasis
60:  Leaning same direction
40:  Related but not quite same
20:  Somewhat related topic
0:   Opposite or unrelated
```

### **Tone (0-100):**
```
100: Identical emotional expression
90:  Very similar enthusiasm/energy
80:  Same general vibe
70:  Similar but different intensity
50:  Neutral, no strong emotion
30:  Different emotional tone
0:   Opposite emotions
```

### **Factual (0-100):**
```
100: Mention exactly same facts
80:  Very similar factual claims
60:  Related facts, same category
40:  Some factual overlap
20:  Different facts
0:   No factual claims OR opposite facts
```

---

## 📊 Breakdown Display

```
╔════════════════════════════════════════════════╗
║ Question: "Work or food?"                      ║
║                                                ║
║ 🎯 INTENT MATCH: YES (100/100)                 ║
║ ├─ AI: "Chose WORK, values productivity"      ║
║ └─ Human: "Chose WORK, focused on career"     ║
║                                                ║
║ 🎭 TONE SIMILARITY: 90/100                     ║
║ ├─ AI: Casual, motivated tone                 ║
║ └─ Human: Casual, determined tone              ║
║                                                ║
║ 📚 FACTUAL ALIGNMENT: 0/100                    ║
║ └─ No factual claims in either response        ║
║                                                ║
║ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                ║
║ 📊 FINAL CALCULATION                           ║
║                                                ║
║ Intent (60%):    100 × 0.6 = 60 points        ║
║ Tone (30%):       90 × 0.3 = 27 points        ║
║ Factual (10%):     0 × 0.1 =  0 points        ║
║                               ─────────        ║
║ TOTAL SCORE:                  87/100          ║
╚════════════════════════════════════════════════╝
```

---

## 🔧 Updated Judge Prompt

```python
prompt = f"""Judge if AI and Human understood and answered the question the same way.

Question: "{question}"
AI Response: "{ai_resp}"
Human Response: "{human_resp}"

PRIORITY 1: INTENT (60% weight)
Extract what each person chose/preferred:
- AI_INTENT: What did AI choose? (one sentence)
- HUMAN_INTENT: What did human choose? (one sentence)
- INTENT_SCORE: 0-100 (100 = exact same choice, 0 = opposite)

PRIORITY 2: TONE (30% weight)
Compare emotional expression:
- Similar energy/enthusiasm?
- TONE_SCORE: 0-100 (100 = identical vibe, 0 = opposite)

PRIORITY 3: FACTUAL (10% weight)
Check if they mentioned similar facts:
- FACTUAL_SCORE: 0-100 (0 if no facts mentioned)

IGNORE: typos, grammar, exact wording

OUTPUT:
AI_INTENT: [what AI chose]
HUMAN_INTENT: [what human chose]
INTENT_SCORE: [0-100]
TONE_SCORE: [0-100]
FACTUAL_SCORE: [0-100]
INTENT_MATCH: YES or NO
REASONING: Brief explanation
"""
```

---

## 💡 Why This Works Better

### **Old System:**
```
Preference: 50%
Emotional: 30%
Factual: 20%

Problem: "Preference" was vague
```

### **New System:**
```
Intent: 60%  ← Clear: did they choose the same?
Tone: 30%    ← How they said it
Factual: 10% ← Lowest priority

Benefit: Intent is crystal clear!
```

---

## 📊 Research Presentation

### **Table: Intent-First Analysis**
```
| Q# | Question  | Intent Match | Intent Score | Tone | Factual | Final |
|----|-----------|--------------|--------------|------|---------|-------|
| 1  | work/food | ✅ YES       | 100          | 90   | 0       | 87    |
| 2  | day/night | ❌ NO        | 0            | 70   | 60      | 27    |
| 3  | study/fun | ✅ YES       | 90           | 80   | 50      | 83    |
```

### **Key Metrics:**
- **Intent Match Rate:** 92% (23/25)
- **Avg Score (Intent Match):** 89.3
- **Avg Score (Intent Mismatch):** 28.5
- **Intent Contribution:** Avg 54.2/60 points
- **Tone Contribution:** Avg 24.1/30 points

---

## ✅ Implementation Priority

1. **Extract Intent** - Most important!
2. **Score Intent** - 0-100, clear matching
3. **Measure Tone** - Emotional similarity
4. **Check Facts** - Lowest weight
5. **Calculate:** (Intent×0.6) + (Tone×0.3) + (Factual×0.1)

---

**This puts INTENT FIRST, giving it maximum weight!** 🎯✨

The formula now clearly shows:
- **60% = Did they choose the same thing?**
- **30% = Did they say it the same way?**
- **10% = Did they mention similar facts?**
