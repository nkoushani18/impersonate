# 🧮 Score Calculation - Detailed Explanation

## How Your Score is Calculated

Every time you submit a human response, here's exactly what happens:

---

## 📊 5-Step Scoring Process

### **Step 1: LLM Analysis** 
```
Ollama LLaMA 3.2 receives:
- User's original question
- AI's response
- Your (human) response
```

The LLM is told:
> "The HUMAN response is GROUND TRUTH. Check if AI matches it."

### **Step 2: Intent Matching**
```
LLM determines: Do they mean the SAME thing?
- Ignores typos (collage = college)
- Ignores slang differences
- Ignores tone (casual vs formal)
- Focuses ONLY on core meaning
```

**Output:** `INTENT_MATCH: YES` or `NO`

### **Step 3: Component Scoring**
```
LLM evaluates 3 aspects (0-100 each):

1. Factual Alignment
   - Do they state the same facts?
   - Example: Both say "college" vs both say "school"

2. Emotional Alignment  
   - Same emotional tone?
   - Example: Both excited vs both neutral

3. Preference Alignment
   - Same opinion/preference?
   - Example: Both prefer coffee vs both prefer tea
```

### **Step 4: Base Score Assignment**
```
LLM assigns SEMANTIC_SCORE: 0-100

Score Ranges:
90-100 = Nearly identical meaning
70-89  = Same direction, minor differences  
50-69  = Partial alignment
0-49   = Different core intents
```

### **Step 5: Smart Boosting** 🚀
```python
Step 5a: Intent Match Boost
if intent_match == YES and score < 85:
    score = max(85, score)
    print("Intent matched - boosted to minimum 85")

Step 5b: Preference Boost  
if preference_alignment >= 80 and score < 80:
    score = max(80, score)
    print("High preference alignment - boosted to 80")
```

---

## 📈 Visual Scoring Flow

```
[Question] → [AI Response] → [Human Response]
                    ↓
            [Send to Ollama LLaMA 3.2]
                    ↓
        ┌───────────────────────┐
        │   Intent Matching     │
        │   ✅ YES  or  ❌ NO   │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │  Component Analysis   │
        │  • Factual: 0-100     │
        │  • Emotional: 0-100   │
        │  • Preference: 0-100  │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   Base Score: 0-100   │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │   Smart Boosting      │
        │   (if applicable)     │
        └───────────────────────┘
                    ↓
        ┌───────────────────────┐
        │ ⭐ FINAL SCORE: XX/100 │
        └───────────────────────┘
```

---

## 🔍 Example Breakdown

### **Example 1: High Score (95/100)**

**Question:** "school or college?"
**AI:** "College, obviously yaar. Way more freedom."
**Human:** "collage is better"

```
Step 1: LLM Analysis ✓
Step 2: Intent Match → YES (both prefer college, typo ignored)
Step 3: Components:
  - Factual: N/A (no facts stated)
  - Emotional: N/A (no strong emotion)
  - Preference: 95 (both clearly prefer college)
Step 4: Base Score → 70 (LLM's initial score)
Step 5: Boosting:
  - Intent match boost: 70 → 85 (minimum 85)
  - Preference boost: 85 → 95 (high preference alignment)
  
FINAL: 95/100 ✅
```

### **Example 2: Low Score (20/100)**

**Question:** "cats or dogs?"
**AI:** "Dogs all the way!"
**Human:** "cats are way better"

```
Step 1: LLM Analysis ✓
Step 2: Intent Match → NO (opposite preferences)
Step 3: Components:
  - Factual: 0 (no facts)
  - Emotional: 20 (both express preference, different)
  - Preference: 0 (opposite preferences)
Step  4: Base Score → 20
Step 5: Boosting:
  - NO boosting (intent mismatch, low scores)
  
FINAL: 20/100 ❌
```

---

## 🎯 Scoring Guarantees

### **Automatic Minimums:**

1. **Intent Match = YES**
   ```
   → Minimum score: 85/100
   → Reason: If core meaning is same, high score guaranteed
   ```

2. **Preference Alignment ≥ 80**
   ```
   → Minimum score: 80/100
   → Reason: High preference match = strong alignment
   ```

3. **Both Minimums Apply**
   ```
   → Takes the HIGHER of the two
   → Example: Intent=YES (85) + Pref=90 (80) → uses 85 minimum
   ```

---

## 🧪 What Gets Ignored

The judge **intelligently ignores**:

✅ **Typos:** collage = college, schoo = school  
✅ **Slang:** "yaar" vs formal English  
✅ **Tone:** casual vs serious  
✅ **Length:** short vs detailed  
✅ **Exact wording:** different words, same meaning  

---

## 🎨 Breakdown Scores Explained

### **Factual Alignment (0-100)**
Measures if the same FACTS/INFORMATION are stated.

```
Example HIGH score (90):
AI: "Paris is the capital of France"
Human: "Paris capital of france"

Example LOW score (10):
AI: "Paris is the capital"  
Human: "London is the capital"
```

### **Emotional Alignment (0-100)**
Measures if the EMOTIONAL TONE is similar.

```
Example HIGH score (95):
AI: "I'm so excited about this!"
Human: "really excited!!"

Example LOW score (20):
AI: "I'm thrilled!"
Human: "meh, it's okay"
```

### **Preference Alignment (0-100)**
Measures if the OPINION/PREFERENCE is the same.

```
Example  HIGH score (100):
AI: "Coffee is better than tea"
Human: "coffee > tea"

Example LOW score (0):
AI: "I prefer coffee"
Human: "tea is way better"
```

---

## 📊 Why "N/A" Sometimes?

You might see `N/A` for some breakdowns when:

❌ **No factual info provided** → Factual = N/A  
❌ **No emotional expression** → Emotional = N/A  
❌ **Question not about preference** → Preference = N/A  

**This is normal!** Not all questions require all three aspects.

---

## 🔧 Technical Details

### **Judge Prompt (Simplified)**
```
You are judging if an AI matches a human's response.
The HUMAN response is the GROUND TRUTH.

Question: "{question}"
AI: "{ai_response}"
Human: "{human_response}"

IMPORTANT:
1. Ignore typos  
2. Ignore slang/tone
3. Focus: Do they BOTH prefer the same thing?

Output:
INTENT_MATCH: YES or NO
SEMANTIC_SCORE: 0-100
Factual: 0-100
Emotional: 0-100
Preference: 0-100
```

### **Post-Processing**
```python
# Parse LLM output
result = parse_judge_output(llm_response)

# Apply smart boosting
if result.intent_match and result.score < 85:
    result.score = max(85, result.score)
    
if result.preference_alignment >= 80 and result.score < 80:
    result.score = max(80, result.score)

# Return final score
return result
```

---

## 🎯 Summary

**Your score reflects:**
1. ✅ How well AI captured your CORE MEANING
2. ✅ Not exact words, but INTENT
3. ✅ Typos and slang are ignored
4. ✅ Automatic boosts for clear matches

**Goal:** Train the AI to think and respond like YOU!

---

## 💡 Tips for High Scores

1. **Be yourself** - Respond authentically
2. **Don't overthink** - Natural responses work best
3. **Express your preference** - Clear opinions score well
4. **Typos are fine** - The judge ignores them
5. **Short or long** - Both work, focus on meaning

---

## 🔍 View Your Score Details

On the judge results page (`/judge`), click:

📚 **"How Scoring Works"** - See this methodology  
🔍 **"Show Raw Judge Output"** - See LLM's actual response  
🧮 **"Score Calculation"** - See if boosting was applied  

---

**Questions?** Check the raw judge output to see exactly what the LLM said!
