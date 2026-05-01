# STEP-BY-STEP: Implement Proper Scoring Formula

## File 1: Update Judge Prompt

**File:** `f:\Penta-PersonaAI\engine\judge_handler.py`  
**Function:** `_create_judge_prompt` (around line 124-157)

**Replace the entire prompt with:**

```python
def _create_judge_prompt(
    self, 
    question: str, 
    ai_resp: str, 
    human_resp: str,
    persona_name: str
) -> str:
    """Create prompt with weighted scoring formula."""
    prompt = f"""You are judging how well an AI response matches a human's response.

Question: "{question}"
AI Response: "{ai_resp}"
Human Response (GROUND TRUTH): "{human_resp}"

Score each category 0-100:

1. PREFERENCE (50% weight):
   - Do they choose/prefer the same thing?
   - Same choice = 90-100, Similar = 60-80, Different = 0-50

2. EMOTIONAL (30% weight):
   - Same tone/enthusiasm/feeling?
   - Similar emotion = 80-100, Neutral = 50, Opposite = 0-40

3. FACTUAL (20% weight):
   - Same facts/reasons? (Use 0 if no factual claims)
   - Same facts = 80-100, Related = 50-70, Different = 0-40

IGNORE: typos, slang, grammar, formality

OUTPUT EXACTLY:
PREFERENCE: [number 0-100]
EMOTIONAL: [number 0-100]
FACTUAL: [number 0-100]
INTENT_MATCH: YES or NO
REASONING: Brief explanation of each score"""
    return prompt
```

---

## File 2: Update Parser to Calculate Score

**File:** `f:\Penta-PersonaAI\engine\judge_handler.py`  
**Function:** `_parse_judge_output` (around line 159-257)

**Add AFTER extracting all scores (around line 252, before smart scoring):**

```python
# Calculate final score using weighted formula
pref = result["breakdown"].get("preference_alignment", 0)
emot = result["breakdown"].get("emotional_alignment", 0)
fact = result["breakdown"].get("factual_alignment", 0)

# FORMULA: (Preference × 0.5) + (Emotional × 0.3) + (Factual × 0.2)
calculated_score = int((pref * 0.5) + (emot * 0.3) + (fact * 0.2))

# Override LLM's score with our calculated one
result["score"] = calculated_score

print(f"[JUDGE] Formula: ({pref}×0.5) + ({emot}×0.3) + ({fact}×0.2) = {calculated_score}", flush=True)
```

---

## File 3: Add Parsing for New Format

**File:** `f:\Penta-PersonaAI\engine\judge_handler.py`  
**In `_parse_judge_output`, ADD these patterns (around line 200):**

```python
elif line.startswith("PREFERENCE:"):
    try:
        score_str = line.replace("PREFERENCE:", "").strip()
        result["breakdown"]["preference_alignment"] = int(''.join(filter(str.isdigit, score_str)))
    except:
        pass

elif line.startswith("EMOTIONAL:"):
    try:
        score_str = line.replace("EMOTIONAL:", "").strip()
        result["breakdown"]["emotional_alignment"] = int(''.join(filter(str.isdigit, score_str)))
    except:
        pass

elif line.startswith("FACTUAL:"):
    try:
        score_str = line.replace("FACTUAL:", "").strip()
        result["breakdown"]["factual_alignment"] = int(''.join(filter(str.isdigit, score_str)))
    except:
        pass
```

---

## File 4: Display Formula on Frontend

**File:** `f:\Penta-PersonaAI\templates\judging_live.html`  
**In reasoning section (around line 515-518), ADD:**

```html
<!-- Reasoning -->
<div class="reasoning">
    <div class="reasoning-title">🧠 Judge's Reasoning</div>
    <div class="reasoning-text" id="reasoningText">--</div>
    
    <!-- ADD THIS: Formula breakdown -->
    <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(248,244,227,0.2);">
        <div style="font-size: 0.9rem; color: rgba(248,244,227,0.6); margin-bottom: 12px;">
            📐 Scoring Formula
        </div>
        <div style="font-family: 'Courier New', monospace; font-size: 0.95rem; color: var(--orange);">
            Score = (Pref × 50%) + (Emot × 30%) + (Fact × 20%)
        </div>
        <div id="formulaCalc" style="margin-top: 8px; font-size: 0.9rem; color: rgba(248,244,227,0.7);"></div>
    </div>
</div>
```

**And in the `showResult()` function, ADD:**

```javascript
// Show formula calculation
const pref = judge.breakdown.preference_alignment || 0;
const emot = judge.breakdown.emotional_alignment || 0;
const fact = judge.breakdown.factual_alignment || 0;

const calc = `= (${pref} × 0.5) + (${emot} × 0.3) + (${fact} × 0.2) = ${judge.score}`;
document.getElementById('formulaCalc').textContent = calc;
```

---

## Testing

### **Test Case 1:**
```
Question: "Work or food?"
AI: "Work, yaar"
Human: "Work for sure"

Expected:
- Preference: 95
- Emotional: 85
- Factual: 0
- Score = (95×0.5) + (85×0.3) + (0×0.2) = 73
```

### **Test Case 2:**
```
Question: "College or school?"
AI: "College, more freedom"
Human: "College, better opportunities"

Expected:
- Preference: 95
- Emotional: 80
- Factual: 75
- Score = (95×0.5) + (80×0.3) + (75×0.2) = 86
```

---

## Summary

1. **Update prompt** - Ask for PREFERENCE, EMOTIONAL, FACTUAL
2. **Parse new format** - Extract all three scores
3. **Calculate score** - Use weighted formula
4. **Display formula** - Show calculation on frontend

**Result:** Transparent, reproducible, research-ready scoring! ✅

---

Want me to make these changes directly? Or do you want to apply them manually?
