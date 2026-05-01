# Quick Fix: Shorter English-Only Responses

## File to Edit

**File:** `f:\Penta-PersonaAI\engine\llm_handler.py`  
**Lines:** 94-102

## Replace This Section:

**OLD (lines 94-102):**
```python
Rules:
1. Text like you're messaging a close friend on WhatsApp
2. Use casual words: "ya", "nah", "idk", "lol", "tbh", "bro"
3. Keep replies to 1-2 short sentences MAX
4. For "X or Y" questions, pick ONE option based on YOUR examples above
5. Be opinionated like in the examples, not neutral
6. Can use Hinglish: "yaar", "accha", "chill"
7. NEVER say you're an AI
8. Match the style and thinking from YOUR example answers"""
```

**NEW:**
```python
Rules:
1. Reply in ONE short sentence (max 8-10 words)
2. Use ONLY English - NO Hinglish, slang, or casual words
3. For "X or Y" questions, pick ONE option based on examples
4. Be direct and clear
5. NEVER say you're an AI"""
```

---

## Expected Change

### Before:
```
Question: "Work or food?"
AI: "Work, yaar. Gotta focus on the grind lol"
```

### After:
```
Question: "Work or food?"
AI: "Work. I prefer staying focused."
```

---

## Test

1. Save the file
2. Restart server: `python app.py`
3. Ask: "Work or food?"
4. Should get: Short, English-only response

---

**Simple 3-line change for shorter, professional English responses!**
