# ✅ JUDGING PAGE FIX - Quick Solution

## Problem

Backend judged successfully (Score: 95/100) but frontend showed "Judging Timed Out"

**Root Cause:** API returns `latest_comparison` and `history`, not `comparisons`

---

## 🔧 Quick Fix

**File:** `f:\Penta-PersonaAI\templates\judging_live.html`  
**Lines:** Replace the `pollForResults()` function (lines 441-472)

### **Copy This Entire Function:**

```javascript
async function pollForResults() {
    const maxAttempts = 60; 
    let attempts = 0;

    const interval = setInterval(async () => {
        attempts++;
        console.log(`[POLL] Attempt ${attempts}/60`);

        try {
            const personaId = localStorage.getItem('activePersonaId');
            const response = await fetch(`/api/judge/results?persona_id=${personaId}`);
            const data = await response.json();
            
            console.log('[POLL] API Response:', data);

            // Just get the latest_comparison (most recent result)
            let result = data.latest_comparison;

            if (result && result.judge_result) {
                console.log('[POLL] ✅ Found result! Score:', result.judge_result.score);
                clearInterval(interval);
                showResult(result);
            } else if (attempts >= maxAttempts) {
                console.log('[POLL] ❌ Timeout');
                clearInterval(interval);
                showError();
            } else {
                console.log('[POLL] No result yet, waiting...');
            }
        } catch (error) {
            console.error('[POLL] Error:', error);
            if (attempts >= maxAttempts) {
                clearInterval(interval);
                showError();
            }
        }
    }, 1000); // Poll every second
}
```

---

## 🎯 What This Does

**Before:** Looked for `data.comparisons.find(...)` ← field doesn't exist!  
**After:** Uses `data.latest_comparison` ← correct field!

---

## 🚀 Test Steps

1. **Edit `judging_live.html`** - Replace the function
2. **Save the file**
3. **Go to `/human`**
4. **Hard refresh** (Ctrl+Shift+R)
5. **Submit a response**
6. **Watch the new tab:**
   - Animation should play
   - After ~10-30s → **Score reveals!** ✅

---

## 📊 Expected Console Output

```
[POLL] Attempt 1/60
[POLL] API Response: {latest_comparison: {...}, history: [...]}
[POLL] No result yet, waiting...

[POLL] Attempt 2/60
[POLL] API Response: {latest_comparison: {...}, history: [...]}
[POLL] No result yet, waiting...

...

[POLL] Attempt 15/60
[POLL] API Response: {latest_comparison: {judge_result: {score: 95}}}
[POLL] ✅ Found result! Score: 95
```

Then the beautiful result card reveals!

---

## Alternative: Test Manually

Don't want to edit yet? **Test the Judge Results page:**

1. Submit a response on `/human`
2. Wait ~10-30s for backend to finish
3. Go to: `http://localhost:5000/judge`
4. You'll see the score there!

The live judging page will work once you apply the fix.

---

**Apply this fix and the animation will work perfectly!** 🎉✨
