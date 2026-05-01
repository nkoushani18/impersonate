# 🎬 Auto-Open Judging Page Fix

## Quick Manual Edit Needed

The judging page is created, but you need to add ONE line to `human.html`:

---

## 📝 Edit Location

File: `f:\Penta-PersonaAI\templates\human.html`  
Function: `submitResponse` (around line 903-930)

---

## ✏️ What to Add

Find this line (around line 915):
```javascript
button.disabled = true;
```

**Add this line RIGHT BEFORE it:**
```javascript
// Auto-open judging page in new tab
window.open(`/judging/live?q=${questionId}`, '_blank');
```

---

## 📄 Full Context

**Before:**
```javascript
async function submitResponse(questionId) {
    const input = document.getElementById(`input-${questionId}`);
    const resultDiv = document.getElementById(`result-${questionId}`);
    const button = event.target.closest('.submit-btn');

    const response = input.value.trim();
    if (!response) {
        resultDiv.innerHTML = '<div class="error-message">Please enter a response</div>';
        setTimeout(() => resultDiv.innerHTML = '', 3000);
        return;
    }

    button.disabled = true;  // ← ADD NEW CODE BEFORE THIS LINE
    button.innerHTML = '<span>⚖️ Judging...</span>';
```

**After:**
```javascript
async function submitResponse(questionId) {
    const input = document.getElementById(`input-${questionId}`);
    const resultDiv = document.getElementById(`result-${questionId}`);
    const button = event.target.closest('.submit-btn');

    const response = input.value.trim();
    if (!response) {
        resultDiv.innerHTML = '<div class="error-message">Please enter a response</div>';
        setTimeout(() => resultDiv.innerHTML = '', 3000);
        return;
    }

    // Auto-open judging page in new tab
    window.open(`/judging/live?q=${questionId}`, '_blank');

    button.disabled = true;
    button.innerHTML = '<span>⚖️ Submitted!</span>';  // ← Also change to "Submitted!"
    resultDiv.innerHTML = '<div class="loading">Check the judging tab for results...</div>';
```

---

## 🎯 Complete Flow After This Fix

1. **User submits on `/human`**
   ```
   Click "Submit" → window.open() fires
   ```

2. **NEW TAB OPENS: `/judging/live?q=<question_id>`**
   ```
   Shows animated scales
   "Judging..."
   Dots pulsing
   ```

3. **Backend judges** (10-30 seconds)

4. **Judging page polls every second** for results

5. **Animation fades out**

6. **Beautiful result card reveals:**
   ```
   ╔═══════════════════════════════════╗
   ║    Judgment Complete              ║
   ║                                   ║
   ║          85/100                   ║
   ║                                   ║
   ║      ✅ Intent Match              ║
   ║                                   ║
   ║  [Factual] [Emotional] [Prefer]  ║
   ║     N/A        N/A        90      ║
   ║                                   ║
   ║  🧠 Judge's Reasoning             ║
   ║  Both responses indicate...       ║
   ║                                   ║
   ║      [Close & Return]             ║
   ╚═══════════════════════════════════╝
   ```

7. **User closes tab** → Back to `/human` to answer more!

---

## ✅ What's Already Done

✅ Created `/judging/live` page with animations  
✅ Added Flask route  
✅ Polling logic to get results  
✅ Beautiful minimal design  
✅ Auto-reveals score after judging  

❌ Just need the `window.open()` call added!

---

## 🚀 Test Steps After Fix

1. Restart server: `python app.py`
2. Go to `/human`
3. Answer a question
4. Click "Submit Response"
5. **NEW TAB SHOULD AUTO-OPEN** showing judging animation
6. Watch the beautiful animation
7. See the result reveal after ~10-30 seconds

---

## 💡 Alternative: Test the Page Manually

Don't want to edit? Test the page directly:

1. Submit a response on `/human` first (to create a pending judgment)
2. Manually open: `http://localhost:5000/judging/live?q=<question_id>`
   - Replace `<question_id>` with an actual ID from your session
3. You'll see the animation and then the result!

---

**Almost done! Just add that one `window.open()` line!** 🎉
