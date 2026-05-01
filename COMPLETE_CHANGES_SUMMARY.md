# 🎯 COMPLETE SUMMARY - All Changes Needed

## What You Want

1. ✅ **Intent-First Formula** - Intent 60%, Tone 30%, Factual 10%
2. ✅ **Show Calculation** - Display formula, not just scores
3. ✅ **Hide Zeros** - Only show non-zero components
4. ✅ **Shorter English Responses** - No Hinglish, max 10 words

---

## 📋 Files to Edit

### **File 1: `llm_handler.py`**
**Location:** Lines 94-102  
**Change:** Remove Hinglish, shorter responses

**FROM:**
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

**TO:**
```python
Rules:
1. Reply in ONE short sentence (max 10 words)
2. Use ONLY English - NO Hinglish or slang
3. For "X or Y" questions, pick ONE based on examples
4. Be direct and clear
5. NEVER say you're an AI"""
```

---

### **File 2: `judging_live.html`**
**Location:** Lines 497-512  
**Change:** Replace breakdown grid with formula calculation

**Find this block:**
```javascript
// Breakdown
const breakdownGrid = document.getElementById('breakdownGrid');
breakdownGrid.innerHTML = `
    <div class="breakdown-item">
        <div class="breakdown-label">Factual</div>
        <div class="breakdown-value">${judge.breakdown.factual_alignment || 'N/A'}</div>
    </div>
    <div class="breakdown-item">
        <div class="breakdown-label">Emotional</div>
        <div class="breakdown-value">${judge.breakdown.emotional_alignment || 'N/A'}</div>
    </div>
    <div class="breakdown-item">
        <div class="breakdown-label">Preference</div>
        <div class="breakdown-value">${judge.breakdown.preference_alignment || 'N/A'}</div>
    </div>
`;
```

**Replace with this COMPLETE block:**
```javascript
// Breakdown - Formula Calculation
const breakdownGrid = document.getElementById('breakdownGrid');

const intent = judge.breakdown.preference_alignment || 0;
const tone = judge.breakdown.emotional_alignment || 0;
const factual = judge.breakdown.factual_alignment || 0;

// Build only non-zero components
let calcHTML = '';
let formulaParts = [];

if (intent > 0) {
    const pts = (intent * 0.6).toFixed(1);
    formulaParts.push(`(${intent} × 0.6)`);
    calcHTML += `
        <div style="margin-bottom: 16px; display: flex; justify-content: space-between;">
            <span style="color: rgba(248,244,227,0.7);">🎯 Intent (60%)</span>
            <span style="font-size: 1.4rem; font-weight: 700; color: #8ac4ff;">
                ${intent} × 0.6 = ${pts} pts
            </span>
        </div>
    `;
}

if (tone > 0) {
    const pts = (tone * 0.3).toFixed(1);
    formulaParts.push(`(${tone} × 0.3)`);
    calcHTML += `
        <div style="margin-bottom: 16px; display: flex; justify-content: space-between;">
            <span style="color: rgba(248,244,227,0.7);">🎭 Tone (30%)</span>
            <span style="font-size: 1.4rem; font-weight: 700; color: #8ac4ff;">
                ${tone} × 0.3 = ${pts} pts
            </span>
        </div>
    `;
}

if (factual > 0) {
    const pts = (factual * 0.1).toFixed(1);
    formulaParts.push(`(${factual} × 0.1)`);
    calcHTML += `
        <div style="margin-bottom: 16px; display: flex; justify-content: space-between;">
            <span style="color: rgba(248,244,227,0.7);">📚 Factual (10%)</span>
            <span style="font-size: 1.4rem; font-weight: 700; color: #8ac4ff;">
                ${factual} × 0.1 = ${pts} pts
            </span>
        </div>
    `;
}

const total = ((intent * 0.6) + (tone * 0.3) + (factual * 0.1)).toFixed(0);
const formulaStr = formulaParts.join(' + ') + ` = ${total}`;

breakdownGrid.innerHTML = `
    <div class="breakdown-item" style="grid-column: 1/-1; padding: 30px; text-align: left;">
        <div style="font-size: 1.1rem; color: var(--orange); margin-bottom: 20px; font-weight: 600;">
            📐 Score Calculation
        </div>
        
        ${calcHTML}
        
        <div style="border-top: 2px solid var(--orange); margin: 20px 0;"></div>
        
        <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
            <span style="font-size: 1.2rem; color: var(--cream); font-weight: 700;">TOTAL</span>
            <span style="font-size: 2rem; font-weight: 900; color: var(--orange);">
                ${total}/100
            </span>
        </div>
        
        <div style="padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: rgba(248,244,227,0.5); margin-bottom: 6px;">Formula</div>
            <div style="font-family: monospace; color: var(--orange);">
                ${formulaStr}
            </div>
        </div>
    </div>
`;
```

---

## 🎯 Result

### **Before:**
```
FACTUAL: N/A
EMOTIONAL: 5
PREFERENCE: 92

FINAL: 95/100
```

### **After:**
```
📐 Score Calculation

🎯 Intent (60%)    92 × 0.6 = 55.2 pts
🎭 Tone (30%)       5 × 0.3 = 1.5 pts
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL              57/100

Formula: (92 × 0.6) + (5 × 0.3) = 57
```

---

## ✅ Testing

1. **Edit both files**
2. **Restart server:** `python app.py`
3. **Submit response on `/human`**
4. **Check judging tab** - should show formula!
5. **Ask in chat** - should get short English response!

---

## 📊 For Research Paper

Now you can cite:
- **Scoring Formula:** (Intent × 0.6) + (Tone × 0.3) + (Factual × 0.1)
- **Intent Weight:** 60% (primary)
- **Tone Weight:** 30% (secondary)
- **Factual Weight:** 10% (tertiary)
- **Transparent Calculation:** Shown in UI

Perfect for methodological rigor!

---

**Apply these 2 file changes and you're done!** 🎉
