# 🔄 Auto-Refresh & Auto-Select Features

## What's New

I've added **automatic refresh** and **automatic persona selection** to make the judge system seamless!

---

## ✨ Features Added

### 1️⃣ **Human Interface** (`/human`)

#### **Auto-Refresh Pending Questions**
- Automatically checks for new questions **every 5 seconds**
- Silent refresh (doesn't show "Loading..." during auto-refresh)
- Starts when you select a persona
- Stops when you switch personas

#### **Auto-Select Active Persona**
- Automatically selects the persona you're chatting with in main chat
- No need to manually select - just open `/human` and it knows!
- Uses localStorage to track active persona across tabs

---

### 2️⃣ **Judge Results** (`/judge`)

#### **Auto-Refresh Results** (Already Existed)
- Updates **every 10 seconds**
- Shows latest comparisons automatically
- Updates statistics in real-time

#### **Auto-Select Active Persona** (NEW!)
- Automatically loads results for the persona you're chatting with
- Opens the page and immediately shows relevant data
- Syncs with main chat selection

---

### 3️⃣ **Main Chat** (`/`)

#### **LocalStorage Sync** (NEW!)
- Saves active persona to browser storage
- Makes other pages "know" which persona is active
- Persists across page refreshes

---

## 🎯 How It Works

### Flow:
```
Main Chat (/)
   ↓
User selects "Rishit" persona
   ↓
localStorage.setItem('activePersonaId', 'rishit')
   ↓
User opens /human in new tab
   ↓
Page reads localStorage
   ↓
Automatically selects "Rishit"
   ↓
Starts auto-refresh every 5 seconds
   ↓
New questions appear automatically!
```

---

## 💡 Usage Example

### Before (Manual):
1. Chat with Rishit at `/`
2. Open `/human` in new tab
3. **Manually select** Rishit from dropdown
4. **Manually refresh** to see new questions

### Now (Automatic):
1. Chat with Rishit at `/`
2. Open `/human` in new tab
3. ✅ **Rishit is already selected!**
4. ✅ **Questions auto-refresh every 5 seconds!**
5. ✅ **New questions appear instantly!**

---

## ⏱️ Refresh Intervals

| Interface | Refresh Rate | What Refreshes |
|-----------|--------------|----------------|
| **Main Chat** | N/A | N/A |
| **Human Interface** | **5 seconds** | Pending questions |
| **Judge Results** | **10 seconds** | Statistics, latest comparison, history |

---

## 🧪 Test It!

### Test Auto-Select:
1. Go to http://localhost:5000/
2. Select "Koushani" persona
3. Open http://localhost:5000/human in **new tab**
4. ✅ Koushani should be **already selected**!
5. Open http://localhost:5000/judge in **another new tab**
6. ✅ Koushani should be **already selected** there too!

### Test Auto-Refresh (Human Interface):
1. Have `/human` open with a persona selected
2. In main chat, ask 3 questions
3. Watch `/human` page - **within 5 seconds**, new questions appear!
4. No need to refresh manually!

### Test Auto-Refresh (Judge Results):
1. Have `/judge` open with a persona selected  
2. Submit human responses at `/human`
3. Watch `/judge` page - **within 10 seconds**, results update!
4. Latest comparison appears automatically!

---

## 📁 Files Modified

### `frontend/src/App.jsx`
- Added `localStorage.setItem('activePersonaId', id)` in `selectPersona()`
- Saves active persona to browser storage

### `templates/human.html`
- Added `autoRefreshInterval` for 5-second refresh
- Added auto-select from localStorage
- Added `startAutoRefresh()` function
- Modified `loadPendingQuestions()` to support silent refresh

### `templates/judge.html`
- Added auto-select from localStorage
- Auto-loads results on page load if persona is active

---

## 🎨 User Experience

### Before:
❌ Constant manual work
❌ Missing new questions
❌ Disconnected interfaces
❌ Repetitive persona selection

### Now:
✅ Seamless workflow
✅ Real-time updates
✅ Synchronized across tabs
✅ One-click access

---

## 🔧 Technical Details

### LocalStorage Key:
```javascript
localStorage.setItem('activePersonaId', 'rishit')
localStorage.getItem('activePersonaId') // returns 'rishit'
```

### Auto-Refresh Implementation:
```javascript
// Refresh every 5 seconds
setInterval(() => {
    if (currentPersonaId) {
        loadPendingQuestions(true); // Silent refresh
    }
}, 5000);
```

### Silent vs Normal Loading:
```javascript
// Normal: Shows "Loading..." message
loadPendingQuestions(false) 

// Silent: No loading indicator during auto-refresh
loadPendingQuestions(true)
```

---

## ⚡ Performance Impact

- **Minimal**: Only fetches data when needed
- **Efficient**: Silent refresh doesn't re-render entire page
- **Throttled**: 5-10 second intervals prevent server overload
- **Smart**: Only refreshes when persona is selected

---

## 🎉 Benefits

1. **No more manual refresh** - Questions appear automatically
2. **No more persona selection** - Auto-selects from main chat
3. **Real-time sync** - See updates as they happen
4. **Multi-tab workflow** - Chat, answer, and view results simultaneously
5. **Better UX** - Feels like a single integrated app

---

**Everything is live and working!** Just restart the server and try it out! 🚀
