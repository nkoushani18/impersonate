# ✅ Testing Checklist for LLM Judge System

## Pre-Testing Setup

- [ ] Ollama is installed
- [ ] LLaMA 3.2 model is downloaded (`ollama pull llama3.2`)
- [ ] Ollama server is running (`ollama serve`)
- [ ] Flask app is running (`python app.py`)
- [ ] Server accessible at http://localhost:5000

---

## Test 1: Main Chat Interface (/)

### Steps:
1. [ ] Open http://localhost:5000/
2. [ ] Select a persona (e.g., "Rishit")
3. [ ] Ask a simple question: "school or college?"
4. [ ] Verify AI responds
5. [ ] Check terminal logs show: `[JUDGE] Created comparison session: rishit_X_XXXXX`

### Expected Result:
✅ AI responds with persona-appropriate answer
✅ Console shows question session was created

---

## Test 2: Human Interface (/human)

### Steps:
1. [ ] Open http://localhost:5000/human in new tab/window
2. [ ] Select the same persona (e.g., "Rishit")
3. [ ] Verify pending questions appear
4. [ ] Check that:
   - [ ] User's question is displayed
   - [ ] AI's response is shown
   - [ ] Text input box is available
5. [ ] Type a response (e.g., "college bro")
6. [ ] Click "Submit Response"
7. [ ] Wait for judge evaluation (5-10 seconds)

### Expected Result:
✅ Question appears immediately
✅ AI response is visible
✅ After submission, score appears (0-100)
✅ Breakdown shows: Factual, Emotional, Preference alignment
✅ Success message with green background
✅ "Submit Response" button disappears
✅ Input field becomes disabled

---

## Test 3: Judge Results (/judge)

### Steps:
1. [ ] Open http://localhost:5000/judge in new tab/window
2. [ ] Select persona (e.g., "Rishit")
3. [ ] Verify statistics section shows:
   - [ ] Total Comparisons
   - [ ] Average Score
   - [ ] Highest Score
   - [ ] Latest Score
4. [ ] Verify latest comparison section shows:
   - [ ] User question
   - [ ] AI response
   - [ ] Human response
   - [ ] Overall score (large display)
   - [ ] Intent match badge (✅ or ❌)
   - [ ] Breakdown scores (3 types)
   - [ ] Judge's reasoning
   - [ ] Core intents (AI and Human)
5. [ ] Verify history section shows:
   - [ ] List of previous comparisons
   - [ ] Each with question, both responses, score
6. [ ] Wait 10 seconds and verify auto-refresh works

### Expected Result:
✅ All statistics display correctly
✅ Latest comparison shows complete details
✅ Score breakdown is visible
✅ Judge's reasoning is displayed
✅ History shows all past comparisons
✅ Page auto-updates without manual refresh

---

## Test 4: Multiple Questions Flow

### Steps:
1. [ ] In main chat (/), ask 3 different questions:
   - [ ] "Do you like programming?"
   - [ ] "Coffee or tea?"
   - [ ] "What's your dream job?"
2. [ ] Switch to human interface (/human)
3. [ ] Verify all 3 questions appear
4. [ ] Answer all 3 questions
5. [ ] Switch to judge results (/judge)
6. [ ] Verify all 3 comparisons are in history

### Expected Result:
✅ All questions appear in human interface
✅ Can submit multiple responses in sequence
✅ Judge results shows all 3 in history
✅ Statistics update accordingly

---

## Test 5: Different Personas

### Steps:
1. [ ] Main chat: Select "Koushani" persona
2. [ ] Ask: "What do you think about relationships?"
3. [ ] Human interface: Select "Koushani"
4. [ ] Verify only Koushani's questions appear (not Rishit's)
5. [ ] Answer as Koushani
6. [ ] Judge results: Select "Koushani"
7. [ ] Verify only Koushani's comparisons shown

### Expected Result:
✅ Questions are persona-specific
✅ Rishit's questions don't appear in Koushani's queue
✅ Results are separated by persona

---

## Test 6: Semantic Similarity Evaluation

### Test Case A: Very Similar Responses
**Question**: "Do you prefer iOS or Android?"
**AI Response**: "iOS is better, much smoother"
**Human Response**: "iPhone for sure, way better UX"

**Expected Score**: 85-100 (high similarity)

---

### Test Case B: Opposite Responses
**Question**: "Morning person or night owl?"
**AI Response**: "I love mornings, early bird all the way"
**Human Response**: "Night owl, I hate waking up early"

**Expected Score**: 0-30 (opposite intents)

---

### Test Case C: Same Intent, Different Style
**Question**: "Pizza or burger?"
**AI Response**: "Pizza yaar, no competition"
**Human Response**: "Pizza, obviously"

**Expected Score**: 90-100 (same intent, different wording)

---

## Test 7: Edge Cases

### No Questions Yet
1. [ ] Open human interface before asking any questions
2. [ ] Select a persona
3. [ ] Should see: "No pending questions yet"

### No Comparisons Yet
1. [ ] Open judge results before any human responses
2. [ ] Select a persona
3. [ ] Should see: "No comparisons yet"

### Already Answered Question
1. [ ] Answer a question in human interface
2. [ ] Refresh the page
3. [ ] That question should not appear again

---

## Test 8: API Endpoints

### Get Pending Questions
```bash
curl http://localhost:5000/api/human/pending?persona_id=rishit
```
**Expected**: JSON with list of pending questions

### Submit Human Response
```bash
curl -X POST http://localhost:5000/api/human/respond \
  -H "Content-Type: application/json" \
  -d '{"question_id": "rishit_1_1234567890", "response": "test response"}'
```
**Expected**: JSON with success=true and judge_result

### Get Judge Results
```bash
curl http://localhost:5000/api/judge/results?persona_id=rishit
```
**Expected**: JSON with latest_comparison, history, statistics

---

## Test 9: Ollama Connection Issues

### Simulate Ollama Down
1. [ ] Stop Ollama server (Ctrl+C on `ollama serve`)
2. [ ] Submit a human response
3. [ ] Should see error about Ollama connection

### Recovery
1. [ ] Restart Ollama (`ollama serve`)
2. [ ] Submit response again
3. [ ] Should work correctly

---

## Test 10: Server Restart (Data Persistence)

1. [ ] Create several comparisons
2. [ ] Note the statistics
3. [ ] Stop Flask server (Ctrl+C)
4. [ ] Restart Flask server (`python app.py`)
5. [ ] Check judge results

**Expected**: 
⚠️ All history is CLEARED (in-memory storage)
✅ Can start fresh comparisons

---

## Performance Checks

- [ ] Human response submission takes 5-15 seconds (judge evaluation time)
- [ ] Judge results page auto-refreshes every 10 seconds
- [ ] No errors in browser console
- [ ] No errors in Flask terminal logs
- [ ] Ollama terminal shows activity when judging

---

## Visual Checks

### Human Interface (/human)
- [ ] Purple gradient background
- [ ] White cards with proper spacing
- [ ] Question cards have blue left border
- [ ] Submit button has gradient background
- [ ] Success message is green
- [ ] Score is displayed prominently

### Judge Results (/judge)
- [ ] Pink gradient background
- [ ] Stats cards in grid layout
- [ ] Big score display (48px font)
- [ ] Breakdown items have borders
- [ ] History items have timestamps
- [ ] Intent badges (green for match, red for mismatch)

---

## Final Verification

- [ ] All 3 interfaces work independently
- [ ] Can switch between tabs without issues
- [ ] Questions flow correctly through the system
- [ ] Judge evaluations are reasonable
- [ ] History accumulates correctly
- [ ] Statistics calculate properly
- [ ] No crashes or freezes
- [ ] Browser and server logs are clean

---

## Common Issues & Solutions

### Issue: "persona_id required" error
**Solution**: Make sure to select a persona from dropdown first

### Issue: Questions not appearing in human interface
**Solution**: Ask questions in main chat first

### Issue: Judge results empty
**Solution**: Submit human responses first via /human interface

### Issue: Very long wait time for judge
**Solution**: Normal for LLaMA 3.2, can take 10-15 seconds

### Issue: Score seems wrong
**Solution**: Judge focuses on semantic meaning, not exact match. Different wording but same intent = high score

---

## Success Criteria

✅ All 3 interfaces load without errors
✅ Questions flow from chat → human interface
✅ Human responses trigger judge evaluation
✅ Scores display with detailed breakdown
✅ History accumulates correctly
✅ Auto-refresh works on judge page
✅ Multiple personas work independently
✅ Semantic similarity is evaluated correctly

---

**Testing Complete!** 🎉

If all checkboxes are ✅, your LLM Judge System is working perfectly!
