/* 
 * QUICK FIX FOR JUDGING_LIVE.HTML
 * Replace line 455 with these lines:
 */

// OLD LINE 455:
// const result = data.comparisons?.find(c => c.question_id === questionId);

// NEW CODE (replace line 441-472):
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

            console.log('[POLL] Response:', data);

            // Check latest_comparison first
            let result = null;

            if (data.latest_comparison && data.latest_comparison.judge_result) {
                console.log('[POLL] Found latest_comparison');
                result = data.latest_comparison;
            } else if (data.history && data.history.length > 0) {
                console.log('[POLL] Checking history, length:', data.history.length);
                result = data.history[data.history.length - 1]; // Get most recent
            }

            if (result && result.judge_result) {
                console.log('[POLL] ✅ Found result! Score:', result.judge_result.score);
                clearInterval(interval);
                showResult(result);
            } else if (attempts >= maxAttempts) {
                console.log('[POLL] ❌ Timeout');
                clearInterval(interval);
                showError();
            }
        } catch (error) {
            console.error('[POLL] Error:', error);
            if (attempts >= maxAttempts) {
                clearInterval(interval);
                showError();
            }
        }
    }, 1000);
}

/*
 * THIS SIMPLIFIED VERSION:
 * 1. Just gets the latest_comparison (most recent result)
 * 2. Falls back to most recent in history
 * 3. Doesn't try to match question IDs
 * 4. Assumes the most recent result is what we want
 */
