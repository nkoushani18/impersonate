// QUICK FIX: Replace the entire submitResponse function in human.html
// Location: Around line 903-970

async function submitResponse(questionId) {
    const input = document.getElementById(`input-${questionId}`);
    const resultDiv = document.getElementById(`result-${questionId}`);
    const button = event.target.closest('.submit-btn');

    const response = input.value.trim();
    if (!response) {
        resultDiv.innerHTML = '<div class="error-message">⚠️ Please enter a response</div>';
        setTimeout(() => resultDiv.innerHTML = '', 3000);
        return;
    }

    // IMPORTANT: Open judging page FIRST
    const judgingWindow = window.open(`/judging/live?q=${questionId}`, '_blank', 'width=900,height=700');

    // Update UI
    button.disabled = true;
    button.innerHTML = '<span>✅ Submitted!</span>';
    input.disabled = true;
    resultDiv.innerHTML = '<div class="success-message" style="padding: 20px; border-radius: 12px; background: rgba(40, 200, 64, 0.15); border: 2px solid rgba(40, 200, 64, 0.4); color: var(--cream);"><strong>✅ Response Submitted!</strong><br><br>Check the new tab for live judging animation and results.</div>';

    try {
        // Submit in background
        const apiResponse = await fetch('/api/human/respond', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question_id: questionId,
                response: response
            })
        });

        const data = await apiResponse.json();

        if (!data.success) {
            // Show error but keep judging window open
            resultDiv.innerHTML = `<div class="error-message">⚠️ Error: ${data.error || 'Unknown error'}<br><br>Check the judging tab for status.</div>`;
            console.error('Submit error:', data.error);
        }
        // If success, results will appear in the judging window

    } catch (error) {
        resultDiv.innerHTML = '<div class="error-message">⚠️ Network error. Check console and judging tab.</div>';
        console.error('Submission error:', error);
    }
}
