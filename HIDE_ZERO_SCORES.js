// UPDATE: Only show non-zero components in calculation

// Replace in judging_live.html showResult() function:

const intent = judge.breakdown.preference_alignment !== undefined ? judge.breakdown.preference_alignment : 0;
const tone = judge.breakdown.emotional_alignment !== undefined ? judge.breakdown.emotional_alignment : 0;
const factual = judge.breakdown.factual_alignment !== undefined ? judge.breakdown.factual_alignment : 0;

// Build calculation HTML - only include non-zero components
let calculationHTML = '';

if (intent > 0) {
    const pts = (intent * 0.6).toFixed(1);
    calculationHTML += `
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: rgba(248,244,227,0.7);">🎯 Intent (60%)</span>
                <span style="font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #8ac4ff, #86a5d9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ${intent} × 0.6 = ${pts} pts
                </span>
            </div>
        </div>
    `;
}

if (tone > 0) {
    const pts = (tone * 0.3).toFixed(1);
    calculationHTML += `
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: rgba(248,244,227,0.7);">🎭 Tone (30%)</span>
                <span style="font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #8ac4ff, #86a5d9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ${tone} × 0.3 = ${pts} pts
                </span>
            </div>
        </div>
    `;
}

if (factual > 0) {
    const pts = (factual * 0.1).toFixed(1);
    calculationHTML += `
        <div style="margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: rgba(248,244,227,0.7);">📚 Factual (10%)</span>
                <span style="font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #8ac4ff, #86a5d9); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ${factual} × 0.1 = ${pts} pts
                </span>
            </div>
        </div>
    `;
}

// Calculate total
const totalPts = ((intent * 0.6) + (tone * 0.3) + (factual * 0.1)).toFixed(0);

// Build formula string - only non-zero components
let formulaParts = [];
if (intent > 0) formulaParts.push(`${intent} × 0.6`);
if (tone > 0) formulaParts.push(`${tone} × 0.3`);
if (factual > 0) formulaParts.push(`${factual} × 0.1`);
const formulaStr = formulaParts.join(' + ') + ` = ${totalPts}`;

breakdownGrid.innerHTML = `
    <div class="breakdown-item" style="grid-column: 1/-1; text-align: left; padding: 30px;">
        <div style="font-size: 1.1rem; color: var(--orange); margin-bottom: 20px; font-weight: 600;">
            📐 Score Calculation
        </div>
        
        ${calculationHTML}
        
        <!-- Divider -->
        <div style="border-top: 2px solid var(--orange); margin: 20px 0;"></div>
        
        <!-- Total -->
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 1.2rem; color: var(--cream); font-weight: 700;">FINAL SCORE</span>
            <span style="font-size: 2rem; font-weight: 900; background: linear-gradient(135deg, #ff9900, #ffb347); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ${totalPts}/100
            </span>
        </div>
        
        <!-- Formula -->
        <div style="margin-top: 16px; padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; text-align: center;">
            <div style="font-size: 0.85rem; color: rgba(248,244,227,0.5); margin-bottom: 6px;">Formula</div>
            <div style="font-family: 'Courier New', monospace; color: var(--orange); font-size: 0.95rem;">
                ${formulaStr}
            </div>
        </div>
    </div>
`;
