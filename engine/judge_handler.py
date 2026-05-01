"""
LLM Judge Handler
Judges semantic similarity between AI and human responses.
Primary: Ollama (LLaMA 3.2) for local use.
Fallback: Gemini API — used automatically when Ollama is unavailable (e.g. on Render).
"""

import os
import requests
import json
from typing import Dict, Optional


class JudgeHandler:
    """Handler for LLM-based response judging. Uses Ollama locally, Gemini in production."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3.2"

        # Gemini fallback — uses the same keys already loaded in chat_handler
        self._gemini_model = None
        self._init_gemini_fallback()

    def _init_gemini_fallback(self):
        """Initialize Gemini as a fallback judge."""
        try:
            import google.generativeai as genai
            from dotenv import load_dotenv
            load_dotenv()

            # Try keys in order until one works
            for i in range(1, 5):
                key = os.environ.get(f"GEMINI_API_KEY_{i}", "")
                if key.strip():
                    genai.configure(api_key=key)
                    self._gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                    print(f"[JUDGE] Gemini fallback ready (key {i})", flush=True)
                    break
        except Exception as e:
            print(f"[JUDGE] Gemini fallback init failed: {e}", flush=True)
            self._gemini_model = None

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def judge_responses(
        self,
        user_question: str,
        ai_response: str,
        human_response: str,
        persona_name: str
    ) -> Dict:
        """
        Judge semantic similarity between AI and human responses.
        Tries Ollama first, falls back to Gemini automatically.
        """
        prompt = self._create_judge_prompt(user_question, ai_response, human_response, persona_name)

        # --- Try Ollama first ---
        print("[JUDGE] Checking Ollama connection...", flush=True)
        if self.check_connection():
            result = self._judge_with_ollama(prompt)
            if result is not None:
                return result
            print("[JUDGE] Ollama failed — falling back to Gemini...", flush=True)
        else:
            print("[JUDGE] Ollama not available — using Gemini fallback.", flush=True)

        # --- Gemini fallback ---
        if self._gemini_model:
            return self._judge_with_gemini(prompt)

        # --- Both unavailable ---
        return {
            "error": "No judge available. Ollama is offline and Gemini key is missing.",
            "score": 0,
            "breakdown": {},
            "reasoning": "No LLM judge could be reached."
        }

    # ------------------------------------------------------------------
    # Ollama judge
    # ------------------------------------------------------------------

    def _judge_with_ollama(self, prompt: str) -> Optional[Dict]:
        """Call Ollama. Returns None on failure so caller can fall back."""
        try:
            print("[JUDGE] Sending to Ollama LLaMA 3.2...", flush=True)
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 300,
                    }
                },
                timeout=120
            )
            if response.status_code != 200:
                return None

            judge_output = response.json().get("response", "")
            parsed = self._parse_judge_output(judge_output)
            print(f"[JUDGE] ✅ Ollama judging complete! Score: {parsed.get('score', 0)}/100", flush=True)
            return parsed

        except requests.exceptions.Timeout:
            print("[JUDGE] Ollama timeout.", flush=True)
            return None
        except Exception as e:
            print(f"[JUDGE] Ollama error: {e}", flush=True)
            return None

    # ------------------------------------------------------------------
    # Gemini fallback judge
    # ------------------------------------------------------------------

    def _judge_with_gemini(self, prompt: str) -> Dict:
        """Call Gemini as judge fallback."""
        try:
            print("[JUDGE] Sending to Gemini judge...", flush=True)
            response = self._gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 300,
                }
            )
            judge_output = response.text.strip() if response.text else ""
            parsed = self._parse_judge_output(judge_output)
            parsed["_judge"] = "gemini-fallback"
            print(f"[JUDGE] ✅ Gemini judging complete! Score: {parsed.get('score', 0)}/100", flush=True)
            return parsed

        except Exception as e:
            print(f"[JUDGE] Gemini judge error: {e}", flush=True)
            return {
                "error": f"Gemini judge failed: {str(e)[:100]}",
                "score": 0,
                "breakdown": {},
                "reasoning": "Judge error"
            }

    # ------------------------------------------------------------------
    # Prompt + parser (unchanged from original)
    # ------------------------------------------------------------------

    def _create_judge_prompt(
        self,
        question: str,
        ai_resp: str,
        human_resp: str,
        persona_name: str
    ) -> str:
        """Ultra-strict judge prompt for X or Y selection."""
        return f"""You are a strict, logical judge. Compare the AI and Human answers.

Q: "{question}"
AI: "{ai_resp}"
Human: "{human_resp}"

--- RULES ---
1. IF "X or Y" QUESTION:
   - Check which option they chose.
   - If AI chose X and Human chose Y -> PREFERENCE: 0 (They Disagree).
   - If AI chose X and Human chose X -> PREFERENCE: 100 (They Agree).

2. NO HALLUCINATIONS:
   - "Lion" is NOT "Mummy".
   - "Apple" is NOT "Car".
   - If the words are different and mean different things, Score 0.

3. SCORING GUIDELINES:
   - PREFERENCE: 100 (Same Choice) or 0 (Different Choice). Use 50 only for partial match.
   - EMOTIONAL: 80 usually. 0 if completely unrelated words.
   - FACTUAL: 
     - If they give NO reasons (just 1 word answer) -> Score 0.
     - If they give DIFFERENT reasons -> Score 0.
     - If they give SAME reasons -> Score 80.
   - CONSISTENCY:
     - If PREFERENCE < 50, INTENT_MATCH must be NO.
     - If PREFERENCE > 50, INTENT_MATCH must be YES.

OUTPUT ONLY THIS FORMAT:
PREFERENCE: [0-100]
EMOTIONAL: [0-100]
FACTUAL: [0-100]
INTENT_MATCH: [YES/NO]
REASONING: [1 sentence explanation]"""

    def _parse_judge_output(self, output: str) -> Dict:
        """Parse strict judge output format."""
        output = output.strip()
        print(f"[JUDGE DEBUG] Raw LLM output:\n{output}\n", flush=True)

        result = {
            "score": 0,
            "intent_match": False,
            "breakdown": {
                "preference_alignment": 0,
                "emotional_alignment": 0,
                "factual_alignment": 0
            },
            "reasoning": "",
            "raw_output": output
        }

        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            upper_line = line.upper()

            if "PREFERENCE:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    result["breakdown"]["preference_alignment"] = int(''.join(filter(str.isdigit, score_part)))
                except: pass

            elif "EMOTIONAL:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    result["breakdown"]["emotional_alignment"] = int(''.join(filter(str.isdigit, score_part)))
                except: pass

            elif "FACTUAL:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    result["breakdown"]["factual_alignment"] = int(''.join(filter(str.isdigit, score_part)))
                except: pass

            elif "INTENT_MATCH:" in upper_line:
                if "YES" in upper_line:
                    result["intent_match"] = True

            elif "REASONING:" in upper_line:
                try:
                    result["reasoning"] = line.split(':', 1)[1].strip()
                except:
                    result["reasoning"] = line

        # Consistency corrections
        p = result["breakdown"]["preference_alignment"]
        if p <= 40 and result["intent_match"]:
            print(f"[JUDGE] ⚠️ Preference={p} (Low) but Intent=YES → correcting to NO.")
            result["intent_match"] = False
        elif p >= 80 and not result["intent_match"]:
            print(f"[JUDGE] ⚠️ Preference={p} (High) but Intent=NO → correcting to YES.")
            result["intent_match"] = True

        # Final score formula (unchanged)
        p = result["breakdown"]["preference_alignment"]
        e = result["breakdown"]["emotional_alignment"]
        f = result["breakdown"]["factual_alignment"]

        if f == 0:
            final_score = int((p * 0.7) + (e * 0.3))
            print(f"[JUDGE] Factual=0 → P:{p}×0.7 + E:{e}×0.3 = {final_score}", flush=True)
        else:
            final_score = int((p * 0.6) + (e * 0.3) + (f * 0.1))
            print(f"[JUDGE] P:{p} E:{e} F:{f} → {final_score}", flush=True)

        result["score"] = final_score
        return result
