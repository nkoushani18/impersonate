"""
Gemini LLM Handler
Fast, intelligent persona-based responses using Google Gemini.
Supports multi-key + multi-model rotation to maximize free quota.
"""

import google.generativeai as genai
from typing import Dict, List, Optional


# Persona demographic info
PERSONA_DEMOGRAPHICS = {
    "Persona_1": {"gender": "female", "age": 21, "nickname": "Koushani"},
    "Persona_2": {"gender": "male",   "age": 21, "nickname": "Rishit"},
    "Persona_3": {"gender": "male",   "age": 21, "nickname": "Harsh"},
    "Persona_4": {"gender": "female", "age": 21, "nickname": "Manya"},
    "Persona_5": {"gender": "male",   "age": 21, "nickname": "Salil"},
}


class GeminiHandler:
    """
    Handles LLM-based persona responses using Gemini.

    Rotation strategy:
      1. On a quota / rate-limit error, rotate to the next MODEL (same key).
      2. When all models on the current key are exhausted, rotate to the next KEY
         and reset the model index back to 0.
      3. If every key × model combination fails, return a graceful fallback.
    """

    # Models confirmed to have free quota from your AI Studio dashboard.
    # Priority: highest RPD first. 0-quota models excluded.
    # gemini-3.1-flash-lite → 500 RPD (best!) | gemini-2.5-flash → 20 RPD
    MODEL_LIST = [
        "gemini-2.5-flash-lite-preview-06-17",  # 10 RPM | 250K TPM | 20 RPD
        "gemini-2.5-flash",                      #  5 RPM | 250K TPM | 20 RPD
        "gemini-1.5-flash",                      # reliable fallback, large quota
        "gemini-1.5-flash-8b",                   # lightest fallback
    ]

    def __init__(self, api_keys: List[str]):
        """
        Args:
            api_keys: List of Gemini API keys to rotate through.
        """
        if not api_keys:
            raise ValueError("At least one API key is required.")

        self.api_keys = api_keys
        self.current_key_index = 0
        self.current_model_index = 0

        # Total combinations tried in one request — prevents infinite loops
        self._max_rotations = len(api_keys) * len(self.MODEL_LIST)

        self._configure()

        # Generation config — identical to original (temperature, top_p, max_output_tokens unchanged)
        self.generation_config = genai.GenerationConfig(
            temperature=0.9,
            top_p=0.95,
            max_output_tokens=500,
        )

        print(
            f"[LLM] GeminiHandler ready | "
            f"{len(api_keys)} key(s) × {len(self.MODEL_LIST)} model(s) = "
            f"{self._max_rotations} fallback combinations",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure(self):
        """Apply current key + model to genai."""
        key   = self.api_keys[self.current_key_index]
        model = self.MODEL_LIST[self.current_model_index]
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)
        print(
            f"[LLM] Using key[{self.current_key_index + 1}] + "
            f"model '{model}'",
            flush=True,
        )

    def _rotate(self) -> bool:
        """
        Advance to the next model, then next key when models wrap around.

        Returns:
            True  — a new combination is available.
            False — all combinations exhausted.
        """
        self.current_model_index += 1

        if self.current_model_index >= len(self.MODEL_LIST):
            # All models on this key exhausted → next key
            self.current_model_index = 0
            self.current_key_index += 1

            if self.current_key_index >= len(self.api_keys):
                print("[LLM] ❌ All keys × models exhausted.", flush=True)
                return False

            print(
                f"[LLM] Key[{self.current_key_index}] quota done → "
                f"rotating to key[{self.current_key_index + 1}]",
                flush=True,
            )

        self._configure()
        return True

    def _is_quota_error(self, error_str: str) -> bool:
        quota_signals = ["429", "quota", "rate", "resource_exhausted", "exhausted"]
        low = error_str.lower()
        return any(s in low for s in quota_signals)

    # ------------------------------------------------------------------
    # Persona context builder (unchanged logic)
    # ------------------------------------------------------------------

    def _build_persona_context(self, persona: Dict, persona_id: str) -> str:
        """Build context from persona traits AND original Q&A examples."""
        name   = persona.get("Name", "Unknown")
        traits = persona.get("Traits", {})
        demo   = PERSONA_DEMOGRAPHICS.get(persona_id, {"gender": "person", "age": 21})
        gender = demo.get("gender", "person")

        raw_responses = persona.get("Raw_Responses", {})

        style = (
            "You're a 21-year-old Indian girl texting casually."
            if gender == "female"
            else "You're a 21-year-old Indian guy texting casually."
        )

        examples_text = ""
        if raw_responses:
            examples_text = "\n\nHere are examples of how YOU answered similar questions:\n"
            for question, answer in list(raw_responses.items())[:6]:
                q = question.replace("'", "").strip()
                a = str(answer).strip()
                if a and a.lower() not in ["nan", "none", ""]:
                    examples_text += f'Q: "{q}"\nYOU: "{a}"\n\n'

        traits_text = ""
        if not examples_text:
            traits_text = "\n\nYour personality traits:\n"
            traits_text += "\n".join([f"- {k}: {v}" for k, v in traits.items()])

        return f"""You are {name}. {style}
{examples_text}{traits_text}
Rules:
1. "X or Y" questions: REPLY WITH ONLY ONE WORD - JUST X OR Y. NO EXPLANATIONS. NO PUNCTUATION. ONE WORD ONLY.
2. For other questions: Maximum 10 words, casual English.
3. Be direct, match YOUR examples.
4. NEVER say you're an AI.
5. Don't be evasive. Pick one side clearly."""

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_response(
        self,
        persona: Dict,
        user_message: str,
        include_reasoning: bool = False,
        persona_id: str = None,
    ) -> Dict:
        """
        Generate a persona-based response with automatic key+model rotation.
        """
        print(f"[LLM] Generating for: '{user_message}'", flush=True)

        context    = self._build_persona_context(persona, persona_id or "Persona_1")
        is_xor     = " or " in user_message.lower()
        constraint = (
            " (CRITICAL: Reply with ONLY ONE WORD - either the first option or the second."
            " NO punctuation. NO explanations. JUST THE CHOICE.)"
            if is_xor else ""
        )

        prompt = f"""{context}

Friend: "{user_message}"

Your reply{constraint}:"""

        generation_config = self.generation_config  # set once in __init__, unchanged

        rotations_tried = 0

        while rotations_tried <= self._max_rotations:
            try:
                response = self.model.generate_content(
                    prompt, generation_config=generation_config
                )

                if not response.text:
                    return {
                        "response": "idk man, that's a tough one",
                        "success": False,
                        "error": "Empty response from Gemini",
                    }

                text = response.text.strip().strip('"')

                if is_xor:
                    first_word = text.split()[0] if text.split() else text
                    text = first_word.strip('.,!?;:-"\'')

                print(f"[LLM] ✅ Response: '{text}'", flush=True)
                return {"response": text, "reasoning": None, "success": True}

            except Exception as e:
                error_str = str(e)
                print(f"[LLM] ⚠️  Error: {type(e).__name__}: {error_str[:120]}", flush=True)

                if self._is_quota_error(error_str):
                    rotations_tried += 1
                    print(f"[LLM] Quota hit — rotation attempt {rotations_tried}/{self._max_rotations}", flush=True)
                    if not self._rotate():
                        break  # truly exhausted
                else:
                    # Non-quota error — don't rotate, just fail fast
                    return {
                        "response": f"Error: {error_str[:100]}",
                        "reasoning": None,
                        "success": False,
                        "error": error_str,
                    }

        # All combinations exhausted
        return {
            "response": "Sorry, all API quotas are currently exhausted. Try again later!",
            "reasoning": None,
            "success": False,
            "error": "All API key × model combinations exhausted.",
        }
