"""
Gemini LLM Handler
Fast, intelligent persona-based responses using Google Gemini.
"""

import google.generativeai as genai
from typing import Dict, Optional


# Persona demographic info
PERSONA_DEMOGRAPHICS = {
    "Persona_1": {"gender": "female", "age": 21, "nickname": "Koushani"},
    "Persona_2": {"gender": "male", "age": 21, "nickname": "Rishit"},
    "Persona_3": {"gender": "male", "age": 21, "nickname": "Harsh"},
    "Persona_4": {"gender": "female", "age": 21, "nickname": "Manya"},
    "Persona_5": {"gender": "male", "age": 21, "nickname": "Salil"},
}


class GeminiHandler:
    """Handles LLM-based persona responses using Gemini with model rotation."""
    
    # Models to try in order (when one hits quota, try the next)
    MODEL_LIST = [
        'models/gemini-2.5-flash',
        'models/gemini-2.0-flash',
        'models/gemini-2.5-flash-lite',
        'models/gemini-2.0-flash-lite',
        'models/gemini-flash-latest',
    ]
    
    def __init__(self, api_key: str):
        """Initialize Gemini with API key."""
        genai.configure(api_key=api_key)
        
        self.current_model_index = 0
        self._init_model()
        
        self.generation_config = genai.GenerationConfig(
            temperature=0.9,
            top_p=0.95,
            max_output_tokens=500,
        )
        print(f"[LLM] GeminiHandler initialized with {self.MODEL_LIST[self.current_model_index]}", flush=True)
    
    def _init_model(self):
        """Initialize the current model."""
        model_name = self.MODEL_LIST[self.current_model_index]
        self.model = genai.GenerativeModel(model_name)
        print(f"[LLM] Using model: {model_name}", flush=True)
    
    def _rotate_model(self):
        """Switch to the next model in the list."""
        self.current_model_index = (self.current_model_index + 1) % len(self.MODEL_LIST)
        self._init_model()
        print(f"[LLM] Rotated to model index {self.current_model_index}", flush=True)
    
    def _build_persona_context(self, persona: Dict, persona_id: str) -> str:
        """Build context from persona traits AND original Q&A examples."""
        name = persona.get("Name", "Unknown")
        traits = persona.get("Traits", {})
        
        demo = PERSONA_DEMOGRAPHICS.get(persona_id, {"gender": "person", "age": 21})
        gender = demo.get("gender", "person")
        
        # Get original question-answer pairs if available
        raw_responses = persona.get("Raw_Responses", {})
        
        if gender == "female":
            style = "You're a 21-year-old Indian girl texting casually."
        else:
            style = "You're a 21-year-old Indian guy texting casually."
        
        # Build examples from actual Q&A if available
        examples_text = ""
        if raw_responses:
            examples_text = "\n\nHere are examples of how YOU answered similar questions:\n"
            for question, answer in list(raw_responses.items())[:6]:  # Use top 6 examples
                # Clean up question text
                q = question.replace("'", "").strip()
                a = str(answer).strip()
                if a and a.lower() not in ['nan', 'none', '']:
                    examples_text += f'Q: "{q}"\nYOU: "{a}"\n\n'
        
        # Fall back to trait descriptions if no raw responses
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
    
    def generate_response(
        self, 
        persona: Dict, 
        user_message: str,
        include_reasoning: bool = False,
        persona_id: str = None
    ) -> Dict:
        """Generate a persona-based response."""
        print(f"[LLM-HANDLER] Starting generation for: '{user_message}'", flush=True)
        
        context = self._build_persona_context(persona, persona_id or "Persona_1")
        
        # Dynamic constraint for "X or Y" questions
        constraint = ""
        if " or " in user_message.lower():
            constraint = " (CRITICAL: Reply with ONLY ONE WORD - either the first option or the second. NO punctuation. NO explanations. JUST THE CHOICE.)"
        
        prompt = f"""{context}

Friend: "{user_message}"

Your reply{constraint}:"""
        
        print(f"[LLM-HANDLER] Prompt length: {len(prompt)} chars", flush=True)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            print(f"[LLM-HANDLER] Got response object", flush=True)
            
            # Check if response has text
            if not response.text:
                print(f"[LLM-HANDLER] Response has no text!", flush=True)
                return {
                    "response": "idk man, that's a tough one",
                    "success": False,
                    "error": "Empty response from Gemini"
                }
            
            text = response.text.strip()
            print(f"[LLM-HANDLER] Raw response: '{text}'", flush=True)
            
            # Clean quotes
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            
            # For "X or Y" questions, extract ONLY the first word (the choice)
            if " or " in user_message.lower():
                # Remove punctuation and get first meaningful word
                first_word = text.split()[0] if text.split() else text
                # Remove trailing punctuation
                first_word = first_word.strip('.,!?;:-"\'')
                print(f"[LLM-HANDLER] Extracted choice: '{first_word}'", flush=True)
                text = first_word
            
            return {
                "response": text,
                "reasoning": None,
                "success": True
            }
                
        except Exception as e:
            error_str = str(e)
            print(f"[LLM-HANDLER] ERROR: {type(e).__name__}: {error_str}", flush=True)
            
            # Check for quota/rate limit errors - try rotating model
            if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                print(f"[LLM-HANDLER] Quota exceeded! Rotating to next model...", flush=True)
                self._rotate_model()
                
                # Try once more with new model
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=self.generation_config
                    )
                    if response.text:
                        text = response.text.strip()
                        if text.startswith('"') and text.endswith('"'):
                            text = text[1:-1]
                        
                        # For "X or Y" questions, extract ONLY the first word
                        if " or " in user_message.lower():
                            first_word = text.split()[0] if text.split() else text
                            first_word = first_word.strip('.,!?;:-"\'')
                            text = first_word
                        
                        print(f"[LLM-HANDLER] Success with rotated model!", flush=True)
                        return {
                            "response": text,
                            "reasoning": None,
                            "success": True
                        }
                except Exception as e2:
                    print(f"[LLM-HANDLER] Retry also failed: {e2}", flush=True)
            
            return {
                "response": f"Error: {error_str[:100]}",
                "reasoning": None,
                "success": False,
                "error": error_str
            }

