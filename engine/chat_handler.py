"""
Chat Handler
Main orchestrator for persona-based conversations.
Now powered by Gemini LLM for intelligent responses.
"""

import os
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
from engine.persona_loader import PersonaLoader
from engine.intent_detector import IntentDetector
from engine.llm_handler import GeminiHandler

# Load .env file for local development (ignored in production where env vars are set on the platform)
load_dotenv()

# Read API key from environment variable — set this in Render's dashboard
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class ChatHandler:
    """Main chat handler for persona-based conversations."""
    
    def __init__(self, csv_path: str = None):
        self.persona_loader = PersonaLoader()
        self.intent_detector = IntentDetector()
        
        # Initialize Gemini LLM with API key
        try:
            self.llm = GeminiHandler(GEMINI_API_KEY)
            self.use_llm = True
            print("[*] Gemini LLM initialized successfully!")
        except Exception as e:
            self.llm = None
            self.use_llm = False
            print(f"[!] LLM init failed: {e}. Using templates.")
        
        self.active_persona_id: Optional[str] = None
        self.active_persona: Optional[dict] = None
        
        # Load personas from CSV if provided
        if csv_path:
            self.persona_loader.load_from_csv(csv_path)
            self.persona_loader.save_personas_to_json()
    
    def set_active_persona(self, persona_id: str) -> bool:
        """
        Set the active persona for conversations.
        
        Returns:
            True if persona was found and loaded, False otherwise
        """
        persona = self.persona_loader.load_persona(persona_id)
        if persona:
            self.active_persona_id = persona_id
            self.active_persona = persona
            return True
        return False
    
    def get_active_persona_info(self) -> dict:
        """Get information about the currently active persona."""
        if not self.active_persona:
            return {"error": "No active persona set"}
        
        return {
            "persona_id": self.active_persona_id,
            "name": self.active_persona.get("Name", "Unknown"),
            "traits": self.active_persona.get("Traits", {})
        }
    
    def list_available_personas(self) -> list:
        """List all available persona IDs."""
        return self.persona_loader.list_personas()
    
    def handle_message(self, user_message: str, debug: bool = False) -> dict:
        """
        Handle a user message and generate a persona-consistent response.
        
        Args:
            user_message: The user's input message
            debug: If True, include trait reasoning in response
            
        Returns:
            Dict with response and optional debug info
        """
        # Check for small talk first
        is_small_talk, small_talk_response = self.intent_detector.detect_small_talk(user_message)
        
        if is_small_talk:
            return {
                "response": small_talk_response,
                "type": "small_talk",
                "debug": None
            }
        
        # Check if persona is set
        if not self.active_persona:
            return {
                "response": "Please select a persona first before asking questions.",
                "type": "error",
                "debug": None
            }
        
        # Use LLM if available, otherwise fall back to templates
        if self.use_llm and self.llm:
            try:
                return self._handle_with_llm(user_message, debug)
            except Exception as e:
                print(f"[ERROR] LLM handling failed: {e}. Falling back to templates.", flush=True)
                return self._handle_with_templates(user_message, debug)
        else:
            return self._handle_with_templates(user_message, debug)
    
    def _handle_with_llm(self, user_message: str, debug: bool) -> dict:
        """Handle message using Gemini LLM."""
        import sys
        print(f"[LLM] Calling Gemini for: '{user_message}'", flush=True)
        print(f"[LLM] Active persona: {self.active_persona_id}", flush=True)
        
        result = self.llm.generate_response(
            persona=self.active_persona,
            user_message=user_message,
            include_reasoning=debug,
            persona_id=self.active_persona_id
        )
        
        print(f"[LLM] Success: {result.get('success')}", flush=True)
        print(f"[LLM] Response: {result.get('response')}", flush=True)
        if result.get('error'):
            print(f"[LLM] ERROR: {result.get('error')}", flush=True)
        
        if result["success"]:
            debug_info = None
            if debug and result.get("reasoning"):
                intent, relevant_traits, confidence = self.intent_detector.detect_intent(user_message)
                debug_info = {
                    "intent": intent,
                    "confidence": confidence,
                    "traits_used": {t: self.active_persona.get("Traits", {}).get(t, "N/A") 
                                   for t in relevant_traits[:3]},
                    "llm_reasoning": result.get("reasoning")
                }
            
            return {
                "response": result["response"],
                "type": "llm_response",
                "debug": debug_info
            }
        else:
            # Check if it's a rate limit error - fall back to templates
            error_msg = result.get("error", "")
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                print("[LLM] Rate limit hit! Falling back to templates...", flush=True)
                return self._handle_with_templates(user_message, debug)
            
            # For other errors, return the error message
            return {
                "response": result.get("response", "hmm let me think..."),
                "type": "llm_error",
                "debug": {"error": result.get("error")} if debug else None
            }
    
    def _handle_with_templates(self, user_message: str, debug: bool) -> dict:
        """Handle message using template-based responses (fallback)."""
        from engine.response_generator import ResponseGenerator
        response_generator = ResponseGenerator()
        
        # Detect intent
        intent, relevant_traits, confidence = self.intent_detector.detect_intent(user_message)
        
        # Get persona traits
        traits = self.active_persona.get("Traits", {})
        
        # Generate response
        if debug:
            result = response_generator.generate_with_explanation(
                intent, traits, relevant_traits, user_message
            )
            return {
                "response": result["response"],
                "type": intent,
                "debug": {
                    "intent": result["intent"],
                    "confidence": confidence,
                    "traits_used": result["traits_used"]
                }
            }
        else:
            response = response_generator.generate_response(
                intent, traits, relevant_traits, user_message
            )
            return {
                "response": response,
                "type": intent,
                "debug": None
            }
    
    def get_persona_summary(self, persona_id: str) -> str:
        """Get a summary of a specific persona."""
        return self.persona_loader.get_persona_summary(persona_id)
