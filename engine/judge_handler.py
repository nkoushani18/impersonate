"""
LLM Judge Handler
Interfaces with Ollama LLaMA 3.2 to judge semantic similarity between AI and human responses.
"""

import requests
import json
from typing import Dict, Optional


class JudgeHandler:
    """Handler for LLM-based response judging using Ollama."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """
        Initialize the judge handler.
        
        Args:
            ollama_url: URL of the Ollama server (default: localhost:11434)
        """
        self.ollama_url = ollama_url
        self.model = "llama3.2"  # LLaMA 3.2 model
        
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
        Judge the semantic similarity between AI and human responses.
        
        Args:
            user_question: The original user question
            ai_response: The AI agent's response
            human_response: The human's response
            persona_name: Name of the persona being evaluated
            
        Returns:
            Dict with:
                - score: 0-100 semantic accuracy score
                - breakdown: Detailed analysis
                - intent_match: Whether core intents match
                - reasoning: Explanation of the score
        """
        # Check if Ollama is running first
        print("[JUDGE] Checking Ollama connection...", flush=True)
        if not self.check_connection():
            return {
                "error": "⚠️ Ollama is not running! Please start it with: ollama serve",
                "score": 0,
                "breakdown": {},
                "reasoning": "Cannot connect to Ollama server at localhost:11434"
            }
        
        # Construct the judge prompt
        prompt = self._create_judge_prompt(
            user_question, ai_response, human_response, persona_name
        )
        
        try:
            print(f"[JUDGE] Sending to Ollama LLaMA 3.2... (this may take 10-30 seconds)", flush=True)
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent judging
                        "top_p": 0.9,
                        "num_predict": 300,  # Limit response length for faster results
                    }
                },
                timeout=120  # Increased to 2 minutes for slower systems
            )
            
            print(f"[JUDGE] Got response from Ollama (status: {response.status_code})", flush=True)
            
            if response.status_code != 200:
                return {
                    "error": f"Ollama API error: {response.status_code}",
                    "score": 0,
                    "breakdown": {},
                    "reasoning": "Failed to get response from judge"
                }
            
            result = response.json()
            judge_output = result.get("response", "")
            
            print(f"[JUDGE] Parsing judge output...", flush=True)
            # Parse the judge's output
            parsed = self._parse_judge_output(judge_output)
            
            print(f"[JUDGE] ✅ Judging complete! Score: {parsed.get('score', 0)}/100", flush=True)
            return parsed
            
        except requests.exceptions.Timeout:
            return {
                "error": "⏱️ Ollama took too long to respond (>2 minutes). Your system might be too slow for LLaMA 3.2.",
                "score": 0,
                "breakdown": {},
                "reasoning": "Timeout waiting for judge"
            }
        except Exception as e:
            print(f"[JUDGE] ❌ Error: {str(e)}", flush=True)
            return {
                "error": f"Judge error: {str(e)}",
                "score": 0,
                "breakdown": {},
                "reasoning": "Failed to evaluate responses"
            }
    
    def _create_judge_prompt(
        self, 
        question: str, 
        ai_resp: str, 
        human_resp: str,
        persona_name: str
    ) -> str:
        """Ultra-strict judge prompt for X or Y selection."""
        prompt = f"""You are a strict, logical judge. Compare the AI and Human answers.

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
        
        return prompt
    
    def _parse_judge_output(self, output: str) -> Dict:
        """Parse strict judge output format."""
        # Clean output
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

        # Parse line by line
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            upper_line = line.upper()

            # Parse Scores
            if "PREFERENCE:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    score = int(''.join(filter(str.isdigit, score_part)))
                    result["breakdown"]["preference_alignment"] = score
                except: pass
            
            elif "EMOTIONAL:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    score = int(''.join(filter(str.isdigit, score_part)))
                    result["breakdown"]["emotional_alignment"] = score
                except: pass

            elif "FACTUAL:" in upper_line:
                try:
                    score_part = line.split(':', 1)[1]
                    score = int(''.join(filter(str.isdigit, score_part)))
                    result["breakdown"]["factual_alignment"] = score
                except: pass
            
            # Parse Intent
            elif "INTENT_MATCH:" in upper_line:
                if "YES" in upper_line:
                    result["intent_match"] = True

            # Parse Reasoning
            elif "REASONING:" in upper_line:
                try:
                    result["reasoning"] = line.split(':', 1)[1].strip()
                except:
                    result["reasoning"] = line

        # LOGIC FIX: Trust explicit Preference Score over binary Intent Match
        # If the LLM says Preference: 0 (Disagreement), it typically means it detected a conflict.
        # We should NOT override this with a potentially hallucinated "INTENT_MATCH: YES".
        
        if result["breakdown"]["preference_alignment"] <= 40:
            if result["intent_match"]:
                print(f"[JUDGE] ⚠️ Inconsistency detected: Preference={result['breakdown']['preference_alignment']} (Low) but Intent=YES. Correcting Intent to NO.")
                result["intent_match"] = False
        
        # Conversely, if Preference is high (>= 80), Intent Match should probably be YES.
        elif result["breakdown"]["preference_alignment"] >= 80:
             if not result["intent_match"]:
                 print(f"[JUDGE] ⚠️ Inconsistency detected: Preference={result['breakdown']['preference_alignment']} (High) but Intent=NO. Correcting Intent to YES.")
                 result["intent_match"] = True

        # Calculate Final Score using Formula
        p = result["breakdown"]["preference_alignment"]
        e = result["breakdown"]["emotional_alignment"]
        f = result["breakdown"]["factual_alignment"]
        
        if f == 0:
            # If Factual is 0 (N/A), redistribute weight: Intent 70%, Tone 30%
            final_score = int((p * 0.7) + (e * 0.3))
            print(f"[JUDGE] Factual is 0 -> Redistributed weights: P:{p}*0.7 + E:{e}*0.3 = {final_score}", flush=True)
        else:
            final_score = int((p * 0.6) + (e * 0.3) + (f * 0.1))
            print(f"[JUDGE] Standard weights -> P:{p} E:{e} F:{f} = {final_score}", flush=True)
            
        result["score"] = final_score

        return result
