"""
Judge Manager
Manages the comparison sessions and stores history of judgments.
"""

from typing import Dict, List, Optional
from datetime import datetime
from engine.judge_handler import JudgeHandler


class JudgeManager:
    """Manages judge comparisons and maintains history."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """Initialize the judge manager."""
        self.judge_handler = JudgeHandler(ollama_url)
        
        # Store comparisons in memory (resets on server restart)
        self.comparisons: Dict[str, List[Dict]] = {}
        # Key: persona_id, Value: List of comparison results
        
        # Pending questions waiting for human response
        self.pending_questions: Dict[str, Dict] = {}
        # Key: question_id, Value: {question, ai_response, persona_id, timestamp}
        
        self._question_counter = 0
    
    def create_comparison_session(
        self,
        persona_id: str,
        user_question: str,
        ai_response: str
    ) -> str:
        """
        Create a new comparison session when AI responds.
        Returns a question_id for the human to respond to.
        """
        self._question_counter += 1
        question_id = f"{persona_id}_{self._question_counter}_{int(datetime.now().timestamp())}"
        
        self.pending_questions[question_id] = {
            "question_id": question_id,
            "persona_id": persona_id,
            "user_question": user_question,
            "ai_response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "human_response": None,
            "judge_result": None
        }
        
        return question_id
    
    def submit_human_response(
        self,
        question_id: str,
        human_response: str,
        persona_name: str
    ) -> Dict:
        """
        Submit human response and trigger judge evaluation.
        
        Returns:
            Judge result with score and breakdown
        """
        if question_id not in self.pending_questions:
            return {
                "error": "Question ID not found",
                "success": False
            }
        
        session = self.pending_questions[question_id]
        session["human_response"] = human_response
        
        # Call the judge
        judge_result = self.judge_handler.judge_responses(
            user_question=session["user_question"],
            ai_response=session["ai_response"],
            human_response=human_response,
            persona_name=persona_name
        )
        
        session["judge_result"] = judge_result
        session["judged_at"] = datetime.now().isoformat()
        
        # Store in history
        persona_id = session["persona_id"]
        if persona_id not in self.comparisons:
            self.comparisons[persona_id] = []
        
        self.comparisons[persona_id].append({
            "question_id": question_id,
            "user_question": session["user_question"],
            "ai_response": session["ai_response"],
            "human_response": human_response,
            "judge_result": judge_result,
            "timestamp": session["timestamp"],
            "judged_at": session["judged_at"]
        })
        
        # Remove from pending
        # Keep it for now so we can show it in results
        # del self.pending_questions[question_id]
        
        return {
            "success": True,
            "judge_result": judge_result,
            "question_id": question_id
        }
    
    def get_pending_questions(self, persona_id: str) -> List[Dict]:
        """Get all pending questions for a specific persona."""
        pending = []
        for qid, session in self.pending_questions.items():
            if session["persona_id"] == persona_id and session["human_response"] is None:
                pending.append({
                    "question_id": qid,
                    "user_question": session["user_question"],
                    "ai_response": session["ai_response"],
                    "timestamp": session["timestamp"]
                })
        return pending
    
    def get_latest_comparison(self, persona_id: str) -> Optional[Dict]:
        """Get the most recent comparison for a persona."""
        if persona_id not in self.comparisons or not self.comparisons[persona_id]:
            return None
        
        return self.comparisons[persona_id][-1]
    
    def get_comparison_history(self, persona_id: str) -> List[Dict]:
        """Get all comparisons for a persona."""
        return self.comparisons.get(persona_id, [])
    
    def get_statistics(self, persona_id: str) -> Dict:
        """Get statistics for a persona's comparisons."""
        comparisons = self.comparisons.get(persona_id, [])
        
        if not comparisons:
            return {
                "total_comparisons": 0,
                "average_score": 0,
                "highest_score": 0,
                "lowest_score": 0
            }
        
        scores = [c["judge_result"]["score"] for c in comparisons if "score" in c["judge_result"]]
        
        return {
            "total_comparisons": len(comparisons),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "highest_score": max(scores) if scores else 0,
            "lowest_score": min(scores) if scores else 0,
            "latest_score": scores[-1] if scores else 0
        }
