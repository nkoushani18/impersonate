"""
Intent Detector
Classifies user questions into intents for trait-based response generation.
"""

import re
from typing import Tuple, List


class IntentDetector:
    """Detects the intent behind user questions."""
    
    # Intent definitions with keywords and patterns
    INTENTS = {
        "career_move": {
            "keywords": [
                "job", "career", "work", "promotion", "move", "relocate", 
                "opportunity", "salary", "office", "boss", "colleague",
                "resign", "quit", "hire", "interview", "company", "business",
                "profession", "occupation", "employ", "workplace"
            ],
            "patterns": [
                r"should i take.*(job|offer|opportunity)",
                r"career.*(advice|decision|change|path)",
                r"move.*(city|country|abroad|away).*for.*work",
                r"leave.*(job|company|work)"
            ],
            "relevant_traits": [
                "Career_Orientation", "Risk_Tolerance", "Decision_Style",
                "Future_Preference", "Success_Driver", "Independence"
            ]
        },
        "relationship_conflict": {
            "keywords": [
                "partner", "relationship", "boyfriend", "girlfriend", "spouse",
                "husband", "wife", "conflict", "fight", "argue", "disagree",
                "dating", "love", "marriage", "breakup", "divorce", "trust"
            ],
            "patterns": [
                r"partner.*(disagree|conflict|fight|upset)",
                r"relationship.*(problem|issue|trouble)",
                r"(boyfriend|girlfriend|spouse).*(angry|upset|disagree)"
            ],
            "relevant_traits": [
                "Relationship_Priority", "Conflict_Handling", "Emotional_Orientation",
                "Core_Value", "Communication_Style"
            ]
        },
        "family_decision": {
            "keywords": [
                "family", "parent", "mother", "father", "mom", "dad",
                "sibling", "brother", "sister", "child", "children", "kid",
                "grandparent", "relative", "home", "hometown"
            ],
            "patterns": [
                r"(family|parent).*(or|vs|versus).*",
                r"move.*away.*from.*(family|parent|home)",
                r"choose.*(between|family)"
            ],
            "relevant_traits": [
                "Family_Priority", "Core_Value", "Future_Preference",
                "Decision_Style", "Emotional_Orientation"
            ]
        },
        "future_choice": {
            "keywords": [
                "future", "goal", "dream", "plan", "life", "success",
                "ambition", "aspire", "hope", "wish", "achieve", "vision",
                "5 years", "10 years", "long term"
            ],
            "patterns": [
                r"(future|life).*(look|plan|goal|dream)",
                r"what.*want.*(life|future|career)",
                r"where.*see.*yourself"
            ],
            "relevant_traits": [
                "Future_Preference", "Success_Driver", "Core_Value",
                "Risk_Tolerance", "Career_Orientation"
            ]
        },
        "values_inquiry": {
            "keywords": [
                "value", "priority", "important", "believe", "principle",
                "moral", "ethic", "matter", "care", "worth", "meaning",
                "purpose", "why"
            ],
            "patterns": [
                r"what.*matter.*(most|you)",
                r"what.*value.*(most|you)",
                r"(important|priority).*(to you|in life)"
            ],
            "relevant_traits": [
                "Core_Value", "Success_Driver", "Emotional_Orientation",
                "Personality_Type"
            ]
        },
        "stress_decision": {
            "keywords": [
                "stress", "pressure", "overwhelm", "anxiety", "worried",
                "uncertain", "confused", "difficult", "hard", "struggle",
                "cope", "handle", "manage", "deal"
            ],
            "patterns": [
                r"how.*deal.*(stress|pressure)",
                r"when.*(stressed|overwhelmed|uncertain)",
                r"difficult.*(decision|choice|situation)"
            ],
            "relevant_traits": [
                "Stress_Response", "Uncertainty_Response", "Risk_Tolerance",
                "Decision_Style", "Personality_Type"
            ]
        },
        "value_comparison": {
            "keywords": [
                "or", "vs", "versus", "choose", "prefer", "between",
                "which", "better", "priority"
            ],
            "patterns": [
                r"(\w+)\s+(or|vs|versus)\s+(\w+)",
                r"choose\s+between\s+(\w+)\s+and\s+(\w+)",
                r"(\w+)\s+or\s+(\w+)\??",
                r"would you (pick|choose|prefer)\s+(\w+)\s+or\s+(\w+)",
                r"(family|career|money|love|freedom|stability|work|friends|partner)\s+(or|vs)\s+(\w+)"
            ],
            "relevant_traits": [
                "Family_Priority", "Relationship_Priority", "Career_Orientation",
                "Core_Value", "Future_Preference", "Independence", "Success_Driver"
            ],
            "comparison_mappings": {
                "family": ["Family_Priority"],
                "career": ["Career_Orientation", "Success_Driver"],
                "work": ["Career_Orientation", "Future_Preference"],
                "money": ["Success_Driver", "Future_Preference"],
                "love": ["Relationship_Priority", "Core_Value"],
                "relationship": ["Relationship_Priority"],
                "partner": ["Relationship_Priority"],
                "friends": ["Relationship_Priority"],
                "freedom": ["Independence", "Core_Value"],
                "stability": ["Future_Preference", "Risk_Tolerance"],
                "growth": ["Future_Preference", "Career_Orientation"],
                "security": ["Core_Value", "Risk_Tolerance"],
                "happiness": ["Core_Value", "Emotional_Orientation"],
                "success": ["Success_Driver", "Career_Orientation"]
            }
        }
    }
    
    # Small talk patterns
    SMALL_TALK = {
        "greeting": {
            "patterns": [r"^(hi|hey|hello|howdy|greetings|yo)[\s!?.,]*$"],
            "response": "Hey there."
        },
        "how_are_you": {
            "patterns": [r"^how\s+(are|r)\s+(you|u)", r"^what'?s\s*up", r"^wassup", r"^how.+doing\??$"],
            "response": "I'm good. You?"
        },
        "im_fine": {
            "patterns": [
                r"^i'?m\s*(fine|good|great|okay|ok|well|alright|doing good|doing well)",
                r"^(fine|good|great|okay|ok|alright)[\s!?.,]*$",
                r"^(doing|feeling)\s*(fine|good|great|okay|well)",
                r"^not\s*bad",
                r"^all\s*good"
            ],
            "response": "Nice! So what's on your mind?"
        },
        "thanks": {
            "patterns": [r"^(thank|thanks|thx|ty)[\s!?.,]*", r"^appreciate"],
            "response": "No problem."
        },
        "bye": {
            "patterns": [r"^(bye|goodbye|see you|cya|later|farewell)[\s!?.,]*$"],
            "response": "Later."
        },
        "identity": {
            "patterns": [r"^who\s+(are|r)\s+(you|u)\??$", r"^what\s+(are|r)\s+(you|u)\??$"],
            "response": "I'm simulating the persona you selected. Ask me about life, career, or relationships!"
        }
    }
    
    def detect_small_talk(self, message: str) -> Tuple[bool, str]:
        """
        Check if message is small talk.
        
        Returns:
            Tuple of (is_small_talk, response)
        """
        message_lower = message.lower().strip()
        
        for talk_type, data in self.SMALL_TALK.items():
            for pattern in data["patterns"]:
                if re.search(pattern, message_lower):
                    return True, data["response"]
        
        return False, ""
    
    def detect_intent(self, message: str) -> Tuple[str, List[str], float]:
        """
        Detect the intent of a user message.
        
        Returns:
            Tuple of (intent_name, relevant_traits, confidence)
        """
        message_lower = message.lower()
        
        # Check for value comparison pattern first (X or Y?)
        comparison_match = self._detect_value_comparison(message_lower)
        if comparison_match:
            return comparison_match
        
        best_intent = None
        best_score = 0
        best_traits = []
        
        for intent_name, intent_data in self.INTENTS.items():
            score = 0
            
            # Check keywords
            for keyword in intent_data["keywords"]:
                if keyword in message_lower:
                    score += 1
            
            # Check patterns
            for pattern in intent_data["patterns"]:
                if re.search(pattern, message_lower):
                    score += 3  # Patterns are weighted higher
            
            if score > best_score:
                best_score = score
                best_intent = intent_name
                best_traits = intent_data["relevant_traits"]
        
        # Calculate confidence (0.0 to 1.0)
        confidence = min(best_score / 5.0, 1.0) if best_score > 0 else 0.0
        
        if best_intent is None or confidence < 0.2:
            return "unknown", [], confidence
        
        return best_intent, best_traits, confidence
    
    def _detect_value_comparison(self, message: str) -> Tuple[str, List[str], float]:
        """
        Detect 'X or Y' style value comparison questions.
        
        Returns:
            Tuple of (intent, traits, confidence) or None if not a comparison
        """
        comparison_data = self.INTENTS.get("value_comparison", {})
        mappings = comparison_data.get("comparison_mappings", {})
        
        # Pattern to detect "X or Y" structure
        patterns = [
            r"(\w+)\s+(?:or|vs|versus)\s+(\w+)",
            r"choose\s+(?:between\s+)?(\w+)\s+(?:and|or)\s+(\w+)",
            r"(?:pick|prefer)\s+(\w+)\s+or\s+(\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                option_a = match.group(1).lower()
                option_b = match.group(2).lower()
                
                # Get relevant traits for both options
                traits = []
                if option_a in mappings:
                    traits.extend(mappings[option_a])
                if option_b in mappings:
                    traits.extend(mappings[option_b])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_traits = []
                for t in traits:
                    if t not in seen:
                        seen.add(t)
                        unique_traits.append(t)
                
                if unique_traits:
                    return "value_comparison", unique_traits, 0.9
                else:
                    # Generic comparison without specific mapping
                    return "value_comparison", comparison_data.get("relevant_traits", []), 0.7
        
        return None
