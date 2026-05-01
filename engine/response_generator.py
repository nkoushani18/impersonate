"""
Response Generator
Generates human-like, casual responses based on persona traits.
Responses sound like 21-year-old Indian college students texting.
"""

from typing import Dict, List, Optional
import random


class ResponseGenerator:
    """Generates casual, human-like responses based on persona traits."""
    
    # Response templates - casual Indian youth texting style
    RESPONSE_TEMPLATES = {
        "career_move": {
            "Family_Priority:High": [
                "nah fam, family comes first for me. job can wait",
                "wouldn't take it if it messes with family time tbh",
                "family >> everything else. i'd stay close to home"
            ],
            "Family_Priority:Medium": [
                "i'd try to balance it somehow yaar",
                "tough call tbh... depends on the specifics",
                "maybe i can make both work? idk"
            ],
            "Family_Priority:Low": [
                "career all the way. gotta do what's best for me",
                "i'd take it. my growth matters more rn",
                "yeah i'd go for it. can't hold back my career"
            ],
            "Career_Orientation:Visionary": [
                "if it's exciting? hell yeah i'm in",
                "big opportunities don't come twice. i'd take it",
                "yesss i chase the dream always"
            ],
            "Career_Orientation:Logical": [
                "depends on the salary and benefits tbh",
                "show me the numbers first lol",
                "i'd calculate the pros and cons properly"
            ],
            "Risk_Tolerance:Low": [
                "sounds risky... i'd need to think about it",
                "i prefer the stable path honestly",
                "idk man that's too uncertain for me"
            ],
            "Risk_Tolerance:High": [
                "you gotta take risks to grow yaar",
                "life's boring if you play it safe lol",
                "i'd take the leap. why not?"
            ]
        },
        "relationship_conflict": {
            "Conflict_Handling:Communication-based": [
                "i'd just talk it out tbh. communication is key",
                "we need to sit and discuss this properly",
                "i believe in talking through problems yaar"
            ],
            "Conflict_Handling:Avoidance": [
                "i'd probably just chill for a bit and let it cool down",
                "not in the mood for drama rn lol",
                "i'd wait it out honestly"
            ],
            "Conflict_Handling:Confrontation": [
                "i'd call them out directly. no bs",
                "we're settling this now. i don't let things slide",
                "direct confrontation works best for me"
            ],
            "Relationship_Priority:High": [
                "i'd do whatever it takes to fix it tbh",
                "hate fighting with people i care about",
                "i'd compromise. the relationship matters more"
            ],
            "Emotional_Orientation:High": [
                "ngl i'd get pretty emotional about it",
                "it would hurt me a lot tbh",
                "i go with my heart in these situations"
            ]
        },
        "family_decision": {
            "Family_Priority:High": [
                "family first yaar. no question",
                "i'd do whatever's best for my family",
                "their opinion matters most to me honestly"
            ],
            "Family_Priority:Medium": [
                "it's complicated... i want to help but also have my life",
                "i'd try to balance both somehow",
                "love them but need my space too lol"
            ],
            "Family_Priority:Low": [
                "my life my rules tbh",
                "i make my own decisions. sorry not sorry",
                "they'll understand eventually"
            ],
            "Independence:High": [
                "i decide for myself. period",
                "independence is everything to me yaar",
                "my life my choices"
            ]
        },
        "future_choice": {
            "Future_Preference:Stability": [
                "stability for sure. i like being secure",
                "i'd go for the safe option tbh",
                "peace of mind matters more to me"
            ],
            "Future_Preference:Growth": [
                "growth all the way! comfort zone is a trap",
                "i pick the challenging path always",
                "how else will i grow lol"
            ],
            "Risk_Tolerance:Low": [
                "the safer choice for me",
                "i don't like taking big risks honestly",
                "better safe than sorry yaar"
            ],
            "Risk_Tolerance:High": [
                "the risky one obviously lol",
                "high risk high reward baby",
                "what's life without some adventure"
            ],
            "Decision_Style:Quick": [
                "i'd just decide fast and move on",
                "overthinking is waste of time tbh",
                "gut feeling and go"
            ],
            "Decision_Style:Delayed": [
                "i need time to think about it properly",
                "can't rush big decisions yaar",
                "let me sleep on it first"
            ]
        },
        "values_inquiry": {
            "Core_Value:Emotional Security": [
                "peace of mind honestly. i need to feel safe",
                "emotional stability matters most to me",
                "being mentally at peace > everything"
            ],
            "Core_Value:Freedom": [
                "freedom to do my own thing yaar",
                "being independent and free matters most",
                "can't stand being controlled lol"
            ],
            "Family_Priority:High": [
                "my family tbh. they mean everything",
                "family first always for me",
                "nothing matters more than family yaar"
            ],
            "Career_Orientation:Visionary": [
                "chasing my dreams matters most",
                "building something big. that's what drives me",
                "making my vision real is everything"
            ],
            "Relationship_Priority:High": [
                "the people i love tbh",
                "relationships matter most to me",
                "love and connection > everything else"
            ],
            "Personality_Type:Visionary": [
                "thinking about the future yaar",
                "i'm always dreaming about what could be",
                "the big picture matters most to me"
            ],
            "Personality_Type:Practical": [
                "practical results honestly",
                "what works is what matters",
                "no fluff. just real solutions"
            ]
        },
        "stress_decision": {
            "Stress_Response:Logical": [
                "i just break it down step by step",
                "panicking doesn't help. logic does",
                "i analyze and figure it out calmly"
            ],
            "Stress_Response:Emotional": [
                "ngl i freak out a little first lol",
                "i need to vent before i can think clearly",
                "it hits me hard tbh. need time to process"
            ],
            "Stress_Response:Cautious": [
                "i slow down and think carefully",
                "can't afford to make mistakes under stress",
                "i triple check everything when stressed"
            ],
            "Uncertainty_Response:Optimistic": [
                "it'll work out yaar. it always does",
                "positive vibes only lol",
                "i trust the universe honestly"
            ],
            "Uncertainty_Response:Planning": [
                "i plan for every possible scenario",
                "preparation kills anxiety for me",
                "lists and backups. that's how i cope"
            ]
        },
        "value_comparison": {
            "Family_Priority:High": [
                "family yaar. no doubt",
                "family 100%. career can wait",
                "family first. always"
            ],
            "Family_Priority:Medium": [
                "tbh depends on the situation",
                "i'd try to balance both honestly",
                "tough one... whichever needs me more rn"
            ],
            "Family_Priority:Low": [
                "career for sure. gotta chase my goals",
                "my work tbh. i need to focus on me",
                "career. can't sacrifice my dreams"
            ],
            "Relationship_Priority:High": [
                "my partner. love over everything",
                "relationship obviously",
                "the person i love. easy choice"
            ],
            "Career_Orientation:Visionary": [
                "career. chasing the dream",
                "work. i'm building something",
                "my goals first"
            ],
            "Career_Orientation:Logical": [
                "whichever makes more sense logically",
                "the practical choice tbh",
                "whatever has better returns lol"
            ],
            "Core_Value:Emotional Security": [
                "family. i need that stability",
                "the safer option for me",
                "whatever gives me peace of mind"
            ],
            "Core_Value:Freedom": [
                "whatever gives me more freedom",
                "independence always",
                "the option where i'm free"
            ],
            "Future_Preference:Stability": [
                "the stable option for sure",
                "whatever's more reliable",
                "stability wins for me"
            ],
            "Future_Preference:Growth": [
                "the challenging one. growth matters",
                "whichever helps me grow more",
                "i pick the harder path"
            ],
            "Independence:High": [
                "whatever lets me be independent",
                "my own path. always",
                "me over everything else lol"
            ]
        }
    }
    
    # Fallback responses
    FALLBACK_RESPONSES = [
        "hmm idk yaar depends on the situation",
        "that's tough... i'd have to think about it",
        "tbh i handle things as they come",
        "not sure rn but i'd figure it out"
    ]
    
    # Out of scope response
    OUT_OF_SCOPE_RESPONSE = "lol idk about that. ask me about life, career or relationships?"
    
    def __init__(self):
        pass
    
    def generate_response(
        self, 
        intent: str, 
        persona_traits: Dict, 
        relevant_traits: List[str],
        user_message: str = ""
    ) -> str:
        """Generate response based on intent and traits."""
        
        if intent not in self.RESPONSE_TEMPLATES:
            return self.OUT_OF_SCOPE_RESPONSE
        
        intent_templates = self.RESPONSE_TEMPLATES[intent]
        
        # Find matching templates
        matched_responses = []
        for trait in relevant_traits:
            trait_value = persona_traits.get(trait.split(":")[0] if ":" in trait else trait, None)
            if trait_value:
                trait_key = f"{trait.split(':')[0] if ':' in trait else trait}:{trait_value}"
                if trait_key in intent_templates:
                    matched_responses.extend(intent_templates[trait_key])
        
        # Check full trait keys
        for trait_key, responses in intent_templates.items():
            trait_name = trait_key.split(":")[0]
            trait_val = trait_key.split(":")[1] if ":" in trait_key else None
            
            if trait_name in persona_traits:
                if persona_traits[trait_name] == trait_val:
                    matched_responses.extend(responses)
        
        if matched_responses:
            return random.choice(list(set(matched_responses)))
        
        return random.choice(self.FALLBACK_RESPONSES)
    
    def generate_with_explanation(
        self,
        intent: str,
        persona_traits: Dict,
        relevant_traits: List[str],
        user_message: str = ""
    ) -> Dict:
        """Generate response with reasoning."""
        response = self.generate_response(intent, persona_traits, relevant_traits, user_message)
        
        used_traits = {}
        for trait in relevant_traits[:3]:
            if trait in persona_traits:
                used_traits[trait] = persona_traits[trait]
        
        return {
            "response": response,
            "intent": intent,
            "traits_used": used_traits
        }
