"""
Trait Mapping Rules
Maps survey answers to persona traits (no ML, just rule-based mapping)
"""

# Answer to Trait Value Mapping
TRAIT_MAPPINGS = {
    # Q1: Family Priority
    "Feel emotionally conflicted and prioritize relationships": {"Family_Priority": "High", "Relationship_Priority": "High"},
    "Analyze long-term benefits and plan how to manage both": {"Family_Priority": "Medium", "Relationship_Priority": "Medium"},
    "Decide independently without external pressure": {"Family_Priority": "Low", "Independence": "High"},
    
    # Q2: Conflict Handling
    "Explain your reasoning and expect understanding": {"Conflict_Handling": "Communication-based"},
    "Avoid the discussion until emotions settle": {"Conflict_Handling": "Avoidance"},
    "Address it immediately and directly": {"Conflict_Handling": "Confrontation"},
    
    # Q3: Career Orientation
    "Proven paths and real-world outcomes": {"Career_Orientation": "Logical", "Risk_Tolerance": "Low"},
    "Future possibilities and what excites you": {"Career_Orientation": "Visionary", "Risk_Tolerance": "High"},
    "What others have advised": {"Career_Orientation": "Guided", "Independence": "Low"},
    
    # Q4: Future Preference
    "Stable job, secure income, clear routine": {"Future_Preference": "Stability"},
    "Meaningful work with freedom and growth": {"Future_Preference": "Growth"},
    "Balanced life that adapts as things change": {"Future_Preference": "Flexibility"},
    
    # Q5: Decision Style
    "Stick to your decision regardless": {"Decision_Style": "Firm", "Independence": "High"},
    "Re-evaluate logically and discuss trade-offs": {"Decision_Style": "Analytical"},
    "Prioritize their feelings over your choice": {"Decision_Style": "Accommodating", "Relationship_Priority": "High"},
    
    # Q6: Risk Tolerance / Uncertainty Response
    "Trust things will work out with time": {"Uncertainty_Response": "Optimistic", "Risk_Tolerance": "Medium"},
    "Create a plan to reduce risk": {"Uncertainty_Response": "Planning", "Risk_Tolerance": "Low"},
    "Seek advice from trusted people": {"Uncertainty_Response": "Collaborative", "Independence": "Low"},
    
    # Q7: Personality Type
    "Caring, sensitive, and understanding": {"Personality_Type": "Empathetic", "Emotional_Orientation": "High"},
    "Visionary, curious, and idea-driven": {"Personality_Type": "Visionary", "Creativity": "High"},
    "Practical, reliable, and grounded": {"Personality_Type": "Practical", "Emotional_Orientation": "Low"},
    
    # Q8: Stress Response / Decision Making
    "Decide quickly once logic is clear": {"Stress_Response": "Logical", "Decision_Style": "Quick"},
    "Take time to reflect on emotions": {"Stress_Response": "Emotional", "Decision_Style": "Reflective"},
    "Delay until more options appear": {"Stress_Response": "Cautious", "Decision_Style": "Delayed"},
    
    # Q9: Core Value
    "Emotional security and trust": {"Core_Value": "Emotional Security"},
    "Freedom and flexibility": {"Core_Value": "Freedom"},
    "Achievement and recognition": {"Core_Value": "Achievement"},
    
    # Q10: Success Driver
    "Discipline, planning, and consistency": {"Success_Driver": "Consistency"},
    "Passion, exploration, and adaptability": {"Success_Driver": "Exploration"},
    "Connections and support from others": {"Success_Driver": "Relationships"}
}


def map_answer_to_traits(answer: str) -> dict:
    """
    Map a survey answer to its corresponding traits.
    Returns empty dict if answer not found.
    """
    # Clean the answer (strip whitespace)
    answer = answer.strip()
    
    # Direct lookup
    if answer in TRAIT_MAPPINGS:
        return TRAIT_MAPPINGS[answer]
    
    # Fuzzy matching - check if answer is contained in any key
    for key, traits in TRAIT_MAPPINGS.items():
        if answer.lower() in key.lower() or key.lower() in answer.lower():
            return traits
    
    return {}


def build_persona_traits(survey_answers: dict) -> dict:
    """
    Build a complete trait profile from all survey answers.
    
    Args:
        survey_answers: Dict mapping question number to answer string
        
    Returns:
        Complete trait profile dict
    """
    traits = {}
    
    for question_num, answer in survey_answers.items():
        mapped_traits = map_answer_to_traits(answer)
        traits.update(mapped_traits)
    
    return traits
