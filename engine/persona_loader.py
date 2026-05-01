"""
Persona Loader
Loads persona profiles from CSV survey data and converts to trait profiles.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# Column name to question mapping (based on the uploaded CSV)
COLUMN_TO_QUESTION = {
    "When a major career opportunity requires moving away from family, you ": "Q1",
    "  In a relationship conflict caused by work pressure, you usually  ": "Q2",
    "  When choosing a career path, you trust more  ": "Q3",
    "  Your ideal future life looks like  ": "Q4",
    "  If your partner strongly disagrees with your career choice, you  ": "Q5",
    "  Under uncertainty about career and relationships, you  ": "Q6",
    "  People close to you often describe you as  ": "Q7",
    "  When making an important life decision, you  ": "Q8",
    "  In both career and relationships, you value more  ": "Q9",
    "  Looking at your future, you believe success comes from  ": "Q10"
}


class PersonaLoader:
    """Loads and manages persona profiles."""
    
    def __init__(self, personas_dir: str = None):
        if personas_dir is None:
            personas_dir = Path(__file__).parent.parent / "personas"
        self.personas_dir = Path(personas_dir)
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        self.personas: Dict[str, dict] = {}
    
    def load_from_csv(self, csv_path: str) -> List[dict]:
        """
        Load personas from survey response CSV.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of persona profiles
        """
        from data.trait_mapping import map_answer_to_traits
        
        personas = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                persona_id = row.get('persona_id', '').strip()
                name = row.get('Name', '').strip()
                
                # Skip empty rows
                if not persona_id or persona_id == '':
                    continue
                
                # Clean persona_id (remove any newlines)
                persona_id = persona_id.replace('\n', '').replace('\r', '').strip()
                
                # Build traits from survey answers
                traits = {}
                survey_answers = {}
                raw_responses = {}  # Store actual question-answer pairs
                
                for column, question_id in COLUMN_TO_QUESTION.items():
                    answer = row.get(column, '').strip()
                    if answer:
                        survey_answers[question_id] = answer
                        # Store the actual question text with answer
                        question_text = column.strip()
                        raw_responses[question_text] = answer
                        mapped = map_answer_to_traits(answer)
                        traits.update(mapped)
                
                # Create persona profile
                persona = {
                    "Persona_ID": persona_id,
                    "Name": name,
                    "Traits": traits,
                    "Survey_Answers": survey_answers,
                    "Raw_Responses": raw_responses  # Include for LLM context
                }
                
                personas.append(persona)
                self.personas[persona_id] = persona
        
        return personas
    
    def save_personas_to_json(self):
        """Save all loaded personas to individual JSON files."""
        for persona_id, persona in self.personas.items():
            filepath = self.personas_dir / f"{persona_id}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(persona, f, indent=2)
    
    def load_persona(self, persona_id: str) -> Optional[dict]:
        """Load a specific persona by ID."""
        # Check memory first
        if persona_id in self.personas:
            return self.personas[persona_id]
        
        # Try loading from JSON file
        filepath = self.personas_dir / f"{persona_id}.json"
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                persona = json.load(f)
                self.personas[persona_id] = persona
                return persona
        
        return None
    
    def list_personas(self) -> List[str]:
        """List all available persona IDs."""
        persona_ids = set(self.personas.keys())
        
        # Also scan JSON files
        for filepath in self.personas_dir.glob("*.json"):
            persona_ids.add(filepath.stem)
        
        return sorted(list(persona_ids))
    
    def get_persona_summary(self, persona_id: str) -> str:
        """Get a human-readable summary of a persona."""
        persona = self.load_persona(persona_id)
        if not persona:
            return f"Persona {persona_id} not found."
        
        traits = persona.get("Traits", {})
        name = persona.get("Name", "Unknown")
        
        summary = f"**{persona_id}** ({name})\n"
        summary += "Traits:\n"
        for trait, value in traits.items():
            summary += f"  - {trait}: {value}\n"
        
        return summary
