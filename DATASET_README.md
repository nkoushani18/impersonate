# Dataset README — Extra Dataset for Penta-PersonaAI

---# 1. Name of the Dataset

**Penta-PersonaAI: A Survey Dataset Based on Personas for Career and Relationship Advice**

---# 2. Description of the Dataset

This dataset was gathered as part of the research project **"Penta-PersonaAI"**, a multi-agent AI system that maps individual behavioural traits to one of five unique user personas in order to provide personalised career and relationship guidance.

A structured 10-question psychographic survey covering real-life decision-making situations pertaining to personal values, risk tolerance, relationship conflicts, and career choices was completed by participants. A deterministic rule-based trait engine was then used to map each participant's responses to a set of personality traits that characterise their persona archetype.

The dataset allows the persona classification pipeline described in the paper to be fully reproducible by capturing both the **derived trait profiles** and the **raw survey responses** for each of the five participants.

The method used to collect data was a structured survey using Google Forms.
Relationship behaviour and career decisions are the survey domain.
Rule-based trait mapping (no machine learning) is the method used for persona classification.
There were five participants.
Ten multiple-choice questions with three possible answers each make up the survey.

--- ##3. File Details Property | Value | |---|---| | **Filename** | `penta_persona_dataset.csv` | | **Format** | Comma-Separated Values (CSV), UTF-8 encoded | | **Number of Columns** | 25 | **Number of Rows** | 5 (one row for each participant/persona) | **Header Row** | Yes (Row 1) |---
## 4. Column Dictionary

### Section A — Participant Identification (Columns 1–2)

| Column Name | Type | Description |
|---|---|---|
| `Persona_ID` | String | Unique identifier assigned to each participant (Persona_1 to Persona_5) |
| `Participant_Name` | String | First name or pseudonym of the survey respondent |

---

### Section B — Raw Survey Responses (Columns 3–12)

Each column corresponds to one survey question. Values are the exact answer option selected by the participant.

| Column Name | Maps To Trait | Survey Question |
|---|---|---|
| `Q1 - When a major career opportunity requires moving away from family you` | `Family_Priority` | Tests emotional vs. analytical response to family-career conflict |
| `Q2 - In a relationship conflict caused by work pressure you usually` | `Conflict_Handling` | Identifies communication style under relationship stress |
| `Q3 - When choosing a career path you trust more` | `Career_Orientation` | Distinguishes logical vs. visionary vs. guided career thinking |
| `Q4 - Your ideal future life looks like` | `Future_Preference` | Captures preference for stability, growth, or flexibility |
| `Q5 - If your partner strongly disagrees with your career choice you` | `Decision_Style` | Measures independence and firmness in decision-making |
| `Q6 - Under uncertainty about career and relationships you` | `Risk_Tolerance` | Identifies coping mechanism under uncertainty |
| `Q7 - People close to you often describe you as` | `Personality_Type` | Self-reported personality archetype |
| `Q8 - When making an important life decision you` | `Stress_Response` | Captures cognitive/emotional approach to high-stakes decisions |
| `Q9 - In both career and relationships you value more` | `Core_Value` | Identifies the participant's primary motivational value |
| `Q10 - Looking at your future you believe success comes from` | `Success_Driver` | Captures the participant's philosophy on achieving success |

---

### Section C — Derived Persona Traits (Columns 13–25)

These values are automatically derived from the survey answers using the rule-based `TRAIT_MAPPINGS` engine (`data/trait_mapping.py`). Empty cells indicate the trait is not applicable or not triggered for that persona.

| Column Name | Possible Values | Description |
|---|---|---|
| `Family_Priority` | High / Medium / Low | Degree to which family proximity influences career decisions |
| `Relationship_Priority` | High / Medium | Importance placed on romantic relationships in life choices |
| `Conflict_Handling` | Communication-based / Avoidance / Confrontation | Style used to manage interpersonal conflict |
| `Career_Orientation` | Logical / Visionary / Guided | How the participant approaches career decision-making |
| `Risk_Tolerance` | High / Medium / Low | Willingness to accept uncertainty in career/life decisions |
| `Future_Preference` | Stability / Growth / Flexibility | Preferred life trajectory |
| `Decision_Style` | Quick / Reflective / Delayed / Firm / Analytical / Accommodating | How decisions are made under pressure |
| `Independence` | High / Low | Degree of self-reliance in major life decisions |
| `Uncertainty_Response` | Optimistic / Planning / Collaborative | Behavioral response when facing ambiguous situations |
| `Personality_Type` | Empathetic / Visionary / Practical | Dominant personality archetype as perceived by peers |
| `Emotional_Orientation` | High / Low | Degree to which emotions influence decision-making |
| `Creativity` | High | Present only for Visionary personality types |
| `Stress_Response` | Logical / Emotional / Cautious | Cognitive style when making high-stakes decisions |
| `Core_Value` | Emotional Security / Freedom / Achievement | The participant's primary motivational value |
| `Success_Driver` | Consistency / Exploration / Relationships | The participant's believed path to long-term success |

---

## 5. Persona Archetypes Summary

| Persona_ID | Participant | Personality Type | Career Orientation | Future Preference | Core Value |
|---|---|---|---|---|---|
| Persona_1 | Koushani Nath | Empathetic | Logical | Stability | Emotional Security |
| Persona_2 | Rishit Tandon | Visionary | Visionary | Growth | Emotional Security |
| Persona_3 | Harshv. | Practical | Visionary | Flexibility | Freedom |
| Persona_4 | Manya Sgarma | Empathetic | Visionary | Growth | Emotional Security |
| Persona_5 | SALIL SHEKHAR | Practical | Visionary | Stability | Emotional Security |

---

## 6. How to Use This Dataset

### Loading the CSV (Python)
```python
import pandas as pd

df = pd.read_csv("penta_persona_dataset.csv")
print(df.shape)        # (5, 25)
print(df.columns)      # All 25 column names
print(df["Persona_ID"])
```

### Reproducing Persona Classification
The trait derivation logic is fully documented in `data/trait_mapping.py`. To reproduce the trait profile for any participant:
1. Take any Q1–Q10 answer string
2. Look it up in the `TRAIT_MAPPINGS` dictionary
3. The returned dict contains all traits triggered by that answer

### Extending the Dataset
To add new participants, append a new row with:
- A new `Persona_ID` (e.g., `Persona_6`)
- Answers selected from the valid option sets defined in `data/survey_questions.py`
- Derived traits computed via `data/trait_mapping.py`

---

## 7. Ethical Considerations

- All participants provided informed consent for their responses to be used in academic research.
- Participant names are included as pseudonyms/first names only, with no additional personally identifiable information (PII) collected.
- No sensitive demographic data (age, gender, location, etc.) was collected.
- The dataset is released strictly for academic and research purposes.

---

## 8. Citation

If you use this dataset in your research, please cite the associated paper:

```
[Authors], "Penta-PersonaAI: A Multi-Agent AI System for Persona-Based 
Career and Relationship Guidance," [Conference Name], [Year].
```

---

## 9. Contact

For questions regarding the dataset or the research system, please contact the corresponding author via the IEEE paper submission portal.

---

*Last Updated: March 2026*
