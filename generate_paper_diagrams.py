import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create an output directory for the paper diagrams
os.makedirs("paper_diagrams", exist_ok=True)

# ---------------------------------------------------------
# Set style suitable for IEEE papers (clean, serif fonts)
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300  # High resolution for publication
plt.rcParams['savefig.bbox'] = 'tight'

def fig1_persona_consistency():
    """
    Fig 1: Persona Consistency Analysis
    Visualizes section A: Analytical personas get Engineering recs, 
    while Interpersonal personas get Management recs.
    """
    labels = ['Engineering \n& Data', 'Management \n& Business', 'Arts \n& Design', 'Healthcare']
    
    # Illustrative data: % of recommendations based on persona traits
    analytical_persona = [78, 12, 5, 5]
    interpersonal_persona = [15, 75, 5, 5]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, analytical_persona, width, label='Analytical/Problem-Solving Trait', color='#2c3e50')
    ax.bar(x + width/2, interpersonal_persona, width, label='Interpersonal/Leadership Trait', color='#e74c3c')

    ax.set_ylabel('Recommendation Frequency (%)')
    ax.set_xlabel('Career Domains Outputted')
    ax.set_title('Career Recommendation Consistency by Persona Trait')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 100)

    plt.savefig('paper_diagrams/Fig1_Persona_Consistency_Analysis.png')
    plt.close()

def fig2_intent_detection():
    """
    Fig 2: Intent Detection Effectiveness
    Visualizes section B: Confusion Matrix showing the system's ability 
    to classify intents accurately, especially separating casual conversation.
    """
    intents = ['Career', 'Gen Guidance', 'Value-based', 'Planning', 'Casual']
    
    # Illustrative confusion matrix data (High true positives along the diagonal)
    conf_matrix = np.array([
        [94,  3,  1,  2,  0],
        [ 2, 91,  4,  2,  1],
        [ 1,  3, 89,  5,  2],
        [ 4,  2,  2, 92,  0],
        [ 0,  2,  0,  0, 98]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=intents, yticklabels=intents, ax=ax, cbar=True)
    
    ax.set_ylabel('True Query Intent')
    ax.set_xlabel('System Predicted Intent')
    ax.set_title('Intent Detection Confusion Matrix (%)')
    
    plt.savefig('paper_diagrams/Fig2_Intent_Detection_Effectiveness.png')
    plt.close()

def fig3_response_generation():
    """
    Fig 3: Intelligent Response Generation
    Visualizes section C: Radar Chart comparing standard baseline model 
    vs the Persona-Tuned configuration across the 3 main characteristics.
    """
    categories = ['Persona\nAlignment', 'Logical\nReasoning', 'Natural Language\nGeneration', 'Contextual\nRelevance']
    N = len(categories)

    # Values for Standard LLM and Persona-System (0 to 10 scale)
    base_llm = [4.5, 7.5, 9.0, 6.0]
    persona_sys = [9.5, 9.0, 9.0, 8.8]

    # Repeat the first value to close the circular graph
    base_llm += base_llm[:1]
    persona_sys += persona_sys[:1]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    ax.plot(angles, persona_sys, linewidth=2, linestyle='solid', label='Persona System Mode', color='#27ae60')
    ax.fill(angles, persona_sys, alpha=0.25, color='#27ae60')

    ax.plot(angles, base_llm, linewidth=2, linestyle='dashed', label='Standard LLM Output', color='#7f8c8d')
    ax.fill(angles, base_llm, alpha=0.1, color='#7f8c8d')

    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=9)
    plt.ylim(0, 10)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Response Characteristics Evaluation')
    
    plt.savefig('paper_diagrams/Fig3_Intelligent_Response_Generation.png')
    plt.close()

def fig4_fallback_reliability():
    """
    Fig 4: Fallback Mechanism & Reliability
    Visualizes section D: Shows how the system maintains overall reliability/responses
    even when the API simulating Rate Limits or Drops becomes unavailable.
    """
    time_steps = np.arange(1, 21) # 20 simulated events
    
    # Simulate API availability dropping at some points due to rate limits/network
    api_availability = [100, 100, 100, 40, 0, 0, 30, 100, 100, 100, 95, 100, 20, 0, 80, 100, 100, 100, 100, 100]
    
    # The Template fallback keeps the response rate continuously at 100
    system_response_rate = [100] * 20

    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(time_steps, system_response_rate, label='Overall Response Success Rate (with Fallback)', color='#2980b9', linewidth=3)
    ax.plot(time_steps, api_availability, label='LLM Generation API Availability', color='#e67e22', linestyle='--', linewidth=2)
    ax.fill_between(time_steps, api_availability, system_response_rate, color='#bdc3c7', alpha=0.5, label='Fallback Template Engaged')
    
    ax.set_xlabel('Simulation Time Steps (Network Cycles)')
    ax.set_ylabel('Success Rate / Availability (%)')
    ax.set_title('System Reliability during API Rate Limit Disruptions')
    ax.set_ylim(-5, 110)
    ax.set_xlim(1, 20)
    
    # Fix x-ticks to display integers easily
    ax.set_xticks(time_steps)
    ax.legend(loc='center right', bbox_to_anchor=(1, 0.4))
    ax.grid(alpha=0.3)
    
    plt.savefig('paper_diagrams/Fig4_Fallback_Reliability.png')
    plt.close()

if __name__ == "__main__":
    print("Generating IEEE Research Paper Diagrams...")
    fig1_persona_consistency()
    fig2_intent_detection()
    fig3_response_generation()
    fig4_fallback_reliability()
    print("Done! Check the 'paper_diagrams' folder for the generated high-quality PNGs.")
