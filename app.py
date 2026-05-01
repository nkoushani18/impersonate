"""
Penta-PersonaAI Flask Application
Web interface for persona-based conversations.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from engine.chat_handler import ChatHandler
from engine.judge_manager import JudgeManager

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize chat handler with the survey CSV
CSV_PATH = project_root / "Persona Survey (Career and Relationship) (Responses) - Form Responses 1.csv"
chat_handler = ChatHandler(str(CSV_PATH) if CSV_PATH.exists() else None)

# Initialize judge manager
judge_manager = JudgeManager()


@app.route('/')
def index():
    """Main chat interface."""
    return render_template('index.html')


@app.route('/human')
def human_interface():
    """Human response interface."""
    return render_template('human.html')


@app.route('/judge')
def judge_results():
    """Judge results viewer."""
    return render_template('judge.html')


@app.route('/judging/live')
def judging_live():
    """Live judging animation page."""
    return render_template('judging_live.html')


@app.route('/api/personas', methods=['GET'])
def get_personas():
    """Get list of available personas."""
    personas = chat_handler.list_available_personas()
    
    # Get details for each persona
    persona_details = []
    for pid in personas:
        persona = chat_handler.persona_loader.load_persona(pid)
        if persona:
            persona_details.append({
                "id": pid,
                "name": persona.get("Name", "Unknown"),
                "traits": persona.get("Traits", {})
            })
    
    return jsonify({"personas": persona_details})


@app.route('/api/persona/select', methods=['POST'])
def select_persona():
    """Select active persona."""
    data = request.json
    persona_id = data.get('persona_id')
    
    if not persona_id:
        return jsonify({"error": "persona_id required"}), 400
    
    success = chat_handler.set_active_persona(persona_id)
    
    if success:
        return jsonify({
            "success": True,
            "persona": chat_handler.get_active_persona_info()
        })
    else:
        return jsonify({"error": f"Persona {persona_id} not found"}), 404


@app.route('/api/persona/active', methods=['GET'])
def get_active_persona():
    """Get currently active persona info."""
    info = chat_handler.get_active_persona_info()
    if "error" in info:
        return jsonify(info), 400
    return jsonify(info)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat message."""
    data = request.json
    message = data.get('message', '')
    debug = data.get('debug', False)
    
    if not message:
        return jsonify({"error": "message required"}), 400
    
    # Log for debugging
    print(f"[CHAT] Message received: '{message}'", flush=True)
    print(f"[CHAT] LLM Enabled: {chat_handler.use_llm}", flush=True)
    print(f"[CHAT] Active persona: {chat_handler.active_persona_id}", flush=True)
    
    result = chat_handler.handle_message(message, debug=debug)
    print(f"[CHAT] Response type: {result.get('type')}", flush=True)
    response_text = result.get('response', '') or ''
    print(f"[CHAT] Response: {response_text[:50]}...", flush=True)
    
    # Create a comparison session if we have an active persona
    if chat_handler.active_persona_id and result.get('type') not in ['error', 'small_talk']:
        try:
            question_id = judge_manager.create_comparison_session(
                persona_id=chat_handler.active_persona_id,
                user_question=message,
                ai_response=response_text
            )
            result['question_id'] = question_id
            print(f"[JUDGE] Created comparison session: {question_id}", flush=True)
        except Exception as e:
            print(f"[JUDGE] Failed to create comparison session: {e}", flush=True)
    
    return jsonify(result)


@app.route('/api/human/pending', methods=['GET'])
def get_pending_questions_for_human():
    """Get pending questions for the human to answer."""
    persona_id = request.args.get('persona_id')
    
    if not persona_id:
        return jsonify({"error": "persona_id required"}), 400
    
    pending = judge_manager.get_pending_questions(persona_id)
    
    return jsonify({
        "persona_id": persona_id,
        "pending_questions": pending,
        "count": len(pending)
    })


@app.route('/api/human/respond', methods=['POST'])
def human_respond():
    """Submit human response and trigger judge evaluation."""
    data = request.json
    question_id = data.get('question_id')
    human_response = data.get('response', '')
    
    if not question_id or not human_response:
        return jsonify({"error": "question_id and response required"}), 400
    
    # Get persona name for the question
    if question_id not in judge_manager.pending_questions:
        return jsonify({"error": "Question ID not found"}), 404
    
    session = judge_manager.pending_questions[question_id]
    persona_id = session['persona_id']
    persona = chat_handler.persona_loader.load_persona(persona_id)
    persona_name = persona.get("Name", "Unknown") if persona else "Unknown"
    
    # Submit response and get judge result
    result = judge_manager.submit_human_response(
        question_id=question_id,
        human_response=human_response,
        persona_name=persona_name
    )
    
    print(f"[JUDGE] Human responded to {question_id}, Score: {result.get('judge_result', {}).get('score', 'N/A')}", flush=True)
    
    return jsonify(result)


@app.route('/api/judge/results', methods=['GET'])
def get_judge_results():
    """Get judge results and history for a persona."""
    persona_id = request.args.get('persona_id')
    
    if not persona_id:
        # If no persona specified, return error
        return jsonify({"error": "persona_id required"}), 400
    
    # Get comparison history
    history = judge_manager.get_comparison_history(persona_id)
    latest = judge_manager.get_latest_comparison(persona_id)
    stats = judge_manager.get_statistics(persona_id)
    
    # Get persona info
    persona = chat_handler.persona_loader.load_persona(persona_id)
    persona_name = persona.get("Name", "Unknown") if persona else "Unknown"
    
    return jsonify({
        "persona_id": persona_id,
        "persona_name": persona_name,
        "latest_comparison": latest,
        "history": history,
        "comparisons": history,  # Add this for compatibility with judging_live.html
        "statistics": stats,
        "stats": stats  # Also add stats alias
    })


@app.route('/api/system/identity', methods=['GET'])
def system_identity():
    """Return system identity statement."""
    return jsonify({
        "identity": "I am a persona-based agent that reasons using structured traits derived from survey responses, not a learning system.",
        "principle": "The system models latent human traits from structured survey data and uses them for consistent, responsible, agentic reasoning at inference time."
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("Penta-PersonaAI with LLM Judge System Starting...")
    print("="*70)
    print(f"[*] Loading personas from: {CSV_PATH}")
    print(f"[*] LLM Enabled: {chat_handler.use_llm}")
    print("\n" + "="*70)
    print("\n" + "="*70)
    print("ACCESS THE 3 INTERFACES:")
    print("="*70)
    print("1. Main Chat (AI Agent):     http://127.0.0.1:5000/")
    print("2. Human Interface:          http://127.0.0.1:5000/human")
    print("3. Judge Results:             http://127.0.0.1:5000/judge")
    print("="*70)
    print("WARNING: Make sure Ollama is running: ollama serve")
    print("="*70 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
