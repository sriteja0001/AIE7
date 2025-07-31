from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question')
        api_keys = data.get('apiKeys', {})
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Set API keys from frontend
        if api_keys.get('openai'):
            os.environ["OPENAI_API_KEY"] = api_keys['openai']
        if api_keys.get('tavily'):
            os.environ["TAVILY_API_KEY"] = api_keys['tavily']
        if api_keys.get('langsmith'):
            os.environ["LANGCHAIN_API_KEY"] = api_keys['langsmith']
        
        # Import your agent (import here so API keys are set first)
        from app import agent_chain_with_formatting
        
        # Run the agent
        result = agent_chain_with_formatting.invoke({"question": question})
        
        return jsonify({'answer': result})
        
    except Exception as e:
        return jsonify({'error': 'Failed to process request', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Starting Student Health Advisor Backend...")
    print("📍 Backend running on http://localhost:5001")
    print("🔗 Frontend should connect to this backend")
    app.run(host='0.0.0.0', port=5001, debug=True) 