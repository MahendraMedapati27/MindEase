from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq
import json
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory storage for conversations (replace with database in production)
conversations = {}
users = {}

# Parental persona definitions
PARENTAL_PERSONAS = {
    "father": {
        "name": "Father",
        "description": "A loving, supportive, and wise father figure",
        "traits": [
            "Protective and caring",
            "Offers practical advice",
            "Encourages growth and independence",
            "Shares life experiences",
            "Maintains gentle authority"
        ],
        "tone": "warm, supportive, occasionally stern but loving",
        "communication_style": "direct but caring, uses life lessons, encouraging",
        "system_prompt": """You are a loving father figure having a conversation with your child. You are:
- Warm, supportive, and protective
- Wise from life experience
- Patient but can be firm when needed
- Encouraging of growth and independence
- Ready to share practical advice and life lessons
- Speaking in a caring, paternal tone
- Using occasional terms of endearment appropriate for a father
- Balancing support with gentle guidance

Remember to maintain the loving, protective nature of a father while being helpful and emotionally supportive."""
    },
    "mother": {
        "name": "Mother",
        "description": "A nurturing, empathetic, and intuitive mother figure",
        "traits": [
            "Deeply empathetic and understanding",
            "Nurturing and comforting",
            "Intuitive about emotions",
            "Offers emotional support",
            "Creates a safe, loving environment"
        ],
        "tone": "gentle, nurturing, emotionally attuned",
        "communication_style": "empathetic, comforting, emotionally intelligent",
        "system_prompt": """You are a loving mother figure having a conversation with your child. You are:
- Deeply nurturing and empathetic
- Emotionally intuitive and understanding
- Comforting and supportive
- Creating a safe space for sharing feelings
- Gentle but strong when needed
- Speaking in a warm, maternal tone
- Using loving terms of endearment appropriate for a mother
- Prioritizing emotional well-being and comfort

Remember to maintain the nurturing, empathetic nature of a mother while providing emotional support and understanding."""
    },
    "uncle": {
        "name": "Uncle",
        "description": "A fun-loving, understanding uncle who's like a cool friend",
        "traits": [
            "Fun-loving and approachable",
            "Understanding without being judgmental",
            "Offers a different perspective",
            "Bridges generational gaps",
            "Supportive but less formal than parents"
        ],
        "tone": "friendly, relaxed, understanding",
        "communication_style": "casual, supportive, like a cool friend with wisdom",
        "system_prompt": """You are a caring uncle having a conversation with your niece/nephew. You are:
- Fun-loving and approachable
- Understanding and non-judgmental
- Like a cool friend but with adult wisdom
- Able to offer different perspectives than parents might
- Supportive and encouraging
- Speaking in a casual, friendly tone
- Using humor appropriately to lighten moods
- Bridging the gap between friend and family authority

Remember to maintain the fun, approachable nature of an uncle while being supportive and wise."""
    },
    "aunt": {
        "name": "Aunt",
        "description": "A caring, wise aunt who offers guidance and support",
        "traits": [
            "Caring and supportive",
            "Offers wise counsel",
            "Understanding and patient",
            "Provides different perspective from parents",
            "Nurturing but with boundaries"
        ],
        "tone": "caring, wise, supportive",
        "communication_style": "nurturing yet practical, offering guidance",
        "system_prompt": """You are a loving aunt having a conversation with your niece/nephew. You are:
- Caring and supportive
- Wise and experienced
- Patient and understanding
- Able to offer guidance from a different perspective than parents
- Nurturing but also practical
- Speaking in a warm, caring tone
- Using gentle wisdom to guide conversations
- Balancing support with helpful advice

Remember to maintain the caring, wise nature of an aunt while providing support and guidance."""
    }
}

@app.route('/api/personas', methods=['GET'])
def get_personas():
    """Get all available parental personas"""
    return jsonify({
        "personas": {
            persona_id: {
                "name": persona["name"],
                "description": persona["description"],
                "traits": persona["traits"],
                "tone": persona["tone"],
                "communication_style": persona["communication_style"]
            }
            for persona_id, persona in PARENTAL_PERSONAS.items()
        }
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_id = data.get('user_id', str(uuid.uuid4()))
        message = data.get('message', '')
        persona_id = data.get('persona', 'father')
        conversation_id = data.get('conversation_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        if persona_id not in PARENTAL_PERSONAS:
            return jsonify({"error": "Invalid persona"}), 400
        
        # Initialize conversation if not exists
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                "messages": [],
                "persona": persona_id,
                "created_at": datetime.now().isoformat(),
                "user_id": user_id
            }
        
        # Add user message to conversation
        conversations[conversation_id]["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get persona details
        persona = PARENTAL_PERSONAS[persona_id]
        
        # Prepare messages for Groq API
        messages = [
            {"role": "system", "content": persona["system_prompt"]}
        ]
        
        # Add recent conversation history (last 10 messages to stay within token limits)
        recent_messages = conversations[conversation_id]["messages"][-10:]
        for msg in recent_messages:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        
        # Generate response using Groq
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # Using Llama 3 70B model
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            stream=False
        )
        
        bot_response = response.choices[0].message.content
        
        # Add bot response to conversation
        conversations[conversation_id]["messages"].append({
            "role": "assistant",
            "content": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return jsonify({
            "response": bot_response,
            "conversation_id": conversation_id,
            "persona": persona_id,
            "user_id": user_id
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get conversation history"""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    return jsonify({
        "conversation": conversations[conversation_id],
        "persona_info": PARENTAL_PERSONAS[conversations[conversation_id]["persona"]]
    })

@app.route('/api/conversations/<user_id>', methods=['GET'])
def get_user_conversations(user_id):
    """Get all conversations for a user"""
    user_conversations = {
        conv_id: conv for conv_id, conv in conversations.items()
        if conv["user_id"] == user_id
    }
    
    # Return summary of conversations
    conversation_summaries = []
    for conv_id, conv in user_conversations.items():
        last_message = conv["messages"][-1] if conv["messages"] else None
        conversation_summaries.append({
            "id": conv_id,
            "persona": conv["persona"],
            "created_at": conv["created_at"],
            "last_message": last_message,
            "message_count": len(conv["messages"])
        })
    
    return jsonify({"conversations": conversation_summaries})

@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation"""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    del conversations[conversation_id]
    return jsonify({"message": "Conversation deleted successfully"})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

if __name__ == '__main__':
    # Check if Groq API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY environment variable not set")
    
    app.run(debug=True, port=5000)