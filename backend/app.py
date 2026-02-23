from flask import Flask, request, jsonify
from flask_cors import CORS
from deep_translator import GoogleTranslator
import ollama
# Handle Keras/TensorFlow import safely
try:
    import numpy as np
    import tensorflow as tf
    from keras.layers import TFSMLayer
    from tensorflow.keras.preprocessing import image
    ML_AVAILABLE = True
except ImportError:
    print("Warning: Keras/TensorFlow not fully installed. Crop Disease Detection will be unavailable.")
    ML_AVAILABLE = False

import tempfile
import os

app = Flask(__name__)
CORS(app)

# --- ML Model Setup ---
IMG_SIZE = 224
model = None

try:
    # Attempt to load model if it exists
    model_path = "model_saved"
    if os.path.exists(model_path):
        model = TFSMLayer(model_path, call_endpoint="serving_default")
        print("✅ Crop Disease Model loaded successfully")
    else:
        print(f"⚠️ Model directory '{model_path}' not found. Prediction endpoint will fail.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")


# 🔹 CLASS ID → DISEASE NAME
CLASS_NAMES = {
    0: "Tomato Target Spot",
    1: "Tomato Mosaic Virus",
    2: "Tomato Yellow Leaf Curl Virus",
    3: "Healthy"
}

# 🔹 CLASS ID → ALL RECOMMENDATIONS
RECOMMENDATIONS = {
    0: [
        "Remove infected leaves",
        "Avoid overhead irrigation",
        "Apply suitable fungicide",
        "Improve air circulation",
        "Maintain field hygiene",
        "Monitor crops regularly"
    ],
    1: [
        "Remove infected plants",
        "Control insect vectors",
        "Use resistant varieties",
        "Avoid overcrowding",
        "Maintain proper sanitation",
        "Monitor nearby plants"
    ],
    2: [
        "Use virus-free seedlings",
        "Control whiteflies",
        "Remove affected plants",
        "Apply reflective mulch",
        "Use resistant hybrids",
        "Ensure proper spacing"
    ],
    3: [
        "Continue regular monitoring",
        "Maintain proper irrigation",
        "Apply balanced fertilizers",
        "Keep field weed-free",
        "Follow crop rotation",
        "Use certified seeds"
    ]
}

# 🔹 ROTATION STATE
rotation_index = {}

def predict_image(img_path):
    if not model:
        raise Exception("Model not loaded")
        
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model(img)

    if isinstance(preds, dict):
        preds = list(preds.values())[0]

    preds = preds.numpy()
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return class_id, confidence

def get_rotating_recommendations(class_id):
    all_recs = RECOMMENDATIONS.get(class_id, [])
    idx = rotation_index.get(class_id, 0)

    start = (idx * 2) % len(all_recs)
    selected = all_recs[start:start + 2]

    rotation_index[class_id] = idx + 1
    return selected

# --- End ML Setup ---

# Helper function for free translation
def translate_msg(text, target):
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except:
        return text

@app.route('/get-advice', methods=['POST'])
def get_advice():
    data = request.json
    user_text = data.get('message', '')
    user_lang = data.get('lang', 'en') # 'te', 'hi', or 'en'

    # 1. Translate User Voice -> English for the AI Brain
    english_query = translate_msg(user_text, 'en')
    print(f"Original ({user_lang}): {user_text} -> English: {english_query}")

    # 2. AI Expert Logic
    prompt = f"""
    [ROLE: Indian Agriculture Scientist]
    User Query: {english_query}
    Task: Analyze the soil and crop. Give 2 sentences of specific expert advice.
    """
    
    try:
        response = ollama.generate(model='llama3.2:1b', prompt=prompt)
        english_answer = response['response'].strip()
    except Exception as e:
        print(f"Ollama Error: {e}")
        english_answer = "I am currently unable to access my brain. Please ensure Ollama is running."

    # 3. Translate AI Advice -> User's Original Language
    final_answer = translate_msg(english_answer, user_lang)
    print(f"Final Output ({user_lang}): {final_answer}")

    return jsonify({
        "answer": final_answer,
        "lang_code": f"{user_lang}-IN"
    })

@app.route('/detect', methods=['POST'])
def detect():
    if not ML_AVAILABLE or not model:
        # MOCK MODE: Return simulated response if model is offline
        print("⚠️ Model/TF unavailable. Using Smart Simulation.")
        import random
        
        # 1. Try to guess from filename for better demo experience
        mock_class_id = 3 # Default to Healthy if unsure
        filename = ""
        if "image" in request.files:
            filename = request.files["image"].filename.lower()
        
        if "healthy" in filename:
            mock_class_id = 3
        elif "mosaic" in filename or "virus" in filename:
            mock_class_id = 1
        elif "curl" in filename:
            mock_class_id = 2
        elif "spot" in filename or "blight" in filename:
            mock_class_id = 0
        else:
            # Fallback: Random but biased towards Healthy (40%) to avoid "scaring" farmers
            # 0=Spot, 1=Mosaic, 2=Curl, 3=Healthy
            weights = [20, 20, 20, 40]
            mock_class_id = random.choices([0, 1, 2, 3], weights=weights, k=1)[0]
            
        confidence = random.uniform(0.88, 0.99)
        
        disease_name = CLASS_NAMES.get(mock_class_id, "Unknown")
        recommendations = get_rotating_recommendations(mock_class_id)
        
        return jsonify({
            "class_id": mock_class_id,
            "disease": disease_name,
            "confidence": confidence,
            "recommendations": recommendations,
            "note": "SIMULATION MODE (Real Model Missing)"
        })

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            class_id, confidence = predict_image(tmp.name)
        
        os.unlink(tmp.name)

        disease_name = CLASS_NAMES.get(class_id, "Unknown")
        recommendations = get_rotating_recommendations(class_id)

        return jsonify({
            "class_id": class_id,
            "disease": disease_name,
            "confidence": float(confidence),
            "recommendations": recommendations
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
