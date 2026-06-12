import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration - Update these to match your actual file names
MODELS_DIR = 'models'
MODEL_EN_UR = os.path.join(MODELS_DIR, 'english_to_urdu_new.h5')
MODEL_EN_PA = os.path.join(MODELS_DIR, 'english_to_punjabi_new.h5')
TOKEN_EN = os.path.join(MODELS_DIR, 'english_tokenizer.pkl')
TOKEN_UR = os.path.join(MODELS_DIR, 'urdu_tokenizer.pkl')
TOKEN_PA = os.path.join(MODELS_DIR, 'punjabi_tokenizer.pkl')

# New configuration for Hugging Face custom-trained model
HF_MODEL_DIR = os.getenv('HF_MODEL_DIR', r"D:\final model trained ali\checkpoint-quantized")

# Try to import torch and transformers for the new fine-tuned model
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

def find_m2m_lang_token(tokenizer, code):
    code_lower = code.lower().strip()
    
    # Retrieve language map
    lang_to_id = getattr(tokenizer, 'lang_code_to_id', {})
    if not lang_to_id:
        try:
            lang_to_id = tokenizer.get_vocab()
        except Exception:
            lang_to_id = {}
            
    patterns = {
        'en': ['eng_latn', 'en_', 'en_xx', 'en'],
        'ur': ['urd_arab', 'ur_', 'ur_pk', 'ur'],
        'pa': ['pan_Arab', 'pan_arab', 'pan', 'pa']
    }
    
    candidates = patterns.get(code_lower, [code_lower])
    
    # First pass: look for exact match in vocabulary
    for cand in candidates:
        for k in lang_to_id.keys():
            k_low = k.lower()
            if k_low == cand or k_low == f"__{cand}__" or k_low == f"[{cand}]" or k_low == f"<{cand}>":
                return k
                
    # Second pass: clean the vocabulary k to check exact clean match
    for cand in candidates:
        cand_clean = cand.strip('_[]<>')
        for k in lang_to_id.keys():
            k_clean = k.lower().strip('_[]<>')
            if k_clean == cand_clean:
                return k
                
    # Third pass: starts with or contains as a fallback (but only if candidate length >= 3 to avoid short 2-letter tags matching wrong languages like 'pa' starting with 'pag')
    for cand in candidates:
        if len(cand) >= 3:
            for k in lang_to_id.keys():
                k_low = k.lower()
                if k_low.startswith(cand) or k_low.startswith(f"__{cand}") or k_low.startswith(f"<{cand}"):
                    return k
                    
    # Ultimate fallback in case nothing matches
    for cand in candidates:
        for k in lang_to_id.keys():
            if cand in k.lower():
                return k
                
    return code

def find_m2m_lang_id(tokenizer, code):
    code_lower = code.lower().strip()
    
    # Try calling get_lang_id directly if it exists and works
    if hasattr(tokenizer, 'get_lang_id'):
        try:
            return tokenizer.get_lang_id(code)
        except Exception:
            pass
            
    # Retrieve lang map
    lang_to_id = getattr(tokenizer, 'lang_code_to_id', {})
    if not lang_to_id:
        try:
            lang_to_id = tokenizer.get_vocab()
        except Exception:
            lang_to_id = {}
            
    patterns = {
        'en': ['eng_latn', 'en_', 'en_xx', 'en'],
        'ur': ['urd_arab', 'ur_', 'ur_pk', 'ur'],
        'pa': ['pan_Arab', 'pan_arab', 'pan', 'pa']
    }
    
    candidates = patterns.get(code_lower, [code_lower])
    
    # First pass: exact match
    for cand in candidates:
        for k, v in lang_to_id.items():
            k_low = k.lower()
            if k_low == cand or k_low == f"__{cand}__" or k_low == f"[{cand}]" or k_low == f"<{cand}>":
                return v
                
    # Second pass: clean exact match
    for cand in candidates:
        cand_clean = cand.strip('_[]<>')
        for k, v in lang_to_id.items():
            k_clean = k.lower().strip('_[]<>')
            if k_clean == cand_clean:
                return v
                
    # Third pass: starts with (length >= 3)
    for cand in candidates:
        if len(cand) >= 3:
            for k, v in lang_to_id.items():
                k_low = k.lower()
                if k_low.startswith(cand) or k_low.startswith(f"__{cand}") or k_low.startswith(f"<{cand}"):
                    return v
                    
    # Ultimate fallback
    for cand in candidates:
        for k, v in lang_to_id.items():
            if cand in k.lower():
                return v
                
    try:
        token_id = tokenizer.convert_tokens_to_ids(code)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return token_id
    except Exception:
        pass
        
    return None

# Global storage for both TensorFlow legacy assets and new Hugging Face models
assets = {
    'engine': 'uninitialized', # 'hf' for HuggingFace Transformers, 'tf' for Keras, or 'failed'
    'tf_models': {},
    'tf_tokenizers': {},
    'hf_model': None,
    'hf_tokenizer': None,
    'hf_is_causal': True,
    'hf_device': 'cpu'
}
assets_ready = False

def load_assets():
    global assets_ready
    
    # ----------------- LAYER 1: Try Loading Hugging Face Model -----------------
    if TRANSFORMERS_AVAILABLE:
        # Check if the fine-tuned model directory or its core config exists
        config_path = os.path.join(HF_MODEL_DIR, 'config.json')
        if os.path.exists(HF_MODEL_DIR) and (os.path.exists(config_path) or any(f.endswith('.safetensors') for f in os.listdir(HF_MODEL_DIR) if os.path.isfile(os.path.join(HF_MODEL_DIR, f)))):
            print(f"--- [SYSTEM] Detected Hugging Face Fine-tuned Model at {HF_MODEL_DIR} ---")
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"--- [SYSTEM] Using hardware device: {device} ---")
                
                # Load tokenizer
                print("--- [SYSTEM] Loading HF Tokenizer... ---")
                tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
                
                # Check config to determine model architecture (Seq2Seq vs CausalLM)
                is_causal = True
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        cfg = json.load(f)
                        architectures = cfg.get('architectures', [])
                        print(f"--- [SYSTEM] Model architectures: {architectures} ---")
                        for arch in architectures:
                            if 'Seq2Seq' in arch or 'ConditionalGeneration' in arch or 'Marian' in arch or 'T5' in arch:
                                is_causal = False
                                break
                
                # Load model weight dynamically based on architecture
                print(f"--- [SYSTEM] Loading HF Model (Is Causal/Decoder-only: {is_causal})... ---")
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                if is_causal:
                    model = AutoModelForCausalLM.from_pretrained(
                            HF_MODEL_DIR,
                            dtype=torch.float32,
                            low_cpu_mem_usage=True
                            )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                            HF_MODEL_DIR,
                          dtype=torch.float32,
                         low_cpu_mem_usage=True
                         )
                
                assets['engine'] = 'hf'
                assets['hf_model'] = model
                assets['hf_tokenizer'] = tokenizer
                assets['hf_is_causal'] = is_causal
                assets['hf_device'] = device
                
                print("--- [SYSTEM] Hugging Face Translation Model Loaded Successfully! ---")
                assets_ready = True
                return True
            except Exception as hf_err:
                import traceback
                print(f"--- [WARNING] Hugging Face loader failed: {str(hf_err)} ---")
                traceback.print_exc()
                print("--- Falling back to Legacy-TF engine ---")
        else:
            print(f"--- [INFO] Hugging Face fine-tuned checkpoint directory '{HF_MODEL_DIR}' not found. ---")
    else:
        print("--- [INFO] 'torch' or 'transformers' packages not installed. Run 'pip install torch transformers' to use safetensors engine. ---")

    # ----------------- LAYER 2: Legacy TensorFlow / Keras H5 fallback -----------------
    print("--- [SYSTEM] Trying to load legacy TensorFlow Translation Models... ---")
    try:
        import tensorflow as tf
        required_files = [MODEL_EN_UR, MODEL_EN_PA, TOKEN_EN, TOKEN_UR, TOKEN_PA]
        missing = [f for f in required_files if not os.path.exists(f)]
        
        if missing:
            print("--- [ERROR] Missing Legacy TensorFlow Assets ---")
            for m in missing: print(f"  - {m}")
            assets['engine'] = 'failed'
            return False

        print("--- [SYSTEM] Loading TensorFlow Models... ---")
        assets['tf_models']['en_ur'] = tf.keras.models.load_model(MODEL_EN_UR)
        assets['tf_models']['en_pa'] = tf.keras.models.load_model(MODEL_EN_PA)
        
        print("--- [SYSTEM] Loading Tokenizers... ---")
        with open(TOKEN_EN, 'rb') as f: assets['tf_tokenizers']['en'] = pickle.load(f)
        with open(TOKEN_UR, 'rb') as f: assets['tf_tokenizers']['ur'] = pickle.load(f)
        with open(TOKEN_PA, 'rb') as f: assets['tf_tokenizers']['pa'] = pickle.load(f)
            
        assets['engine'] = 'tf'
        print("--- [SYSTEM] All Legacy TensorFlow Assets Loaded Successfully ---")
        assets_ready = True
        return True
    except Exception as e:
        print(f"--- [ERROR] Failed to load legay TF assets: {str(e)} ---")
        assets['engine'] = 'failed'
        return False

# Initial load attempt
load_assets()

def preprocess_text(text, tokenizer, max_len=50):
    import tensorflow as tf
    sequences = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded

def postprocess_prediction(prediction, tokenizer):
    indices = np.argmax(prediction, axis=-1)[0]
    words = []
    for idx in indices:
        if idx == 0: break 
        word = tokenizer.index_word.get(idx, '')
        if word == '<end>': break 
        if word: words.append(word)
    return ' '.join(words)

@app.route('/', methods=['GET'])
def index():
    if assets['engine'] == 'hf':
        engine_status = f"Running HF PyTorch/safetensors Model ({HF_MODEL_DIR})"
        supported_pairs = "English &harr; Urdu, English &harr; Punjabi (Shahmukhi), Urdu &harr; Punjabi (Shahmukhi)"
    else:
        engine_status = "Running Legacy Keras TF (.h5) Engine"
        supported_pairs = "English &rarr; Urdu, English &rarr; Punjabi"
    return f"<h1>Translation Server is Running!</h1><p>Active Backend Engine: <strong>{engine_status}</strong></p><p>Supported pairs: <strong>{supported_pairs}</strong></p>"

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'GET':
        return jsonify({
            "status": "online",
            "active_engine": assets['engine'],
            "hf_model_path": HF_MODEL_DIR,
            "message": "Send a POST request with your data to translate."
        })
        
    if not assets_ready:
        if not load_assets():
            return jsonify({
                "error": "Models or tokenizers could not be loaded. Please ensure you have placed your model files correctly.",
                "local_setup_tip": "If using HuggingFace model, run: 'pip install torch transformers' and verify your files exist in 'models/checkpoint-67804'."
            }), 500

    try:
        data = request.get_json()
        text = data.get('text', '')
        src = data.get('source_lang', '')
        tgt = data.get('target_lang', '')

        if not text or not text.strip(): return jsonify({"error": "No text provided"}), 400

        # Run translation using active Hugging Face Model
        if assets['engine'] == 'hf':
            model = assets['hf_model']
            tokenizer = assets['hf_tokenizer']
            device = assets['hf_device']
            is_causal = assets['hf_is_causal']
            
            # Format display names for logging / prompt guiding
            lang_map = {
                            'en': 'eng_Latn',
                            'ur': 'urd_Arab',
                            'pa': 'pan_Arab'
                        }
            from_lang = lang_map.get(src, src)
            to_lang = lang_map.get(tgt, tgt)
            
            # If Model is Causal decoder (like Qwen, Gemma, LLaMA), format with prompt template
            if is_causal:
                # Construct standard translation instruction prompt
                prompt = f"Translate from {from_lang} to {to_lang}:\n{text}\n\nTranslation:"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                input_len = inputs.input_ids.shape[1]
                
                with torch.no_grad():
                    # Generate options optimal for translation tasks
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.2,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    )
                
                # Split out input prompt to get pure translated tokens
                new_tokens = outputs[0][input_len:]
                translated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            else:
                # If Seq2Seq encoder-decoder model (like Marian, MT5, M2M100, NLLB)
                is_m2m_nllb = any(kw in str(type(model)).lower() or kw in str(type(tokenizer)).lower() 
                                  for kw in ["m2m100", "nllb", "marian", "mbart", "mt5"])
                
            if is_m2m_nllb:
                    try:
                        flores_map = {
                            'en': 'eng_Latn',
                            'ur': 'urd_Arab',
                            'pa': 'pan_Arab'
                        }
                        tgt_flores = flores_map.get(tgt, tgt)
                        src_flores = flores_map.get(src, src)

                        print(f"--- [SYSTEM] Translating: {src_flores} -> {tgt_flores} using prefix method ---")

                        # Use >>lang_code<< prefix — this is how the model was trained
                        prefixed_text = f">>{tgt_flores}<< {text}"
                        inputs = tokenizer(prefixed_text, return_tensors="pt", 
                                         max_length=128, truncation=True).to(device)

                        # Also try forced_bos_token_id as backup
                        token_id = tokenizer.convert_tokens_to_ids(tgt_flores)
                        gen_kwargs = {
                            "max_new_tokens": 256,
                            "num_beams": 4,
                            "no_repeat_ngram_size": 3,
                            "early_stopping": True
                        }
                        if token_id != tokenizer.unk_token_id:
                            gen_kwargs["forced_bos_token_id"] = token_id

                        with torch.no_grad():
                            outputs = model.generate(**inputs, **gen_kwargs)
                        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    except Exception as m2m_err:
                        print(f"--- [WARNING] Multilingual specific token generation failed: {str(m2m_err)}. Falling back to standard generation ---")
                        import traceback
                        traceback.print_exc()
                        inputs = tokenizer(text, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=256,
                                temperature=0.1
                            )
                        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            else:
                    prefix = ""
                    # Add task prefix if it contains T5
                    if "t5" in str(type(model)).lower():
                        prefix = f"translate {from_lang} to {to_lang}: "
                    
                    inputs = tokenizer(prefix + text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=0.1
                        )
                    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
            return jsonify({"result": translated_text})

        # Run translation using legacy TensorFlow H5 Keras Engine
        elif assets['engine'] == 'tf':
            pair_key = f"{src}_{tgt}"
            
            if pair_key not in assets['tf_models']:
                return jsonify({"error": f"Translation from {src} to {tgt} is not supported by the legacy local model."}), 400

            model = assets['tf_models'][pair_key]
            tokenizer_in = assets['tf_tokenizers'].get(src)
            tokenizer_out = assets['tf_tokenizers'].get(tgt)

            if not tokenizer_in or not tokenizer_out:
                return jsonify({"error": "Tokenizers for this language pair are missing."}), 500

            # 1. Preprocess
            input_data = preprocess_text(text, tokenizer_in)

            # 2. Predict
            prediction = model.predict(input_data)

            # 3. Postprocess
            translated_text = postprocess_prediction(prediction, tokenizer_out)

            return jsonify({"result": translated_text})
            
        else:
            return jsonify({"error": "No active backend inference engine available."}), 500

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
