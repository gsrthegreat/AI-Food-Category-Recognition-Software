# python_model/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, tempfile, joblib, numpy as np, cv2
import traceback
import sys

app = Flask(__name__)
CORS(app)

# Load model and vocab from mounted /app/models or /app/features
MODEL_PATHS = [
    "/app/models/model.pkl",
    "/app/models/model.joblib",
]

VOCAB_PATH = "/app/models/vocab.npy"
CLASSES_PATH = "/app/models/classes.npy"

# Print startup diagnostics
print("="*50)
print("STARTUP DIAGNOSTICS")
print("="*50)
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print("\nChecking for model files:")
print(f"  /app/models exists: {os.path.exists('/app/models')}")
if os.path.exists('/app/models'):
    print(f"  Contents: {os.listdir('/app/models')}")
print(f"  model.pkl exists: {os.path.exists('/app/models/model.pkl')}")
print(f"  model.joblib exists: {os.path.exists('/app/models/model.joblib')}")
print(f"  vocab.npy exists: {os.path.exists(VOCAB_PATH)}")
print(f"  classes.npy exists: {os.path.exists(CLASSES_PATH)}")
print("="*50)

# try load model
model = None
label_encoder = None

if os.path.exists("/app/models/model.pkl"):
    try:
        print("Loading model from model.pkl...")
        data = joblib.load("/app/models/model.pkl")
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            label_encoder = data.get('label_encoder', None)
            print(f"  ✓ Loaded model from dict (type: {type(model).__name__})")
        else:
            model = data
            print(f"  ✓ Loaded model directly (type: {type(model).__name__})")
    except Exception as e:
        print(f"  ✗ Failed to load model.pkl: {e}")
        traceback.print_exc()

# fallback: try model.joblib
if model is None and os.path.exists("/app/models/model.joblib"):
    try:
        print("Loading model from model.joblib...")
        model = joblib.load("/app/models/model.joblib")
        print(f"  ✓ Loaded model (type: {type(model).__name__})")
    except Exception as e:
        print(f"  ✗ Failed to load model.joblib: {e}")
        traceback.print_exc()

# load vocab & classes
vocab = None
if os.path.exists(VOCAB_PATH):
    try:
        vocab = np.load(VOCAB_PATH)
        print(f"✓ Loaded vocab with shape {vocab.shape}")
    except Exception as e:
        print(f"✗ Failed to load vocab: {e}")
        traceback.print_exc()
else:
    print("✗ vocab.npy not found")

classes = None
if os.path.exists(CLASSES_PATH):
    try:
        classes = np.load(CLASSES_PATH, allow_pickle=True)
        print(f"✓ Loaded classes: {classes}")
    except Exception as e:
        print(f"✗ Failed to load classes: {e}")
        traceback.print_exc()
else:
    print("✗ classes.npy not found")

# init sift
try:
    sift = cv2.SIFT_create()
    print("✓ SIFT initialized")
except Exception as e:
    print(f"✗ Failed to initialize SIFT: {e}")
    sift = None

print("="*50)
print(f"READY: model={model is not None}, vocab={vocab is not None}, sift={sift is not None}")
print("="*50)

def build_histogram(descriptors, vocab):
    K = vocab.shape[0]
    if descriptors is None or descriptors.size == 0:
        return np.zeros(K, dtype=np.float32)
    dists = np.linalg.norm(descriptors[:, None] - vocab[None, :], axis=2)
    word_ids = np.argmin(dists, axis=1)
    hist, _ = np.histogram(word_ids, bins=np.arange(K+1))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-7)
    return hist

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/")
def hello():
    status = {
        "message": "Python model service is running",
        "model_loaded": model is not None,
        "vocab_loaded": vocab is not None,
        "sift_initialized": sift is not None,
        "classes_loaded": classes is not None
    }
    return jsonify(status)

@app.route("/predict", methods=["POST"])
def predict():
    temp_path = None
    fd = None
    
    try:
        # Check prerequisites
        if sift is None:
            return jsonify({"error": "SIFT not initialized"}), 500
        if vocab is None:
            return jsonify({"error": "Vocabulary not loaded"}), 500
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        # Check for image file
        if 'image' not in request.files:
            return jsonify({"error": "no image provided"}), 400
        
        f = request.files['image']
        if f.filename == "":
            return jsonify({"error": "empty filename"}), 400

        print(f"\n--- Processing image: {f.filename} ---")

        # Save temp file
        fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(f.filename)[1])
        f.save(temp_path)
        print(f"Saved to temp: {temp_path} (size: {os.path.getsize(temp_path)} bytes)")

        # Read image
        img = cv2.imdecode(np.fromfile(temp_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "cannot read image - invalid format"}), 400
        
        print(f"Image loaded: shape={img.shape}, dtype={img.dtype}")
        
        # Resize and convert
        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"Grayscale: shape={gray.shape}")
        
        # Extract features
        kps, des = sift.detectAndCompute(gray, None)
        print(f"SIFT: {len(kps) if kps else 0} keypoints, descriptors shape={des.shape if des is not None else None}")

        # Build histogram
        if des is None:
            des = np.array([]).reshape(0, 128)
        hist = build_histogram(des, vocab).reshape(1, -1)
        print(f"Histogram: shape={hist.shape}, sum={hist.sum():.4f}")

        # Predict
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(hist)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            print(f"Prediction (proba): idx={idx}, conf={conf:.4f}")
            
            if label_encoder is not None:
                label = label_encoder.inverse_transform([idx])[0]
            elif classes is not None:
                label = str(classes[idx])
            else:
                label = str(idx)
        else:
            pred = model.predict(hist)[0]
            conf = None
            print(f"Prediction: {pred}")
            
            if label_encoder is not None:
                label = label_encoder.inverse_transform([pred])[0]
            elif classes is not None:
                label = str(classes[int(pred)])
            else:
                label = str(pred)

        print(f"Final label: {label}, confidence: {conf}")
        print("--- Done ---\n")
        
        return jsonify({"label": label, "confidence": conf})
        
    except Exception as e:
        print(f"\n!!! ERROR in /predict !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        print("!!!\n")
        return jsonify({"error": str(e), "type": type(e).__name__}), 500
        
    finally:
        # Cleanup
        if fd is not None:
            try:
                os.close(fd)
            except:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)