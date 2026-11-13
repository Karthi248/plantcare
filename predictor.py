"""Offline prediction: leaf photo -> disease + remedy, with a confidence check."""
import json
import numpy as np
from PIL import Image

import os

try:  # lightweight TensorFlow Lite runtime first, full TensorFlow only as a last resort
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIDENCE_THRESHOLD = 0.60  # below this, ask for a clearer photo
IMG_SIZE = (224, 224)

with open(os.path.join(HERE, "labels.json")) as f:
    LABELS = json.load(f)
with open(os.path.join(HERE, "remedies.json"), encoding="utf-8") as f:
    REMEDIES = json.load(f)

_interpreter = Interpreter(model_path=os.path.join(HERE, "plantcare.tflite"))
_interpreter.allocate_tensors()
_in = _interpreter.get_input_details()[0]
_out = _interpreter.get_output_details()[0]

MESSAGES = {
    "en": {"unsure": "I am not sure. Please take a clearer photo of a single leaf in good light.",
           "healthy": "The plant looks healthy.",
           "no_remedy": "No saved remedy for this disease yet. Please ask your local agricultural officer."},
    "ta": {"unsure": "எனக்கு உறுதியாகத் தெரியவில்லை. நல்ல வெளிச்சத்தில் ஒரு இலையை தெளிவாக புகைப்படம் எடுக்கவும்.",
           "healthy": "செடி ஆரோக்கியமாக உள்ளது.",
           "no_remedy": "இந்த நோய்க்கான தீர்வு இன்னும் சேர்க்கப்படவில்லை. உங்கள் வேளாண் அலுவலரை அணுகவும்."},
}

