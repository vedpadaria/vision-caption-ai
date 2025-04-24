import os
import shutil
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms
from data_loader import get_loader
from model import DecoderRNN, EncoderCNN
from nlp_utils import clean_sentence

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cocoapi_dir = r"C:\Users\vedpa\OneDrive\Desktop\sampleProjects\image_captioning"
vocab_path = r"C:\Users\vedpa\OneDrive\Desktop\sampleProjects\image_captioning\vocab.pkl"
model_dir = r"C:\Users\vedpa\OneDrive\Desktop\sampleProjects\image_captioning\models"

def initialize_models():
    """Initialize and load the AI models"""
    if not os.path.exists("vocab.pkl"):
        if os.path.exists(vocab_path):
            shutil.copy(vocab_path, "vocab.pkl")
            print("Copied vocab.pkl to current directory")
        else:
            raise FileNotFoundError(f"vocab.pkl not found at {vocab_path}")

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    
    data_loader = get_loader(
        transform=transform_test,
        mode="test",
        vocab_file="vocab.pkl",
        vocab_from_file=True,
        cocoapi_loc=cocoapi_dir
    )
    
    
    embed_size = 256
    hidden_size = 512
    vocab_size = len(data_loader.dataset.vocab)
    
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    
    encoder_path = os.path.join(model_dir, "encoder-3.pkl")
    decoder_path = os.path.join(model_dir, "decoder-3.pkl")

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder, data_loader, transform_test

encoder, decoder, data_loader, transform_test = initialize_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_caption(image_path):
    """Generate caption for input image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform_test(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = encoder(image).unsqueeze(1)
            output = decoder.sample(features)
        return clean_sentence(output, data_loader.dataset.vocab.idx2word)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        caption = predict_caption(filepath)
        
        return jsonify({
            'caption': caption,
            'image_url': f"/uploads/{filename}"
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def clear_uploads():
    """Clear the uploads folder"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    clear_uploads()
    app.run(host='0.0.0.0', port=5000, debug=True)