import io
import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# TODO: Import or define your model class here
# from your_model_file import YourModelClass
# Move the model to the appropriate device

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Replace this with the path to your trained model
model_path = 'dataset_ebs/model_20231226_062812.pth'  # 'path/to/your/model.pth'

# Replace this with the list of class names
class_names = ['goods', 'rejects']  # ['class1', 'class2', 'class3']

mypretrainedname = 'efficientvit_b0' #Put one of the model name listed in previous cell

# Load pre-trained ViT model with specified dropout rate
model = timm.create_model(mypretrainedname, pretrained=True, num_classes=2, drop_rate=0.0)
model = model.to(device)

# Load the model state dictionary
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        model.eval()
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.max(probabilities, 1)

    predicted_class = class_names[top_catid[0]]
    confidence = top_prob[0].item()

    return jsonify({'class': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
