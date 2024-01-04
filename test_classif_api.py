import requests
import base64
import json

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def send_encoded_image(image_path, url):
    encoded_image = encode_image_to_base64(image_path)
    payload = json.dumps({"image": encoded_image})
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=payload, headers=headers)
    return response.json()

# URL of your Flask API
url = 'http://localhost:5000/predict'  # Replace with your actual URL

# Path to your image file
image_path = 'f:/classif_timm/dataset_ebs/rejects/R-028.jpg'  # Replace with the path to your image

# Send the image and print the response
response = send_encoded_image(image_path, url)
print(response)

