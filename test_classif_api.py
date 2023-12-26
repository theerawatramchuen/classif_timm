
# import requests
# import os

# def send_image(image_path, url):
#     with open(image_path, 'rb') as image_file:
#         response = requests.post(url, files={'image': image_file})
#     return response.json()

# def send_images_in_folder(folder_path, url):
#     for root, dirs, files in os.walk(folder_path):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#                 image_path = os.path.join(root, file)
#                 response = send_image(image_path, url)
#                 print(f"Response for {image_path}: {response}")

# url = 'http://localhost:5000/predict'  # Replace with your Flask app's URL
# folder_path = 'datset_ebs/rejects/'#'path/to/your/folder'  # Path to your folder containing images

# send_images_in_folder(folder_path, url)

import requests

url = 'http://localhost:5000/predict'  # Replace with your Flask app's URL
image_path = 'f:/vit_timm/dataset_ebs/rejects/R-029.jpg'  # Path to your image file

with open(image_path, 'rb') as image_file:
    response = requests.post(url, files={'image': image_file})

print(response.json())
