# Image Classification Using Vision transformer or other model in 'timm' pretrained weight model library
Image Classification Using Vision transformer or various other model with 'timm' pretrained weight 

### Steps to follow:
#### Installation:
Anaconda Python Environment <br/>
version is working for CPU or [GPU] <br/>
Python 3.8 <br/>
torchvision 0.16.1 (pip3 install torchvision==0.16.1) <br/>
torch 2.1.1 or [torch 2.1.1+cu121] (https://pytorch.org/get-started/locally/) <br/>
timm (pip3 install timm) <br/>
scikit-learn <br/>
mathplotlib <br/>

#### Installation (optional for image classification API):
flask (pip3 install flask) <br/>
opencv-python (pip3 install opencv-python) <br/>

#### Setup for demo code on EBS dataset:
1. Unzip "dataset_ebs.zip" under folder "classif_timm" folder<br/>
2. Run "vit_ebs_timm_gpt-4-turbo.ipynb" for the model training, model weight saving and prediction sample images<br/>

#### Running training and inferencing with other pretrained weight model which listed in the 8th cell
1. Open "timm_classif.ipynp" with Jupyter notebook <br/>
2. Define dataset folder in the 2nd cell <br/>
3. Define training hyper parameters in the 3rd cell <br/>
4. Define Data augmentation and normalization for training in the 4th cell <br/>
5. Define pretrained weight model in the 9th cell <br>

#### Running demo API mode:
Make sure flask and opencv-python packages are installed in the python environment.
1. Open new terminal and Run "python classif_api.py" <br/>
2. Open new terminal and Run "python test_classif_api.py" <br/>
#### Reference: 
https://timm.fast.ai/ <br/>
https://github.com/huggingface/pytorch-image-models <br/>
https://youtu.be/mK0CHqLCoXA?si=UXKv2XkihnrjSp0O <br/>

