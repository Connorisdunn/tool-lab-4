#what to do here

from fastapi import FastAPI
from fastapi import File, UploadFile
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import Response
import keras


str_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'bee>
mapping = {
'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
'household electrical device': ['clock', 'computer_keyboard', 'lamp', 'telephone', '>
'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kan>
'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
'people': ['baby', 'boy', 'girl', 'man', 'woman'],
'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

model = keras.models.load_model("trainedModel.h5")
limiter = Limiter(key_func=get_remote_address) # Gets the user's address
app = FastAPI()
app.state.limiter = limiter # Define's the api's limiter object
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # Adds the exception to the api 

@app.get("/")
@limiter.limit("15/minute") #Defines the limit for this specific request
async def homepage(request: Request, response: Response):
  return {"api_info": f"Connor-Dunn's CS280 Image Classifier"}


#@app.post("/receiveFile/")
#async def classify(file: UploadFile = File(...)):
#    fileContent = await file.read() # Reads file information from request as a stream of bytes
#    myFileString = "cuteDog.png"
#    myFileInBytes = open(myFileString) # This is how you'll receive your image from the request.
#    image = Image.open(BytesIO(myFileInBytes)) # Converts file to a python image from a stream of bytes.
#    imageArray = np.asarray(image) #converts image to numpy array
#    smallImage = cv2.resize(imageArray, dsize=(32, 32), interpolation=cv2.INTER_CUBIC) # Resizes the image
#    batchedImage = np.expand_dims(smallImage, axis=0)
#    return {"File Name": f"{file.filename}"} # Returns the filename as a response.


@app.post("/classify/")
@limiter.limit("5/minute")
async def classify(request: Request, response: Response, file: UploadFile = File(...)):
    img = await file.read()
    image = Image.open(BytesIO(img))
    array = np.asarray(image)
    new_image_array = cv2.resize(array, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    batchedImage = np.expand_dims(new_image_array, axis=0)
    output = model.predict(batchedImage)
    max_index = np.argmax(output[0])
    sub_group = str_labels[max_index]
    for category, words in mapping.items():
        for word in words:
            if word == sub_group:
                given_category = category
                break
    return {"classification": {"category": f"{given_category}", "type": f"{sub_group}"}} # Returns the filename as a response.


@app.post("/ImageSize/")
@limiter.limit("2/minute")
async def printImageSize(request:Request, response:Response, file: UploadFile = File(...)):
    img = await file.read()
    image = Image.open(BytesIO(img))
    array = np.asarray(image)
    new_image_array = cv2.resize(array, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    return {"Image shape": f"{new_image_array.shape}"}
