from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", useFast=True)
image = Image.open('./dog.jpg')

depth = pipe(image)["depth"]
depth.save("dog-depth-map.jpg")
