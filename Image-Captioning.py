from transformers import ViTFeatureExtractor, VisionEncoderDecoderModel, GPT2Tokenizer
import requests
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

model_name = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def show_and_generate_caption(url, greedy=True):
    image = Image.open(requests.get(url, stream=True).raw)
    plt.imshow(np.asarray(image))
    plt.axis('off')
    plt.show()
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values

    if greedy:
        generated_ids = model.generate(pixel_values)
    else:
        generated_ids = model.generate(
            pixel_values,
            do_sample=True,
            max_length=30,
            top_k=50,
            top_p=0.95
        )

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Generated Caption:", generated_text)

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR1DlTPzc50gqfjsa4B0svq-fpKFEuU4MFigA&usqp=CAU"
show_and_generate_caption(url, greedy=False)