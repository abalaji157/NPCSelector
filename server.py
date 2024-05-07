from flask import Flask, render_template, request, redirect, url_for

import torch
import pickle
import open_clip
import objaverse
import requests
from io import BytesIO
from PIL import Image, ImageDraw
from IPython.display import display
import os
import re
import openai
import random
import numpy as np
import heapq
from langchain import PromptTemplate
from langchain.llms import OpenAI

app = Flask(__name__)

#@title load model and data
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# load data
object_indices = pickle.load(open("object_indices.p", "rb"))
print("object indices loaded")
object_features = torch.load("objaverse_features.pt", map_location=torch.device('cpu')).to(device)
print("object features loaded")
object_features /= object_features.norm(dim=-1, keepdim=True)

#@title functions
def retrieve_objects(queries, topk=5):
    text = tokenizer(queries).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text).half()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    obj_probs = 100.0 * text_features @ object_features.T

    results = []
    scores = []
    for obj_prob in obj_probs:
        indices = torch.argsort(obj_prob, descending=True)[:topk]
        results.append([object_indices[ind] for ind in indices])
        scores.append([obj_prob[ind] for ind in indices])
    return results, scores

def combine_images(images, image_size):
    # set the width and height of each image
    width, height = image_size

    # create a new image to hold the combined image
    combined_image = Image.new('RGB', (len(images) * width + 30, height + 20), (255, 255, 255))

    # paste each image into the combined image with borders
    for i, img in enumerate(images):
        img = img.resize((width, height))
        new_img = Image.new("RGB", (width, height), (192, 192, 192))
        new_img.paste(img, mask=img.split()[3])
        x_offset = i * (width + 10) + 10
        y_offset = 10
        combined_image.paste(new_img, (x_offset, y_offset))
        draw = ImageDraw.Draw(combined_image)
        draw.rectangle((x_offset, y_offset, x_offset + width, y_offset + height), outline=(255, 255, 255), width=1)

    # return the combined image
    return combined_image

s3_path = "https://objaverse-im.s3.us-west-2.amazonaws.com/"
def get_image(object_index):
    if isinstance(object_index, list):
        urls = [s3_path + f"{ind}/007.png" for ind in object_index]
    else:
        urls = [s3_path + f"{object_index}/{i}.png" for i in ["000", "004", "007", "010"]]
    images = [Image.open(BytesIO(requests.get(url).content)) for url in urls]
    combined_image = combine_images(images, image_size=(200,200))
    return combined_image


import random

def choose_top_objects(imageq, top_object_indices, game_scene):
  annotations = objaverse.load_annotations(top_object_indices[0])
  # res_map = {}
  new_anno = []
  fst_valid = None
  for i, index in enumerate(annotations):
    print("index is ")
    print(index)
    anno = annotations[index]
    name = anno['name']
    link = anno['thumbnails']['images'][0]['url']
    val = False
    image_returned = True
    try:
      val = choose_suitable_image(link, game_scene)
    except:
      image_returned = False
      val = False
      print("failed")
    if val or image_returned:
      print("appended")
      new_anno.append((index, anno))
      fst_valid = index
      if len(new_anno) == 5:
        break
  print("arrived here")
  random_image_number = random.randint(0, len(new_anno)-1)
  print(new_anno)
  print(random_image_number)

  print(new_anno[random_image_number])
  tmp = (new_anno[random_image_number][0])
  print(tmp)
  image = get_image(tmp)
  display(image)
  print("displayed ")
  return new_anno[random_image_number][1]

def choose_top_object(imageq, annotations, game_scene):
  for i, index in enumerate(annotations):
    tmp_query = "Choose one of the following images as the one that best fits into the scene, described here: " + str(game_scene) + ". Return 0, 1, 2, 3, 4, etc. as the index of the image you think is the best for the scene"
    response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": tmp_query},
            {
              "type": "image_url",
              "image_url": {
                "url": i['thumbnails']['images'][0]['url'],
              },
            },
          ],
        }
      ],
      max_tokens=300,
  )
  return response.choices[0]['message']['content']

def describe_image(image_url, game_scene):
  tmp_query = "Describe the following image. Explain how it fits into the following scene - " + str(game_scene)
  response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": tmp_query},
            {
              "type": "image_url",
              "image_url": {
                "url": image_url,
              },
            },
          ],
        }
      ],
      max_tokens=300,
  )
  return response.choices[0]['message']['content']

def choose_suitable_image(image_url, game_scene):
  tmp_query = '''I will give you a game scene. Your task is to analyze the picture in the input and
  respond whether the style of the image would fit well as an object in the game scene. Output "yes" if you think it would or "no" if you think it would not. '''
  if ("jpeg" not in image_url and "jpg" not in image_url and "png" not in image_url):
    print("invalid URL")
    print(image_url)
    return False
  response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": tmp_query},
            {
              "type": "image_url",
              "image_url": {
                "url": image_url,
              },
            },
          ],
        }
      ],
      max_tokens=300,
  )
  if "yes" in response.choices[0]['message']['content']:
    return True
  return False

imagequeries = None
global image_url 
query = None

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == "POST":
      openai.api_key = request.form.get("openai_key")
      global query
      query = request.form.get("scene_description")

      response = openai.ChatCompletion.create(
      model="gpt-4-turbo",
      max_tokens=300,
      messages=[
          {"role": "system", "content": "You will be provided with a scene from a game. You will CHOOSE ONLY FOUR most applicable digital assets to the scene entered by the user from a universe of 3d models. Choose 4 applicable assets from the user response, and return their names as a comma separated list."},
          {"role": "user", "content": query}
      ]
      )

      global imagequeries
      imagequeries = response.choices[0].message.content.split(",")
      firstimagedesc = ""
      firstfeedback = ""

      # for imageq in imagequeries:
      imageq = imagequeries[0]
      print("Retrieving images for: " + str(imageq))
      top_object_indices = retrieve_objects([imageq], topk=6)[0]
      zeta = choose_top_objects(imageq, top_object_indices, query)
      image_url = zeta['thumbnails']['images'][0]['url']
      print(f"image url is {image_url}\n\n\n\n\n\n")
      return redirect(url_for('results1', url_for_image = image_url, cur_object = imageq))
      # return render_template('temp.html')
    else: 
      # return render_template('index.html')
      return render_template('index.html')
    # response = openai.ChatCompletion.create(
    # model="gpt-4-turbo",
    # max_tokens=300,
    # messages=[
    #     {"role": "system", "content": "You will be provided with a scene from a game. You will CHOOSE ONLY FOUR most applicable digital assets to the scene entered by the user from a universe of 3d models. Choose 4 applicable assets from the user response, and return their names as a comma separated list."},
    #     {"role": "user", "content": query}
    # ]
    # )

    # imagequeries = response.choices[0].message.content.split(",")
    # print(imagequeries)

    # firstimagedesc = ""
    # firstfeedback = ""

    # for imageq in imagequeries:
    #     print("Retrieving images for: " + str(imageq))
    #     top_object_indices = retrieve_objects([imageq], topk=6)[0]
    #     zeta = choose_top_objects(imageq, top_object_indices, query)
    #     satisfied = input("Do you like this object? (yes/no)")
    #     if satisfied == "no":
    #         feedback = input("Give specific reasons that you do not like this object. For example say this object is not friendly enough or is too cartoony.")
    #         while firstfeedback == "":
    #             firstfeedback = feedback







# Results1 endpoint
@app.route('/results1', methods=['GET', 'POST'])
def results1():
    if request.method == 'POST':
        # imagequeries = requests.get('imagequeries')
        user_decision = request.form.get('decision')
        if user_decision == 'no':
            # Handle user's input if they select 'no'
            # user_thoughts = request.form.get('user_thoughts')
            return redirect(url_for('results1'))
        elif user_decision == 'yes':
            global imagequeries
            imageq = imagequeries[1]
            print("Retrieving images for: " + str(imagequeries[1]))
            top_object_indices = retrieve_objects([imageq], topk=6)[0]
            global query 
            zeta = choose_top_objects(imageq, top_object_indices, query)
            image_url = zeta['thumbnails']['images'][0]['url']
            print(image_url)
            return redirect(url_for('results2', url_for_image = image_url, cur_object = imageq))
    else:
        # Render the template with images and user thoughts
        # imageq = imagequeries[1]
        # print("Retrieving images for: " + str(imageq))
        # top_object_indices = retrieve_objects([imageq], topk=6)[0]
        # zeta = choose_top_objects(imageq, top_object_indices, query)
        # image_url = zeta['thumbnails']['images'][0]['url']
        # print(f"image url is {image_url}\n\n\n\n\n\n")
        return render_template('results1.html')

# Results2, Results3, Results4 endpoints (similar logic to Results1)
@app.route('/results2', methods=['GET', 'POST'])
def results2():
    if request.method == 'POST':
        user_decision = request.form.get('decision')
        if user_decision == 'no':
            return redirect(url_for('results2'))
        elif user_decision == 'yes':
            global imagequeries
            imageq = imagequeries[2]
            print("Retrieving images for: " + str(imagequeries[2]))
            top_object_indices = retrieve_objects([imageq], topk=6)[0]
            global query 
            zeta = choose_top_objects(imageq, top_object_indices, query)
            image_url = zeta['thumbnails']['images'][0]['url']
            print(image_url)
            return redirect(url_for('results3', url_for_image = image_url, cur_object = imageq))
    else:
        # Render the template with images and user thoughts
        return render_template('results2.html')


@app.route('/results3', methods=['GET', 'POST'])
def results3():
    if request.method == 'POST':
        user_decision = request.form.get('decision')
        if user_decision == 'no':
            # Handle user's input if they select 'no'
            # user_thoughts = request.form.get('user_thoughts')
            return redirect(url_for('results3'))
        elif user_decision == 'yes':
            global imagequeries
            imageq = imagequeries[3]
            print("Retrieving images for: " + str(imagequeries[3]))
            top_object_indices = retrieve_objects([imageq], topk=6)[0]
            global query 
            zeta = choose_top_objects(imageq, top_object_indices, query)
            image_url = zeta['thumbnails']['images'][0]['url']
            print(image_url)
            return redirect(url_for('results4', url_for_image = image_url, cur_object = imageq))
    else:
        return render_template('results3.html')


@app.route('/results4', methods=['GET', 'POST'])
def results4():
    if request.method == 'POST':
        user_decision = request.form.get('decision')
        if user_decision == 'no':
            # Handle user's input if they select 'no'
            # user_thoughts = request.form.get('user_thoughts')
            return redirect(url_for('results4'))
        elif user_decision == 'yes':
            return "Thank you so much for trying out NPC Selector!"
    else:
        # Render the template with images and user thoughts
        return render_template('results4.html')




if __name__ == '__main__':
    app.run(debug=True)
