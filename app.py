# from flask import Flask, request, render_template, redirect, url_for
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
# from PIL import Image
# import torch
# import os
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer

# app = Flask(__name__)


# # Path to your client_secrets.json file
# CLIENT_SECRETS_PATH = "client_secrets.json"

# # Authenticate and create the PyDrive client
# gauth = GoogleAuth()
# gauth.LoadClientConfigFile(CLIENT_SECRETS_PATH)
# gauth.LocalWebserverAuth()
# drive = GoogleDrive(gauth)

# # Load pre-trained model and feature extractor
# model = VisionEncoderDecoderModel.from_pretrained(
#     "nlpconnect/vit-gpt2-image-captioning"
# )
# feature_extractor = ViTFeatureExtractor.from_pretrained(
#     "nlpconnect/vit-gpt2-image-captioning"
# )
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# # Initialize the sentence transformer model
# caption_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Initialize FAISS index
# dimension = 384  # Dimension of the sentence embeddings
# index = faiss.IndexFlatL2(dimension)

# # Store image paths separately
# image_paths = []


# def generate_caption(image_path):
#     image = Image.open(image_path).convert("RGB")
#     pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
#     output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
#     caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return caption


# def process_images(image_list):
#     for image in image_list:
#         image.GetContentFile(image["title"])
#         caption = generate_caption(image["title"])
#         vector = caption_model.encode(caption).astype("float32")
#         index.add(np.array([vector]))
#         image_paths.append(image["title"])


# @app.route("/", methods=["GET", "POST"])
# def home():
#     folder_id = request.args.get("folder_id", "root")
#     folders = drive.ListFile(
#         {
#             "q": f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"
#         }
#     ).GetList()
#     images = drive.ListFile(
#         {"q": f"'{folder_id}' in parents and mimeType contains 'image/'"}
#     ).GetList()

#     return render_template("index.html", folders=folders, images=images)


# @app.route("/select_folder", methods=["POST"])
# def select_folder():
#     folder_id = request.form["folder_id"]
#     return redirect(url_for("home", folder_id=folder_id))


# @app.route("/process_folder", methods=["POST"])
# def process_folder():
#     folder_id = request.form["folder_id"]
#     images = drive.ListFile(
#         {"q": f"'{folder_id}' in parents and mimeType contains 'image/'"}
#     ).GetList()
#     process_images(images)
#     return redirect(url_for("home", folder_id=folder_id))


# @app.route("/search_image", methods=["POST"])
# def search_image():
#     query = request.form["query"]
#     query_vector = caption_model.encode(query).astype("float32")
#     D, I = index.search(np.array([query_vector]), k=1)
#     result_image_path = image_paths[I[0][0]]
#     return render_template("index.html", image_path=result_image_path)


# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, request, render_template, redirect, url_for
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
import torch
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Path to your client_secrets.json file
CLIENT_SECRETS_PATH = "client_secrets.json"

# Authenticate and create the PyDrive client
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
gauth.LoadClientConfigFile(CLIENT_SECRETS_PATH)
drive = GoogleDrive(gauth)

# Set redirect URI for Flask app
gauth.settings["oauth_scope"] = ["https://www.googleapis.com/auth/drive"]
gauth.settings["client_config_file"] = CLIENT_SECRETS_PATH
gauth.settings["get_refresh_token"] = True
gauth.settings["redirect_uri"] = "http://127.0.0.1:5000/oauth2callback"


@app.route("/oauth2callback")
def oauth2callback():
    gauth.Authenticate()
    return redirect(url_for("home"))


# Load pre-trained model and feature extractor
model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initialize the sentence transformer model
caption_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index
dimension = 384  # Dimension of the sentence embeddings
index = faiss.IndexFlatL2(dimension)

# Store image paths separately
image_paths = []


@app.route("/home", methods=["GET", "POST"])
def home():
    folder_id = request.args.get("folder_id", "root")
    folders = drive.ListFile(
        {
            "q": f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder'"
        }
    ).GetList()
    images = drive.ListFile(
        {"q": f"'{folder_id}' in parents and mimeType contains 'image/'"}
    ).GetList()

    return render_template("index.html", folders=folders, images=images)


@app.route("/select_folder", methods=["POST"])
def select_folder():
    folder_id = request.form["folder_id"]
    return redirect(url_for("home", folder_id=folder_id))


@app.route("/process_folder", methods=["POST"])
def process_folder():
    folder_id = request.form["folder_id"]
    images = drive.ListFile(
        {"q": f"'{folder_id}' in parents and mimeType contains 'image/'"}
    ).GetList()
    process_images(images)
    return redirect(url_for("home", folder_id=folder_id))


@app.route("/search_image", methods=["POST"])
def search_image():
    query = request.form["query"]
    query_vector = caption_model.encode(query).astype("float32")
    D, I = index.search(np.array([query_vector]), k=1)
    result_image_path = image_paths[I[0][0]]
    return render_template("index.html", image_path=result_image_path)


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption


def process_images(image_list):
    for image in image_list:
        image.GetContentFile(image["title"])
        caption = generate_caption(image["title"])
        vector = caption_model.encode(caption).astype("float32")
        index.add(np.array([vector]))
        image_paths.append(image["title"])


if __name__ == "__main__":
    app.run(debug=True)
