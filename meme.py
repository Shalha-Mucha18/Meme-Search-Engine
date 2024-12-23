# import necessary library
import os
import json
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Explicitly set the Tesseract path 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Meme images directory
MEME_DIR = r"C:/Users/ASUS/Desktop/Fringe Core/meme"

def preprocess_image(image_path):
    """Preprocess the image to improve OCR."""
    img = Image.open(image_path)
    img = img.convert("L")  # convert to grayscale
    img = img.filter(ImageFilter.MedianFilter())  # remove noise
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2) 
    return img

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        # preprocess the image
        img = preprocess_image(image_path)
        # use Tesseract with custom configurations
        text = pytesseract.image_to_string(img, config="--psm 6 --oem 3", lang="eng")
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""

def load_meme_data(meme_dir):
    """Load meme data, extracting features from images"""
    meme_data = []
    for file in os.listdir(meme_dir):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".webp"):
            image_path = os.path.join(meme_dir, file)
            text = ""
            # try loading metadata if available
            json_path = image_path.replace(".jpg", ".json").replace(".png", ".json").replace(".webp", ".json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    text = metadata.get("text", "")
            # fallback to extracting text from the image
            if not text:
                text = extract_text_from_image(image_path)

            meme_data.append({"file": file, "text": text})
    return meme_data    

def find_related_memes(keyword, meme_data):
    """Find the memes related to the given keyword."""
    texts = [meme["text"] for meme in meme_data]
    files = [meme["file"] for meme in meme_data]
    # append keyword
    texts.append(keyword)
    # compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    # Compute cosine similarity between the keyword and all meme texts
    keyword_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(keyword_vector, tfidf_matrix[:-1]).flatten()
    # get top related memes
    related_memes = [(files[i], similarity_scores[i]) for i in range(len(files))]
    related_memes = sorted(related_memes, key=lambda x: x[1], reverse=True)[:5]
    return related_memes

if __name__ == "__main__":
    print("Processed 5 images")
    meme_data = load_meme_data(MEME_DIR)

    while True:
        # user input
        keyword = input("Enter a query (or 'exit' to quit): ").strip()
        
        if keyword.lower() == 'exit':
            break

        print("Search Results:")
        related_memes = find_related_memes(keyword, meme_data)

        if related_memes:
            for meme, score in related_memes:
                print(f"('{meme}', np.float32({score:.8f}))")
        else:
            print("No related memes found.")


