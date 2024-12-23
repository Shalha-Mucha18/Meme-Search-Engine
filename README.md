# Meme Search Engine

This project is a **Meme Search Engine** that processes meme images, extracts text using OCR, and allows users to find memes related to a given keyword. It uses `pytesseract` for Optical Character Recognition (OCR), `Pillow` for image preprocessing, and `scikit-learn` for keyword matching via TF-IDF and cosine similarity.

---

## Features
- **OCR (Optical Character Recognition)**: Extracts text from meme images.
- **Text Preprocessing**: Enhances images to improve OCR accuracy.
- **Keyword Search**: Finds memes related to a user-provided query.
- **Similarity Matching**: Uses TF-IDF and cosine similarity to rank memes by relevance.

---

## Requirements

### Prerequisites
- Python 3.7 or higher
- Tesseract OCR installed on your system.

### Python Libraries
Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
