# Face Recognition System using CLIP & Neon DB

A stable face recognition project built with Python to detect faces, extract feature vectors, and match identities using a cloud-based serverless vector database.

## 🔄 Project Upgrades: Old vs. New

This project was updated to replace legacy local file storage with a modern serverless vector cloud setup:

- **Database:** Shifted from storing cropped image files locally to a serverless Neon PostgreSQL database.
- **Search System:** Replaced slow manual image file looping with fast pgvector similarity search.
- **Mathematical Matching:** Upgraded from basic pixel comparisons to L2 vector space calculations (`<->`).
- **Feature Extraction:** Replaced older pixel-based methods with OpenAI's CLIP Vision Transformer model via Hugging Face.
- **Stability:** Fixed third-party package conflicts to create a reliable and working environment in Google Colab.

## 📌 Key Features

- **Cloud Vector Storage:** Uses Neon PostgreSQL with the pgvector extension for data indexing.
- **Transformer Embeddings:** Generates accurate 512-dimensional face vectors using OpenAI's pre-trained CLIP model (`clip-vit-base-patch32`).
- **Nearest Neighbor Search:** Leverages the Euclidean L2 distance operator (`<->`) to quickly retrieve the closest matching identity profile from the database.
- **Clean Execution Code:** Fully contained in a structured Python script for easy replication.

## 🛠️ System Workflow

1. **Face Detection:** The system takes an input image and isolates the face boundaries using an OpenCV Cascade Classifier.
2. **Feature Extraction:** The cropped face image is processed through the Hugging Face `CLIPProcessor`.
3. **Vector Generation:** The `CLIPVisionModel` converts the face image into a 512-point numeric array.
4. **Database Matching:** The system queries the Neon database using an L2 distance vector search to instantly fetch the name and filename of the closest match.

## 🎛️ Technology Stack

- **Language:** Python 3.x
- **Deep Learning Framework:** PyTorch
- **Computer Vision:** OpenCV (Open Source Computer Vision Library)
- **Feature Extractor Model:** OpenAI CLIP (Vision Transformer)
- **Cloud Database:** Neon PostgreSQL Serverless Instance
- **Database Extension:** pgvector plugin (`vector(512)`)

## 📊 Database Schema

```sql
CREATE TABLE face_profiles (
    id SERIAL PRIMARY KEY,
    person_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    embedding vector(512) NOT NULL
);
```

## 🚀 Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install opencv-python transformers psycopg2-binary pillow numpy torch
   ```
2. **Initialize Database:** Configure your Neon DB URL and execute the table schema creation method.
3. **Register Face Profiles:** Pass your reference images through the registration function to save names and vector arrays into the cloud database.
4. **Run Face Matching:** Pass unknown test images into the recognition engine to query the serverless database and display the closest identity match.
