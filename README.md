# Face Recognition System using CLIP & Neon DB

A Python-based face recognition system that detects faces, generates embeddings, and performs similarity matching using a cloud-based vector database.

##Project Upgrades (Old vs New)

This project improves the traditional face recognition approach by introducing a cloud-based vector search system.

* **Database:** Local image storage → Neon PostgreSQL (serverless)
* **Search Method:** Loop-based comparison → pgvector similarity search
* **Feature Extraction:** Pixel-based methods → CLIP embeddings
* **Matching:** Euclidean pixel comparison → L2 vector distance (`<->`)
* **Reliability:** Improved dependency management for Colab execution

##Key Features

* Cloud-based vector storage using Neon PostgreSQL
* Face embeddings using CLIP Vision Transformer
* Fast similarity search using pgvector
* Scalable and modular pipeline

##System Workflow

1. Input image is provided to the system
2. Face is detected and cropped using OpenCV
3. Cropped image is passed to CLIP model for embedding generation
4. Embedding is stored in PostgreSQL database
5. New image embedding is compared using L2 distance (`<->`)
6. Closest match is returned from database

##Technology Stack

* Python 3.x
* OpenCV
* PyTorch
* OpenAI CLIP (Hugging Face)
* Neon PostgreSQL
* pgvector extension


## Database Schema

```sql
CREATE TABLE face_profiles (
    id SERIAL PRIMARY KEY,
    person_name TEXT NOT NULL,
    filename TEXT NOT NULL,
    embedding vector(512) NOT NULL
);
```

## Setup Instructions

1. Install dependencies:

```bash
pip install opencv-python transformers psycopg2-binary pillow numpy torch
```

2. Configure Neon DB connection strings inside the script.

3. Create the database table using the schema provided above.

4. Run the script to add reference images and generate embeddings.

5. Execute the recognition function on test images to query the vector database.
