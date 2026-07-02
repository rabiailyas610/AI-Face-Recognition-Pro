# Face Recognition System – OpenCV SFace

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen.svg)](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w48/Zhong_SFace_Sigmoid-Constrained_Hypersphere_Loss_for_Robust_Face_Recognition_CVPRW_2020_paper.pdf)

A robust, offline‑capable face recognition system built with **OpenCV's SFace** deep learning model and a **Streamlit** web interface. It supports automatic enrolment from a folder hierarchy, manual registration, and real‑time identification with cosine similarity matching.


## Features

- **Pretrained SFace Model** – State‑of‑the‑art face recognition (99%+ accuracy on LFW) without any external training.
- **Automatic Enrolment** – Place person‑specific subfolders inside `train/` and the app computes average face embeddings on first launch.
- **Manual Registration** – Add new identities directly through the UI.
- **Registered Persons List** – View all enrolled individuals in a dedicated tab.
- **Cosine Similarity Matching** – Efficient and explainable nearest‑neighbour search using 128‑dimensional feature vectors.
- **Persistent Storage** – Face embeddings are stored in a lightweight JSON database (`face_db.json`) for instant recall across sessions.
- **Offline Operation** – The model file (35 MB) is downloaded once and then runs completely offline.
- **Adjustable Threshold** – Fine‑tune the matching strictness via the confidence slider (implemented internally with a cosine threshold of 0.4).


## Tech Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Face Detection     | OpenCV Haar Cascade                 |
| Face Recognition   | OpenCV SFace (ONNX)                 |
| Frontend           | Streamlit                           |
| Database           | JSON (file‑based)                   |
| Language           | Python 3.9+                         |
| Key Libraries      | `opencv-contrib-python`, `numpy`, `urllib` |


## Quick Start

### Prerequisites
- Python 3.9 or later
- `pip` package manager

### Installation
1. Clone the repository or copy the project folder.
2. Install dependencies:
   ```bash
   pip install opencv-contrib-python streamlit numpy
