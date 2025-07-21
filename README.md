# LipRead
## 🗣️ LipRead: Deep Learning-Based Lip Reading System

This project implements a deep learning pipeline that performs **visual speech recognition** — i.e., predicting spoken words from lip movements in videos — using **CTC loss** and a CNN-LSTM architecture.

It is built and trained using the **GRID corpus** dataset and includes a full pipeline for:

* Preprocessing `.mpg` videos and `.align` files
* Training a model using CTC (Connectionist Temporal Classification)
* Running inference and decoding predictions
* Streamlit interface for easy interaction

---

### 📁 Folder Structure

```
LipRead/
├── LipReading_final.ipynb   # Final Colab notebook
├── decoder.py               # Charset and decoding logic
├── model.py                 # Model architecture using CTC
├── train_ctc.py             # CTC training script
├── train_model.py           # Alternate training entry point
├── utils.py                 # Preprocessing utilities
├── lip_detector.py          # Mouth/lip detection from video
├── streamlit_app.py         # Web app interface
├── models/                  # Trained model weights (add your .h5 file here)
└── data/                    # Folder containing 'videos' and 'labels'
```

---

### 🚀 How to Run (Colab)

1. Open the notebook: [📓 LipReading\_final.ipynb](LipReading_final.ipynb)
2. Run cells to:

   * Install dependencies
   * Download and extract GRID dataset
   * Preprocess videos and alignments
   * Train model with CTC
   * Run inference on test clips

---

### 🧠 Model Architecture

* **3D CNN** for spatiotemporal feature extraction from video
* **Bidirectional LSTM** for temporal modeling
* **CTC Loss** for alignment-free sequence training
* Greedy decoding for converting model outputs into text

---

### 📦 Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

Includes:
`tensorflow`, `keras`, `opencv-python`, `streamlit`, `imageio`, `numpy`, `scipy`, etc.

---

### 🎥 Sample Input

* Video clips: `.mpg` format
* Alignments: `.align` files for word-level timing

### 🧪 Output

Model returns a text sequence matching the spoken content in the silent video.

---

### 🌐 Optional: Streamlit App

To launch the UI:

```bash
streamlit run streamlit_app.py
```

---

### 📚 Dataset

* GRID Corpus (subset)
* Format: `data/videos/*.mpg`, `data/labels/*.align`

You can download it automatically in the notebook or manually via Google Drive.

---

### 🤝 Contributors

* 👤 **Priyanshul** ([@Priyanshul02](https://github.com/Priyanshul02))

---

### 🏁 Future Improvements

* Beam search decoding
* Add speaker adaptation
* Real-time webcam inference
* Integrate with whisper or audio-text fusion

---

Would you like me to:

* Save this as a `README.md` file and push it to your GitHub repo?
* Include images, architecture diagrams, or sample output?

Let me know!
