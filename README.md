# 📸 Image Captioning AI: ResNet-50 + Transformers

A deep learning application that automatically generates descriptive captions for images. This project implements an **Encoder-Decoder** architecture where a pre-trained CNN extracts visual features and a Transformer model generates natural language descriptions.

🚀 **[Link to your Streamlit App if hosted]**

## 🛠️ Features

- **Real-time Captioning:** Upload any image (JPG/PNG) to see the AI's interpretation.
- **Transformer-based Logic:** Uses multi-head self-attention for superior context understanding compared to traditional LSTMs.
- **Visual Feedback:** Displays the uploaded image alongside the generated text.
- **Optimized for Deployment:** Uses `@st.cache_resource` for fast model loading and memory efficiency.

## 📊 The Machine Learning Pipeline

1. **Feature Extraction (Encoder):** A pre-trained **ResNet-50** backbone (with the final classification layer removed) encodes images into 2048-dimensional feature maps.
2. **Sequential Modeling (Decoder):** A **Transformer Decoder** with 4 layers and 4 attention heads. It uses **Sinusoidal Positional Encoding** to maintain the spatial/sequential order of tokens.
3. **Inference (Greedy Search):** The model predicts the next word in the sequence by calculating the highest probability for each token until the `<EOS>` (End of Sentence) tag is reached.
4. **Vocabulary Mapping:** A custom tokenizer built from the training corpus to map integer predictions back to human-readable words.

## 💻 Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Torchvision
- **Computer Vision:** PIL (Pillow)
- **Web Framework:** Streamlit
- **NLP:** NLTK
- **Large File Management:** Git LFS (Large File Storage)

## 📂 Project Structure

```text
├── app/
│   └── main.py          # Streamlit UI and Inference logic
├── encoder.pth          # Saved ResNet-50 Encoder weights (Git LFS)
├── decoder.pth          # Saved Transformer Decoder weights (Git LFS)
├── vocab.pkl            # Serialized Vocabulary object (Mapping IDs to Words)
├── requirements.txt     # Dependency list
├── README.md            # Project documentation
└── .gitattributes       # Git LFS configuration
```
