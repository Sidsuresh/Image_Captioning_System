import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
import torchvision.models as models
import torch.nn as nn
import math
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

# --- PASTE THE CLASS DEFINITIONS FROM YOUR NOTEBOOK HERE ---
# Creating a custom class for the Vocabulary - This class will handle the tokenization and encoding of the captions
class Vocabulary:
    # Initializing the Vocabulary class with a minimum threshold for word frequency
    def __init__(self, min_threshold):
        self.min_threshold = min_threshold
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {val: key for key, val in self.itos.items()}
    
    # Defining the length of the vocabulary
    def __len__(self):
        return len(self.itos)
    
    # Tokenizing the text using NLTK
    @staticmethod
    def tokenizer(text):
        return nltk.tokenize.word_tokenize(text.lower())
    
    # Building the vocabulary from the sentences
    def build_vocab(self, sentences):
        freq = Counter()
        idx = 4
        for sentence in sentences:
            for word in self.tokenizer(sentence):
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1
                
                if freq[word] == self.min_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    # Encoding the text into token IDs
    def encode(self, text):
        return [self.stoi.get(word, self.stoi['<UNK>']) for word in self.tokenizer(text)]
    
    # Decoding the token IDs back into text
    def decode(self, token_ids, skip_special_tokens=True):
        tokens = []
        for idx in token_ids:
            word = self.itos.get(idx, '<UNK>')
            if skip_special_tokens and word in ['<PAD>', '<SOS>', '<EOS>']:
                continue
            tokens.append(word)
        return ' '.join(tokens)
    
# Creating a custom class for the Encoder CNN - This class will handle the encoding of the images using a pre-trained CNN model (ResNet50)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, use_spatial_features = False, train_cnn = False):
        super(EncoderCNN, self).__init__()
        self.train_cnn = train_cnn
        # The CNN model will provide spatial features, which are required for Transformer, if True
        # The CNN model will provide global features, which are required for LSTM, if False
        self.use_spatial_features = use_spatial_features

        # Load the pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2 if use_spatial_features else -1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze the parameters of the CNN model
        for param in self.resnet.parameters():
            param.requires_grad = self.train_cnn
        
        # The final fully connected layer will be used to project the features to the desired embedding size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
    
    # Forward pass through the CNN model to extract features from the images
    def forward(self, images):
        features = self.resnet(images)

        if self.use_spatial_features:
            features = features.flatten(2).permute(0, 2, 1)
            features = self.fc(features)
        else:
            features = features.view(features.size(0), -1)
            features = self.fc(features)

        return features

# Creating a custom class for the Positional Encoding - This class will handle the positional encoding for the input embeddings within the Transformer
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        # The positional encoding is initialized with zeros
        pe = torch.zeros(max_len, embed_dim)
        # The position is a tensor of shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # The positional encoding is calculated using sine and cosine functions
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    # Forward pass through the positional encoding layer where the positional encoding is added to the input embeddings
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
# Creating a custom class for the Residual Block - This class will handle the residual connections within the Transformer
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        
        # The residual block consists of two linear layers with ReLU activation in between
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    # Forward pass through the residual block where the input is added to the output of the layer
    def forward(self, x):
        return x + self.layer(x)

# Creating a custom class for the Decoder Transformer - This class will handle the decoding of the image features into captions using a Transformer model
# The Transformer model consists of multiple layers of self-attention and feed-forward networks which are used to process the input sequence 
# and generate the output sequence

class DecoderTransformer(nn.Module):
    def __init__(self, feature_size=512, hidden_size=512, vocab_size=2002, num_layers=4, num_heads=4, max_len=50):
        super(DecoderTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_size, max_len=max_len)
        self.res_block = ResidualBlock(hidden_size)

        # The TransformerDecoderLayer is a single layer of the Transformer decoder which consists of 4 heads
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        # The TransformerDecoder is the full decoder which consists of multiple layers of the Transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Clamp caption token values within vocab range
        captions = torch.clamp(captions, min=0, max=2001)

        # Get embedding + sinusoidal position encoding
        embedded = self.embedding(captions)
        embedded = self.pos_encoder(embedded)

        # Generate causal mask for decoder to prevent attending to future tokens
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embedded.size(1)).to(embedded.device)

        # Reshape features to match the expected input shape for the transformer decoder
        tgt = embedded.permute(1, 0, 2)
        memory = features.permute(1, 0, 2)

        # Decode and project
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)  
        output = output.permute(1, 0, 2)           
        return self.fc(output)                     


# Constants from your notebook
EMBED_SIZE = 512
NUM_HEADS = 4
NUM_LAYERS = 4
MAX_LEN = 50

@st.cache_resource # Efficiently loads models only once
def load_assets():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    device = torch.device("cpu")
    
    # Initialize Encoder
    encoder = EncoderCNN(embed_size=EMBED_SIZE, use_spatial_features=True)
    encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
    encoder.eval()
    
    # Initialize Decoder
    decoder = DecoderTransformer(
        feature_size=EMBED_SIZE,
        hidden_size=EMBED_SIZE,
        vocab_size=len(vocab),
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_len=MAX_LEN
    )
    decoder.load_state_dict(torch.load('decoder.pth', map_location=device))
    decoder.eval()
    
    return encoder, decoder, vocab

# Logic to generate captions
def get_caption(image, encoder, decoder, vocab):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = [vocab.stoi['<SOS>']]
        
        for _ in range(MAX_LEN):
            input_tensor = torch.tensor(outputs).unsqueeze(0)
            output = decoder(features, input_tensor)
            next_token = output[0, -1, :].argmax().item()
            
            if next_token == vocab.stoi['<EOS>']:
                break
            outputs.append(next_token)
            
    return vocab.decode(outputs)

# --- Streamlit UI ---
st.title("Image Captioning with Transformers")
st.write("Upload an image to see the model-generated description.")

encoder, decoder, vocab = load_assets()

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    
    if st.button("Caption This Image"):
        with st.spinner("Generating..."):
            result = get_caption(img, encoder, decoder, vocab)
            st.subheader(f"Generated Caption: {result}")