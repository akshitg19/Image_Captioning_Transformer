import torch
import torch.nn as nn
import timm

class ImageCaptioner(nn.Module):
    """
    A Transformer-based model for image captioning.

    Args:
        context_length (int): The maximum length of a caption.
        vocabulary_size (int): The total number of unique words in the vocabulary.
        num_blocks (int): The number of Transformer decoder blocks.
        model_dim (int): The dimensionality of the model's embeddings.
        num_heads (int): The number of attention heads in the Transformer.
        prob (float): The dropout probability.
    """
    def __init__(self, context_length, vocabulary_size, num_blocks, model_dim, num_heads, prob):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CNN Encoder
        self.cnn_encoder = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0) # num_classes=0 gets features
        
        # Determine the output feature size from the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            cnn_output = self.cnn_encoder(dummy_input)
            in_features = cnn_output.shape[1]
        
        self.project = nn.Linear(in_features, model_dim)

        # Transformer Decoder
        self.word_embeddings = nn.Embedding(vocabulary_size, model_dim)
        self.pos_embeddings = nn.Embedding(context_length, model_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=2 * model_dim, 
            dropout=prob, 
            batch_first=True, 
            norm_first=True
        )
        self.blocks = nn.TransformerDecoder(decoder_layer, num_layers=num_blocks)
        self.vocab_projection = nn.Linear(model_dim, vocabulary_size)

    def forward(self, images, true_labels):
        """
        Performs a forward pass for training (teacher forcing).
        """
        B, T = true_labels.shape
        
        # 1. Encode image
        with torch.no_grad():
            img_features = self.cnn_encoder(images)
        encoded_image = self.project(img_features)
        img_for_attention = torch.unsqueeze(encoded_image, 1)

        # 2. Embed captions
        tok_embedded = self.word_embeddings(true_labels)
        positions = torch.arange(T, device=self.device)
        pos_embedded = self.pos_embeddings(positions)
        total_embeddings = tok_embedded + pos_embedded

        # 3. Generate attention mask for the decoder
        attention_mask = nn.Transformer.generate_square_subsequent_mask(T, device=self.device)
        
        # 4. Pass through Transformer decoder
        block_output = self.blocks(
            tgt=total_embeddings, 
            memory=img_for_attention, 
            tgt_mask=attention_mask
        )
        
        # 5. Project to vocabulary space
        vocabulary_vector = self.vocab_projection(block_output)

        return vocabulary_vector
