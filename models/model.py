import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import torch
import torch.nn as nn
import timm

from config import (
    D_MODEL, IMG_SIZE, PATCH_SIZE, NUM_QUERIES, NUM_CHARS,
    NUM_DECODER_LAYERS, NUM_HEADS, DROPOUT, DROP_PATH,
)
from models.decoder import TransformerDecoder, ColorHead, CharHead


class CaptchaModel(nn.Module):
    def __init__(self, backbone: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()

        encoder = None
        if pretrained:
            try:
                import socket
                socket.setdefaulttimeout(30)
                encoder = timm.create_model(backbone, pretrained=True, img_size=IMG_SIZE[0], drop_path_rate=DROP_PATH)
                print(f"Loaded pretrained weights for {backbone}")
            except Exception as e:
                print(f"Could not load pretrained weights ({type(e).__name__}), using random init.")
        if encoder is None:
            encoder = timm.create_model(backbone, pretrained=False, img_size=IMG_SIZE[0], drop_path_rate=DROP_PATH)

        self.encoder = encoder
        self.encoder.reset_classifier(0)

        encoder_dim = self._get_encoder_dim(backbone)

        if encoder_dim != D_MODEL:
            self.encoder_proj = nn.Linear(encoder_dim, D_MODEL)
        else:
            self.encoder_proj = nn.Identity()

        self.decoder = TransformerDecoder(
            num_layers=NUM_DECODER_LAYERS,
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            num_queries=NUM_QUERIES,
            dropout=DROPOUT,
        )

        self.color_head = ColorHead(d_model=D_MODEL)
        self.char_head = CharHead(d_model=D_MODEL, num_chars=NUM_CHARS)

    @staticmethod
    def _get_encoder_dim(backbone: str) -> int:
        dims = {
            "vit_base_patch16_224": 768,
            "vit_large_patch16_224": 1024,
            "swin_base_patch4_window7_224": 1024,
            "swin_tiny_patch4_window7_224": 768,
            "convnext_tiny": 768,
        }
        return dims.get(backbone, 768)

    def forward_encoder(self, x):
        features = self.encoder.forward_features(x)
        if isinstance(features, (list, tuple)):
            features = features[0]
        return self.encoder_proj(features)

    def forward(self, x, return_aux=False):
        memory = self.forward_encoder(x)

        decoder_out = self.decoder(memory, return_aux=return_aux)
        if return_aux:
            queries, aux_outputs = decoder_out
        else:
            queries = decoder_out
            aux_outputs = []

        color_logits = self.color_head(queries)
        char_logits = self.char_head(queries)

        if return_aux and aux_outputs:
            aux_color = [self.color_head(q) for q in aux_outputs]
            aux_char = [self.char_head(q) for q in aux_outputs]
            return {"char": char_logits, "color": color_logits,
                    "aux_char": aux_char, "aux_color": aux_color}

        return {"char": char_logits, "color": color_logits}

    @torch.no_grad()
    def predict(self, x, threshold: float = 0.5):
        outputs = self.forward(x)
        color_prob = torch.sigmoid(outputs["color"])
        char_prob = torch.softmax(outputs["char"], dim=-1)

        is_red = color_prob >= threshold
        char_pred = char_prob.argmax(dim=-1)

        return is_red, char_pred
