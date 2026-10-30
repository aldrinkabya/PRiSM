import torch
import torch.nn as nn
from transformers import BertModel


class CLSClassifier(nn.Module):
    """BERT with a classifier over the [CLS] vectors of all its layers."""

    def __init__(self, n_classes, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.num_layers = self.bert.config.num_hidden_layers
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden * self.num_layers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def train_only(self, layers):
        """Freezes the encoder except the given layers. The head stays trainable."""
        for p in self.bert.parameters():
            p.requires_grad = False
        for i in layers:
            for p in self.bert.encoder.layer[int(i)].parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = torch.stack([h[:, 0, :] for h in out.hidden_states[1:]], dim=1)
        return self.head(self.dropout(cls.view(cls.size(0), -1)))
