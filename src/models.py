# CNN and LSTM text classifiers (3 classes: gpt / claude / gemini).
# Both take (token_ids, lengths) so the same training code works for both.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=100,
                 kernel_sizes=(3, 4, 5), num_classes=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # one conv layer per kernel size (looks at 3-, 4-, and 5-token windows)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)              # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)            # Conv1d wants (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))        # (batch, num_filters, seq_len - k + 1)
            h = h.max(dim=2).values          # max pooling over the whole sequence
            pooled.append(h)
        h = torch.cat(pooled, dim=1)         # (batch, num_filters * 3)
        return self.fc(self.dropout(h))      # (batch, 3)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)              # (batch, seq_len, embed_dim)
        # packing tells the LSTM where each sequence really ends, so it skips the padding
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]                          # last hidden state, (batch, hidden_dim)
        return self.fc(self.dropout(h))      # (batch, 3)


if __name__ == "__main__":
    # quick shape check with fake data
    x = torch.randint(0, 1000, (4, 512))
    lengths = torch.tensor([512, 300, 50, 10])
    print("CNN output: ", CNNClassifier(vocab_size=1000)(x, lengths).shape)
    print("LSTM output:", LSTMClassifier(vocab_size=1000)(x, lengths).shape)