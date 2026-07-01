"""Optional model variants used by the ASR experiments."""

from __future__ import annotations

import torch
from transformers import Wav2Vec2ForCTC


class BiLSTMCTCHead(torch.nn.Module):
    """A recurrent CTC head that preserves the encoder time resolution."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = torch.nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.lstm(self.dropout(hidden_states))
        return self.projection(self.dropout(hidden_states))


class Wav2Vec2BiLSTMForCTC(Wav2Vec2ForCTC):
    """Wav2Vec2 CTC model with a serializable two-way LSTM head."""

    def __init__(self, config) -> None:
        super().__init__(config)
        hidden_size = int(getattr(config, "ctc_head_hidden_size", 256))
        num_layers = int(getattr(config, "ctc_head_num_layers", 2))
        dropout = float(getattr(config, "ctc_head_dropout", 0.1))
        self.lm_head = BiLSTMCTCHead(
            input_size=config.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            vocab_size=config.vocab_size,
            dropout=dropout,
        )
        # The LSTM keeps PyTorch's stable default initialization. Initialize
        # the new output projection with the same policy as Wav2Vec2.
        self._init_weights(self.lm_head.projection)

