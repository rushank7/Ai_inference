from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Tiny GPT-2-style config. Small enough to run forward passes on CPU in milliseconds."""

    vocab_size: int = 65
    block_size: int = 32   # max sequence length (context window)
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 32

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head
