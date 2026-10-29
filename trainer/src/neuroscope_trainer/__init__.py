"""Train a small OLMo 2 style language model on this Mac (MLX), then export it for the NeuroScope viewer.

The trainer runs as its own process and reports progress as JSON lines on stdout; the contract is
docs/trainer-protocol.md. Subcommands: bench, prepare, pretrain, sft, export, compare.
"""

__version__ = "0.1.0"
PROTOCOL_VERSION = 1
