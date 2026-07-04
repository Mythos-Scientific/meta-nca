import logging

from . import functional, neighbors, nn, strategies, typing
from .nn import TaskNet, TinyCausalLM, update_tasknet

logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jaxlib").setLevel(logging.WARNING)
