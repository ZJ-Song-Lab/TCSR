"""TCSR: Teacher-Corroborated Student Set Reconstruction.

Reference-level post-training quantization of distilled DETR detectors.
The full-precision student P supplies every differentiable target; the frozen
teacher T only supplies detached, bounded corroboration and priority weights.

See the accompanying paper for the method description. All default
hyper-parameters in this package follow the frozen settings reported in the
paper (Table "Frozen method and optimization settings").
"""

__version__ = "1.0.0"
