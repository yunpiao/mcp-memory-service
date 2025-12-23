"""Quality scoring system for memory evaluation."""
import os

# OPTIMIZATION: Skip heavy imports for Cloudflare backend
# Cloudflare uses remote embedding, no local quality scoring needed
_STORAGE_BACKEND = os.getenv('MCP_MEMORY_STORAGE_BACKEND', 'sqlite').lower()

__all__ = []

if _STORAGE_BACKEND != 'cloudflare':
    from .scorer import QualityScorer
    from .onnx_ranker import ONNXRankerModel
    from .ai_evaluator import QualityEvaluator
    from .implicit_signals import ImplicitSignalsEvaluator
    from .config import QualityConfig

    __all__ = [
        'QualityScorer',
        'ONNXRankerModel',
        'QualityEvaluator',
        'ImplicitSignalsEvaluator',
        'QualityConfig'
    ]
else:
    # Provide minimal stubs for Cloudflare mode
    QualityScorer = None
    ONNXRankerModel = None
    QualityEvaluator = None
    ImplicitSignalsEvaluator = None
    QualityConfig = None
