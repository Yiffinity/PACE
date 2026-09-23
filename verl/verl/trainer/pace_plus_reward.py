"""Task-neutral reward for direct PACE+ policy distillation."""


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """Return zero because PACE+ optimizes only the configured distillation loss."""
    del data_source, solution_str, ground_truth, extra_info, kwargs
    return 0.0
