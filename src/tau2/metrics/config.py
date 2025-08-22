from typing import Dict


class MetricsConfig:
    TSR_WEIGHTS: Dict[str, float] = {
        "communicate_info": 0.2,
        "action": 0.4,
        "nl_assertion": 0.4,
    }
    TSR_SUCCESS_THRESHOLD: float = 0.5
    TCRR_WINDOW_SIZE: int = 3
    TUE_WEIGHTS: Dict[str, float] = {"tool_correctness": 0.6, "param_accuracy": 0.4}
    GSRT_DEFAULT_JUDGE_MODEL: str = "gpt-4o-mini"
    GSRT_DEFAULT_JUDGE_ARGS: Dict = {"temperature": 0.0}

    METRICS_DISPLAY_PRECISION: int = 2
    CONFIDENCE_INTERVAL_LEVEL: float = 0.95

    @classmethod
    def validate_config(cls) -> None:
        """Validate that all configuration values are sensible."""

        tsr_sum = sum(cls.TSR_WEIGHTS.values())
        if abs(tsr_sum - 1.0) > 1e-6:
            raise ValueError(f"TSR weights must sum to 1.0, got {tsr_sum}")

        tue_sum = sum(cls.TUE_WEIGHTS.values())
        if abs(tue_sum - 1.0) > 1e-6:
            raise ValueError(f"TUE weights must sum to 1.0, got {tue_sum}")

        if not (0.0 <= cls.TSR_SUCCESS_THRESHOLD <= 1.0):
            raise ValueError(
                f"TSR_SUCCESS_THRESHOLD must be between 0 and 1, got {cls.TSR_SUCCESS_THRESHOLD}"
            )

        if cls.TCRR_WINDOW_SIZE < 1:
            raise ValueError(
                f"TCRR_WINDOW_SIZE must be >= 1, got {cls.TCRR_WINDOW_SIZE}"
            )

        if not (0.0 < cls.CONFIDENCE_INTERVAL_LEVEL < 1.0):
            raise ValueError(
                f"CONFIDENCE_INTERVAL_LEVEL must be between 0 and 1, got {cls.CONFIDENCE_INTERVAL_LEVEL}"
            )


MetricsConfig.validate_config()


def get_tsr_weights() -> Dict[str, float]:
    """Get TSR channel weights."""
    return MetricsConfig.TSR_WEIGHTS.copy()


def get_tue_weights() -> Dict[str, float]:
    """Get TUE component weights."""
    return MetricsConfig.TUE_WEIGHTS.copy()


def get_tcrr_window_size() -> int:
    """Get TCRR window size."""
    return MetricsConfig.TCRR_WINDOW_SIZE


def get_gsrt_judge_config() -> tuple[str, Dict]:
    """Get GSRT judge model and arguments."""
    return (
        MetricsConfig.GSRT_DEFAULT_JUDGE_MODEL,
        MetricsConfig.GSRT_DEFAULT_JUDGE_ARGS.copy(),
    )
