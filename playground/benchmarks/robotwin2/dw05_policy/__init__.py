"""RoboTwin policy package wrapper for DW05."""

from playground.benchmarks.robotwin2.dw05_policy.deploy_policy import (
    DW05RobotWinDeployPolicy,
    eval,
    get_model,
    reset_model,
)

__all__ = ["DW05RobotWinDeployPolicy", "eval", "get_model", "reset_model"]
