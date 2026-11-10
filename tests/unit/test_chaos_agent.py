"""Unit tests for ChaosAgent input validation and dry-run safety boundaries."""

from __future__ import annotations

import pytest

from harness.chaos_agent import CgroupLimits, ChaosAgent, ChaosValidationError, NetemProfile


def test_netem_command_generation_and_bounds() -> None:
    agent = ChaosAgent(dry_run=True)
    cmd = agent.apply_netem(NetemProfile(interface="eth0", delay_ms=15, jitter_ms=3, loss_percent=0.5))
    assert cmd == [
        "tc",
        "qdisc",
        "replace",
        "dev",
        "eth0",
        "root",
        "netem",
        "delay",
        "15ms",
        "3ms",
        "loss",
        "0.50%",
    ]


def test_rejects_shell_injection_interface_name() -> None:
    agent = ChaosAgent(dry_run=True)
    with pytest.raises(ChaosValidationError):
        agent.apply_netem(NetemProfile(interface="eth0; rm -rf /", delay_ms=5))


def test_cgroup_v2_path_confinement() -> None:
    agent = ChaosAgent(dry_run=True)
    writes = agent.apply_cgroup_v2_limits(
        CgroupLimits(
            cgroup_path="/sys/fs/cgroup/vfs_pgvector",
            cpu_quota_us=200_000,
            cpu_period_us=100_000,
            memory_max_bytes=4 * 1024 * 1024 * 1024,
        )
    )
    assert writes["/sys/fs/cgroup/vfs_pgvector/cpu.max"] == "200000 100000\n"

    with pytest.raises(ChaosValidationError):
        agent.apply_cgroup_v2_limits(CgroupLimits(cgroup_path="/tmp/not_a_cgroup"))
