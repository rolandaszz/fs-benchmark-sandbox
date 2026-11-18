"""Host-level Linux fault injection controller wrapping tc netem, cgroups v2, and page-cache drops."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


class ChaosValidationError(ValueError):
    """Raised when chaos parameters violate workstation safety bounds."""


@dataclass(frozen=True)
class NetemProfile:
    """Kernel network emulation parameters applied via tc netem."""

    interface: str = "lo"
    delay_ms: int = 10
    jitter_ms: int = 2
    loss_percent: float = 0.0


@dataclass(frozen=True)
class CgroupLimits:
    """Linux cgroups v2 CPU quota and hard memory limit parameters."""

    cgroup_path: str
    cpu_quota_us: int = 200_000
    cpu_period_us: int = 100_000
    memory_max_bytes: int = 4 * 1024 * 1024 * 1024


def _validate_identifier(value: str, label: str) -> str:
    if not SAFE_IDENTIFIER_RE.match(value):
        raise ChaosValidationError(f"unsafe {label} identifier: {value!r}")
    return value


def _parse_memory_mib(raw: str) -> float:
    """Parse Docker stats memory string (e.g. '412.5MiB / 4GiB' or '1.2GiB / 4GiB') into MiB."""
    usage_part = raw.split("/")[0].strip()
    match = re.match(r"^([0-9.]+)\s*([a-zA-Z]+)$", usage_part)
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"gib", "gb", "g"}:
        return val * 1024.0
    if unit in {"mib", "mb", "m"}:
        return val
    if unit in {"kib", "kb", "k"}:
        return val / 1024.0
    return val


class ChaosAgent:
    """Executes or dry-runs strictly bounded host chaos primitives without shell interpolation."""

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run

    def build_netem_apply_cmd(self, profile: NetemProfile) -> list[str]:
        iface = _validate_identifier(profile.interface, "network interface")
        if profile.delay_ms < 0 or profile.delay_ms > 2_000:
            raise ChaosValidationError("delay_ms must be between 0 and 2000")
        if profile.jitter_ms < 0 or profile.jitter_ms > 1_000:
            raise ChaosValidationError("jitter_ms must be between 0 and 1000")
        if profile.loss_percent < 0.0 or profile.loss_percent > 50.0:
            raise ChaosValidationError("loss_percent must be between 0.0 and 50.0")

        cmd = [
            "tc",
            "qdisc",
            "replace",
            "dev",
            iface,
            "root",
            "netem",
            "delay",
            f"{profile.delay_ms}ms",
            f"{profile.jitter_ms}ms",
        ]
        if profile.loss_percent > 0.0:
            cmd.extend(["loss", f"{profile.loss_percent:.2f}%"])
        return cmd

    def build_netem_clear_cmd(self, interface: str = "lo") -> list[str]:
        iface = _validate_identifier(interface, "network interface")
        return ["tc", "qdisc", "del", "dev", iface, "root"]

    def apply_netem(self, profile: NetemProfile) -> list[str]:
        cmd = self.build_netem_apply_cmd(profile)
        if not self.dry_run:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        return cmd

    def clear_netem(self, interface: str = "lo") -> list[str]:
        cmd = self.build_netem_clear_cmd(interface)
        if not self.dry_run:
            subprocess.run(cmd, check=False, capture_output=True, text=True)
        return cmd

    def apply_cgroup_v2_limits(self, limits: CgroupLimits) -> dict[str, str]:
        if limits.cpu_quota_us <= 0 or limits.cpu_period_us <= 0:
            raise ChaosValidationError("CPU quota and period must be positive integers")
        if limits.memory_max_bytes < 64 * 1024 * 1024:
            raise ChaosValidationError("memory_max_bytes must be at least 64 MiB")

        cg_dir = Path(limits.cgroup_path).resolve()
        if not cg_dir.is_relative_to(Path("/sys/fs/cgroup")):
            raise ChaosValidationError(f"cgroup_path must reside under /sys/fs/cgroup: {cg_dir}")

        payload = {
            str(cg_dir / "cpu.max"): f"{limits.cpu_quota_us} {limits.cpu_period_us}\n",
            str(cg_dir / "memory.max"): f"{limits.memory_max_bytes}\n",
        }
        if not self.dry_run:
            for file_path, content in payload.items():
                Path(file_path).write_text(content, encoding="utf-8")
        return payload

    def drop_os_page_cache(self) -> str:
        action = "sync && write 3 -> /proc/sys/vm/drop_caches"
        if not self.dry_run:
            subprocess.run(["sync"], check=True)
            Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="utf-8")
        return action

    def sample_container_resources(self, engine: str) -> tuple[float, float]:
        """Sample live container (memory_mib, cpu_percent) via docker stats without shell interpolation."""
        safe_eng = _validate_identifier(engine, "engine")
        container_name = f"vfs-{safe_eng}"
        cmd = ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}|{{.CPUPerc}}", container_name]
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3.0)
            line = res.stdout.strip()
            if "|" not in line:
                return (0.0, 0.0)
            mem_str, cpu_str = line.split("|", 1)
            mem_mib = _parse_memory_mib(mem_str)
            cpu_pct = float(cpu_str.strip().rstrip("%") or 0.0)
            return (round(mem_mib, 2), round(cpu_pct, 2))
        except (subprocess.SubprocessError, OSError, ValueError):
            return (0.0, 0.0)
