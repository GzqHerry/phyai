"""Robot-side RPC configuration. No checkpoint or model imports."""

from dataclasses import dataclass

from phyai_robot.remote import RemotePolicy


@dataclass
class RemotePolicyConfig:
    """Client endpoint and per-exchange waits, independent of robot control Hz.

    timeout_s covers metadata, prediction and end-run calls. The first warmup
    gets its own larger budget for lazy model initialization/compilation.
    """

    endpoint: str = "tcp://127.0.0.1:5555"
    timeout_s: float = 2.0
    warmup_timeout_s: float = 120.0

    def create(self):
        """Construct a client without connecting; begin_run performs the handshake."""
        return RemotePolicy(
            self.endpoint,
            timeout_s=self.timeout_s,
            warmup_timeout_s=self.warmup_timeout_s,
        )

    def validate(self):
        if not self.endpoint.strip():
            raise ValueError("remote.endpoint must not be empty")
        self.create()
