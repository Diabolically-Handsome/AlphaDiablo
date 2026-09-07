"""The original v6 shopping algorithm under an explicit wider time contract."""
from .completion_clock import COMPLETION_L2_V1
from .resource_sustain_armor import SustainEquipmentReadinessService


class SustainCompletionService(SustainEquipmentReadinessService):
    @property
    def collect_microstep_cap(self):
        return COMPLETION_L2_V1.collect_command_window_microsteps

    @property
    def service_microstep_cap(self):
        return COMPLETION_L2_V1.service_microsteps

    def telemetry(self):
        result = super().telemetry()
        result.update(time_protocol=COMPLETION_L2_V1.protocol,
                      collect_command_window_microsteps=self.collect_microstep_cap,
                      service_microstep_cap=self.service_microstep_cap)
        return result
