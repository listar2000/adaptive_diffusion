from abc import ABC, abstractmethod
import numpy as np

from abc import ABC, abstractmethod


class BasicSchedule(ABC):
    """
    An abstract base class for different types of parameter schedules.
    """

    def __init__(self, init_value: float):
        self.value = init_value
        self.step_count = 0

    @abstractmethod
    def step(self) -> float:
        """
        Proceeds to the next step in the schedule and returns the current value.

        Must be implemented by subclasses.

        Returns:
            float: The updated value after the step.
        """
        pass

    def get_value(self) -> float:
        """
        Gets the current value of the schedule.

        Returns:
            float: The current value of the schedule.
        """
        return self.value

    def reset(self) -> None:
        """
        Resets the schedule to its initial state.
        """
        self.step_count = 0
        self._reset_value()

    def _reset_value(self) -> None:
        """
        Resets the specific value to its initial state.

        Can be overridden by subclasses if they hold state that should
        be reset beyond the `self.value`.
        """
        self.value = self.init_value

    def __call__(self) -> float:
        """
        Allows the schedule to be called as a function.

        Returns:
            float: The updated value after the step.
        """
        return self.step()


class ConstantSchedule(BasicSchedule):
    def step(self) -> float:
        return self.value


class TimeBasedDecaySchedule(BasicSchedule):
    def __init__(self, init_value: float, decay_rate: float):
        super().__init__(init_value)
        self.decay_rate = decay_rate

    def step(self) -> float:
        self.value *= (1 / (1 + self.decay_rate * self.step_count))
        self.step_count += 1
        return self.value


class StepDecaySchedule(BasicSchedule):
    def __init__(self, init_value: float, step_size: int, decay_factor: float):
        super().__init__(init_value)
        self.step_size = step_size
        self.decay_factor = decay_factor

    def step(self) -> float:
        if self.step_count % self.step_size == 0:
            self.value *= self.decay_factor
        self.step_count += 1
        return self.value


class ExponentialDecaySchedule(BasicSchedule):
    def __init__(self, init_value: float, decay_rate: float):
        super().__init__(init_value)
        self.decay_rate = decay_rate

    def step(self) -> float:
        self.value *= self.decay_rate ** self.step_count
        self.step_count += 1
        return self.value


class ExponentialGrowthSchedule(BasicSchedule):
    def __init__(self, init_value: float, growth_rate: float):
        """
        Creates an exponential growth schedule for the inverse temperature.

        Parameters:
        - init_value: The initial value for beta.
        - growth_rate: The rate at which beta grows each step. A higher rate means a quicker reduction in temperature.
        """
        super().__init__(init_value)
        self.growth_rate = growth_rate

    def step(self) -> float:
        """
        Updates the inverse temperature beta exponentially.

        Returns:
        - The updated beta value.
        """
        self.value *= np.exp(self.growth_rate * self.step_count)
        self.step_count += 1
        return self.value
