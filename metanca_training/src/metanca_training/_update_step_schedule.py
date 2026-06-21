import logging
from typing import Callable

logger = logging.getLogger(__name__)


def constant_update_step(n: int) -> Callable[[int, int], int]:
    def schedule(metaepoch: int, current_num_update_steps: int) -> int:
        return n

    return schedule


def increment_update_step(rate: int, max_steps: int | None = None) -> Callable[[int, int], int]:
    def schedule(metaepoch: int, current_num_update_steps: int) -> int:
        if max_steps is not None and current_num_update_steps >= max_steps:
            return max_steps

        if metaepoch > 0 and (metaepoch % rate) == 0:
            next_num_update_steps = current_num_update_steps + 1
            if max_steps is not None:
                next_num_update_steps = min(next_num_update_steps, max_steps)

            logger.info(
                f"Metaepoch {metaepoch}: " f"incrementing # update steps to {next_num_update_steps}"
            )
            return next_num_update_steps

        return current_num_update_steps

    return schedule
