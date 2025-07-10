import asyncio
from typing import Callable, Any, Tuple

class TaskManager:
    """Simple async task queue for background processing."""
    def __init__(self, workers: int = 1):
        self.queue: asyncio.Queue[Tuple[Callable, tuple, dict]] = asyncio.Queue()
        self.workers = workers
        self.worker_tasks = []

    async def start(self):
        """Start worker tasks."""
        for _ in range(self.workers):
            self.worker_tasks.append(asyncio.create_task(self._worker()))

    async def _worker(self):
        while True:
            func, args, kwargs = await self.queue.get()
            try:
                await asyncio.to_thread(func, *args, **kwargs)
            finally:
                self.queue.task_done()

    async def add_task(self, func: Callable, *args: Any, **kwargs: Any):
        await self.queue.put((func, args, kwargs))


# Global task manager instance
# The number of workers can be tuned according to your hardware
# One worker is often enough since Whisper models are GPU intensive
manager = TaskManager()
