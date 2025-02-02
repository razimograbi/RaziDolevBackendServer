import asyncio
import time

class AudioAccumulator:
    """
    Asynchronous version of AudioAccumulator using asyncio.Lock.
    Accumulates raw PCM data until threshold_bytes or time_threshold is reached.
    """

    def __init__(
        self,
        sample_rate=16000,
        sample_width=2,
        channels=1,
        threshold_bytes=50 * 1024,  # 50 KB
        time_threshold=5             # 5 seconds
    ):
        self.sample_rate = sample_rate
        self.sample_width = sample_width  # 2 bytes = 16-bit
        self.channels = channels
        self.chunks = bytearray()

        self.threshold_bytes = threshold_bytes
        self.time_threshold = time_threshold

        self.last_transcribe_time = time.time()
        self.lock = asyncio.Lock()  # Async Lock

    async def add_chunk(self, pcm_data: bytes) -> bool:
        """
        Asynchronously adds a chunk of PCM data.
        Returns True if threshold (size or time) is reached.
        """
        async with self.lock:  # Use async lock
            self.chunks.extend(pcm_data)
            current_time = time.time()

            # Check thresholds
            size_met = len(self.chunks) >= self.threshold_bytes
            time_met = (current_time - self.last_transcribe_time) >= self.time_threshold

            return size_met or time_met

    async def flush(self) -> bytes:
        """
        Asynchronously retrieves and clears the buffer with async lock protection.
        Returns the accumulated data.
        """
        async with self.lock:
            data = bytes(self.chunks)  # Copy buffer
            self.chunks.clear()  # Clear buffer
            self.last_transcribe_time = time.time()  # Update timestamp
            return data

    @property
    async def current_size(self) -> int:
        """Asynchronously returns buffer size."""
        async with self.lock:
            return len(self.chunks)

