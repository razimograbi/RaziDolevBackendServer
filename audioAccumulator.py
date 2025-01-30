import time
import threading

class AudioAccumulator:
    """
    Thread-safe Accumulates raw PCM data for a participant until
    either threshold_bytes or time_threshold is reached.
    """

    def __init__(
        self,
        sample_rate=16000,
        sample_width=2,
        channels=1,
        threshold_bytes=50 * 1024,  # 15 KB
        time_threshold=5             # 5 seconds
    ):
        self.sample_rate = sample_rate
        self.sample_width = sample_width  # 2 bytes = 16-bit
        self.channels = channels
        self.chunks = bytearray()

        self.threshold_bytes = threshold_bytes
        self.time_threshold = time_threshold

        self.last_transcribe_time = time.time()
        self.lock = threading.Lock()

    def add_chunk(self, pcm_data: bytes) -> bool:
        """
        Thread-safe chunk addition with combined size/time check.
        Returns True if processing threshold is met.
        """
        with self.lock:
            # Atomic modification of buffer and timestamp
            self.chunks.extend(pcm_data)
            current_time = time.time()

            # Check thresholds while holding the lock
            size_met = len(self.chunks) >= self.threshold_bytes
            time_met = (current_time - self.last_transcribe_time) >= self.time_threshold

            return size_met or time_met

    def flush(self) -> bytes:
        """
        Atomically retrieves and resets the buffer with lock protection.
        Returns empty bytes if called concurrently.
        """
        with self.lock:
            data = bytes(self.chunks)
            self.chunks.clear()
            self.last_transcribe_time = time.time()
            return data

    @property
    def current_size(self) -> int:
        """Thread-safe buffer size check"""
        with self.lock:
            return len(self.chunks)
