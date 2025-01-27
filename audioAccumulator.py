import time

class AudioAccumulator:
    """
    Accumulates raw PCM data for a participant until
    either threshold_bytes or time_threshold is reached.
    """

    def __init__(
        self,
        sample_rate=16000,
        sample_width=2,
        channels=1,
        threshold_bytes=15 * 1024,  # 15 KB
        time_threshold=5             # 5 seconds
    ):
        self.sample_rate = sample_rate
        self.sample_width = sample_width  # 2 bytes = 16-bit
        self.channels = channels
        self.chunks = bytearray()

        self.threshold_bytes = threshold_bytes
        self.time_threshold = time_threshold

        self.last_transcribe_time = time.time()

    def add_chunk(self, pcm_data: bytes) -> bool:
        """
        Append new PCM data to the buffer, then check if we should auto-transcribe.
        Returns True if threshold is reached; False otherwise.
        """
        self.chunks.extend(pcm_data)
        now = time.time()

        if (len(self.chunks) >= self.threshold_bytes) or \
                ((now - self.last_transcribe_time) >= self.time_threshold):
            return True
        return False

    def flush(self) -> bytes:
        """
        Returns the accumulated PCM and resets the buffer + timer.
        """
        data = bytes(self.chunks)
        self.chunks.clear()
        self.last_transcribe_time = time.time()
        return data
