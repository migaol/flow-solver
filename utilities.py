import time

class Timestamp:
    '''Record labeled timestamps.'''
    def __init__(self) -> None:
        self.ts = []
        self.start = time.time()
        self.prev = self.start

    def add(self, name: str) -> None:
        '''Add a timestamp with the given name.'''
        now = time.time()
        self.ts.append((name, now - self.prev))
        self.prev = now

    def to_str(self) -> str:
        max_name_length = max(len(name) for name,_ in self.ts)
        formatted_ts = [f"{name.rjust(max_name_length)}: {t:.4f} s" for name,t in self.ts]
        formatted_ts.append(f"{'total'.rjust(max_name_length)}: {sum(t for _,t in self.ts):.4f} s")
        return '\n'.join(formatted_ts)