from src.Fetch import Fetch
from src.Process import Process

fetcher = Fetch()
df = fetcher.fetch("SPY")
processor = Process()
result = processor.preprocess(df)
print(result.tail())
