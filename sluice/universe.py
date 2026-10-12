CORE = ("NVDA", "AMD", "AVGO", "TSM", "MSFT", "GOOGL", "META", "ORCL", "AMZN")
CONTROL = ("KO", "JNJ")
BENCHMARK = ("SPY", "QQQ")
UNIVERSE = CORE + CONTROL + BENCHMARK

# Other share classes of a company in the universe, mapped to the one we track.
ALIASES = {"GOOG": "GOOGL"}


def canonical(ticker: str) -> str:
    symbol = ticker.strip().upper()
    return ALIASES.get(symbol, symbol)
