from src import config

def mask(s: str) -> str:
    if len(s) < 8:
        return "***"
    return s[:4] + "*" * (len(s) - 8) + s[-4:]

print("LLM_PROVIDER:", config.LLM_PROVIDER)
print("OPENAI_MODEL:", config.OPENAI_MODEL)
print("EMBEDDING_MODEL:", config.EMBEDDING_MODEL)
print("DATA_DIR:", config.DATA_DIR)
print("INDEX_DIR:", config.INDEX_DIR)
print("API key looks loaded:", mask(config.OPENAI_API_KEY))