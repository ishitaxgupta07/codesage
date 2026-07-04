import time
import re
from groq import RateLimitError

def safe_invoke(llm, messages, max_retries=15):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            error_msg = str(e)
            match = re.search(r"try again in (\d+)m([\d.]+)s", error_msg)
            if match:
                wait_seconds = int(match.group(1)) * 60 + float(match.group(2))
            else:
                wait_seconds = 90
            wait_seconds += 10
            print(f"[rate limit] Waiting {wait_seconds:.0f}s before retrying (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_seconds)
    raise RuntimeError("Max retries exceeded for rate limit")