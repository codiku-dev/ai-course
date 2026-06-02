import re
import time

with open("corpus17mb-training-data.txt", "r", encoding="utf-8") as f:
    text = f.read()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence in sentences:
        print(sentence)
        time.sleep(0.5)
        input("Press Enter to continue...")
