from pathlib import Path

PROMPT_TEMPLATE_ID = "textvqa_short_v1"
PROMPT = "Answer the question using the image. Reply with only the short answer, with no explanation. Question: {question}"


def make_messages(question: str, image_path: str) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image", "image": Path(image_path).resolve().as_uri()},
        {"type": "text", "text": PROMPT.format(question=question)},
    ]}]
