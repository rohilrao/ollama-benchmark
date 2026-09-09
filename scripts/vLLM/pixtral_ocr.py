import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


BASE_URL = "http://localhost:8555/v1"
MODEL_NAME = "/models/pixtral"
IMAGE_PATH = "test_page.png"

client = OpenAI(
    base_url=BASE_URL,
    api_key="dummy",
)


def encode_image(path):
    image_bytes = Path(path).read_bytes()
    return base64.b64encode(image_bytes).decode("utf-8")


image_b64 = encode_image(IMAGE_PATH)


def run_ocr(request_id):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all text from this image. "
                            "Preserve the reading order and formatting where possible."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        },
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=2048,
    )

    return request_id, response.choices[0].message.content


# ------------------------------------------------------------
# Single request test
# ------------------------------------------------------------

request_id, text = run_ocr(0)

print("OCR result:")
print(text)


# ------------------------------------------------------------
# Parallel request test
# ------------------------------------------------------------

NUM_PARALLEL_REQUESTS = 10

with ThreadPoolExecutor(max_workers=NUM_PARALLEL_REQUESTS) as executor:
    futures = [
        executor.submit(run_ocr, i)
        for i in range(NUM_PARALLEL_REQUESTS)
    ]

    for future in as_completed(futures):
        request_id, text = future.result()
        print(f"\nRequest {request_id} completed")
        print(text[:500])
