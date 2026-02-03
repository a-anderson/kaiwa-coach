# llm_smoke.py
"""
LLM smoke test for KaiwaCoach.
Loads Qwen3 via mlx-lm and tests strict JSON-schema output with a single retry repair.

Usage:
  python llm_smoke.py --language ja
  python llm_smoke.py --language fr

Install:
  pip install -U mlx-lm pydantic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

from pydantic import BaseModel, ValidationError

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL_ID = "mlx-community/Qwen3-14B-bf16"


class ConversationReply(BaseModel):
    reply: str


REPAIR_PROMPT = """Rewrite the following output so that it strictly follows the required JSON format.
Do not add or remove fields.
Output ONLY valid JSON.

Original output:
{bad_output}
"""


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from a string that may contain
    extra text, code fences, or multiple JSON objects.
    """
    s = text.strip()

    # Remove common code-fence wrappers if present
    if "```" in s:
        s = s.replace("```json", "").replace("```", "").strip()

    start = s.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", s, 0)
    s = s[start:]

    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(s)

    if not isinstance(obj, dict):
        raise json.JSONDecodeError("First JSON value is not an object", s, 0)

    return obj



def render_conversation_prompt(language: str) -> str:
    return f"""You are KaiwaCoach, a conversational language partner.
You respond concisely but naturally.
You output ONLY valid JSON matching this schema: {{"reply": "..."}}.
Do not output anything except JSON.

Language: {language}
User: Give me one short, natural greeting and ask how I am.
"""


def generate_json_with_retry(
    model,
    tokenizer,
    prompt: str,
    schema_model: type[BaseModel],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 256,
    retry_once: bool = True,
) -> BaseModel:
    sampler = make_sampler(temperature, top_p=top_p)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)

    try:
        parsed = extract_first_json_object(raw)
        return schema_model.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError):
        if not retry_once:
            raise

        repair = REPAIR_PROMPT.format(bad_output=raw)
        sampler2 = make_sampler(0.0, top_p=1.0)
        raw2 = generate(model, tokenizer, prompt=repair, max_tokens=max_tokens, sampler=sampler2)

        parsed2 = extract_first_json_object(raw2)
        return schema_model.model_validate(parsed2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["ja", "fr"], required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID)
    args = p.parse_args()

    print(f"[llm_smoke] Loading model: {args.model}")
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"[llm_smoke] Loaded in {time.time()-t0:.1f}s")

    prompt = render_conversation_prompt(args.language)
    print("[llm_smoke] Generating...")
    t1 = time.time()
    out = generate_json_with_retry(model, tokenizer, prompt, ConversationReply)
    dt = time.time() - t1

    print("\n=== JSON OUTPUT ===")
    print(out.model_dump_json(indent=2, ensure_ascii=False))
    print("===================\n")
    print(f"[llm_smoke] Latency: {dt:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[llm_smoke] Cancelled.")
        sys.exit(130)
