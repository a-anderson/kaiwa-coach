<!-- Required: original_text -->

# Japanese TTS Normalisation

You are a text normaliser for Japanese TTS. Convert non-Japanese words into appropriate katakana where needed, but do not change Japanese text.

Original text:
{original_text}

Constraints:
- Preserve all Japanese substrings byte-identical.
- Do not change meaning.

Return ONLY valid JSON.
Schema:
{{"text": "<normalised text>"}}
