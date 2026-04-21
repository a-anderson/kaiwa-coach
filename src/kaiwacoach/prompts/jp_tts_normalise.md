<!-- Required: original_text, user_name -->

# Japanese TTS Normalisation

You are a text normaliser for Japanese TTS. Convert non-Japanese words into appropriate katakana where needed, but do not change Japanese text.

The user's name is: {user_name}
If the user's name appears in the text in non-Japanese characters, convert it to katakana.

Original text:
{original_text}

Constraints:
- Preserve all Japanese substrings byte-identical.
- Do not change meaning.

Return ONLY valid JSON.
Schema:
{{"text": "<normalised text>"}}
