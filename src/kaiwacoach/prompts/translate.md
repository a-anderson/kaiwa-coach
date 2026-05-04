<!-- Required: source_language, target_language, text -->

# Translate

Translate the following {source_language} text into natural sounding {target_language}.
Preserve the meaning and tone faithfully. Do not add explanations or commentary.
You MUST return a single JSON object and nothing else.
The response must start with '{{' and end with '}}'.
Do not include markdown, code fences, or extra keys.

Text to translate:
{text}

Return ONLY valid JSON.
Schema:
{{"translation": "<translated text in {target_language}>"}}
