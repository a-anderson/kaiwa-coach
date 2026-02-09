<!-- Required: language, user_text -->

# Native Reformulation

You are a native speaker of {language}. Rewrite the user's sentence to sound natural while preserving meaning. Ensure the correct grammar and punctuation are used.
You MUST reply only in {language}.
You MUST return a single JSON object and nothing else.
The response must start with '{{' and end with '}}'.
Do not include markdown, code fences, or extra keys.

User text:
{user_text}

Return ONLY valid JSON.
Schema:
{{"native": "<natural rewrite>"}}
