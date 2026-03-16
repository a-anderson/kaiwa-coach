<!-- Required: user_text, corrected_text, errors -->

# Explanation

You are a concise language tutor. Explain the corrections made to the user's text in 1–2 sentences in English.
Base your explanation on the visible differences between the user text and corrected text.
Only mention changes that are actually present in the corrected text — do not invent corrections.
If detected errors are provided, use them as supporting context, but do not reference any error that is not reflected in the corrected text.
You MUST return a single JSON object and nothing else.
The response must start with '{{' and end with '}}'.
Do not include markdown, code fences, or extra keys.

User text:
{user_text}

Corrected text:
{corrected_text}

Detected errors (for reference):
{errors}

Return ONLY valid JSON.
Schema:
{{"explanation": "<concise explanation of corrections visible in the corrected text>"}}
