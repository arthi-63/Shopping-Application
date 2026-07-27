from google import genai
from config.settings import GEMINI_API_KEY


class GeminiService:

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def generate_recommendation(self, user_query, products):

        prompt = f"""
You are an intelligent shopping assistant.

User Query:
{user_query}

Here are the top matching products:

{products}

Your task:
1. Recommend the best product.
2. Explain why it matches the user's request.
3. Mention one or two alternatives.
4. Keep the response friendly and concise.
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            return response.text

        except Exception as e:
            return f"Error: {e}"