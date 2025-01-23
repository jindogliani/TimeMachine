from abc import abstractmethod
# from langchain.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

with open("Open_AI_api_key.txt", "r") as file:
    open_ai_api_key = file.read().strip()

class GPT4o:
    def __init__(self, temperature=0.1):
        super().__init__()
        api_key=open_ai_api_key
        if not api_key:
            raise ValueError("OpenAI API Key is missing. Set it as an environment variable.")
        self.openai = ChatOpenAI(
            api_key=api_key,
            model_name="gpt-4o",
            temperature=temperature,
        )

    def invoke(self, message):
        try:
            response = self.openai.invoke(message)
            return response.content.strip()
        except Exception as e:
            return f"Error occurred: {e}"

if __name__ == "__main__":
    gpt4o = GPT4o()
    print(gpt4o.invoke("Hello, how are you?"))
