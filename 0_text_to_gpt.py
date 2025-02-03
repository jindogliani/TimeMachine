from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI

# OpenAI API 키를 파일에서 읽음
with open("Open_AI_api_key.txt", "r") as file:
    open_ai_api_key = file.read().strip()

# ReFlexion Prompt 파일에서 읽음
with open("ReFlexion_prompt(Action).txt", "r") as file:
    reflexion_prompt_action = file.read().strip()
with open("ReFlexion_prompt(Scene).txt", "r") as file:
    reflexion_prompt_scene = file.read().strip()

class GPT4o(ABC):
    def __init__(self, temperature=0.1, system_message=None):
        super().__init__()
        self.api_key = open_ai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API Key is missing. Set it as an environment variable.")
        
        self.openai = ChatOpenAI(
            api_key=self.api_key,
            model_name="gpt-4o",
            temperature=temperature,
        )
        self.system_message = system_message

    @abstractmethod
    def process_message(self, message):
        """Abstract method to define how the input message will be processed."""
        pass

    def invoke(self, message):
        try:
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": message}
            ]
            response = self.openai(messages=messages)
            return response.content
        except Exception as e:
            return f"Error occurred: {e}"

class SceneGraphGPT(GPT4o):
    def __init__(self, temperature=0.1):
        super().__init__(temperature, system_message=reflexion_prompt_scene)

class ActionSceneGraphGPT(GPT4o):
    def __init__(self, temperature=0.1):
        super().__init__(temperature, system_message=reflexion_prompt_action)

    def process_message(self, message):
        # 입력 문장을 그대로 반환 (추가 처리 없음)
        return message

if __name__ == "__main__":
    gpt4o = ActionSceneGraphGPT()
    
    user_input = input("Enter a sentence to generate Scene Graphs: ")
    response = gpt4o.invoke(user_input)
    
    print("\n=== Generated Scene Graph ===")
    print(response)
