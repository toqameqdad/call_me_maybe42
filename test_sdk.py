from llm_sdk import Small_LLM_Model


model = Small_LLM_Model()

tokens = model.encode("hello")

print("Encoded:", tokens)