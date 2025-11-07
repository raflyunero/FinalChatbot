from openai import OpenAI
client = OpenAI(api_key="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")

models = client.models.list()

for model in models.data:
    print(model.id)