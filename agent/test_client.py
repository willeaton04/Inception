import ollama_client


if (ollama_client.is_available()):
    user = input('Ask ollama anything: ')
    response = ollama_client.generate(prompt=user, system='Answer the question in a concise manner')
    print(response)


