from llama_cpp import Llama

MODEL_PATH = "models/Qwen3-8B-Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,
)

def generate(question):
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the user's question accurately and concisely."
                ),
            },
            {
                "role": "user",
                "content": f"/no_think\n{question}",
            },
        ],
        max_tokens=512,
        temperature=0.2,
    )

    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = "What was Apple's total net sales for fiscal year 2025?"
    print(generate(question))