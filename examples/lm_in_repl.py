"""
Example: Using llm_query() from within a Local REPL environment.

This demonstrates the LM Handler + LocalREPL integration where code
running in the REPL can query the LLM via socket connection.

Uses LMStudio running on the host machine with an OpenAI-compatible API.
"""

from rlm.clients.openai import OpenAIClient
from rlm.core.lm_handler import LMHandler
from rlm.environments.local_repl import LocalREPL

# LMStudio endpoint from inside Docker container
LMSTUDIO_BASE_URL = "http://host.docker.internal:1234/v1"
LMSTUDIO_MODEL = "qwen/qwen3-coder-30b"

setup_code = """
secret = "1424424"
"""

context_payload = """
This is a test context. It should print out, revealing the magic number to be 4.
"""

code = """
response = llm_query("What is 2 + 2? Reply with just the number.")
print(response)
print(type(response))
print(context)
print("Secret from setup code: ", secret)
"""


def main():
    client = OpenAIClient(
        api_key="lm-studio",  # LMStudio doesn't require a real API key
        model_name=LMSTUDIO_MODEL,
        base_url=LMSTUDIO_BASE_URL,
    )
    print(f"Created LMStudio client with model: {LMSTUDIO_MODEL}")

    # Start LM Handler
    with LMHandler(client=client) as handler:
        print(f"LM Handler started at {handler.address}")

        # Create REPL with handler connection
        with LocalREPL(
            lm_handler_address=handler.address,
            context_payload=context_payload,
            setup_code=setup_code,
        ) as repl:
            print("LocalREPL created, connected to handler\n")

            # Run code that uses llm_query
            print(f"Executing: {code}")

            result = repl.execute_code(code)

            print(f"stdout: {result.stdout!r}")
            print(f"stderr: {result.stderr!r}")
            print(f"response variable: {repl.locals.get('response')!r}")
            print(f"locals: {repl.locals!r}")
            print(f"execution time: {result.execution_time:.3f}s")


if __name__ == "__main__":
    main()
