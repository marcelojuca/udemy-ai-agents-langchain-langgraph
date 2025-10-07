import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

async def main() -> None:
    print("Hello from langchain-course!")



if __name__ == "__main__":
    asyncio.run(main())
