## Project Goals
Original source code: https://github.com/emarco177/langchain-course/tree/project/code-interpreter

1. Implement a slim custom version of OpenAI's **Code Interpreter** using GPT-4.  
2. Understand how a code execution agent operates, including:  
   - Writing and running code dynamically.  
   - Returning execution results and generated artifacts.  
3. Explore the **limitations** of using AI agents in production.  
4. Learn and apply **OpenAI Function Calling** for code execution tasks.  
5. Replicate core capabilities demonstrated by OpenAI’s Code Interpreter:  
   - Generate and save files (e.g., Python-created QR codes).  
   - Perform data analysis on CSV files using Python and Pandas.  
   - Process and transform datasets based on natural language prompts.  
6. Understand how the agent:  
   - Plans tasks from a user request.  
   - Selects appropriate tools (e.g., Python interpreter).  
   - Executes code in an isolated environment.  
   - Transforms output into a usable response.  
7. Identify pitfalls in data parsing (e.g., counting CSV rows with multi-value fields incorrectly).  
8. Implement backend logic to mimic the operational flow of the Code Interpreter.  

Required:
- pandas
- tabulate
- qrcode
- pillow