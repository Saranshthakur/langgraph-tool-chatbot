#  LangGraph Intelligent Chatbot

In this project we have build a chatbot that can both **chat naturally** and **look up real information** when it needs to.

It uses **LangGraph** to handle the conversation flow and **Groq’s Gemma-2-9B model** to generate responses. When you ask a factual question, it automatically uses **Wikipedia** or **ArXiv** to find real answers before replying.

---

##  What It Does

The chatbot can do two main things:

1. **Talk like a normal assistant.**  
   Ask it open questions or general topics — it just replies conversationally using the Groq model.

2. **Look things up when needed.**  
   If your question needs real data (like “Who invented backpropagation?” or “What is LangGraph?”),  
   the model decides on its own to call one of two tools:
   - **Wikipedia** → for short factual summaries  
   - **ArXiv** → for academic paper abstracts  

LangGraph manages the logic — when to use the tools, how to pass the query, and how to merge the result into the final reply.

---

##  How It Works

Here’s the basic flow:

1. **Setup the tools**  
   - `ArxivQueryRun` calls the ArXiv API for research papers.  
   - `WikipediaQueryRun` pulls short summaries from Wikipedia.

2. **Keep track of the conversation**  
   - The `State` class saves messages between the user and the bot using LangGraph’s `add_messages` helper.

3. **Build the workflow**  
   - A `StateGraph` defines how the messages move through nodes:  
     - `chatbot` → the main LLM logic  
     - `tools` → runs when the model requests external data  
   - Conditional edges (`tools_condition`) decide when to jump between them.

4. **Run it**  
   - You send a message like “What is LangGraph?”   
   - The final answer is printed at the end.

In short, the model figures out when it needs extra info and uses the right tool to get it before replying.

