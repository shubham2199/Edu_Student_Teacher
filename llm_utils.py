import os
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# Initialize GROQ Client and LLM
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Chat function to ask LLM based on student-specific context
def ask_student_agent(context_data, user_prompt):
    system_msg = (
        "You are a student assistant AI. Only answer questions "
        "that relate to the following student's data: "
        f"{context_data}. If asked questions not related to this data, politely refuse."
    )

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    return response.content
