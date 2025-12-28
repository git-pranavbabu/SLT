from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
# from langchain_community.chat_models import ChatOllama # Use this if local
from agent.agent_state import AgentState
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



# --- SETUP LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile")

SYSTEM_PROMPT = """
You are a Sign Language Interpreter. Convert the following sequence of Gloss words into fluent English.
Input: "ME HUNGRY" -> Output: "I am hungry."
"""

# --- NODE 1: THE ACCUMULATOR ---
# This node doesn't do much logic, it just holds the state.
# In a real app, we might clean duplicates here.
def accumulator_node(state: AgentState):
    # This node just passes the state forward. 
    # The actual appending happens in the graph edge (we'll see in Lesson 4).
    # For now, we just return the current state.
    current_glosses = state.get("gloss_sequence", [])
    print(f"📝 Current Buffer: {current_glosses}")
    return {"gloss_sequence": current_glosses}

# --- NODE 2: THE TRANSLATOR ---
def translator_node(state: AgentState):
    glosses = state["gloss_sequence"]
    
    # 1. Filter out the Command words (like "SEND")
    clean_glosses = [word for word in glosses if word != "SEND"]
    
    if not clean_glosses:
        return {"final_translation": "..."}
    
    # 2. Construct the Prompt
    gloss_text = " ".join(clean_glosses)
    print(f"🚀 Triggering LLM with: {gloss_text}")
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=gloss_text)
    ]
    
    # 3. Call Brain
    response = llm.invoke(messages)
    
    # 4. Clear the buffer after translation!
    # We return an empty list to reset the memory for the next sentence.
    return {
        "final_translation": response.content,
        "gloss_sequence": [] 
    }
