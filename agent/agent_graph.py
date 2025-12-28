from langgraph.graph import StateGraph, END
from agent.agent_state import AgentState
from agent.agent_nodes import accumulator_node, translator_node

# 1. Initialize the Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("accumulator", accumulator_node)
workflow.add_node("translator", translator_node)

# 3. Define the Entry Point
workflow.set_entry_point("accumulator")

# 4. THE CONDITIONAL LOGIC (The Router)
def should_translate(state: AgentState):
    glosses = state["gloss_sequence"]
    
    if not glosses:
        return "wait"
    
    last_word = glosses[-1]
    
    # --- THIS IS YOUR SEND GESTURE LOGIC ---
    if last_word == "SEND":
        return "translate"
    else:
        return "wait"

# 5. Add Edges
# From accumulator, we verify the condition
workflow.add_conditional_edges(
    "accumulator",
    should_translate,
    {
        "wait": END,           # Stop processing and wait for next user input
        "translate": "translator" # Go to LLM
    }
)

# From translator, we end the cycle (and wait for new input)
workflow.add_edge("translator", END)

# 6. Compile
app = workflow.compile()

# --- SIMULATION ---
print("--- STARTING AGENT SIMULATION ---")

# Step 1: User signs "ME"
print("\nUser signs: ME")
current_state = app.invoke({"gloss_sequence": ["ME"]})
# Result: Buffer=['ME'], Translation=None

# Step 2: User signs "HUNGRY" (Append to previous state)
# In a real app, we manage this appending outside. 
# Here we simulate the growing list.
print("\nUser signs: HUNGRY")
current_state = app.invoke({"gloss_sequence": ["ME", "HUNGRY"]})
# Result: Buffer=['ME', 'HUNGRY'], Translation=None

# Step 3: User signs "SEND" (The Trigger!)
print("\nUser signs: SEND")
result = app.invoke({"gloss_sequence": ["ME", "HUNGRY", "SEND"]})

print(f"\n🎉 FINAL OUTPUT: {result['final_translation']}")
print(f"🧹 Buffer Status: {result['gloss_sequence']} (Should be empty)")