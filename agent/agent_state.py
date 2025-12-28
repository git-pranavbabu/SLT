from typing import List, TypedDict

# This defines the structure of our Graph's memory
class AgentState(TypedDict):
    # The raw words coming from the LSTM
    # We use 'operator.add' logic usually, but for now just a list
    gloss_sequence: List[str] 
    
    # The final English output
    final_translation: str