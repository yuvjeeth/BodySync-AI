import os
import json
from groq import Groq
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from tools import webcam_capture_image, capture_and_extract_nutrition

class LLMResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=LLMResponse)

# 1. Set your API Key
client = Groq(api_key='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')

# Define available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "webcam_capture_image",
            "description": "Opens the webcam, allows the user to take a picture, and saves it. The user must press SPACE to capture the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Path where the captured image will be saved. Default is 'captured_image.jpg'"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_and_extract_nutrition",
            "description": "Opens the webcam to capture an image of a nutrition label, then uses OCR and AI to extract nutrition information from the label. Returns parsed nutritional values like calories, fat, carbs, and protein.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Map tool names to functions
available_functions = {
    "webcam_capture_image": webcam_capture_image,
    "capture_and_extract_nutrition": capture_and_extract_nutrition
}

formatInstructions = parser.get_format_instructions()

systemMessage = {"role": "system", "content":f"""You are a helpful, enthusiastic fitness assistant. 
                 You will provide information about fitness topics, suggest diets, 
                 and suggest workout routines based on user queries. Your general tone should be welcoming and motivational. 
                 You should gather the following information from the user initially:
                 Their name, age, gender, height, weight, what their current activity level is, and what is their fitness goal.
                 Take that response and then ask them what their current meals are, and ask for dietary restrictions.
                 Suggest them home-cookable meals by taking their grocery store preference.
                 
                 You have access to a webcam tool that can take photos. Use it when the user asks you to click a photo or take a picture."""}

userMessages = [
            {"role": "user", "content": "Hello! Can you confirm you are working? You're awesome!"}
        ]

try:
    # Conversation loop
    print("Chat started! Type 'quit' to exit.\n")
    while True:
        # Get user input
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Exiting chat...")
            break
        
        # Add user message to conversation history
        userMessages.append({"role": "user", "content": user_input})
        
        # Agentic loop - keep calling until no more tool calls
        while True:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[systemMessage] + userMessages,
                tools=tools,
                temperature=0.5,
                max_tokens=1024,
            )
            
            response_message = completion.choices[0].message
            
            # Check if the LLM wants to call a tool
            if response_message.tool_calls:
                # Add assistant response to messages
                userMessages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }
                        for tool_call in response_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"\n[Calling tool: {tool_name}]")
                    
                    # Execute the tool
                    if tool_name in available_functions:
                        tool_result = available_functions[tool_name](**tool_args)
                        
                        # Add tool result to messages
                        userMessages.append({
                            "role": "user",
                            "content": f"Tool '{tool_name}' result: {tool_result}"
                        })
                    else:
                        userMessages.append({
                            "role": "user",
                            "content": f"Error: Tool '{tool_name}' not found"
                        })
                
                # Continue the loop to get the next response
                continue
            else:
                # No tool calls, we have the final response
                break
        
        # Extract and display final response
        raw_response = response_message.content
        print(f"\nAssistant: {raw_response}\n")
        
        # Add assistant response to conversation history
        userMessages.append({"role": "assistant", "content": raw_response})

except Exception as e:
    print(f"Error: {e}")