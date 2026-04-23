import os
import json
from groq import Groq
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from bodysync_tools import (
    webcam_capture_image,
    capture_and_extract_nutrition,
    capture_and_analyze_body,
    extract_nutrition_from_file,
    analyze_body_from_file
)

class LLMResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


parser = PydanticOutputParser(pydantic_object=LLMResponse)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; environment variables can be provided by the shell instead.
    pass

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Create a .env file with GROQ_API_KEY=... or set it in your environment."
    )

client = Groq(api_key=api_key)

# Define available tools for the agent
tools = [
    # ============================================================
    # IMAGE CAPTURE TOOL
    # ============================================================
    {
        "type": "function",
        "function": {
            "name": "webcam_capture_image",
            "description": "Opens the webcam and allows the user to capture a photo. Press SPACE to capture, Q to quit. Useful for taking photos of nutrition labels, body images, or general photography.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "File path where the captured image will be saved (e.g., 'my_photo.jpg')"
                    }
                },
                "required": []
            }
        }
    },
    # ============================================================
    # NUTRITION OCR TOOLS
    # ============================================================
    {
        "type": "function",
        "function": {
            "name": "capture_and_extract_nutrition",
            "description": "Captures a nutrition label image from the webcam and automatically extracts nutritional information using OCR and AI. Detects: calories, total fat, carbohydrates, and protein. Best for food packaging nutrition labels.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_nutrition_from_file",
            "description": "Extracts nutrition information from a nutrition label image using OCR and AI. Provide an image file path containing a nutrition label. Returns: calories, fat, carbohydrates, and protein values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the nutrition label image file (e.g., 'nutrition_label.jpg' or '/path/to/label.png')"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    # ============================================================
    # POSE ESTIMATION & BODY ANALYSIS TOOLS
    # ============================================================
    {
        "type": "function",
        "function": {
            "name": "capture_and_analyze_body",
            "description": "Captures a full-body image from the webcam and analyzes body measurements using pose estimation. Computes body ratios (shoulder width, hip width, torso length, shoulder-to-hip ratio) and estimates somatotype (ectomorph, mesomorph, or endomorph). User should stand straight and face the camera.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_body_from_file",
            "description": "Analyzes body measurements and pose from an existing image file using pose estimation. Computes normalized body ratios and estimates somatotype. Requires a clear full-body image where the person faces the camera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the full-body image file (e.g., 'body_photo.jpg' or '/path/to/body.png')"
                    }
                },
                "required": ["image_path"]
            }
        }
    }
]

# Map tool names to function implementations
available_functions = {
    "webcam_capture_image": webcam_capture_image,
    "capture_and_extract_nutrition": capture_and_extract_nutrition,
    "extract_nutrition_from_file": extract_nutrition_from_file,
    "capture_and_analyze_body": capture_and_analyze_body,
    "analyze_body_from_file": analyze_body_from_file,
}

formatInstructions = parser.get_format_instructions()

systemMessage = {"role": "system", "content": f"""You are a helpful, enthusiastic fitness and nutrition assistant. You provide personalized fitness and dietary guidance.

Your capabilities:
1. NUTRITION ANALYSIS: Use OCR tools to extract and analyze nutrition information from food labels
2. BODY ANALYSIS: Use pose estimation tools to assess body composition and estimates somatotype
3. FITNESS GUIDANCE: Provide workout recommendations, diet plans, and fitness tips

Initial Assessment Process:
- Ask the user for: name, age, gender, height, weight, current activity level, and fitness goals
- Once you have basic info, ask about: current meals, dietary restrictions, and grocery preferences
- Use your tools when the user asks to analyze nutrition labels or body measurements

Tool Usage Guidelines:
- When user asks to "take a photo", "capture an image", or "scan a label" → use webcam_capture_image or the specialized capture tools
- When user has an existing photo → ask for the file path and use the extract_nutrition_from_file or analyze_body_from_file tools
- Explain what each tool will do before using it
- After getting results, provide personalized advice based on the data

Tone: Be motivational, supportive, and practical. Avoid making definitive health claims."""}

userMessages = [
    {"role": "user", "content": "Hello! Can you confirm you are working? You're awesome!"}
]

try:
    # Conversation loop
    print("=" * 60)
    print("FITNESS & NUTRITION ASSISTANT")
    print("=" * 60)
    print("\nChat started! Type 'quit' to exit.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Thank you for using the fitness assistant. Stay healthy!")
            break
        
        # Add user message to conversation history
        userMessages.append({"role": "user", "content": user_input})
        
        # Agentic loop - keep calling until no more tool calls
        while True:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[systemMessage] + userMessages,
                tools=tools,
                temperature=0.7,
                max_tokens=2048,
            )
            
            response_message = completion.choices[0].message
            
            # Check if the LLM wants to call a tool
            if response_message.tool_calls:
                # Add assistant response to messages
                userMessages.append({
                    "role": "assistant",
                    "content": response_message.content if response_message.content else "",
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
                        try:
                            tool_result = available_functions[tool_name](**tool_args)
                            print(f"[Tool result]: Success\n")
                            
                            # Add tool result to messages
                            userMessages.append({
                                "role": "user",
                                "content": f"Tool result from '{tool_name}':\n{tool_result}"
                            })
                        except Exception as tool_error:
                            error_msg = f"Tool execution error: {str(tool_error)}"
                            print(f"[Tool error]: {error_msg}\n")
                            userMessages.append({
                                "role": "user",
                                "content": f"Error executing tool '{tool_name}': {error_msg}"
                            })
                    else:
                        error_msg = f"Tool '{tool_name}' not found in available functions"
                        print(f"[Tool error]: {error_msg}\n")
                        userMessages.append({
                            "role": "user",
                            "content": error_msg
                        })
                
                # Continue the loop to get the next response
                continue
            else:
                # No tool calls, we have the final response
                break
        
        # Extract and display final response
        raw_response = response_message.content or "(No response)"
        print(f"\nAssistant: {raw_response}\n")
        
        # Add assistant response to conversation history
        userMessages.append({"role": "assistant", "content": raw_response})

except KeyboardInterrupt:
    print("\n\nChat interrupted. Goodbye!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()