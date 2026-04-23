# LLM Agent Tool Integration Guide

## Overview

The BodySync-AI fitness assistant agent now has integrated tools for **OCR nutrition extraction** and **pose-based body analysis**. The agent can autonomously use these tools to help users with fitness and nutrition guidance.

## Available Tools

### 1. **Webcam Image Capture**
- **Function**: `webcam_capture_image(output_path)`
- **Description**: Opens webcam and captures an image
- **Usage**: Agent uses this when user asks to "take a photo" or "capture an image"
- **Example**: User → "Can you take a photo of my nutrition label?"
  - Agent → Opens webcam, user presses SPACE to capture
  - Image saved to specified path

---

### 2. **Nutrition Extraction (Webcam)**
- **Function**: `capture_and_extract_nutrition()`
- **Description**: Captures nutrition label image and extracts values using EasyOCR
- **Extracted Values**:
  - Calories (kcal)
  - Total Fat (g)
  - Carbohydrates (g)
  - Protein (g)
- **Usage**: Agent uses this for immediate nutrition label scanning
- **Example**: User → "Can you scan my food's nutrition label?"
  - Agent → Captures image, runs EasyOCR detection, returns nutrients

---

### 3. **Nutrition Extraction (From File)**
- **Function**: `extract_nutrition_from_file(image_path)`
- **Description**: Extracts nutrition info from an existing image file
- **Requires**: Path to nutrition label image file
- **Usage**: Agent uses this when user provides a previously captured image
- **Example**: User → "I have a photo at 'nutrition_label.jpg', analyze it"
  - Agent → Extracts data from file path provided

---

### 4. **Body Analysis (Webcam)**
- **Function**: `capture_and_analyze_body()`
- **Description**: Captures full-body image and analyzes using MediaPipe pose estimation
- **Analysis Results**:
  - Pose detection (detected/not detected)
  - Keypoints detected (number of body landmarks)
  - Normalized body ratios:
    - Shoulder width (normalized to image)
    - Hip width (normalized to image)
    - Torso length (normalized to image)
    - Shoulder-to-hip ratio
  - Somatotype estimate (ectomorph, mesomorph, endomorph)
- **Best Practices**:
  - Stand straight and face camera
  - Full body visible in frame
  - Good lighting
- **Usage**: Agent uses this for body composition assessment
- **Example**: User → "Can you analyze my body type?"
  - Agent → Captures full-body photo, runs pose estimation, returns analysis

---

### 5. **Body Analysis (From File)**
- **Function**: `analyze_body_from_file(image_path)`
- **Description**: Analyzes body measurements from an existing image file
- **Requires**: Path to full-body image file
- **Usage**: Agent uses this when user provides an existing body photo
- **Example**: User → "I have a body photo at 'my_photo.jpg', analyze it"
  - Agent → Runs pose estimation on provided file

---

## How the Agent Uses These Tools

### System Message Guidance
The agent is instructed to:
1. **Assess User**: Gather initial info (age, weight, activity level, fitness goals)
2. **Use Tools Strategically**: 
   - Ask before using webcam tools
   - Explain what each tool will do
   - Provide personalized advice based on results
3. **Provide Value**: Use OCR and pose data to tailor fitness/nutrition recommendations

### Typical Conversation Flow

```
User: "Hello! I need help with my fitness."

Agent: "I'd love to help! To get started, I need some basic info..."
       [Asks for: name, age, gender, height, weight, activity level, goals]

User: "I'm 30, male, 180lbs, want to build muscle."

Agent: "Great! To better understand your current diet, could I scan a nutrition label?
        Or I can analyze your body composition. Which would help more?"

User: "Yes, scan my nutrition label please!"

Agent: [Uses capture_and_extract_nutrition tool]
       "I can see this product has 250 calories, 8g fat, 35g carbs, 12g protein.
        Based on your muscle-building goal, this is a good option if consumed
        as part of a balanced diet. Let me ask about your current meals..."
```

## Tool Integration Architecture

### File Structure
```
main.py                      # LLM agent with tool definitions
├─ bodysync_tools.py        # Tool implementations
│  ├─ webcam_capture_image()
│  ├─ capture_and_extract_nutrition()
│  ├─ extract_nutrition_from_file()
│  ├─ capture_and_analyze_body()
│  └─ analyze_body_from_file()
│
├─ extract_nutrition_ocr.py  # EasyOCR-based nutrition extraction
│  ├─ extract_nutrition()    # Main extraction function
│  ├─ extract_text_pipeline()
│  └─ extract_nutrients_from_rows()
│
└─ body_analysis.py          # MediaPipe-based pose estimation
   ├─ analyze_body_from_image_bgr()
   ├─ compute_body_ratios_from_pose()
   └─ estimate_somatotype_heuristic()
```

### Tool Mapping in main.py
```python
available_functions = {
    "webcam_capture_image": webcam_capture_image,
    "capture_and_extract_nutrition": capture_and_extract_nutrition,
    "extract_nutrition_from_file": extract_nutrition_from_file,
    "capture_and_analyze_body": capture_and_analyze_body,
    "analyze_body_from_file": analyze_body_from_file,
}
```

## Using the Agent

### Setup
1. **Set GROQ_API_KEY**:
   ```bash
   # Option 1: Create .env file
   echo "GROQ_API_KEY=your_key_here" > .env
   
   # Option 2: Set environment variable
   export GROQ_API_KEY=your_key_here
   ```

2. **Run the agent**:
   ```bash
   python main.py
   ```

### Example Interactions

**Example 1: Nutrition Label Scanning**
```
You: Can you scan a food label for me?
Assistant: Of course! I'll use the webcam to capture your nutrition label.
           [Opens webcam - press SPACE to capture, Q to quit]
           [Extracts: Calories: 120, Fat: 4g, Carbs: 18g, Protein: 6g]
           This is a good light snack option. Is this something you eat regularly?
```

**Example 2: Body Analysis**
```
You: Analyze my body composition
Assistant: I'll capture a full-body photo and analyze your measurements.
           Please stand straight and face the camera directly.
           [Opens webcam - press SPACE to capture, Q to quit]
           [Analysis: Shoulder/Hip ratio: 1.12, Somatotype: Mesomorph]
           Great! Your mesomorphic body type responds well to resistance training.
           I recommend... [personalized workout suggestions]
```

**Example 3: Using Existing Photos**
```
You: I have nutrition_label.jpg, can you extract the data?
Assistant: Absolutely! Let me analyze that file for you.
           [Extracts from file]
           Here's what I found: Calories: 200, Fat: 8g, Carbs: 25g, Protein: 10g
```

## Technical Details

### OCR Technology: EasyOCR
- **Language**: English OCR
- **Processing**: Image preprocessing → Text detection → Nutrient parsing
- **Advantages**:
  - No external API required (runs locally)
  - Works offline
  - Fast inference (CPU or GPU)
  - No Tesseract dependency

### Pose Estimation: MediaPipe
- **Model**: pose_landmarker_lite (downloaded on first use)
- **Features**: 33 body landmarks with confidence scores
- **Outputs**: Normalized body ratios and somatotype heuristic
- **Advantages**:
  - Real-time performance
  - Accurate body keypoint detection
  - No special hardware required

## Error Handling

The tools include robust error handling for:
- ❌ Missing image files → Helpful error message
- ❌ Unreadable images → Clear feedback
- ❌ Failed detection → Suggestions for better photo
- ❌ Webcam unavailable → Fallback to file input
- ✓ All errors logged with context for debugging

## Limitations & Future Improvements

### Current Limitations
- Somatotype estimation uses basic heuristic (not NHANES-trained classifier)
- Body ratios are normalized to image size (not real-world measurements)
- Nutrition extraction works best with clear, well-lit labels
- Single-pose detection (optimized for one person per image)

### Future Enhancements
- [ ] ML-based somatotype classifier with NHANES data
- [ ] Real-world body measurement calibration
- [ ] Multi-person pose detection
- [ ] Meal photo recognition and calorie estimation
- [ ] Workout form analysis and correction
- [ ] Integration with fitness tracking APIs

## Testing

Run the tool integration test:
```bash
python test_tools.py
```

This verifies:
- ✓ All tools import successfully
- ✓ Tool functions are accessible
- ✓ LLM tool definitions are properly formatted
- ✓ Tool mapping is complete

## Support

For issues:
1. Check tool logs (printed to console)
2. Verify webcam permissions
3. Ensure good lighting for OCR/pose estimation
4. Check image file paths are correct
5. Verify GROQ_API_KEY is set for agent mode
