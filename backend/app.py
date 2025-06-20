# agentic_app.py
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, Tool, tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
def clean_response(text):
    # Remove all variants of code blocks
    text = text.strip()
    for marker in ["```json", "```JSON", "```"]:
        if text.startswith(marker):
            text = text[len(marker):].strip()
        if text.endswith(marker):
            text = text[:-len(marker)].strip()
    return text
# Initialize MongoDB
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["learning_correction_db"]

# Collections
learners_col = db["learners"]
misconceptions_col = db["misconceptions"]
roadmaps_col = db["roadmaps"]
progress_col = db["progress"]

# Initialize Gemini LLM
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3,
    max_output_tokens=2048
)

# Create RAG corpus
RAG_CORPUS = [
    "Misconception: Variable scope confusion. Learners often think variables declared inside loops or conditionals are accessible outside those blocks.",
    "Concept: Recursion base case. Many learners forget to define termination conditions, leading to infinite recursion.",
    "Pattern: Off-by-one errors. Common in loop conditions where indexes start at 0 instead of 1 or vice versa.",
    "Antipattern: Modifying list while iterating. Causes unexpected behavior and skipped elements.",
    "Misunderstanding: Pass-by-reference vs pass-by-value. Learners struggle with mutable vs immutable types in function arguments.",
    "Concept: Asynchronous programming. Common mistakes include not awaiting async calls and misunderstanding event loops.",
    "Pattern: Proper error handling. Learners often use broad except clauses that hide real issues.",
    "Antipattern: Deeply nested conditionals. Leads to code that's hard to read and maintain (callback hell).",
    "Misconception: Equality vs identity. Confusion between '==' and 'is' operators in Python.",
    "Concept: Database transactions. Learners often forget to commit transactions or handle rollbacks properly."
]

# Create RAG vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(RAG_CORPUS, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ============== AGENT TOOLS ============== #

@tool
def detect_misconceptions_tool(code: str, quiz_logs: str) -> list:
    """Detects potential misconceptions in learner's code and quiz logs."""
    prompt = f"""
    Analyze the following learner materials and identify potential misconceptions:
    
    CODE:
    {code}
    
    QUIZ LOGS:
    {quiz_logs}
    
    Identify 2-4 potential misconceptions. For each:
    - Tag: Short descriptive tag
    - Description: 1-2 sentence explanation
    - Confidence: 0-100% confidence
    - Location: Where it occurs
    
    Output ONLY as JSON array with keys: tag, description, confidence, location
    """
    response = gemini.invoke(prompt)
    response.content = clean_response(response.content)
    
    print(f"Detection response: {response.content}")
    try:
        return json.loads(response.content)
    except:
        return [{"tag": "Detection Error", "description": "Failed to parse output", "confidence": 0, "location": "N/A"}]

@tool
def classify_misconception_tool(candidate: dict) -> dict:
    """Classifies a misconception candidate using RAG-enhanced analysis."""
    # Ensure required fields exist with defaults
    candidate.setdefault("tag", "Unknown")
    candidate.setdefault("description", f"Potential misconception about {candidate.get('tag', 'Unknown')}")
    candidate.setdefault("confidence", 50)
    candidate.setdefault("location", "General")
    
    # Get RAG context
    rag_results = retriever.get_relevant_documents(candidate["description"])
    rag_context = "\n".join([doc.page_content for doc in rag_results])
    
    prompt = f"""
    Classify this programming misconception using the context below:
    
    MISCONCEPTION CANDIDATE:
    {json.dumps(candidate, indent=2)}
    
    RELEVANT CONTEXT:
    {rag_context}
    
    Provide:
    - Concept: Specific concept name
    - Category: Fundamental|Syntax|Logic|Pattern|Antipattern
    - Explanation: 2-3 sentence technical explanation
    - Confidence: 0-100% confidence score
    - RAG_Sources: Array of source excerpts used
    
    Output ONLY as JSON with these exact keys:
    concept, category, explanation, confidence, rag_sources
    """
    
    try:
        response = gemini.invoke(prompt)
        cleaned = clean_response(response.content)
        result = json.loads(cleaned)
        
        # Validate required fields in response
        result.setdefault("concept", candidate["tag"])
        result.setdefault("category", "Uncategorized")
        result.setdefault("explanation", "No explanation available")
        result.setdefault("confidence", candidate["confidence"])
        result.setdefault("rag_sources", [])
        
        result["original_tag"] = candidate["tag"]
        return result
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return {
            "concept": candidate["tag"],
            "category": "System Error",
            "explanation": f"Classification failed: {str(e)}",
            "confidence": 0,
            "rag_sources": [],
            "original_tag": candidate["tag"]
        }
@tool
def generate_intervention_tool(misconception: dict) -> dict:
    """Generates learning interventions for a classified misconception."""
    # Ensure required fields exist with defaults
    misconception.setdefault("concept", "Unknown concept")
    misconception.setdefault("category", "Uncategorized")
    misconception.setdefault("explanation", "No explanation available")
    misconception.setdefault("confidence", 50)
    
    prompt = f"""
    Create learning intervention for:
    {json.dumps(misconception, indent=2)}
    
    Provide:
    - Analogy: Real-world analogy
    - MicroChallenges: Array of 3 short coding challenges
    - Explanation: Technical explanation with examples
    - EstimatedTime: Minutes required
    
    Output ONLY as JSON with keys: analogy, micro_challenges, explanation, estimated_time
    """
    
    try:
        response = gemini.invoke(prompt)
        response.content = clean_response(response.content)
        result = json.loads(response.content)
        
        # Validate required fields in response
        result.setdefault("analogy", "N/A")
        result.setdefault("micro_challenges", ["Challenge generation failed"])
        result.setdefault("explanation", "No explanation generated")
        result.setdefault("estimated_time", 0)
        
        return result
        
    except Exception as e:
        print(f"Intervention generation error: {str(e)}")
        return {
            "analogy": "N/A",
            "micro_challenges": ["System error occurred"],
            "explanation": f"Failed to generate intervention: {str(e)}",
            "estimated_time": 0
        }
@tool
def adjust_roadmap_tool(learner_id: str, interventions: list) -> dict:
    """Adjusts learning roadmap based on interventions. 
    Args:
        learner_id: The ID of the learner
        interventions: List of intervention objects containing:
            - concept: The concept being addressed
            - analogy: The analogy used
            - micro_challenges: List of challenges
            - explanation: Technical explanation
            - estimated_time: Time required
    Returns:
        Dictionary containing:
            - version: The new roadmap version
            - modules: List of learning modules
    """
    current_roadmap = roadmaps_col.find_one(
        {"learner_id": learner_id}, 
        sort=[("version", -1)]
    ) or DEFAULT_ROADMAP
    
    # Extract concepts from interventions
    concepts = ", ".join([i.get("concept", "Unknown concept") for i in interventions])
    
    prompt = f"""
    Adjust roadmap for learner {learner_id} based on:
    {concepts}
    
    CURRENT ROADMAP:
    {json.dumps(current_roadmap['modules'])}
    
    Instructions:
    - Reorder modules to address misconceptions first
    - Flag adjusted modules with status: 'adjusted'
    - Keep 80% of original structure
    
    Output ONLY as JSON with keys:
    - version: Increment by 1
    - modules: Array of {{title, description, status, tags}}
    """
    
    try:
        response = gemini.invoke(prompt)
        response.content = clean_response(response.content)
        new_roadmap = json.loads(response.content)
        new_roadmap["version"] = current_roadmap.get("version", 0) + 1
        new_roadmap["learner_id"] = learner_id
        return new_roadmap
    except Exception as e:
        print(f"Roadmap adjustment error: {str(e)}")
        return current_roadmap
@tool
def track_recovery_tool(learner_id: str, roadmap: dict) -> dict:
    """Tracks confidence recovery progress for a learner.
    Args:
        learner_id: ID of the learner
        roadmap: Dictionary containing:
            - modules: List of module dictionaries with status
            - version: Roadmap version
    Returns:
        Dictionary with:
            - confidence_index: 0-100 score
            - timeline: Array of recovery events
            - flags: Array of critical issues
    """
    try:
        # Calculate completion metrics
        completed = sum(1 for m in roadmap.get("modules", []) 
                       if m.get("status") == "completed")
        adjusted = sum(1 for m in roadmap.get("modules", []) 
                      if m.get("status") == "adjusted")
        total = len(roadmap.get("modules", []))
        
        progress = int((completed / total) * 100) if total > 0 else 0
        
        # Generate recovery metrics
        prompt = f"""
        Generate recovery metrics for learner {learner_id}:
        
        Progress: {progress}% completed
        Modules: {total}
        Adjusted: {adjusted}
        Version: {roadmap.get("version", 1)}
        
        Output as JSON with these exact keys:
        - confidence_index: 0-100 score
        - timeline: Array of {{date, description, status}}
        - flags: Array of critical issues
        """
        
        response = gemini.invoke(prompt)
        cleaned = clean_response(response.content)
        result = json.loads(cleaned)
        
        # Ensure required fields
        result.setdefault("confidence_index", progress)
        result.setdefault("timeline", [])
        result.setdefault("flags", [])
        
        return result
        
    except Exception as e:
        print(f"Recovery tracking error: {str(e)}")
        return {
            "confidence_index": progress if 'progress' in locals() else 0,
            "timeline": [],
            "flags": [{"type": "system_error", "message": str(e)}]
        }
# Default roadmap
DEFAULT_ROADMAP = {
    "modules": [
        {
            "title": "Python Basics",
            "description": "Variables, data types, and basic operations",
            "status": "completed",
            "tags": ["fundamental"]
        },
        {
            "title": "Control Flow",
            "description": "Conditionals and loops",
            "status": "current",
            "tags": ["fundamental", "logic"]
        },
        {
            "title": "Functions",
            "description": "Defining and using functions",
            "status": "upcoming",
            "tags": ["fundamental"]
        },
        {
            "title": "Data Structures",
            "description": "Lists, dictionaries, and sets",
            "status": "upcoming",
            "tags": ["fundamental"]
        }
    ]
}

# ============== AGENT EXECUTORS ============== #

def create_agent_executor(tools, system_message):
    """Creates an AgentExecutor with tools and system message."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    llm_with_tools = gemini.bind_tools(tools)
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create agent executors for each stage
def create_direct_tool_executor(tool):
    """Creates an executor that directly uses a tool without agent reasoning"""
    def direct_executor(inputs):
        # Extract the actual input parameters from the agent's input format
        tool_input = inputs.get("input", {})
        if isinstance(tool_input, str):
            # If input is a string, try to parse as JSON or use as-is
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {"input": tool_input}
        
        # Call the tool directly
        result = tool.run(tool_input)
        
        # Return in the format the agent expects
        return {"output": result}
    
    return direct_executor

# Modified detection agent setup
detection_executor = create_direct_tool_executor(detect_misconceptions_tool)
detection_agent = create_agent_executor(
    tools=[detect_misconceptions_tool],
   system_message="""You are a misconception detection agent. 
    When given code and quiz logs:
    1. ALWAYS use your tool exactly once
    2. Return the tool's RAW OUTPUT without modification
    3. Do NOT summarize, interpret, or add commentary
    4. The response must be the exact JSON array from the tool"""
)

classification_agent = create_agent_executor(
    tools=[classify_misconception_tool],
    system_message="You classify misconceptions using RAG. Use your tool for each candidate."
)

intervention_agent = create_agent_executor(
    tools=[generate_intervention_tool],
    system_message="You generate learning interventions. Use your tool for each misconception."
)

roadmap_agent = create_agent_executor(
    tools=[adjust_roadmap_tool],
    system_message="You adjust learning roadmaps. Use your tool with learner ID and interventions."
)

recovery_agent = create_agent_executor(
    tools=[track_recovery_tool],
    system_message="You track confidence recovery. Use your tool with learner ID and roadmap."
)

# ============== FLASK ENDPOINTS ============== #



# @app.route('/api/detect', methods=['POST'])
# def detect_endpoint():
#     data = request.json
#     learner_id = data.get('learnerId')
#     code = data.get('code', '')
#     quiz_logs = data.get('quizLogs', '')
    
#     # Run detection agent
#     # agent_input = {
#     #     "input": f"Detect misconceptions in code and quiz logs for learner {learner_id}",
#     #     "code": code,
#     #     "quiz_logs": quiz_logs
#     # }
#     # result = detection_agent.invoke(agent_input)
#     tool_input = {
#         "code": code,
#         "quiz_logs": quiz_logs,
#         "input": f"Detect misconceptions in code: {code[:100]}..."  # Optional context
#     }
    
#     # Execute directly
#     result = detection_executor(tool_input)
#     # Store in MongoDB
#     if learner_id and "output" in result:
#         print("came here")
#         try:
#             candidates = json.loads(result["output"])
#             misconceptions_col.insert_one({
#                 "learner_id": learner_id,
#                 "type": "candidate",
#                 "candidates": candidates,
#                 "timestamp": datetime.utcnow()
#             })
#             return jsonify({"candidates": candidates})
#         except:
#             return jsonify({"error": "Detection failed"}), 500
    
#     return jsonify({"error": "Invalid agent response"}), 500
@app.route('/api/detect', methods=['POST'])
def detect_endpoint():
    data = request.json
    learner_id = data.get('learnerId')
    code = data.get('code', '')
    quiz_logs = data.get('quizLogs', '')
    
    # Prepare the correct input structure for the tool
    tool_input = {
        "code": code,
        "quiz_logs": quiz_logs  # Note: This matches the parameter name in the tool
    }
    
    try:
        # Call the tool directly with properly structured input
        raw_result = detect_misconceptions_tool.run(tool_input)
        print(f"Raw detection output: {raw_result}")
        if isinstance(raw_result, str):
            # If it's a string, clean and parse it
            cleaned_response = clean_response(raw_result)
            candidates = json.loads(cleaned_response)
        elif isinstance(raw_result, list):
            # If it's already a list, use it directly
            candidates = raw_result
        else:
            raise ValueError(f"Unexpected response type: {type(raw_result)}")
        
       
        
        # Store in MongoDB
        if learner_id:
            misconceptions_col.insert_one({
                "learner_id": learner_id,
                "type": "candidate",
                "candidates": candidates,
                "timestamp": datetime.utcnow()
            })
        
        return jsonify({"candidates": candidates})
    except json.JSONDecodeError:
        return jsonify({
            "error": "Failed to parse tool output",
            "raw_output": raw_result
        }), 500
    except Exception as e:
        return jsonify({
            "error": "Detection failed",
            "details": str(e)
        }), 500
@app.route('/api/classify', methods=['POST'])
def classify_endpoint():
    data = request.json
    learner_id = data.get('learnerId')
    candidates = data.get('candidates', [])
    
    misconceptions = []
    for candidate in candidates:
        try:
            # Ensure candidate is properly structured for the tool
            tool_input = {
                "candidate": {
                    "tag": candidate.get("tag", "Unknown"),
                    "description": candidate.get("description", ""),
                    "confidence": candidate.get("confidence", 50),
                    "location": candidate.get("location", "General")
                }
            }
            
            # Call the classification tool with properly structured input
            result = classify_misconception_tool.run(tool_input)
            
            # Parse the result if it's a string
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {
                        "concept": candidate.get("tag", "Parse Error"),
                        "category": "System",
                        "explanation": "Failed to parse classification",
                        "confidence": 0,
                        "rag_sources": []
                    }
            
            # Store in MongoDB
            if learner_id:
                misconceptions_col.insert_one({
                    "learner_id": learner_id,
                    "type": "classified",
                    "misconception": result,
                    "timestamp": datetime.utcnow()
                })
                
            misconceptions.append(result)
            
        except Exception as e:
            print(f"Classification error: {str(e)}")
            misconceptions.append({
                "concept": candidate.get("tag", "Error") if isinstance(candidate, dict) else "Error",
                "category": "System",
                "explanation": f"Classification failed: {str(e)}",
                "confidence": 0,
                "rag_sources": []
            })
    
    return jsonify({"misconceptions": misconceptions})
@app.route('/api/correct', methods=['POST'])
def correct_endpoint():
    data = request.json
    learner_id = data.get('learnerId')
    misconceptions = data.get('misconceptions', [])
    
    interventions = []
    for misconception in misconceptions:
        try:
            # Ensure we have a proper misconception dictionary
            if not isinstance(misconception, dict):
                misconception = {
                    "concept": str(misconception),
                    "category": "Uncategorized",
                    "explanation": "No explanation provided",
                    "confidence": 50
                }
            
            # Prepare the proper input structure for the tool
            tool_input = {
                "misconception": {
                    "concept": misconception.get("concept", "Unknown concept"),
                    "category": misconception.get("category", "Uncategorized"),
                    "explanation": misconception.get("explanation", "No explanation available"),
                    "confidence": misconception.get("confidence", 50),
                    "original_tag": misconception.get("original_tag", ""),
                    "rag_sources": misconception.get("rag_sources", [])
                }
            }
            
            # Call the intervention tool directly
            raw_result = generate_intervention_tool.run(tool_input)
            
            # Parse the result
            if isinstance(raw_result, str):
                intervention = json.loads(clean_response(raw_result))
            else:
                intervention = raw_result
            
            # Ensure the intervention has the required structure
            intervention.setdefault("analogy", "N/A")
            intervention.setdefault("micro_challenges", ["Challenge generation failed"])
            intervention.setdefault("explanation", "No explanation generated")
            intervention.setdefault("estimated_time", 0)
            
            interventions.append({
                "concept": misconception["concept"],
                **intervention
            })
            
            # Store in MongoDB
            if learner_id:
                misconceptions_col.update_one(
                    {"learner_id": learner_id, "misconception.concept": misconception["concept"]},
                    {"$set": {"intervention": intervention}},
                    upsert=True
                )
                
        except Exception as e:
            print(f"Intervention error for {misconception.get('concept', 'unknown')}: {str(e)}")
            interventions.append({
                "concept": misconception.get("concept", "Error"),
                "analogy": "N/A",
                "micro_challenges": ["Intervention failed"],
                "explanation": f"System error: {str(e)}",
                "estimated_time": 0
            })
    
    return jsonify({"interventions": interventions})
@app.route('/api/adjust-roadmap', methods=['POST'])
def adjust_roadmap_endpoint():
    data = request.json
    learner_id = data.get('learnerId')
    interventions = data.get('interventions', [])
    
    if not learner_id:
        return jsonify({"error": "learnerId required"}), 400
    
    try:
        # Call the tool directly with properly structured input
        result = adjust_roadmap_tool.run({
            "learner_id": learner_id,
            "interventions": interventions
        })
        
        # Parse the result if it's a string
        if isinstance(result, str):
            result = json.loads(clean_response(result))
        
        # Store in MongoDB
        roadmaps_col.insert_one({
            "learner_id": learner_id,
            "version": result.get("version", 1),
            "modules": result.get("modules", []),
            "timestamp": datetime.utcnow()
        })
        
        return jsonify({"roadmap": result})
    except Exception as e:
        print(f"Roadmap adjustment failed: {str(e)}")
        return jsonify({
            "error": "Roadmap adjustment failed",
            "details": str(e)
        }), 500
@app.route('/api/track-recovery', methods=['POST'])
def track_recovery_endpoint():
    data = request.json
    learner_id = data.get('learnerId')
    roadmap = data.get('roadmap')
    
    if not learner_id or not roadmap:
        return jsonify({"error": "learnerId and roadmap required"}), 400
    
    try:
        # Call the tool directly
        result = track_recovery_tool.run({
            "learner_id": learner_id,
            "roadmap": roadmap
        })
        
        # Parse if needed
        if isinstance(result, str):
            result = json.loads(clean_response(result))
        
        # Store in MongoDB
        progress_col.insert_one({
            "learner_id": learner_id,
            "roadmap_version": roadmap.get("version", 0),
            "confidence_index": result.get("confidence_index", 0),
            "timeline": result.get("timeline", []),
            "flags": result.get("flags", []),
            "timestamp": datetime.utcnow()
        })
        
        return jsonify(result)
    except Exception as e:
        print(f"Recovery tracking failed: {str(e)}")
        return jsonify({
            "error": "Recovery tracking failed",
            "details": str(e)
        }), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)