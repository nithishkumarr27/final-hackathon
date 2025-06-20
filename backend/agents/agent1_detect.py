"""
Misconception Detection Agent
Analyzes learner interactions to detect recurring errors and conceptual misunderstandings
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MisconceptionCandidate:
    misconception_type: str
    topic: str
    confidence_score: float
    error_pattern: str
    suggested_correction: str
    frequency: int
    learning_impact: str

class MisconceptionDetectionAgent:
    def __init__(self):
        # Initialize Gemini 1.5 Flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define tools for misconception detection
        self.tools = [
            Tool(
                name="code_pattern_analyzer",
                description="Analyzes code submissions for syntax and logic patterns",
                func=self._analyze_code_patterns
            ),
            Tool(
                name="quiz_error_detector",
                description="Detects recurring errors in quiz responses",
                func=self._analyze_quiz_errors
            ),
            Tool(
                name="behavior_pattern_tracker",
                description="Tracks learning behavior patterns for misconception indicators",
                func=self._analyze_behavior_patterns
            ),
            Tool(
                name="misconception_classifier",
                description="Classifies detected errors into misconception categories",
                func=self._classify_misconceptions
            )
        ]
        
        # Create agent prompt
        self.prompt = PromptTemplate(
            input_variables=["input", "chat_history", "agent_scratchpad"],
            template="""
You are an expert Misconception Detection Agent. Your role is to analyze learner interactions and identify conceptual misunderstandings.

ANALYSIS FRAMEWORK:
1. Examine code submissions for recurring syntax/logic errors
2. Analyze quiz responses for conceptual gaps
3. Track behavioral patterns indicating confusion
4. Classify misconceptions with confidence scores

DETECTION CRITERIA:
- Minimum 85% confidence threshold
- Focus on both syntax and logic-level errors
- Detect patterns within 2 consecutive errors
- Associate with specific learning topics

Previous conversation:
{chat_history}

Available tools: {tools}
Tool names: {tool_names}

Current analysis task: {input}

Thought: I need to analyze the learner data to detect misconceptions systematically.
{agent_scratchpad}
"""
        )
        
        # Create agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _analyze_code_patterns(self, code_data: str) -> str:
        """Analyze code submissions for error patterns"""
        try:
            data = json.loads(code_data) if isinstance(code_data, str) else code_data
            patterns = []
            
            for submission in data.get('code_submissions', []):
                code = submission.get('code', '')
                errors = submission.get('errors', [])
                
                # Common misconception patterns
                misconception_patterns = {
                    'off_by_one': {
                        'regex': r'for\s+\w+\s+in\s+range\([^)]*\):|while\s+\w+\s*<=?\s*len\(',
                        'topic': 'loops_iteration',
                        'description': 'Off-by-one error in loop boundaries'
                    },
                    'scope_confusion': {
                        'regex': r'return\s+\w+.*\n.*return\s+\w+|def\s+\w+.*\n.*\w+\s*=',
                        'topic': 'function_scope',
                        'description': 'Variable scope misunderstanding'
                    },
                    'indentation_logic': {
                        'regex': r'if\s+.*:\s*\n\s*.*\n\w+',
                        'topic': 'conditional_logic',
                        'description': 'Indentation affecting logic flow'
                    },
                    'type_confusion': {
                        'regex': r'int\(.*\)\s*\+\s*["\']|str\(.*\)\s*\*\s*\d+',
                        'topic': 'data_types',
                        'description': 'Type conversion misconception'
                    }
                }
                
                for pattern_name, pattern_info in misconception_patterns.items():
                    if re.search(pattern_info['regex'], code, re.IGNORECASE):
                        patterns.append({
                            'pattern': pattern_name,
                            'topic': pattern_info['topic'],
                            'description': pattern_info['description'],
                            'code_snippet': code[:100] + '...' if len(code) > 100 else code
                        })
            
            return json.dumps({
                'detected_patterns': patterns,
                'pattern_count': len(patterns),
                'analysis_status': 'completed'
            })
            
        except Exception as e:
            return json.dumps({'error': f'Code analysis failed: {str(e)}'})
    
    def _analyze_quiz_errors(self, quiz_data: str) -> str:
        """Analyze quiz responses for recurring errors"""
        try:
            data = json.loads(quiz_data) if isinstance(quiz_data, str) else quiz_data
            error_patterns = {}
            
            for quiz in data.get('quiz_logs', []):
                topic = quiz.get('topic', 'unknown')
                responses = quiz.get('responses', [])
                
                for response in responses:
                    if not response.get('correct', True):
                        error_type = response.get('error_type', 'conceptual')
                        
                        if topic not in error_patterns:
                            error_patterns[topic] = {}
                        
                        if error_type not in error_patterns[topic]:
                            error_patterns[topic][error_type] = 0
                        
                        error_patterns[topic][error_type] += 1
            
            # Identify recurring patterns (frequency >= 2)
            recurring_errors = []
            for topic, errors in error_patterns.items():
                for error_type, frequency in errors.items():
                    if frequency >= 2:
                        recurring_errors.append({
                            'topic': topic,
                            'error_type': error_type,
                            'frequency': frequency,
                            'misconception_likelihood': min(frequency * 0.3, 0.95)
                        })
            
            return json.dumps({
                'recurring_errors': recurring_errors,
                'total_topics_affected': len(error_patterns),
                'analysis_status': 'completed'
            })
            
        except Exception as e:
            return json.dumps({'error': f'Quiz analysis failed: {str(e)}'})
    
    def _analyze_behavior_patterns(self, behavior_data: str) -> str:
        """Analyze learning behavior for misconception indicators"""
        try:
            data = json.loads(behavior_data) if isinstance(behavior_data, str) else behavior_data
            behavioral_indicators = []
            
            patterns = data.get('learning_patterns', [])
            
            for pattern in patterns:
                # Indicators of potential misconceptions
                time_on_task = pattern.get('time_spent', 0)
                attempts = pattern.get('attempts', 0)
                help_requests = pattern.get('help_requests', 0)
                topic = pattern.get('topic', 'unknown')
                
                # Calculate misconception probability based on behavioral signals
                misconception_score = 0.0
                
                if attempts > 3:
                    misconception_score += 0.3
                if time_on_task > 1800:  # More than 30 minutes
                    misconception_score += 0.2
                if help_requests > 2:
                    misconception_score += 0.3
                
                if misconception_score >= 0.5:
                    behavioral_indicators.append({
                        'topic': topic,
                        'misconception_probability': min(misconception_score, 0.95),
                        'indicators': {
                            'excessive_attempts': attempts > 3,
                            'prolonged_struggle': time_on_task > 1800,
                            'frequent_help_seeking': help_requests > 2
                        }
                    })
            
            return json.dumps({
                'behavioral_indicators': behavioral_indicators,
                'high_risk_topics': [bi['topic'] for bi in behavioral_indicators if bi['misconception_probability'] > 0.7],
                'analysis_status': 'completed'
            })
            
        except Exception as e:
            return json.dumps({'error': f'Behavior analysis failed: {str(e)}'})
    
    def _classify_misconceptions(self, analysis_results: str) -> str:
        """Classify detected patterns into misconception categories"""
        try:
            # Use Gemini to classify and provide detailed analysis
            classification_prompt = f"""
            Based on the following analysis results, classify the detected patterns into specific misconception categories:
            
            {analysis_results}
            
            Provide a structured classification with:
            1. Misconception type (syntax, logic, conceptual, procedural)
            2. Confidence score (0.0-1.0)
            3. Topic association
            4. Suggested correction approach
            5. Learning impact assessment
            
            Format as JSON with detailed explanations.
            """
            
            response = self.llm.invoke(classification_prompt)
            
            return json.dumps({
                'classification_result': response.content,
                'analysis_status': 'completed',
                'classification_timestamp': 'current'
            })
            
        except Exception as e:
            return json.dumps({'error': f'Classification failed: {str(e)}'})
    
    def detect_misconceptions(self, learner_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main method to detect misconceptions from learner data"""
        try:
            # Prepare input for agent
            agent_input = f"""
            Analyze the following learner data for misconceptions:
            
            Code Submissions: {json.dumps(learner_data.get('code_submissions', []))}
            Quiz Logs: {json.dumps(learner_data.get('quiz_logs', []))}
            Learning Patterns: {json.dumps(learner_data.get('learning_patterns', []))}
            
            Detect and classify all misconceptions with confidence >= 0.85.
            Provide detailed analysis and correction recommendations.
            """
            
            # Execute agent
            result = self.agent_executor.invoke({
                "input": agent_input
            })
            
            # Process and structure the results
            misconceptions = self._process_agent_output(result['output'])
            
            return {
                'success': True,
                'misconceptions': misconceptions,
                'detection_confidence': self._calculate_overall_confidence(misconceptions),
                'topics_analyzed': list(set([m.get('topic', 'unknown') for m in misconceptions])),
                'agent_reasoning': result['output']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Misconception detection failed: {str(e)}',
                'misconceptions': [],
                'detection_confidence': 0.0
            }
    
    def _process_agent_output(self, agent_output: str) -> List[Dict[str, Any]]:
        """Process agent output into structured misconceptions"""
        misconceptions = []
        
        # Extract misconception patterns from agent output
        # This is a simplified version - in production, you'd have more sophisticated parsing
        try:
            # Look for JSON-like structures in the output
            import re
            json_matches = re.findall(r'\{[^}]+\}', agent_output)
            
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if 'topic' in data or 'misconception' in data:
                        misconceptions.append(data)
                except:
                    continue
            
            # If no structured data found, create basic misconceptions from text analysis
            if not misconceptions:
                lines = agent_output.split('\n')
                for line in lines:
                    if 'misconception' in line.lower() or 'error' in line.lower():
                        misconceptions.append({
                            'misconception_type': 'detected_pattern',
                            'topic': 'general',
                            'confidence_score': 0.85,
                            'description': line.strip(),
                            'suggested_correction': 'Review fundamental concepts'
                        })
            
        except Exception as e:
            # Fallback misconceptions
            misconceptions = [{
                'misconception_type': 'analysis_completed',
                'topic': 'general',
                'confidence_score': 0.85,
                'description': 'Analysis completed with agent processing',
                'suggested_correction': 'Review detected patterns'
            }]
        
        return misconceptions
    
    def _calculate_overall_confidence(self, misconceptions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        if not misconceptions:
            return 0.0
        
        total_confidence = sum(m.get('confidence_score', 0.0) for m in misconceptions)
        return total_confidence / len(misconceptions)

# Test function
def test_agent():
    """Test the misconception detection agent"""
    agent = MisconceptionDetectionAgent()
    
    # Test data
    test_data = {
        'code_submissions': [
            {
                'code': 'for i in range(len(arr)):\n    print(arr[i+1])',
                'errors': ['IndexError: list index out of range'],
                'topic': 'loops'
            }
        ],
        'quiz_logs': [
            {
                'topic': 'loops',
                'responses': [
                    {'correct': False, 'error_type': 'off_by_one'},
                    {'correct': False, 'error_type': 'off_by_one'}
                ]
            }
        ],
        'learning_patterns': [
            {
                'topic': 'loops',
                'attempts': 5,
                'time_spent': 2400,
                'help_requests': 3
            }
        ]
    }
    
    result = agent.detect_misconceptions(test_data)
    return result

if __name__ == "__main__":
    # Test the agent
    result = test_agent()
    print(json.dumps(result, indent=2))