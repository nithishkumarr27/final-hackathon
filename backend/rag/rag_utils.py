import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAI
from config import Config
import logging

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.3
        )
        self.corpus_data = None
        self.embeddings = None
        self.index = None
        self._load_corpus()
        
    def _load_corpus(self):
        """Load and index the RAG corpus"""
        try:
            corpus_path = os.path.join(os.path.dirname(__file__), 'rag_corpus.json')
            with open(corpus_path, 'r') as f:
                self.corpus_data = json.load(f)
            
            # Create embeddings for misconceptions
            texts = []
            for misconception in self.corpus_data['misconceptions']:
                text = f"{misconception['title']} {misconception['description']} {' '.join(misconception['common_symptoms'])}"
                texts.append(text)
            
            self.embeddings = self.model.encode(texts)
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"Loaded {len(texts)} misconceptions into RAG system")
            
        except Exception as e:
            logger.error(f"Failed to load RAG corpus: {e}")
            raise
    
    def retrieve_similar_misconceptions(self, query, k=3):
        """Retrieve similar misconceptions based on query"""
        query_embedding = self.model.encode([query])
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            misconception = self.corpus_data['misconceptions'][idx]
            results.append({
                'misconception': misconception,
                'similarity_score': float(scores[0][i])
            })
        
        return results
    
    def generate_explanation(self, misconception_data, user_context=""):
        """Generate explanation using Gemini 1.5 Flash"""
        prompt = f"""
        As an expert programming educator, explain this misconception clearly and provide corrective guidance.
        
        Misconception: {misconception_data['title']}
        Description: {misconception_data['description']}
        
        User Context: {user_context}
        
        Provide:
        1. Clear explanation of why this misconception occurs
        2. Specific corrective strategies
        3. A simple analogy or example
        4. Next steps for the learner
        
        Keep the explanation concise but comprehensive.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Unable to generate explanation at this time."

# Global RAG instance
rag_system = RAGSystem()