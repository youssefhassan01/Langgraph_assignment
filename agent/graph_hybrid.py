from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Any, Dict
import json
from .dspy_signatures import Router, ConstraintExtractor, SQLGenerator, AnswerSynthesizer
from .rag.retrieval import retriever
from .tools.sqlite_tool import sql_tool
import dspy

# Configure DSPy with local model
lm = dspy.OllamaLocal(model='phi3.5:3.8b-mini-instruct-q4_K_M', timeout_s=120)
dspy.configure(lm=lm)

class GraphState(TypedDict):
    question: str
    format_hint: str
    query_id: str
    query_type: Optional[str]
    retrieved_chunks: List[Dict]
    constraints: Optional[Dict]
    sql_query: Optional[str]
    sql_success: bool
    sql_result: Optional[str]
    sql_error: Optional[str]
    final_answer: Optional[Any]
    explanation: Optional[str]
    citations: List[str]
    confidence: float
    repair_count: int
    trace: List[str]

class HybridAgent:
    def __init__(self):
        self.router = Router()
        self.constraint_extractor = ConstraintExtractor()
        self.sql_generator = SQLGenerator()
        self.answer_synthesizer = AnswerSynthesizer()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):

        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("router", self.route_query)
        workflow.add_node("retriever", self.retrieve_documents)
        workflow.add_node("planner", self.extract_constraints)
        workflow.add_node("sql_generator", self.generate_sql)
        workflow.add_node("executor", self.execute_sql)
        workflow.add_node("synthesizer", self.synthesize_answer)
        
        # Define edges
        workflow.set_entry_point("router")
        
        workflow.add_edge("router", "retriever")
        workflow.add_edge("retriever", "planner")
        workflow.add_conditional_edges(
            "planner",
            self.should_generate_sql,
            {
                "sql": "sql_generator",
                "rag": "synthesizer"
            }
        )
        workflow.add_edge("sql_generator", "executor")
        workflow.add_conditional_edges(
            "executor",
            self.should_repair_sql,
            {
                "retry": "sql_generator",
                "continue": "synthesizer"
            }
        )
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def route_query(self, state: GraphState) -> GraphState:

        state["trace"] = state.get("trace", []) + ["Routing query"]
        
        try:
            prediction = self.router(question=state["question"])
            state["query_type"] = prediction.query_type.lower()
            state["trace"].append(f"Routed as: {state['query_type']}")
        except Exception as e:
            state["query_type"] = "hybrid"  # Default to hybrid on error
            state["trace"].append(f"Routing failed, defaulting to hybrid: {str(e)}")
        
        return state
    
    def retrieve_documents(self, state: GraphState) -> GraphState:

        state["trace"].append("Retrieving documents")
        
        try:
            results = retriever.retrieve(state["question"], top_k=3)
            state["retrieved_chunks"] = [
                {"id": chunk_id, "content": content, "score": score}
                for chunk_id, content, score in results
            ]
            state["trace"].append(f"Retrieved {len(results)} document chunks")
        except Exception as e:
            state["retrieved_chunks"] = []
            state["trace"].append(f"Retrieval failed: {str(e)}")
        
        return state
    
    def extract_constraints(self, state: GraphState) -> GraphState:

        state["trace"].append("Extracting constraints")
        
        context_str = "\n".join([chunk["content"] for chunk in state["retrieved_chunks"]])
        
        try:
            prediction = self.constraint_extractor(
                question=state["question"],
                retrieved_context=context_str
            )
            
            # Parse constraints JSON
            constraints = json.loads(prediction.constraints)
            state["constraints"] = constraints
            state["trace"].append(f"Extracted constraints: {constraints}")
        except Exception as e:
            state["constraints"] = {}
            state["trace"].append(f"Constraint extraction failed: {str(e)}")
        
        return state
    
    def generate_sql(self, state: GraphState) -> GraphState:

        state["trace"].append("Generating SQL")
        
        try:
            # Get relevant schema info
            relevant_tables = sql_tool.get_relevant_tables(state["question"])
            schema_info = sql_tool.get_schema_info(relevant_tables)
            
            prediction = self.sql_generator(
                question=state["question"],
                constraints=json.dumps(state.get("constraints", {})),
                schema_info=schema_info
            )
            
            state["sql_query"] = prediction.sql_query
            state["trace"].append(f"Generated SQL: {prediction.sql_query}")
        except Exception as e:
            state["sql_query"] = ""
            state["trace"].append(f"SQL generation failed: {str(e)}")
        
        return state
    
    def execute_sql(self, state: GraphState) -> GraphState:

        state["trace"].append("Executing SQL")
        
        if not state.get("sql_query"):
            state["sql_success"] = False
            state["sql_error"] = "No SQL query generated"
            return state
        
        try:
            success, df, error = sql_tool.execute_query(state["sql_query"])
            state["sql_success"] = success
            
            if success:
                state["sql_result"] = df.to_json(orient='records')
                state["sql_error"] = None
                state["trace"].append(f"SQL executed successfully, returned {len(df)} rows")
            else:
                state["sql_result"] = None
                state["sql_error"] = error
                state["trace"].append(f"SQL execution failed: {error}")
        except Exception as e:
            state["sql_success"] = False
            state["sql_result"] = None
            state["sql_error"] = str(e)
            state["trace"].append(f"SQL execution error: {str(e)}")
        
        return state
    
    def synthesize_answer(self, state: GraphState) -> GraphState:

        state["trace"].append("Synthesizing answer")
        
        # Prepare inputs
        sql_result = state.get("sql_result", "")
        rag_context = "\n".join([chunk["content"] for chunk in state["retrieved_chunks"]])
        
        try:
            prediction = self.answer_synthesizer(
                question=state["question"],
                sql_result=sql_result,
                rag_context=rag_context,
                format_hint=state["format_hint"]
            )
            
            state["final_answer"] = prediction.final_answer
            state["explanation"] = prediction.explanation
            
            # Build citations
            state["citations"] = self._build_citations(state)
            
            # Calculate confidence
            state["confidence"] = self._calculate_confidence(state)
            
            state["trace"].append("Answer synthesized successfully")
        except Exception as e:
            state["final_answer"] = None
            state["explanation"] = f"Error synthesizing answer: {str(e)}"
            state["citations"] = []
            state["confidence"] = 0.0
            state["trace"].append(f"Answer synthesis failed: {str(e)}")
        
        return state
    
    def should_generate_sql(self, state: GraphState) -> str:

        query_type = state.get("query_type", "hybrid")
        
        if query_type == "rag":
            return "rag"
        else:
            return "sql"
    
    def should_repair_sql(self, state: GraphState) -> str:

        repair_count = state.get("repair_count", 0)
        
        if not state.get("sql_success", False) and repair_count < 2:
            state["repair_count"] = repair_count + 1
            state["trace"].append(f"SQL repair attempt {repair_count + 1}")
            return "retry"
        else:
            return "continue"
    
    def _build_citations(self, state: GraphState) -> List[str]:

        citations = []
        
        # Add document citations
        for chunk in state.get("retrieved_chunks", []):
            if chunk["score"] > 0.1:  # Only cite relevant chunks
                citations.append(chunk["id"])
        
        # Add SQL table citations (simplified - in practice, parse SQL to find tables)
        if state.get("sql_query"):
            # Simple heuristic to find table names in SQL
            sql_lower = state["sql_query"].lower()
            tables = ['orders', 'order_items', 'products', 'customers', 'categories']
            for table in tables:
                if table in sql_lower:
                    citations.append(table)
        
        return list(set(citations))  # Remove duplicates
    
    def _calculate_confidence(self, state: GraphState) -> float:

        confidence = 0.5  # Base confidence
        
        # Boost for successful SQL execution
        if state.get("sql_success", False):
            confidence += 0.3
        
        # Boost for high retrieval scores
        retrieval_scores = [chunk["score"] for chunk in state.get("retrieved_chunks", [])]
        if retrieval_scores:
            avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)
            confidence += avg_retrieval_score * 0.2
        
        # Penalize for repairs
        repair_count = state.get("repair_count", 0)
        confidence -= repair_count * 0.1
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def run(self, question: str, format_hint: str, query_id: str) -> Dict:

        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "query_id": query_id,
            "query_type": None,
            "retrieved_chunks": [],
            "constraints": {},
            "sql_query": None,
            "sql_success": False,
            "sql_result": None,
            "sql_error": None,
            "final_answer": None,
            "explanation": None,
            "citations": [],
            "confidence": 0.0,
            "repair_count": 0,
            "trace": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "id": query_id,
            "final_answer": final_state["final_answer"],
            "sql": final_state.get("sql_query", ""),
            "confidence": final_state["confidence"],
            "explanation": final_state["explanation"],
            "citations": final_state["citations"]
        }