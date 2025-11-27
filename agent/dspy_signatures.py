import dspy
from typing import Literal

class RouteQuery(dspy.Signature):
    # router classifier
    question: str = dspy.InputField(desc="User question about retail analytics")
    query_type: Literal["rag", "sql", "hybrid"] = dspy.OutputField(desc="Type of query: rag-only, sql-only, or hybrid")

class ExtractConstraints(dspy.Signature):
    # checks required constarints based on query
    question: str = dspy.InputField(desc="User question")
    retrieved_context: str = dspy.InputField(desc="Relevant document context")
    constraints: str = dspy.OutputField(desc="JSON string with date_ranges, categories, kpis, etc.")

class GenerateSQL(dspy.Signature):
    # generates sql based on query and schema
    question: str = dspy.InputField(desc="User question requiring SQL")
    constraints: str = dspy.InputField(desc="Extracted constraints in JSON")
    schema_info: str = dspy.InputField(desc="Relevant database schema information")
    sql_query: str = dspy.OutputField(desc="SQLite-compatible SQL query")

class SynthesizeAnswer(dspy.Signature):
    # 
    question: str = dspy.InputField(desc="Original user question")
    sql_result: str = dspy.InputField(desc="Result from SQL execution if any")
    rag_context: str = dspy.InputField(desc="Relevant document context")
    format_hint: str = dspy.InputField(desc="Expected output format")
    final_answer: str = dspy.OutputField(desc="Formatted final answer matching format_hint")
    explanation: str = dspy.OutputField(desc="Brief explanation of the answer")

class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classifier = dspy.Predict(RouteQuery)
    
    def forward(self, question):
        return self.classifier(question=question)

class ConstraintExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(ExtractConstraints)
    
    def forward(self, question, retrieved_context):
        return self.extractor(question=question, retrieved_context=retrieved_context)

class SQLGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(GenerateSQL)
    
    def forward(self, question, constraints, schema_info):
        return self.generator(question=question, constraints=constraints, schema_info=schema_info)

class AnswerSynthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesizer = dspy.Predict(SynthesizeAnswer)
    
    def forward(self, question, sql_result, rag_context, format_hint):
        return self.synthesizer(
            question=question,
            sql_result=sql_result,
            rag_context=rag_context,
            format_hint=format_hint
        )