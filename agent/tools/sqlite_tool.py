import sqlite3
import pandas as pd
from typing import List, Optional, Tuple

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):

        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
    
    def close(self):

        if self.connection:
            self.connection.close()
    
    def get_schema_info(self, table_names: List[str] = None) -> str:

        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        
        if table_names is None:
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            table_names = [row[0] for row in cursor.fetchall()]
        
        schema_info = []
        for table_name in table_names:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_info = [f"{col[1]} ({col[2]})" for col in columns]
            schema_info.append(f"Table: {table_name}\nColumns: {', '.join(column_info)}")
        
        return "\n\n".join(schema_info)
    
    def execute_query(self, query: str) -> Tuple[bool, Optional[pd.DataFrame], str]:

        if not self.connection:
            self.connect()
        
        try:
            df = pd.read_sql_query(query, self.connection)
            return True, df, "Success"
        except Exception as e:
            return False, None, str(e)
    
    def get_relevant_tables(self, question: str) -> List[str]:

        question_lower = question.lower()
        relevant_tables = []
        
        table_keywords = {
            'orders': ['order', 'purchase', 'buy'],
            'order_items': ['item', 'detail', 'quantity', 'unitprice', 'discount'],
            'products': ['product', 'item', 'category', 'price'],
            'customers': ['customer', 'client', 'company'],
            'categories': ['category', 'beverage', 'condiment']
        }
        
        for table, keywords in table_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                relevant_tables.append(table)
        
        return relevant_tables if relevant_tables else ['orders', 'order_items', 'products', 'customers']

# Global SQL tool instance
sql_tool = SQLiteTool()