import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from typing import List, Tuple

class SimpleRetriever:
    def __init__(self, docs_directory: str = "docs"):
        self.docs_directory = docs_directory
        self.chunks = []
        self.chunk_ids = []
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = None
        
    def load_and_chunk_documents(self):

        chunks = []
        chunk_ids = []
        
        for filename in os.listdir(self.docs_directory):
            if filename.endswith('.md'):
                filepath = os.path.join(self.docs_directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple chunking by sections (lines starting with #)
                sections = re.split(r'\n#+ ', content)
                for i, section in enumerate(sections):
                    if section.strip():
                        # Clean up the section
                        section = section.strip()
                        lines = section.split('\n')
                        if lines:
                            title = lines[0].replace('#', '').strip()
                            content_text = '\n'.join(lines[1:]).strip()
                            
                            chunk_text = f"{title}\n{content_text}" if content_text else title
                            chunks.append(chunk_text)
                            chunk_ids.append(f"{filename}::chunk{i}")
        
        self.chunks = chunks
        self.chunk_ids = chunk_ids
        return chunks, chunk_ids
    
    def build_index(self):

        if not self.chunks:
            self.load_and_chunk_documents()
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        return self.tfidf_matrix
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:

        if self.tfidf_matrix is None:
            self.build_index()
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((
                    self.chunk_ids[idx],
                    self.chunks[idx],
                    float(similarities[idx])
                ))
        
        return results

# Global retriever instance
retriever = SimpleRetriever()