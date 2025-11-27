from rank_bm25 import BM25Okapi
import os
import re

class Retriever:
    def __init__(self, docs_directory="docs"):
        self.docs_directory = docs_directory
        self.chunks = []
        self.bm25 = None
        self.load_all_documents()
    
    def chunk_markdown_file(self, file_path):
        # generate chunks for bm52
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = os.path.basename(file_path).replace('.md', '')
        
       
        heading_pattern = r'(^#+ .+?)(?=^#+ |\Z)'
        sections = re.findall(heading_pattern, content, re.MULTILINE | re.DOTALL)
        
        for i, section in enumerate(sections):
          
            lines = section.strip().split('\n')
            heading = lines[0].replace('#', '').strip()
            section_content = '\n'.join(lines[1:]).strip()
            
            if section_content:
                chunk_id = f"{filename}::chunk{i}"
                

                clean_content = self.clean_markdown(section_content)
                
                chunks.append({
                    "id": chunk_id,
                    "content": f"{heading} {clean_content}",
                    "source": filename,
                    "heading": heading,
                    "original_content": section_content
                })
        
        return chunks
    
    def clean_markdown(self, text):
        # remove markdown special characters
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'^\s*[-*]\s*', '', text, flags=re.MULTILINE)
        text = ' '.join(text.split())
        return text
    
    def load_all_documents(self):
        # load all docs in /docs
        self.chunks = []
        
        for filename in os.listdir(self.docs_directory):
            if filename.endswith('.md'):
                file_path = os.path.join(self.docs_directory, filename)
                file_chunks = self.chunk_markdown_file(file_path)
                self.chunks.extend(file_chunks)
        
        self._build_index()
    
    def _build_index(self):
        # helper tool for building bm25 index
        corpus_texts = [chunk["content"] for chunk in self.chunks]
        tokenized_corpus = [self.tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    
    def tokenize(self, text):
        # prepping texts to tokens
        return text.lower().split()
    
    def search(self, query, top_k=3):
        # get top-k results
        
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["id"],
                "content": self.chunks[idx]["content"],
                "source": self.chunks[idx]["source"],
                "score": scores[idx],
                "heading": self.chunks[idx]["heading"],
                "original_content": self.chunks[idx]["original_content"]
            })
        
        return results
