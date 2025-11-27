import click
import json
import sys
import os

# Use the fixed version
from agent.graph_hybrid import HybridAgent



def validate_json_line(line, line_num):
    """Validate and clean a JSON line"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        cleaned_line = ''.join(char for char in line if ord(char) >= 32 or char in '\t\n\r')
        cleaned_line = cleaned_line.strip()
        try:
            return json.loads(cleaned_line)
        except:
            return None

@click.command()
@click.option('--batch', required=True, help='Input JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file for results')
def main(batch, out):
    """Run the hybrid agent on a batch of questions."""
    
    if not os.path.exists(batch):
        print(f"Error: Input file {batch} does not exist")
        sys.exit(1)
    
    # Initialize agent
    try:
        agent = HybridAgent()
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Process questions
    results = []
    
    with open(batch, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            item = validate_json_line(line, line_num)
            if item is None:
                print(f"Skipping invalid line {line_num}")
                continue
            
            print(f"Processing: {item['id']}")
            
            try:
                result = agent.run(
                    question=item['question'],
                    format_hint=item['format_hint'],
                    query_id=item['id']
                )
                results.append(result)
                
                # Write results
                with open(out, 'w', encoding='utf-8') as out_file:
                    for res in results:
                        out_file.write(json.dumps(res) + '\n')
                
                print(f"Completed: {item['id']}")
                
            except Exception as e:
                print(f"Error processing {item['id']}: {e}")
                error_result = {
                    "id": item['id'],
                    "final_answer": None,
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": []
                }
                results.append(error_result)
    
    print(f"All done! Processed {len(results)} questions")
    print(f"Results written to {out}")

if __name__ == '__main__':
    main()