import click
import json
from agent.graph_hybrid import HybridAgent
from agent.tools.sqlite_tool import sql_tool
import time

@click.command()
@click.option('--batch', required=True, help='Input JSONL file with questions')
@click.option('--out', required=True, help='Output JSONL file for results')
def main(batch, out):
    """Run the hybrid agent on a batch of questions."""
    
    # Initialize agent
    agent = HybridAgent()
    
    # Process questions
    results = []
    with open(batch, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                
                print(f"Processing: {item['id']}")
                
                result = agent.run(
                    question=item['question'],
                    format_hint=item['format_hint'],
                    query_id=item['id']
                )
                
                results.append(result)
                
                # Write intermediate results
                with open(out, 'w') as out_file:
                    for res in results:
                        out_file.write(json.dumps(res) + '\n')
                
                print(f"Completed: {item['id']}")
    
    print(f"All done! Results written to {out}")

if __name__ == '__main__':
    main()