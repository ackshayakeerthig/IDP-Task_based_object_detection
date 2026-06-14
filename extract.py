import json
import sys

def extract_notebook(path, out_path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(out_path, 'w', encoding='utf-8') as out:
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                out.write('"""\n')
                out.write(''.join(cell['source']))
                out.write('\n"""\n\n')
            elif cell['cell_type'] == 'code':
                out.write(''.join(cell['source']))
                out.write('\n\n')

if __name__ == '__main__':
    extract_notebook(sys.argv[1], sys.argv[2])
