"""Execute the entire report using this Python environment, with a local kernel."""
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
kernel = ROOT / '.jupyter' / 'kernels' / 'project2'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Project 2 (local environment)', 'language': 'python',
}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(ROOT / '.jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.jupyter' / 'runtime')
os.environ['IPYTHONDIR'] = str(ROOT / '.jupyter' / 'ipython')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.jupyter' / 'matplotlib')
os.environ['MPLBACKEND'] = 'Agg'
path = ROOT / 'actuator_load_sharing.ipynb'
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=300, kernel_name='project2', resources={'metadata': {'path': str(ROOT)}}).execute()
nbformat.validate(nb)
nbformat.write(nb, path)
count = sum(c.cell_type == 'code' for c in nb.cells)
print(f'Executed and saved all {count} code cells: {path.name}')
