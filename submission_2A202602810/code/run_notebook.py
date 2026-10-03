"""Execute lab.ipynb from its code directory and preserve real outputs."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

if __name__ == '__main__':
    path = Path(__file__).with_name('lab.ipynb').resolve()
    nb = nbformat.read(path, as_version=4)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/lab-matplotlib')
    client = NotebookClient(nb, timeout=3600, kernel_name='python3',
                            resources={'metadata': {'path': str(path.parent)}})
    def checkpoint(cell, cell_index, **kwargs):
        nbformat.write(nb, path)
        print(f'Completed cell {cell_index}', flush=True)
    client.on_cell_executed = checkpoint
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    from evidence import refresh_notebook
    from results_table import load_results
    output_root = path.parent.parent
    if not (output_root/'results').exists():
        output_root = output_root/'submission_2A202602810'
    refresh_notebook(nb, load_results(output_root/'results'))
    nbformat.write(nb, path)
    print('Notebook completed:', path, flush=True)
