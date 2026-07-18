import json
import sys

def main():
    notebook_path = "notebooks/Thesis.ipynb"
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Build new cells list
    new_cells = [nb['cells'][0]]  # keep the first cell

    # Read the run_pipeline.py file
    with open('run_pipeline.py', 'r') as f:
        run_pipeline_content = f.read()

    # Create the second cell
    second_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": run_pipeline_content.splitlines(True)
    }
    new_cells.append(second_cell)

    # If there are more than two cells, append the rest
    if len(nb['cells']) > 2:
        new_cells.extend(nb['cells'][2:])

    nb['cells'] = new_cells

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    main()