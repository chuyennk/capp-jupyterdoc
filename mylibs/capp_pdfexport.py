import os
import weasyprint

import re
import nbformat
from pathlib import Path
from nbconvert.preprocessors import Preprocessor, ExecutePreprocessor
from nbconvert.exporters import HTMLExporter
from mylibs.pre_pymarkdown import PyMarkdownPreprocessor

NOTEBOOK_VERSION=4

def load(path:Path) -> nbformat.NotebookNode:
    try:
        return nbformat.read(str(path),as_version=NOTEBOOK_VERSION)
    except  Exception as e:
        raise Exception(f"Error loading notebook from location {path}: \n {e}")

def save(notebook: nbformat.NotebookNode, path:Path)->None:
    nbformat.write(notebook,str(path), version=NOTEBOOK_VERSION)

def trust(notebook: nbformat.NotebookNode)->None:
    nbformat.sign.NotebookNotary().sign(nb=notebook)

def untrust(notebook: nbformat.NotebookNode)->None:
    nbformat.sign.NotebookNotary().unsign(nb=notebook)

def execute(notebook: nbformat.NotebookNode, notebook_path:Path)->None:
    try:
        ep = ExecutePreprocessor(timeout=20000, kernel_name="base")
        ep.preprocess(notebook, {})
        trust(notebook)
        save(notebook, notebook_path)
    except Exception as e:
        raise Exception(f"Error executing notebook: \n {e}")

def html_export(notebook):
    exporter = HTMLExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.register_preprocessor(PyMarkdownPreprocessor, enabled=True)
    try: 
#         Do not execute ]S cells when exporting PDF content
#         for c in notebook.cells:
#             if "%%javascript" in c.source: 
#                 c.outputs = []
        html_content, resources = exporter.from_notebook_node(notebook)
        
        return html_content, resources

    except Exception: 
        raise Exception("nbconvert could not export the notebook to html")

def capp_pdfexport(notebook_path:str)->None:
    notebook_path = Path(notebook_path)
    pdf_path = notebook_path.with_suffix(".pdf")
    notebook = load(notebook_path)
    html_content, resources = html_export(notebook)
    try:
        weasyprint.HTML(string=html_content).write_pdf(str(pdf_path))
    except Exception as e:
        raise Exception(f"Error exporting PDF: \n {e}")