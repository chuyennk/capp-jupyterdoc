To install c-app:

1. python -m venv capp-env
2. source ./capp-env/bin/activate
3. set config, data and runtime directories by setting env variables as follow:
    
    HOME_DIR=/Users/chuyen/projects/model-validation-app/c-app/capp-venv

    export JUPYTER_CONFIG_DIR=$HOME_DIR/share/jupyter/config
    export JUPYTER_DATA_DIR=$HOME_DIR/share/jupyter/data
    export JUPYTER_RUNTIME_DIR=$HOME_DIR/share/jupyter/runtime

    Note: this step to ensure any component install, i.e. nbextensions, will be install in capp-env

4. pip install -c conda-forge jupyter_contrib_nbextensions
5. pip install pandas matplotlib seaborn sklearn
6. install weasyprint:
   - install pango on mac: brew install pango libffi
   - pip install weasyprint

7. Modify nbextension/printview & init_cell for printing & auto-run all cells
    - Using printview to export pdf by modify printview/mainjs, e.g:
        // var command = 'import os; os.system(\'jupyter nbconvert ' + nbconvert_options + ' \"' + name + '\"\')';
        var command = 'from mylibs.capp_pdfexport import capp_pdfexport; capp_pdfexport(\'' + name + '\')';

        // var url = utils.splitext(name)[0] + extension;
        var url = utils.splitext(name)[0] + '.pdf';

        (see file nbextensions/printview/main.js)
    - Using init_cell to auto-run both code & markdown cell by modify:
            // if ((cell instanceof codecell.CodeCell) && cell.metadata.init_cell === true ) {
            //     cell.execute();
            //     num++;
            // }
            // run all cells instead of just the ones marked as init_cell
            if ((cell.cell_type === 'code') || (cell.cell_type === 'markdown') ) {
                cell.execute();
            }
        (see file nbextensions/init_cell/main.js for detail)

8. To ensure jupyter notebook start in correct capp-env, create a zsh file with following:
    #!/bin/zsh

    HOME_DIR=/Users/chuyen/projects/model-validation-app/c-app/capp-venv

    export JUPYTER_CONFIG_DIR=$HOME_DIR/share/jupyter/config
    export JUPYTER_DATA_DIR=$HOME_DIR/share/jupyter/data
    export JUPYTER_RUNTIME_DIR=$HOME_DIR/share/jupyter/runtime

    source $HOME_DIR/bin/activate

    jupyter notebook --no-browser
