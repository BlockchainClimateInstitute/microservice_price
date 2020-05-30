Microservice Price
==============================

MicroService to provide price estimation for real estate investments in the UK. The main objective of 
this service is to ingest information relevant to an AVM (Automatic Valuation Model) which can then 
provide a price estimate for a property. This price could be provided at present time, in the past and 
potentially in the future (projections). 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


How to Start Working: virtualenv, requirements and Notebook
------------

First, clone this repo and cd in this folder. You should be using Python 3.X. Create a virtual environment typing:

```shell
virtualenv venv_price
```

In Linux/Mac activate the virtualenv by typing:

```shell
source venv_price/bin/activate
```

Install all requirements:

```shell
pip instal -r requirements.txt
```

Make the virtualenv available to jupyter notebook:

```shell
python -m ipykernel install --name=venv_price
```

You should now be able to find the "venv_price" kernel in your list of available kernels in jupyter notebook.
Create a notebook in the corresponding folder (/notebooks) and start working on your specific task. You should
first develop/demonstrate your work in the notebook and then we will migrate code to the source folder (/src).
The test/sample data we will use for development/debug will be located in the data folder.


Code Changes/Review Process
------------

This repository has 3 core branches:
- master
- uat
- develop (default)

Make sure to commit code to 'develop' branch only (default). We will review code and
migrate as appropriate to "User Acceptance Test" (uat) and "Production" branches
(master) in a progressive fashion (once tests by multiple stakeholders is complete).


Run the FlaskApp
------------

You can test the sample FlaskRest API by typing in your command line:

```shell
python src/app.py
```

You can check the sample API by opening your browser and going to: http://127.0.0.1:5000/


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
