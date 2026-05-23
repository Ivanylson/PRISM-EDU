# PRISM-EDU

PRISM-EDU is an intelligent educational data analysis platform focused on processing, cross-referencing, and visualizing academic performance indicators (such as ENADE Microdata). The project uses AI techniques for validation and generation of strategic insights.

## Project Structure

The project is organized into sequential phases to ensure data integrity and analysis accuracy:

* **`fase1_pre_processamento`**: Cleaning, initial processing, and standardization of raw data.

* **`fase2_cruzamento_dados`**: Integration of different databases and relationships between tables.

* **`fase3_validacao_ia`**: Application of algorithms to verify inconsistencies and data quality.

* **`fase4_motor_ia`**: Core intelligence of the project, responsible for advanced calculations and predictions.

* **`fase5_visualizacao`**: Generation of graphs, dashboards, and user interface output.

## Technologies Used

* **Language:** Python 3.x (Standard libraries: `os`, `sys`, `pathlib`, `json`, `re`, `threading`, `subprocess`, etc.)
* **Interface & Dashboards:** `menu_principal.py` (System entry point), **Streamlit**
* **Data Processing & Math:** **Pandas**, **NumPy**, **SciPy**
* **Artificial Intelligence & ML:** **Scikit-learn**, **XGBoost**, **OpenCode** 
* **Data Visualization:** **Matplotlib**, **Seaborn**, **Plotly**
* **Document Processing:** **PyMuPDF** (`fitz`), **Docling**
* **Web & Requests:** **Requests**, `urllib`

##  Additional Directories

* `preprocessing/`: Contains the raw files and data dictionaries.

* `generatedfiles/`: CSV outputs and reports resulting from the analyses.

* `attempts/`: Test scripts and experimental versions.

## How to Run
### Clone the Repository
```bash
git clone [https://github.com/Ivanylson/PRISM-EDU.git](https://github.com/Ivanylson/PRISM-EDU.git)
cd PRISM-EDU
```

