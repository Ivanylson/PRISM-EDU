<div align="center"> 

# PRISM-EDU: Mining ENADE Microdata to Diagnose Learning Deficiencies in Brazilian Higher Education

### Ivanylson Honorio Gonçalves, Victor Stroele, Regina Braga, José Maria David

### Postgraduate Program in Computer Science – Federal University of Juiz de Fora (UFJF)

### XXXVII Simpósio Brasileiro de Informática na Educação *(S B I E 2026)* - Goiânia/GO                                                                             

</div>


## Abstract PRISM-EDU

PRISM-EDU is an intelligent educational data analysis platform focused on processing, cross-referencing, and visualizing academic performance indicators (such as ENADE Microdata). The project uses AI techniques for validation and generation of strategic insights.

## Setup
This repository contains the code PRISM-EDU and data for `ENADE`. To set up `PRISM-EDU` for your workflow, please follow these steps.

## How to Run
### 1. Clone the repository
```bash
git clone [https://github.com/Ivanylson/PRISM-EDU.git](https://github.com/Ivanylson/PRISM-EDU.git)
cd PRISM-EDU
```

## 2. Install dependencies technologies used

* **Language:** Python 3.x (Standard libraries: `os`, `sys`, `pathlib`, `json`, `re`, `threading`, `subprocess`, etc.)
* **Interface & Dashboards:** `menu_principal.py` (System entry point), **Streamlit**
* **Data Processing & Math:** **Pandas**, **NumPy**, **SciPy**
* **Artificial Intelligence & ML:** **Scikit-learn**, **XGBoost**, **OpenCode** 
* **Data Visualization:** **Matplotlib**, **Seaborn**, **Plotly**
* **Document Processing:** **PyMuPDF** (`fitz`), **Docling**
* **Web & Requests:** **Requests**, `urllib`


## 3. Download the Agent

**OpenCode** is the open-source AI coding agent that powers these interactions. The simplest and most common way to install it is by using the Node.js package manager directly in the terminal. Learn more at: [https://opencode.ai/](https://opencode.ai/).

### 3.1 Prerequisites

If you do not yet have Node.js on your machine, install it so the commands will work:
1. Download the installer from the [official Node.js website](https://nodejs.org/).
2. After downloading, install it by following the standard steps (clicking "Next").
3. Ensure you have **Python 3.8+** installed on your machine to run the dashboard script.

### 3.2 Installing OpenCode

With Node.js installed, you can install OpenCode via the Terminal (CLI):

1. Open your operating system's Terminal or Command Prompt.
2. Type the following command and press `Enter`:
   ```bash
   npm install -g opencode-ai


### 3.3 Official Documentation

In the documentation, you will find a detailed step-by-step guide and other ways to use the tool: [official documentation website](https://opencode.ai/docs/).

## 4. Prompt Documentation: OpenCode Integration

This document details the instructions and texts (*prompts*) sent to the OpenCode API for analyzing the ENADE dataset. The integration uses two main prompts to establish the context and request data analysis.

---

## 1. Session Initialization Prompt

This prompt is used when connecting to the API to define the initial scope of the conversation with the model.

| Attribute | Description |
| :--- | :--- |
| **Location** | `enviar_prompt_opencode()` function (HTTP `POST` request payload) |
| **Objective** | Initiate communication with the server, creating an active session and assigning it an initial context. |
| **Type** | Static (Fixed text) |

### Prompt Content:
> `"ENADE data analysis"`

---

## 2. Main Prompt for Data Analysis and Explanation Generation

This is the main prompt, generated dynamically at runtime, responsible for instructing the Artificial Intelligence to analyze the data extracted from the file and generate the final report.

| Attribute | Description |
| :--- | :--- |
| **Location** | "TAB 6: OPENCODE + EXPLANATORY AI" section (within the `if gerar:` conditional block) |
| **Objective** | Instruct the AI ​​to act as a data analyst, evaluating metadata (rows, columns), basic mathematical statistics, and a data sample to formulate a human-readable report. |
| **Type** | Dynamic (Python *f-string* with variable injection) |

### Prompt Content (Template):

> Analyze the following ENADE educational dataset and generate an explanation in natural language. >
> **FILE:** `{caminho_csv.name}`
> **TONE:** `{tom}`
> `{'ADDITIONAL FOCUS: ' + foco if foco else ''}`
>
> **DATA SUMMARY:**
> * `{resumo['linhas']}` rows, `{len(resumo['colunas'])}` columns
> * Columns: `{', '.join(resumo['colunas'][:15])}{'...' if len(resumo['colunas']) > 15 else ''}`
> * Numeric columns: `{', '.join(resumo['colunas_numericas'][:8])}`
>
> **BASIC STATISTICS:**
> `{json.dumps(resumo['resumo_estatistico'], indent=2, ensure_ascii=False)}`
>
> **SAMPLE (first 5 rows):**
> `{amostra}`
>
> Based on this data, produce an explanation that:
> 1. Contextualizes what this data is (where it comes from, what it means)
> 2. Highlights key findings, trends, and interesting patterns
> 3. Points out anomalies or values ​​that warrant attention
> 4. Concludes with practical recommendations
> 5. Is accessible to a non-expert audience
>
> Format the response in MARKDOWN, using clear sections and accessible, explanatory language.

## 3. System Prompt — Safety Guardrails

This prompt serves as a foundational instruction for the model (specifically the local/custom **Big Pickle** model). It defines the role assumed by the AI ​​and imposes strict operational rules to prevent hallucinations and ensure technical rigor.

| Attribute | Description |
| :--- | :--- |
| **Location** | "AI Transparency: Guardrails and Prompt Engineering" section |
| **Objective** | Define the AI's persona (Instructional Design Expert) and apply strict *Guardrails* regarding theoretical grounding, strict focus on provided data, and response tone. |
| **Inference Parameters** | `temperature=0.1` (highly deterministic) / `top_p=0.9` |
| **Type** | Static |

### Prompt Content:
> You are an expert in Instructional Design and Educational Assessment.
> Your role is to strictly analyze the provided microdata.
>
> SAFETY RULES (GUARDRAILS):
> 1. GROUNDING: Do not invent pedagogical methodologies or theories that are not widely recognized (e.g., Bloom, Vygotsky).
> 2. ZERO-HALLUCINATION: Base your recommendations EXCLUSIVELY on the subjects and metrics submitted in the user prompt.
> 3. TONE: The tone must be formal, directive, and focused on actionable metrics.
> 4. RESTRICTION: If the data does not contain dropout information, do not suggest anti-dropout actions. Respond with "Insufficient data for this metric."

**Example of a User Prompt executed under these rules:**
> `"Analyze the following performance vector: {class_data}"`

---

## 4. Session Initialization Prompt

| Attribute | Description |
| :--- | :--- |
| **Location** | API request payload for the sending function |
| **Objective** | Initiate communication with the server and establish the initial conversation context. |
| **Type** | Static |

### Prompt Content:
> `"ENADE data analysis"`

---


## 5. Project Structure

The project is organized into sequential phases to ensure data integrity and analysis accuracy:

* **`fase1_pre_processamento`**: Cleaning, initial processing, and standardization of raw data.

* **`fase2_cruzamento_dados`**: Integration of different databases and relationships between tables.

* **`fase3_validacao_ia`**: Application of algorithms to verify inconsistencies and data quality.

* **`fase4_motor_ia`**: Core intelligence of the project, responsible for advanced calculations and predictions.

* **`fase5_visualizacao`**: Generation of graphs, dashboards, and user interface output.



##  6. Additional Directories

* `preprocessing/`: Contains the raw files and data dictionaries.

* `generatedfiles/`: CSV outputs and reports resulting from the analyses.

* `attempts/`: Test scripts and experimental versions.



