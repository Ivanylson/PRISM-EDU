# PRISM-EDU 📊🧠

PRISM-EDU is an intelligent educational data analysis platform focused on processing, cross-referencing, and visualizing academic performance indicators (such as ENADE Microdata). The project uses AI techniques for validation and generation of strategic insights.

## 🚀 Project Structure

The project is organized into sequential phases to ensure data integrity and analysis accuracy:

* **`fase1_pre_processamento`**: Cleaning, initial processing, and standardization of raw data.

* **`fase2_cruzamento_dados`**: Integration of different databases and relationships between tables.

* **`fase3_validacao_ia`**: Application of algorithms to verify inconsistencies and data quality.

* **`fase4_motor_ia`**: Core intelligence of the project, responsible for advanced calculations and predictions.

* **`fase5_visualizacao`**: Generation of graphs, dashboards, and user interface output.

## 🛠️ Technologies Used

* **Language:** Python 3.x
* **Interface:** `main_menu.py` (System entry point)
* **Processing:** Pandas, NumPy (Suggested)
* **Artificial Intelligence:** Scikit-learn / OpenAI API (As per phases 3 and 4)

## 📁 Additional Directories

* `preprocessing/`: Contains the raw files and data dictionaries.

* `generatedfiles/`: CSV outputs and reports resulting from the analyses.

* `attempts/`: Test scripts and experimental versions.

## 🔧 How to Run
### 1. Clone the Repository
```bash
git clone [https://github.com/Ivanylson/PRISM-EDU.git](https://github.com/Ivanylson/PRISM-EDU.git)
cd PRISM-EDU
```

<h3>2. Configure the Virtual Environment</h3>
<p>Using a virtual environment is recommended to isolate dependencies:</p>
<pre># Create the environment\npython -m venv venv\n\n# Activate the environment (On Windows - Git Bash/Mingw):\nsource venv/Scripts/activate</pre>

<h3>3. Install Dependencies</h3>
<p>Install all necessary libraries listed in the requirements file:</p>
<pre>pip install -r requirements.txt</pre>

<h3>4. Start the Application</h3>
<p>Run the main menu to coordinate all phases of the project:</p>
<pre>python menu_principal.py</pre>

<h3>📁 Folder Organization</h3>
<ul>
<li><strong>preprocessing/</strong>: Raw input files.</li>
<li><strong>generated_files/</strong>: CSV outputs and reports.</li>
<li><strong>survey_report_summary/</strong>: Consolidated final documents.</li>
<li><strong>attempts/</strong>: Test scripts and installers.</li>
</ul>

<div class="footer">
Ivanylson Honorio Gonçalves, Regina Braga, José Maria David, Victor Stroele

Postgraduate Program in Computer Science – Federal University of Juiz de Fora (UFJF)
Caixa Postal 20.010 – 36.016-970 – Juiz de Fora – MG - Brazil
ivanylson.honorio@estudante.ufjf.br,regina.braga@ufjf.br, jose.david@ufjf.br, victor.stroele@ufjf.br

Abstract. ENADE microdata are underutilized, and the aggregate score does not indicate areas requiring intervention. PRISM-EDU (Prescriptive Insight System for Mining Educational Data) analyzes, item-by-item, responses from each higher education institution using LCA to identify student profiles, which are validated via bootstrap. The profiles are cross-referenced against INEP’s official item-object-of-knowledge, and XGBoost classifies deficiencies by segment. When applied to ENADE 2023 (230,088 students, 1,880 HEIs, 28 programs), it reveals profiles with complementary areas of knowledge that align with national archetypes and highlights chronically deficient areas of knowledge, offering an actionable diagnosis that the aggregate score misses. 
</div>
