# PRISM-EDU 📊🧠

O **PRISM-EDU** é uma plataforma de análise inteligente de dados educacionais, focada no processamento, cruzamento e visualização de indicadores de desempenho acadêmico (como os Microdados do ENADE). O projeto utiliza técnicas de IA para validação e geração de insights estratégicos.

## 🚀 Estrutura do Projeto

O projeto está organizado em fases sequenciais para garantir a integridade dos dados e a precisão das análises:

* **`fase1_pre_processamento`**: Limpeza, tratamento inicial e padronização dos dados brutos.
* **`fase2_cruzamento_dados`**: Integração de diferentes bases de dados e relacionamentos entre tabelas.
* **`fase3_validacao_ia`**: Aplicação de algoritmos para verificação de inconsistências e qualidade dos dados.
* **`fase4_motor_ia`**: Núcleo de inteligência do projeto, responsável pelos cálculos avançados e predições.
* **`fase5_visualizacao`**: Geração de gráficos, dashboards e interface de saída para o usuário.
* **`levantamento_relatorio_sintese`**: Módulo dedicado à compilação dos resultados finais.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Interface:** `menu_principal.py` (Ponto de entrada do sistema)
* **Processamento:** Pandas, NumPy (Sugerido)
* **Inteligência Artificial:** Scikit-learn / OpenAI API (Conforme as fases 3 e 4)

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/Ivanylson/PRISM-EDU.git](https://github.com/Ivanylson/PRISM-EDU.git)


## 📁 Diretórios Adicionais

* `preprocessamento/`: Contém os arquivos brutos e dicionários de dados.
* `arquivosgerados/`: Saídas em CSV e relatórios resultantes das análises.
* `tentativas/`: Scripts de testes e versões experimentais.

## 🔧 Como Executar
### 1. Clonar o Repositório
```bash
git clone [https://github.com/Ivanylson/PRISM-EDU.git](https://github.com/Ivanylson/PRISM-EDU.git)
cd PRISM-EDU
```

<h3>2. Configurar o Ambiente Virtual</h3>
<p>Recomenda-se o uso de um ambiente virtual para isolar as dependências:</p>
<pre># Criar o ambiente\npython -m venv venv\n\n# Ativar o ambiente (No Windows - Git Bash/Mingw):\nsource venv/Scripts/activate</pre>

<h3>3. Instalar Dependências</h3>
<p>Instale todas as bibliotecas necessárias listadas no arquivo de requisitos:</p>
<pre>pip install -r requirements.txt</pre>

<h3>4. Iniciar a Aplicação</h3>
<p>Execute o menu principal para coordenar todas as fases do projeto:</p>
<pre>python menu_principal.py</pre>

<h3>📁 Organização de Pastas</h3>
<ul>
    <li><strong>preprocessamento/</strong>: Arquivos brutos de entrada.</li>
    <li><strong>arquivosgerados/</strong>: Saídas em CSV e relatórios.</li>
    <li><strong>levantamento_relatorio_sintese/</strong>: Documentos finais consolidados.</li>
    <li><strong>tentativas/</strong>: Scripts de testes e instaladores.</li>
</ul>

<div class="footer">
    Desenvolvido por: <strong>Ivanylson</strong>
</div>

