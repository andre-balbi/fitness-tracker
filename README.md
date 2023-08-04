# Monitor de Exercícios

A falta de foco em programas de treinamento de força na literatura relacionada e a ausência de suporte dos rastreadores de atividade atuais 
para exercícios com pesos livres são problemas que têm sido enfrentados por muitos entusiastas do fitness e profissionais da área. Com o crescente 
interesse em tecnologias e aplicativos de monitoramento de atividades, existe um grande potencial para o desenvolvimento de soluções que possam ajudar 
a preencher essa lacuna. Além disso, o reconhecimento de atividade pode ser empregado para enfrentar desafios sociais significativos, como reabilitação, 
sustentabilidade, cuidados com idosos e saúde. Neste projeto, proponho explorar as possibilidades de aplicativos sensíveis ao contexto no campo do 
treinamento de força, utilizando dados coletados a partir de sensores de pulso (smart watch) para analisar exercícios com pesos livres. O objetivo é 
desenvolver modelos que, assim como treinadores pessoais, acompanhem os exercícios, contem repetições e identifiquem exercícios, bem como formas 
inadequadas de execução, a fim de auxiliar os usuários a atingir seus objetivos de maneira mais eficiente e segura.

#### Objetivos
* Classificar os exercícios básicos com barra
* Contar o número de repetições
* Detectar forma inadequada do movimento

#### Instalação
Criar, instalar e ativar o ambiente  Anaconda: `conda env create -f environment.yml`.

### **Etapa 01 - Preparação dos Dados**
---
Ler todos os arquivos CSV brutos separados, processa-los e mesclá-los em um único conjunto de dados.
01. Conjunto de dados em → data/raw
02. Extrair recursos do nome do arquivo
03. Ler todos os arquivos
04. Trabalhar com datetimes
05. Criar uma função personalizada
06. Mesclar conjuntos de dados
07. Resample dos dados (conversão de frequência)
08. Exportar conjunto de dados processados

Códigos originais: [`make_dataset.ipynb`](src/data/make_dataset.py)

Passo a passo explicado: [`01_make_dataset.ipynb`](notebooks/01_make_dataset.ipynb)
