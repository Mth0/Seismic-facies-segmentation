# 🌊 Trilha 3: Segmentação de Fácies Sísmicas

### HACKATHON AI FOR OIL & GAS

#### Integrantes: **Rock the Net**

**Members:**
* Matheus do Ó
* João Nogueira
* Juan David Nieto
* Hanna Rodrigues

---

# 1. Introdução

## O Desafio

O objetivo desta trilha é **implementar um modelo de aprendizado de máquina para segmentação semântica de fácies sísmicas**.
A partir de um corte sísmico como entrada, o modelo deve gerar um **mapa de segmentação pixel a pixel** que identifica diferentes fácies geológicas.

Fácies sísmicas representam conjuntos de camadas sedimentares que se diferenciam entre si por propriedades como **amplitude**, **frequência** e **continuidade dos refletores**.
A análise dessas fácies permite inferir **litologia**, **ambiente deposicional** e características estruturais do subsuperfície.

---

## Descrição do Conjunto de Dados

### **Origem:**

  * Dados sísmicos públicos do **New Zealand Petroleum & Minerals (NZPM)**
  * Fácies interpretadas fornecidas pela **Chevron U.S.A. Inc. (CC-BY-SA-4.0)**

### **Treino**

* **18.830 cortes sísmicos (224 × 224)** gerados a partir das seções **inline** e **crossline**
* **Formato:** arrays 2D (float32), representando amplitude sísmica
* **Rótulos:** mapas de fácies correspondentes (int32), mesma resolução dos cortes


### **Teste**

* **4.700 cortes sísmicos (224 × 224)**
* **Formato:** arrays float32, sem rótulos
* **Objetivo:** avaliar a capacidade de generalização do modelo

---

## Métricas de Avaliação

As seguintes métricas serão utilizadas para avaliar o desempenho do modelo:

* **Matriz de Confusão (absoluta):** contagem de acertos e erros por classe
* **Matriz de Confusão (normalizada):** desempenho proporcional por classe
* **IoU por classe (Intersection over Union):** métrica principal para segmentação
* **Precisão:** proporção de predições corretas entre os positivos preditos
* **Recall:** capacidade de encontrar corretamente os pixels reais de cada classe
* **F1-Score:** média harmônica entre precisão e recall
* **Support:** número total de pixels reais de cada classe

---

## Objetivo Principal

* **Métrica Primária:** IoU por classe e IoU médio (mIoU)
* **Abordagem:** desenvolver um modelo de segmentação semântica capaz de identificar padrões de fácies sísmicas com robustez, equilíbrio entre classes e boa generalização para o conjunto de teste.
