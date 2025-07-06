import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

hipertensao = pd.read_csv('Hypertension.csv')
diabetes = pd.read_csv('Diabetes.csv')

def estatisticas_gerais():
    # Convertendo colunas
    hipertensao["TPAN"] = hipertensao["TPAN"].astype(float)
    diabetes["BPAN"] = diabetes["BPAN"].astype(float)

    # Estatísticas descritivas para hipertensão
    desc_hiper = hipertensao["TPAN"].describe().round(2)
    print("Estatísticas Descritivas - Hipertensão (TPAN):")
    print(f"Total de Regiões: {int(desc_hiper['count'])}")
    print(f"Média de casos: {desc_hiper['mean']}")
    print(f"Desvio padrão: {desc_hiper['std']}")
    print(f"Valor mínimo: {desc_hiper['min']}")
    print(f"Mediana: {desc_hiper['50%']}")
    print(f"Valor máximo: {desc_hiper['max']}")

    # Estatísticas descritivas para diabetes
    desc_diab = diabetes["BPAN"].describe().round(2)
    print("Estatísticas Descritivas - Diabetes (BPAN):")
    print(f"Total de Regiões: {int(desc_diab['count'])}")
    print(f"Média de casos: {desc_diab['mean']}")
    print(f"Desvio padrão: {desc_diab['std']}")
    print(f"Valor mínimo: {desc_diab['min']}")
    print(f"Mediana: {desc_diab['50%']}")
    print(f"Valor máximo: {desc_diab['max']}\n\n")


def proporcao_por_sexo():
    # Convertendo colunas
    hipertensao["TPAN"] = hipertensao["TPAN"].astype(float)
    diabetes["BPAN"] = diabetes["BPAN"].astype(float)

    # Estatísticas descritivas para hipertensão
    desc_hiper = hipertensao["TPAN"].describe().round(2)
    variancia_hiper = hipertensao["TPAN"].var()
    print("Estatísticas Descritivas - Hipertensão (TPAN):")
    print(f"Total de Regiões: {int(desc_hiper['count'])}")
    print(f"Média de casos: {desc_hiper['mean']}")
    print(f"Desvio padrão: {desc_hiper['std']}")
    print(f"Variância: {variancia_hiper:.2f}")
    print(f"Valor mínimo: {desc_hiper['min']}")
    print(f"Mediana: {desc_hiper['50%']}")
    print(f"Valor máximo: {desc_hiper['max']}")

    # Estatísticas descritivas para diabetes
    desc_diab = diabetes["BPAN"].describe().round(2)
    variancia_diab = diabetes["BPAN"].var()
    print("Estatísticas Descritivas - Diabetes (BPAN):")
    print(f"Total de Regiões: {int(desc_diab['count'])}")
    print(f"Média de casos: {desc_diab['mean']}")
    print(f"Desvio padrão: {desc_diab['std']}")
    print(f"Variância: {variancia_diab:.2f}")
    print(f"Valor mínimo: {desc_diab['min']}")
    print(f"Mediana: {desc_diab['50%']}")


def graficos_distribuicao():

    # Convertendo colunas
    hipertensao["TPAN"] = hipertensao["TPAN"].astype(float)
    diabetes["BPAN"] = diabetes["BPAN"].astype(float)

    # Histogramas
    plt.hist(hipertensao["TPAN"], bins=10, alpha=0.5, label="Hipertensão")
    plt.hist(diabetes["BPAN"], bins=10, alpha=0.5, label="Diabetes")
    plt.title("Distribuição de Casos Diagnósticos")
    plt.xlabel("Nº de Casos")
    plt.ylabel("Frequência")
    plt.legend()
    plt.show()

    # Boxplot
    sns.boxplot(data=[hipertensao["TPAN"], diabetes["BPAN"]], palette="pastel")
    plt.xticks([0, 1], ["Hipertensão", "Diabetes"])
    plt.title("Boxplot das Condições")
    plt.show()



if __name__ == "__main__":
    estatisticas_gerais()
    proporcao_por_sexo()
    graficos_distribuicao()

