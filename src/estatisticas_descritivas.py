import pandas as pd

hipertensao = pd.read_csv('../Data/Hypertension.csv')
diabetes = pd.read_csv('../Data/Diabetes.csv')

def estatisticas_descritivas():
    print("Estatísticas descritivas - Hipertensão (TPAN):")
    print("\nEstatísticas descritivas - Diabetes (BPAN):")


def proporcao_por_sexo():

    print("Hipertensão:")
    print(f"Homens: ")
    print(f"Mulheres: ")

    print("\nDiabetes:")
    print(f"Homens: ")
    print(f"Mulheres: ")


if __name__ == "__main__":
    estatisticas_descritivas()
    proporcao_por_sexo()

