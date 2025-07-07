import pandas as pd


def calcular_probabilidades():
    """
    Calcula as probabilidades marginais e condicionais
    usando a fórmula teórica P(A|B) = P(A∩B)/P(B)
    """
    # Carregar os dados
    hipertensao = pd.read_csv('Hypertension.csv')
    diabetes = pd.read_csv('Diabetes.csv')

    # Totais gerais
    total_populacao = hipertensao['TPAD'].sum()
    total_mulheres = hipertensao['TWAD'].sum()
    total_homens = hipertensao['TMAD'].sum()

    # Verificação de consistência
    assert abs((total_mulheres + total_homens) - total_populacao) < 1, "Dados inconsistentes: soma dos sexos != total"

    # Cálculo das probabilidades
    resultados = {
        # Probabilidades marginais
        'Prevalência de Hipertensão': hipertensao['TPAN'].sum() / total_populacao,
        'Prevalência de Diabetes': diabetes['BPAN'].sum() / total_populacao,

        # Probabilidades condicionais
        'Hipertensão entre Mulheres': (hipertensao['TWAN'].sum() / total_populacao) / (
                    total_mulheres / total_populacao),
        'Hipertensão entre Homens': (hipertensao['TMAN'].sum() / total_populacao) / (total_homens / total_populacao),
        'Diabetes entre Mulheres': (diabetes['BWAN'].sum() / total_populacao) / (total_mulheres / total_populacao),
        'Diabetes entre Homens': (diabetes['BMAN'].sum() / total_populacao) / (total_homens / total_populacao)
    }

    return resultados


def main():
    """Função principal que executa a análise"""
    print("Análise Epidemiológica do Condado de Allegheny\n")
    print("Calculando probabilidades de hipertensão e diabetes por sexo...\n")

    try:
        resultados = calcular_probabilidades()

        # Exibir resultados em formato de tabela
        print("{:<30} {:<15}".format('Indicador', 'Prevalência'))
        print("-" * 45)
        for indicador, valor in resultados.items():
            print("{:<30} {:<15.2%}".format(indicador, valor))

        # Interpretação
        print("\nPrincipais conclusões:")
        print(f"- {resultados['Hipertensão entre Mulheres']:.1%} das mulheres têm hipertensão")
        print(f"- {resultados['Hipertensão entre Homens']:.1%} dos homens têm hipertensão")
        print(
            f"- A diferença absoluta entre sexos é de {abs(resultados['Hipertensão entre Mulheres'] - resultados['Hipertensão entre Homens']):.1%} pontos percentuais")

    except FileNotFoundError:
        print("Erro: Os arquivos 'Hypertension.csv' e 'Diabetes.csv' devem estar no mesmo diretório")
    except Exception as e:
        print(f"Ocorreu um erro: {str(e)}")


if __name__ == "__main__":
    main()