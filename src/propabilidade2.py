import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from math import factorial


def load_and_prepare_data():
    """Carrega e prepara os dados com tratamento de erros"""
    try:
        hipertensao = pd.read_csv('Hypertension.csv')
        diabetes = pd.read_csv('Diabetes.csv')

        # Verificação básica de dados
        assert not hipertensao.empty and not diabetes.empty, "Arquivos vazios"
        assert 'TPAN' in hipertensao.columns and 'BPAN' in diabetes.columns, "Colunas faltando"

        return hipertensao, diabetes

    except FileNotFoundError:
        raise FileNotFoundError("Arquivos CSV não encontrados no diretório")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar dados: {str(e)}")


def calculate_poisson_distributions(hipertensao, diabetes):
    """Calcula as distribuições de Poisson para ambas as condições"""
    lambda_hipertensao = hipertensao['TPAN'].mean()
    lambda_diabetes = diabetes['BPAN'].mean()

    max_cases = int(max(hipertensao['TPAN'].max(), diabetes['BPAN'].max())) + 50
    x = np.arange(0, max_cases)

    pmf_hipertensao = poisson.pmf(x, lambda_hipertensao)
    pmf_diabetes = poisson.pmf(x, lambda_diabetes)

    return {
        'lambda_hipertensao': lambda_hipertensao,
        'lambda_diabetes': lambda_diabetes,
        'x_values': x,
        'pmf_hipertensao': pmf_hipertensao,
        'pmf_diabetes': pmf_diabetes,
        'max_cases': max_cases
    }


def plot_poisson_distributions(results):
    """Plota as distribuições teóricas de Poisson"""
    plt.figure(figsize=(14, 6))

    # Hipertensão
    plt.subplot(1, 2, 1)
    plt.bar(results['x_values'], results['pmf_hipertensao'],
            color='skyblue', alpha=0.7, width=0.8)
    plt.title(f'Distribuição de Poisson - Hipertensão\n(λ = {results["lambda_hipertensao"]:.2f})')
    plt.xlabel('Número de Casos por CT')
    plt.ylabel('Probabilidade')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(0, results["lambda_hipertensao"] * 2)

    # Diabetes
    plt.subplot(1, 2, 2)
    plt.bar(results['x_values'], results['pmf_diabetes'],
            color='salmon', alpha=0.7, width=0.8)
    plt.title(f'Distribuição de Poisson - Diabetes\n(λ = {results["lambda_diabetes"]:.2f})')
    plt.xlabel('Número de Casos por CT')
    plt.ylabel('Probabilidade')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlim(0, results["lambda_diabetes"] * 2)

    plt.tight_layout()
    plt.show()


def plot_comparison(hipertensao, diabetes, results):
    """Plota a comparação entre dados reais e o modelo Poisson"""
    plt.figure(figsize=(14, 6))
    bin_width = 20  # Largura dos intervalos para o histograma

    # Função auxiliar para plotagem
    def plot_condition(data, pmf, condition, color, subplot_pos):
        plt.subplot(1, 2, subplot_pos)
        hist, bins, _ = plt.hist(data,
                                 bins=range(0, results['max_cases'], bin_width),
                                 density=True,
                                 color=color,
                                 alpha=0.6,
                                 label='Dados Reais')

        # Ajuste crucial: multiplicar PMF pela largura do bin
        plt.plot(results['x_values'], pmf * bin_width,
                 'k-', lw=2, label='Poisson')

        plt.title(f'Comparação: {condition}')
        plt.xlabel('Número de Casos')
        plt.ylabel('Densidade')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)

    # Hipertensão
    plot_condition(hipertensao['TPAN'],
                   results['pmf_hipertensao'],
                   'Hipertensão',
                   'skyblue', 1)

    # Diabetes
    plot_condition(diabetes['BPAN'],
                   results['pmf_diabetes'],
                   'Diabetes',
                   'salmon', 2)

    plt.tight_layout()
    plt.show()


def calculate_probabilities(results):
    """Calcula e exibe probabilidades específicas"""

    def print_probs(lambda_val, condition):
        print(f"\nProbabilidades para {condition} (λ = {lambda_val:.2f}):")
        print(f"• P(X = 0): {poisson.pmf(0, lambda_val):.4f}")
        print(f"• P(X ≤ 100): {poisson.cdf(100, lambda_val):.4f}")
        print(f"• P(X ≥ 500): {1 - poisson.cdf(499, lambda_val):.4f}")
        print(f"• P(300 ≤ X ≤ 400): {poisson.cdf(400, lambda_val) - poisson.cdf(299, lambda_val):.4f}")
        print(f"• Moda: {int(lambda_val)} casos (P = {poisson.pmf(int(lambda_val), lambda_val):.4f})")

    print_probs(results['lambda_hipertensao'], "Hipertensão")
    print_probs(results['lambda_diabetes'], "Diabetes")


def main():
    """Função principal de execução"""
    print("=== Análise de Distribuição de Casos por Census Tract ===")
    print("Modelagem com Distribuição de Poisson\n")

    try:
        # 1. Carregar dados
        hipertensao, diabetes = load_and_prepare_data()

        # 2. Calcular distribuições
        results = calculate_poisson_distributions(hipertensao, diabetes)

        print(f"\nMédias calculadas:")
        print(f"- Hipertensão: λ = {results['lambda_hipertensao']:.2f} casos por CT")
        print(f"- Diabetes: λ = {results['lambda_diabetes']:.2f} casos por CT")

        # 3. Plotar distribuições teóricas
        plot_poisson_distributions(results)

        # 4. Plotar comparação com dados reais
        plot_comparison(hipertensao, diabetes, results)

        # 5. Calcular probabilidades específicas
        calculate_probabilities(results)

    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        print("Verifique se os arquivos estão no diretório correto e com o formato esperado.")


if __name__ == "__main__":
    main()