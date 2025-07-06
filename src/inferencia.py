import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm, shapiro, ttest_rel, wilcoxon
from scipy.stats import ttest_ind
from IPython.display import display


diabetes = pd.read_csv('Diabetes.csv')
hipertensao = pd.read_csv('Hypertension.csv')

# Visualizar as 5 primeiras linhas de cada um
print("Diabetes:")
display(diabetes.head())

print("\nHipertensão:")
display(hipertensao.head())

# Verificar colunas e tipos de dados
print("Colunas e tipos de Diabetes:")
print(diabetes.dtypes)
print("\nResumo:")
display(diabetes.describe(include='all'))

print("\n\nColunas e tipos de Hipertensão:")
print(hipertensao.dtypes)
print("\nResumo:")
display(hipertensao.describe(include='all'))

# Converte para numérico (com erros ignorados)
diabetes['BPAD'] = pd.to_numeric(diabetes['BPAD'], errors='coerce')
diabetes['BPAN'] = pd.to_numeric(diabetes['BPAN'], errors='coerce')
hipertensao['TPAD'] = pd.to_numeric(hipertensao['TPAD'], errors='coerce')
hipertensao['TPAN'] = pd.to_numeric(hipertensao['TPAN'], errors='coerce')

# Calcula prevalência (%)
diabetes['Prev_Diabetes'] = diabetes['BPAN'] / diabetes['BPAD'] * 100
hipertensao['Prev_Hipertensao'] = hipertensao['TPAN'] / hipertensao['TPAD'] * 100

# Junta as bases pela coluna 'CT'
df = pd.merge(diabetes[['CT', 'Prev_Diabetes']],
              hipertensao[['CT', 'Prev_Hipertensao']],
              on='CT', how='inner')

# Remove linhas com dados ausentes
df = df.dropna()

# Gráfico de dispersão
plt.figure(figsize=(8,6))
plt.scatter(df['Prev_Diabetes'], df['Prev_Hipertensao'], alpha=0.7)
plt.xlabel('Prevalência de Diabetes Tipo 2 (%)')
plt.ylabel('Prevalência de Hipertensão (%)')
plt.title('Correlação entre Prevalência de Diabetes e Hipertensão')
plt.grid(True)
plt.show()

# Correlação de Pearson
corr, pval = pearsonr(df['Prev_Diabetes'], df['Prev_Hipertensao'])
print(f"Correlação de Pearson: {corr:.3f}")
print(f"Valor-p: {pval:.4f}")

# Interpretação
if pval < 0.05:
    print("➡️ Correlação estatisticamente significativa (p < 0.05).")
else:
    print("ℹ️ Correlação NÃO estatisticamente significativa (p >= 0.05).")


# Converter colunas relevantes para numérico
cols_diabetes = ['BWAD', 'BWAN', 'BMAD', 'BMAN']
cols_hipertensao = ['TWAD', 'TWAN', 'TMAD', 'TMAN']

for col in cols_diabetes:
    diabetes[col] = pd.to_numeric(diabetes[col], errors='coerce')

for col in cols_hipertensao:
    hipertensao[col] = pd.to_numeric(hipertensao[col], errors='coerce')

# Calcular prevalência por sexo (%)
diabetes['Prev_Diabetes_M'] = diabetes['BMAN'] / diabetes['BMAD'] * 100
diabetes['Prev_Diabetes_F'] = diabetes['BWAN'] / diabetes['BWAD'] * 100

hipertensao['Prev_Hipertensao_M'] = hipertensao['TMAN'] / hipertensao['TMAD'] * 100
hipertensao['Prev_Hipertensao_F'] = hipertensao['TWAN'] / hipertensao['TWAD'] * 100

# Remover valores ausentes
diab_m = diabetes['Prev_Diabetes_M'].dropna()
diab_f = diabetes['Prev_Diabetes_F'].dropna()
hiper_m = hipertensao['Prev_Hipertensao_M'].dropna()
hiper_f = hipertensao['Prev_Hipertensao_F'].dropna()

# Teste t de diferença entre sexos
t_diab, p_diab = ttest_ind(diab_m, diab_f)
t_hiper, p_hiper = ttest_ind(hiper_m, hiper_f)

# Visualização
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.boxplot([diab_m, diab_f], labels=['Homens', 'Mulheres'])
plt.title('Prevalência de Diabetes Tipo 2 por Sexo')
plt.ylabel('Prevalência (%)')

plt.subplot(1,2,2)
plt.boxplot([hiper_m, hiper_f], labels=['Homens', 'Mulheres'])
plt.title('Prevalência de Hipertensão por Sexo')
plt.ylabel('Prevalência (%)')

plt.tight_layout()
plt.show()

# Resultados dos testes
print("🧪 Teste t para prevalência de Diabetes:")
print(f"t = {t_diab:.3f}, p = {p_diab:.4f}")
if p_diab < 0.05:
    print("➡️ Diferença significativa entre homens e mulheres (Diabetes)")
else:
    print("ℹ️ Sem diferença estatisticamente significativa (Diabetes)")

print("\n🧪 Teste t para prevalência de Hipertensão:")
print(f"t = {t_hiper:.3f}, p = {p_hiper:.4f}")
if p_hiper < 0.05:
    print("➡️ Diferença significativa entre homens e mulheres (Hipertensão)")
else:
    print("ℹ️ Sem diferença estatisticamente significativa (Hipertensão)")

# Colunas a converter
cols_diab = ['BMAD','BMAN','BWAD','BWAN']
cols_hip  = ['TMAD','TMAN','TWAD','TWAN']

for col in cols_diab:
    diabetes[col] = pd.to_numeric(diabetes[col], errors='coerce')
for col in cols_hip:
    hipertensao[col] = pd.to_numeric(hipertensao[col], errors='coerce')

# --- 2. Cálculo das prevalências --------------------------------------------
diabetes['Prev_Diab_M'] = diabetes['BMAN'] / diabetes['BMAD'] * 100   # homens
diabetes['Prev_Diab_F'] = diabetes['BWAN'] / diabetes['BWAD'] * 100   # mulheres

hipertensao['Prev_Hip_M'] = hipertensao['TMAN'] / hipertensao['TMAD'] * 100  # homens
hipertensao['Prev_Hip_F'] = hipertensao['TWAN'] / hipertensao['TWAD'] * 100  # mulheres

# Mescla pela região (CT)
df = pd.merge(
    diabetes[['CT','Prev_Diab_M','Prev_Diab_F']],
    hipertensao[['CT','Prev_Hip_M','Prev_Hip_F']],
    on='CT', how='inner'
).dropna()

# --- 3. Correlações ---------------------------------------------------------
corr_M, p_M = pearsonr(df['Prev_Diab_M'], df['Prev_Hip_M'])
corr_F, p_F = pearsonr(df['Prev_Diab_F'], df['Prev_Hip_F'])

# --- 4. Gráficos ------------------------------------------------------------
plt.figure(figsize=(12,5))

# Homens
plt.subplot(1,2,1)
plt.scatter(df['Prev_Diab_M'], df['Prev_Hip_M'], alpha=0.7)
plt.title(f'Homens\nr={corr_M:.3f}, p={p_M:.4g}')
plt.xlabel('Prevalência Diabetes (%)')
plt.ylabel('Prevalência Hipertensão (%)')
plt.grid(True)

# Mulheres
plt.subplot(1,2,2)
plt.scatter(df['Prev_Diab_F'], df['Prev_Hip_F'], alpha=0.7)
plt.title(f'Mulheres\nr={corr_F:.3f}, p={p_F:.4g}')
plt.xlabel('Prevalência Diabetes (%)')
plt.ylabel('Prevalência Hipertensão (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 5. Saída interpretativa -----------------------------------------------
print("=== Correlação entre prevalências (Diabetes x Hipertensão) ===")
print(f"Homens   -> r = {corr_M:.3f}  |  p = {p_M:.4g}")
print(f"Mulheres -> r = {corr_F:.3f}  |  p = {p_F:.4g}\n")

# Interpretação simples
def interpreta(r,p):
    if p < 0.05:
        signif = "significativa"
    else:
        signif = "NÃO significativa"
    intensidade = ("fraca" if abs(r)<0.3 else
                   "moderada" if abs(r)<0.5 else
                   "forte")
    return f"Correlação {intensidade} e {signif}."

print("Homens  : " + interpreta(corr_M,p_M))
print("Mulheres: " + interpreta(corr_F,p_F))

# 1. Conversão de colunas necessárias ---------------------------------------
to_num_diab = ['BPAN', 'BPAN2', 'BPAD']
to_num_hip  = ['TPAN', 'TPAN2', 'TPAD']

for col in to_num_diab:
    diabetes[col] = pd.to_numeric(diabetes[col], errors='coerce')
for col in to_num_hip:
    hipertensao[col] = pd.to_numeric(hipertensao[col], errors='coerce')

# 2. Métricas de interesse --------------------------------------------------
#   a) Proporção de DIABÉTICOS medicados  = BPAN2 / BPAN
diabetes['Prop_Med_Diab'] = diabetes['BPAN2'] / diabetes['BPAN'] * 100

#   b) Prevalência de HIPERTENSÃO        = TPAN / TPAD
hipertensao['Prev_Hip']   = hipertensao['TPAN']  / hipertensao['TPAD'] * 100

#   c) Proporção de HIPERTENSOS medicados = TPAN2 / TPAN
hipertensao['Prop_Med_Hip'] = hipertensao['TPAN2'] / hipertensao['TPAN'] * 100

#   d) Prevalência de DIABETES            = BPAN / BPAD
diabetes['Prev_Diab'] = diabetes['BPAN'] / diabetes['BPAD'] * 100

# 3. Mesclar bases por CT e eliminar ausentes ------------------------------
df = pd.merge(
        diabetes[['CT','Prop_Med_Diab','Prev_Diab']],
        hipertensao[['CT','Prop_Med_Hip','Prev_Hip']],
        on='CT', how='inner'
     ).dropna()

# 4. Correlações ------------------------------------------------------------
#   4.1 Prop medicados para DIABETES  vs  Prevalência de HIPERTENSÃO
corr1, p1 = pearsonr(df['Prop_Med_Diab'], df['Prev_Hip'])

#   4.2 Prop medicados para HIPERTENSÃO vs  Prevalência de DIABETES
corr2, p2 = pearsonr(df['Prop_Med_Hip'], df['Prev_Diab'])

# 5. Gráficos ---------------------------------------------------------------
plt.figure(figsize=(12,5))

# Gráfico 1
plt.subplot(1,2,1)
plt.scatter(df['Prop_Med_Diab'], df['Prev_Hip'], alpha=0.7)
plt.xlabel('Proporção de diabéticos medicados (%)')
plt.ylabel('Prevalência de hipertensão (%)')
plt.title(f'Diabetes medicada  x  Hipertensão\nr={corr1:.3f}, p={p1:.4g}')
plt.grid(True)

# Gráfico 2
plt.subplot(1,2,2)
plt.scatter(df['Prop_Med_Hip'], df['Prev_Diab'], alpha=0.7)
plt.xlabel('Proporção de hipertensos medicados (%)')
plt.ylabel('Prevalência de diabetes (%)')
plt.title(f'Hipertensão medicada  x  Diabetes\nr={corr2:.3f}, p={p2:.4g}')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Saída textual ----------------------------------------------------------
print("=== Correlação (Pearson) ===")
print(f"1) Prop. DIAB medicados  vs  Prev. HIPERTENSÃO : r = {corr1:.3f} | p = {p1:.4g}")
print(f"2) Prop. HIPERTENSÃO medicados  vs  Prev. DIABETES : r = {corr2:.3f} | p = {p2:.4g}\n")

def interpreta(r,p):
    sig = "significativa (p<0,05)" if p < 0.05 else "não significativa"
    if abs(r) < 0.3:
        intensidade = "fraca"
    elif abs(r) < 0.5:
        intensidade = "moderada"
    else:
        intensidade = "forte"
    direcao = "positiva" if r > 0 else "negativa"
    return f"{intensidade}, {direcao}, {sig}"

print("Interpretação:")
print(f"1) {interpreta(corr1,p1)}")
print(f"2) {interpreta(corr2,p2)}")

# 1. Conversão de colunas numéricas -----------------------------------------
cols_d = ['BMAD','BMAN','BWAD','BWAN']       # diabetes
cols_h = ['TMAD','TMAN','TWAD','TWAN']       # hipertensão
for c in cols_d: diabetes[c]    = pd.to_numeric(diabetes[c],    errors='coerce')
for c in cols_h: hipertensao[c] = pd.to_numeric(hipertensao[c], errors='coerce')

# 2. Prevalências por sexo ---------------------------------------------------
diabetes['Prev_Diab_M']  = diabetes['BMAN'] / diabetes['BMAD'] * 100
diabetes['Prev_Diab_F']  = diabetes['BWAN'] / diabetes['BWAD'] * 100
hipertensao['Prev_Hip_M'] = hipertensao['TMAN'] / hipertensao['TMAD'] * 100
hipertensao['Prev_Hip_F'] = hipertensao['TWAN'] / hipertensao['TWAD'] * 100

# 3. Mescla por região (CT) --------------------------------------------------
df = pd.merge(
        diabetes[['CT','Prev_Diab_M','Prev_Diab_F']],
        hipertensao[['CT','Prev_Hip_M','Prev_Hip_F']],
        on='CT', how='inner'
     )

# --- HOMENS -----------------------------------------------------------------
m_mask = df[['Prev_Diab_M','Prev_Hip_M']].notna().all(axis=1)
dm = df.loc[m_mask,'Prev_Diab_M']
hm = df.loc[m_mask,'Prev_Hip_M']
r_m, p_m = pearsonr(dm, hm)
n_m = m_mask.sum()          # tamanho da amostra

# --- MULHERES ---------------------------------------------------------------
f_mask = df[['Prev_Diab_F','Prev_Hip_F']].notna().all(axis=1)
df_f = df.loc[f_mask,'Prev_Diab_F']
hf   = df.loc[f_mask,'Prev_Hip_F']
r_f, p_f = pearsonr(df_f, hf)
n_f = f_mask.sum()

# 4. Teste de igualdade de correlações (Fisher z) ----------------------------
def fisher_z(r):      # transformação de Fisher
    return 0.5 * np.log((1+r)/(1-r))

import numpy as np
z_m = fisher_z(r_m)
z_f = fisher_z(r_f)
# estatística z para diferença
z_diff = (z_m - z_f) / np.sqrt(1/(n_m-3) + 1/(n_f-3))
p_diff = 2 * (1 - norm.cdf(abs(z_diff)))     # bicaudal

# 5. Gráficos ----------------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(dm, hm, alpha=0.7)
plt.title(f'Homens\nr={r_m:.3f}, p={p_m:.4g}')
plt.xlabel('Prev. Diabetes (%)')
plt.ylabel('Prev. Hipertensão (%)')
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(df_f, hf, alpha=0.7)
plt.title(f'Mulheres\nr={r_f:.3f}, p={p_f:.4g}')
plt.xlabel('Prev. Diabetes (%)')
plt.ylabel('Prev. Hipertensão (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Resultados --------------------------------------------------------------
print("=== Correlação separada por sexo ===")
print(f"Homens   : r = {r_m:.3f} | p = {p_m:.4g} | n = {n_m}")
print(f"Mulheres : r = {r_f:.3f} | p = {p_f:.4g} | n = {n_f}\n")

print("=== Teste de igualdade das correlações (Fisher z) ===")
print(f"z = {z_diff:.3f} | p = {p_diff:.4g}")

# Interpretação automática
def resume(r,p):
    sig = "significativa" if p<0.05 else "não significativa"
    forca = "forte" if abs(r)>=0.5 else "moderada" if abs(r)>=0.3 else "fraca"
    sentido = "positiva" if r>0 else "negativa"
    return f"{forca}, {sentido} e {sig}"

print("\nInterpretação:")
print(f"• Homens   : {resume(r_m,p_m)}.")
print(f"• Mulheres : {resume(r_f,p_f)}.")
if p_diff < 0.05:
    print(f"➡️ As correlações diferem entre os sexos (p={p_diff:.4g}).")
    print("   Isso sugere que a força da relação Diabetes‑Hipertensão é estatisticamente distinta para homens e mulheres.")
else:
    print(f"ℹ️ As correlações não diferem estatisticamente (p={p_diff:.4g}).")


# 1. Converter colunas numéricas --------------------------------------------
for col in ['BPAN','BPAN2','TPAN','TPAN2']:
    if col in diabetes.columns:
        diabetes[col] = pd.to_numeric(diabetes[col], errors='coerce')
    else:
        hipertensao[col] = pd.to_numeric(hipertensao[col], errors='coerce')

# 2. Calcular proporções por região -----------------------------------------
diabetes['Prop_Med_Diab'] = diabetes['BPAN2'] / diabetes['BPAN'] * 100
hipertensao['Prop_Med_Hip'] = hipertensao['TPAN2'] / hipertensao['TPAN'] * 100

# 3. Mesclar e filtrar -------------------------------------------------------
df = pd.merge(
        diabetes[['CT','Prop_Med_Diab']],
        hipertensao[['CT','Prop_Med_Hip']],
        on='CT', how='inner'
     ).dropna()

prop_diab = df['Prop_Med_Diab']
prop_hip  = df['Prop_Med_Hip']
diff      = prop_diab - prop_hip              # diferença pareada

# 4. Teste de normalidade ----------------------------------------------------
p_norm_d  = shapiro(prop_diab)[1]
p_norm_h  = shapiro(prop_hip)[1]
p_norm_df = shapiro(diff)[1]

print(f"Shapiro‑Wilk p‑valores:")
print(f"  Prop_Med_Diab : {p_norm_d:.4f}")
print(f"  Prop_Med_Hip  : {p_norm_h:.4f}")
print(f"  Diferença     : {p_norm_df:.4f}\n")

# 5. Escolha do teste --------------------------------------------------------
if p_norm_d > 0.05 and p_norm_h > 0.05 and p_norm_df > 0.05:
    stat, p_val = ttest_rel(prop_diab, prop_hip)
    teste = "t pareado"
else:
    stat, p_val = wilcoxon(prop_diab, prop_hip)
    teste = "Wilcoxon signed‑rank"

# 6. Estatísticas descritivas ------------------------------------------------
print("=== Estatísticas ===")
print(f"Média prop. medicados Diabetes   : {prop_diab.mean():.2f}% (±{prop_diab.std():.2f})")
print(f"Média prop. medicados Hipertensão: {prop_hip.mean():.2f}% (±{prop_hip.std():.2f})\n")

print(f"=== {teste} ===")
print(f"estatística = {stat:.3f} | p = {p_val:.4g}")

# 7. Boxplot -----------------------------------------------------------------
plt.figure(figsize=(6,5))
plt.boxplot([prop_diab, prop_hip], labels=['Diabetes', 'Hipertensão'])
plt.ylabel('Proporção medicados (%)')
plt.title('Adesão medicamentosa por condição')
plt.grid(True)
plt.show()

# 8. Interpretação -----------------------------------------------------------
alpha = 0.05
if p_val < alpha:
    if prop_diab.mean() > prop_hip.mean():
        concl = "maior entre diabéticos"
    else:
        concl = "maior entre hipertensos"
    print(f"➡️ Diferença estatisticamente significativa (p < 0,05); adesão é {concl}.")
else:
    print("ℹ️ Não há diferença estatisticamente significativa na proporção de medicados.")