import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# 1. Carregando os dados
df = pd.read_csv('Base_M43_Pratique_LOL_RANKED_WIN (1).csv')

# 2. Limpeza: O gameId não tem valor preditivo, então descartamos
df = df.drop(columns=['gameId'])

# 3. Definição de Alvo (y) e Variáveis (X)
X = df.drop(columns=['blueWins'])
y = df['blueWins']

# 4. Separação treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Criando os Modelos com Pipeline
modelos = {
    "Regressão Logística": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000))
    ]),
    "Naive Bayes": Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianNB())
    ]),
    "Árvore": Pipeline([
        ('scaler', StandardScaler()),  # não é necessário, mas mantemos consistência
        ('model', DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
}

# 6. Validação Cruzada
print("📊 Avaliação por Validação Cruzada (ROC AUC):")
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=5, scoring='roc_auc')
    print(f"- {nome}: média={scores.mean():.3f}, std={scores.std():.3f}")

# 7. Avaliação Final em Conjunto de Teste
print("\n📊 Avaliação Final em Conjunto de Teste:")
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:,1] if hasattr(modelo, "predict_proba") else None
    
    print(f"\n--- {nome} ---")
    print(classification_report(y_test, y_pred))
    if y_prob is not None:
        print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")

# 8. Feature Importance (para Regressão Logística)
log_reg = modelos["Regressão Logística"].fit(X_train, y_train)
coef = log_reg.named_steps['model'].coef_[0]
importancia = pd.DataFrame({
    'Fator': X.columns,
    'Peso': coef
}).sort_values(by='Peso', ascending=False)

print("\n📊 Importância dos Fatores (Regressão Logística):")
print(importancia)
