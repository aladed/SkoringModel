import sys
import pandas as pd
import joblib
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

# === 1. Функция предобработки данных (из data_Analysis.ipynb) ===
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'v' in categorical_columns:
        categorical_columns.remove('v')


    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_columns])
    feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

    # Удаляем последний dummy для каждой категории
    cols_to_drop = []
    for col in categorical_columns:
        col_cols = [c for c in encoded_df.columns if c.startswith(f"{col}_")]
        if col_cols:
            cols_to_drop.append(col_cols[-1])
    encoded_df = encoded_df.drop(columns=cols_to_drop)

    df_encoded = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

    # One-hot для признака 'v'
    encoder_v = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data_v = encoder_v.fit_transform(df_encoded[['v']])
    feature_names_v = encoder_v.get_feature_names_out(['v'])
    df_v = pd.DataFrame(encoded_data_v, columns=feature_names_v, index=df_encoded.index)
    df_encoded = pd.concat([df_encoded.drop(columns=['v']), df_v], axis=1)
    if 'v_No' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['v_No'])

    # Возраст по квантилям
    df_encoded['age_bin'] = pd.qcut(df_encoded['age'], q=10, duplicates='drop')
    df_encoded = df_encoded.drop(columns=['age'])
    df_encoded = pd.get_dummies(df_encoded, columns=['age_bin'], prefix='age_q', drop_first=True)

    # One-hot для 'g' и 'h'
    df_encoded = pd.get_dummies(df_encoded, columns=['g', 'h'], drop_first=True)

    # Масштабирование числовых
    numeric_cols = ['a', 'b', 'c', 'd', 'e', 'f', 'l']
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    

    df_encoded = df_encoded.drop(columns="target",axis=1)
    df_encoded = df_encoded.drop(columns="id",axis=1)

    return df_encoded



def main():
 

    file_path = "data.csv"

    # Загружаем модель
    obj = joblib.load("best_xgb_model_Recall: 0.8367, Precision: 0.0118, F1: 0.0232, ROC-AUC: 0.8675.pkl")
    model = obj["model"]
    features = obj["features"]
    threshold = obj["threshold"]

    # Предобработка
    df = pd.read_csv(file_path)
    X = data.drop("target", axis=1)
    y_true = data["target"]

    X.columns = [f"f{i}" for i in range(X.shape[1])]


    print(X.head(10))

    # Выравниваем фичи под обученную модель
    X = X.reindex(columns=features, fill_value=0)

    # Предсказания
    pos_class_index = list(model.classes_).index(1)  # где именно хранится класс "1"
    y_pred_proba = model.predict_proba(X)[:, pos_class_index]
    y_pred_proba = model.predict_proba(X.values)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    if y_true is not None:
        r = recall_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        cm = confusion_matrix(y_true, y_pred)

        print("\n=== Метрики на данных ===")
        print(f"Recall: {r:.4f}, Precision: {p:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        print("Матрица ошибок:")
        print(cm)
    else:
        print("\nЦелевая переменная отсутствует. Выводим предсказания:")
        print(y_pred)

main()
