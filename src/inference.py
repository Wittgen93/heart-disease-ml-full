import argparse
import joblib
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from model_training import Net


def load_data(path: str) -> pd.DataFrame:
    """
    Загружает CSV с данными и возвращает DataFrame.
    """
    df = pd.read_csv(path)
    return df


def main(args):
    # 1. Загрузка данных
    df = load_data(args.input)
    X = df.drop(columns=['output'])

    # 2. Удаляем VIF-фичи, отброшенные при тренировке
    drop_cols = ['thall_2', 'thall_3', 'slp_2']
    X = X.drop(columns=drop_cols, errors='ignore')

    # 3. Масштабирование числовых признаков
    num_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    scaler: StandardScaler = joblib.load('models/scaler.pkl')
    X[num_cols] = scaler.transform(X[num_cols])

    # 4. Предсказания Logistic Regression и RandomForest
    lr = joblib.load('models/lr_model.pkl')
    rf = joblib.load('models/rf_model.pkl')

    probs = {
        'lr': lr.predict_proba(X)[:, 1],
        'rf': rf.predict_proba(X)[:, 1]
    }

    # 5. Предсказание Neural Network
    # Преобразуем в Tensor
    X_tensor = torch.tensor(X.values.astype('float32'))
    dnn = Net(X_tensor.shape[1])
    dnn.load_state_dict(torch.load('models/dnn_model.pt'))
    dnn.eval()
    with torch.no_grad():
        probs['nn'] = dnn(X_tensor).numpy().ravel()

    # 6. Сохраняем результаты
    results = pd.DataFrame(probs, index=df.index)
    results.to_csv(args.output, index=True)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Heart Disease models")
    parser.add_argument(
        '--input',
        type=str,
        default='data/processed/heart_preprocessed.csv',
        help='Path to preprocessed input CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path where predictions CSV will be saved'
    )
    args = parser.parse_args()
    main(args)