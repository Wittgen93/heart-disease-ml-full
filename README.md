# Heart Disease Prediction

## Введение
Цель проекта - построить модель, которая помогает выявлять у пациентов высокий риск сердечно-сосудистых заболеваний (ССЗ) по клиническим данным.

## Задача
Создать воспроизводимую систему предсказания ССЗ на табличных данных, сравнить несколько моделей (Logistic Regression, Random Forest, Neural Network), подготовить артефакты для инференса и кратко законспектировать научную статью по теме.

---

## Датасет
- **Train:** 600 000 записей  
- **Test:** 400 000 записей  
- **Признаки (сырые):**  
  `ID, age, sex, chest, resting_blood_pressure, serum_cholestoral, fasting_blood_sugar, resting_electrocardiographic_results, maximum_heart_rate_achieved, exercise_induced_angina, oldpeak, slope, number_of_major_vessels, thal`  
- **Target:** `class` (0/1, только в train)

**Предобработка (кратко):**
- удаление дубликатов/NaN;
- приведение типов и диапазонов:  
  `age, chest, resting_blood_pressure, serum_cholestoral, maximum_heart_rate_achieved` → целые; `oldpeak` → вещественный;
- one-hot кодирование категорий с `drop_first=True`;
- выравнивание схемы `test` под `train` (без `class`);
- сохранение: `data/processed/train_preprocessed.csv`, `data/processed/test_preprocessed.csv`.

**Финальная схема признаков (после OHE):**
- Числовые: `ID, age, resting_blood_pressure, serum_cholestoral, maximum_heart_rate_achieved, oldpeak`
- Dummy:  
  `sex_1, chest_2, chest_3, chest_4, fasting_blood_sugar_1, resting_electrocardiographic_results_1, resting_electrocardiographic_results_2, exercise_induced_angina_1, slope_2, slope_3, number_of_major_vessels_1, number_of_major_vessels_2, number_of_major_vessels_3, thal_6, thal_7`  
В `train` дополнительно присутствует `class`.

---

## Исследовательский анализ (EDA)
- Гистограммы и box-plots числовых признаков (выбросы физиологичны, правые хвосты у `chol`, `oldpeak`, частично `BP`).
- Bar-plots по dummy-признакам (несбалансированные уровни: редкие `restecg_1`, `slope_3`, `fbs_1`, `chest_2`).
- Корреляции (heatmap):  
  `oldpeak` положительно с `class`; `maximum_heart_rate_achieved` отрицательно с `class`; `age` - слабая положительная связь.
- Scatter `age` vs `maximum_heart_rate_achieved` подтверждает нисходящий тренд; PCA показывает умеренную перекрываемость классов.

Ключевые изображения сохранены в `artifacts/`:
`corr_heatmap.png`, `scatter_age_maxhr.png`, `pca_scatter.png`, `box_*.png`.

---

## Моделирование
**Валидационный сплит:** Stratified 80/20.  
**Масштабирование:** числовые признаки для LogReg/NN (`StandardScaler` через `ColumnTransformer`/отдельный scaler).  
**Дисбаланс:** `class_weight='balanced'` (LogReg/RF), `pos_weight` в `BCEWithLogitsLoss` (NN).

**Модели:**
- Logistic Regression (`saga`, `max_iter=2000`, регуляризация `C=1.0`)
- Random Forest (`n_estimators=400`, `max_depth=12`, `min_samples_leaf=10`)
- Neural Network (PyTorch, MLP):  
  `128 - 64 - 32`, `ReLU`, `Dropout 0.15/0.15/0.10`, выход — **logits**;  
  лосс: `BCEWithLogitsLoss(pos_weight)`; оптимизатор: Adam; ранняя остановка (patience=3).

**Метрики (валидация):**
| Модель           | ROC-AUC | PR-AUC | F1     | Accuracy |
|------------------|--------:|-------:|-------:|---------:|
| Neural Net (MLP) | **0.9627** | **0.9570** | 0.8864 | 0.8982 |
| Random Forest    | 0.9599 | 0.9535 | 0.8832 | 0.8953 |
| Logistic Reg.    | 0.9557 | 0.9485 | 0.8758 | 0.8886 |

**Вывод:** лучшая модель — **Neural Net (MLP)**. Разрыв невелик, что подтверждает устойчивость признаков и корректную предобработку.

**Графики обучения/сравнения:**  
`artifacts/roc_curves.png`, `artifacts/nn_train_loss.png`, `artifacts/nn_val_roc_auc.png`.


## Инференс

Тестовый набор без таргета — на `test` выполняется только предсказание.

**Скрипт:** `src/infer.py`  
**Вход:** предобработанный CSV (`data/processed/test_preprocessed.csv`)  
**Выход:** `artifacts/preds_test.csv` с колонками `ID, prediction_proba`.

```bash
# Предусловия (универсально для Windows/macOS/Linux)
git lfs pull
pip install -r requirements.txt
python -c "import os; os.makedirs('artifacts', exist_ok=True)"

# Запуск инференса (универсально)
python src/infer.py --input data/processed/test_preprocessed.csv --output artifacts/preds_test.csv --models_dir models --model best