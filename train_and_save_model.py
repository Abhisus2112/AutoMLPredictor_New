def train_and_save_model(file_path, model_name="MyCustomModel"):
    import pandas as pd
    import streamlit as st
    import joblib, json
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, r2_score
    from sklearn import metrics
    import numpy as np

    try:
        # Load dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ Error: The file at {file_path} was not found. Please check the path and try again.")
        return None, None, None

    # Separate features and target
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Detect problem type
    if y.nunique() <= 30:
        problem_type = "classification"
        y_cat = y.astype('category')
        y = y_cat.cat.codes
        target_labels = dict(enumerate(y_cat.cat.categories))
    elif y.dtype == "object":
        problem_type = "classification"
        y_cat = y.astype('category')
        y = y_cat.cat.codes
        target_labels = dict(enumerate(y_cat.cat.categories))
        print("✅ Detected classification problem. Target variable converted to numerical labels.")
    else:
        problem_type = "regression"
        target_labels = None
        print("✅ Detected regression problem.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipelines
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    transformers = []
    if len(numeric_features) > 0:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_features))

    if len(categorical_features) > 0:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers, remainder='passthrough')

    if problem_type == "classification":
        # Define a parameter grid for RandomForestClassifier

        param_grid_lg = [
            # Search Space 1: Focus on the 'l1' penalty
            {
                'model__penalty': ['l1'],
                'model__C': [0.1, 1, 10],
                'model__solver': ['liblinear', 'saga']  # Only solvers that work with 'l1'
            },
            # Search Space 2: Focus on the 'l2' penalty
            {
                'model__penalty': ['l2'],
                'model__C': [0.1, 1, 10],
                'model__solver': ['lbfgs', 'liblinear', 'saga']  # Solvers that work with 'l2'
            }
        ]
        param_grid_rf = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2'],
            'model__criterion': ['gini', 'entropy']
        }
        param_grid_ad = {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.1, 0.5, 1.0],
            'model__estimator__max_depth': [1, 3],
            # Correct: pipeline -> model step -> estimator parameter -> max_depth
            'model__estimator__criterion': ['gini', 'entropy']
        }

        # Define a parameter grid for XGBoostClassifier
        param_grid_xgb = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__subsample': [0.7, 0.8, 1.0],
            'model__colsample_bytree': [0.7, 0.8, 1.0],
            'model__gamma': [0, 0.1, 0.2]
        }

        models = {
            "LogisticRegression": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))]),
                param_grid_lg, cv=5, scoring='accuracy'
            ),
            "RandomForestClassifier": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier())]),
                param_grid_rf, cv=5, scoring='accuracy'
            ),
            "AdaBoostClassifier": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(random_state=42), random_state=42))]),
                param_grid_ad, cv=5, scoring='accuracy'),
            "XGBoostClassifier": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier())]),
                param_grid_xgb, cv=5, scoring='accuracy'
            ),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=4)
        }
        scoring = "f1_weighted"
    else:
        # Define a parameter grid for RandomForestRegressor
        def Accuracy_Score(orig, pred):
            MAPE = np.mean(100 * (np.abs(orig - pred) / orig))
            # print('#'*70,'Accuracy:', 100-MAPE)
            return (100 - MAPE)

        # Custom Scoring MAPE calculation
        from sklearn.metrics import make_scorer
        custom_Scoring = make_scorer(Accuracy_Score, greater_is_better=True)

        param_grid_ad = {
            "model__n_estimators": [50, 100, 200, 300],
            "model__learning_rate": [0.01, 0.1, 0.5, 1],
            "model__loss": ["linear", "square", "exponential"],
            "model__estimator__max_depth": [1, 2, 3, 5],
            "model__estimator__min_samples_split": [2, 5, 10],
            "model__estimator__min_samples_leaf": [1, 2, 5]
        }
        param_grid_rf = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4, 10],
            "model__max_features": ["auto", "sqrt", "log2", 0.5],
            "model__bootstrap": [True, False],
            "model__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
            "model__min_weight_fraction_leaf": [0.0, 0.1, 0.2],
            "model__oob_score": [True, False]
        }

        # Define a parameter grid for XGBRegressor
        param_grid_xgb = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", RandomForestRegressor(criterion='squared_error'))]),
                param_grid_rf, cv=5, scoring=custom_Scoring
            ),
            "AdaBoostRegressor": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(random_state=42), random_state=42))]),
                param_grid_ad, cv=5, scoring=custom_Scoring),
            "XGBRegressor": GridSearchCV(
                Pipeline([("preprocessor", preprocessor),
                          ("model", XGBRegressor(objective='reg:squarederror', booster='gbtree'))]),
                param_grid_xgb, cv=5, scoring=custom_Scoring
            ),
            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=8)
        }
        scoring = custom_Scoring

    print("\n✨ Training models with 5-fold cross-validation...")
    best_score, best_model, best_name = -1, None, None
    for name, model in models.items():
        if isinstance(model, GridSearchCV):
            model.fit(X_train, y_train)
            mean_score = model.best_score_
            print(f"  > {name}: {scoring}={mean_score:.4f} (Best params: {model.best_params_})")
            model = model.best_estimator_
        else:
            clf = Pipeline([("preprocessor", preprocessor), ("model", model)])
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=scoring)
            mean_score = scores.mean()
            print(f"  > {name}: {scoring}={mean_score:.4f}")

        if mean_score > best_score:
            best_score, best_model, best_name = mean_score, model, name

    print("\n🏆 Best Model Found!")
    print(f"Name: {best_name}, Score: {best_score:.4f}")

    if not isinstance(best_model, Pipeline):
        best_model = Pipeline([("preprocessor", preprocessor), ("model", best_model)])

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\n📊 Evaluating on Test Set...")

    if problem_type == "classification":
        y_test_labels = np.array([target_labels[int(i)] for i in y_test])
        y_pred_labels = np.array([target_labels[int(i)] for i in y_pred])

        final_score = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {final_score:.4f}")
        scores = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
        mean_score = scores.mean()
        print(f"Overall Accuracy: {mean_score:.4f}")
        print(metrics.classification_report(y_test, y_pred))
        F1_Score = metrics.f1_score(y_test, y_pred, average='weighted')
        print('Accuracy of the model on Testing Sample Data:', round(F1_Score, 2))
        print("\nExample Predictions:")
        for i in range(5):
            print(f"  - Actual: {y_test_labels[i]}, Predicted: {y_pred_labels[i]}")

        cm = confusion_matrix(y_test, y_pred)

        if target_labels:
            disp_labels = [target_labels[i] for i in sorted(target_labels.keys())]
        else:
            disp_labels = None

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix for {best_name}")
        st.pyplot(plt.gcf())
        plt.clf()

    else:
        final_score = mean_squared_error(y_test, y_pred)
        r2_final_score = r2_score(y_test, y_pred)
        scores = cross_val_score(clf, X, y, cv=5, scoring=scoring)
        mean_score = scores.mean()
        print(f"Test MSE: {final_score:.4f}")
        print(f"Test R-squared (R²): {r2_final_score:.4f}")
        print(f"Accuracy: {mean_score:.4f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        plt.title(f"Predicted vs. Actual Values for {best_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()

        # --- Feature Importance Analysis ---
    print("\n🧐 Analyzing Feature Importance...")

    # Get the names of the features after preprocessing
    preprocessor = best_model.named_steps['preprocessor']
    feature_names = list(numeric_features)

    # Get the names for one-hot encoded features
    if len(categorical_features) > 0:
        ohe_transformer = preprocessor.named_transformers_['cat']
        ohe_step = ohe_transformer.named_steps['onehot']
        ohe_feature_names = ohe_step.get_feature_names_out(categorical_features)
        feature_names.extend(ohe_feature_names)

    trained_model = best_model.named_steps['model']

    if hasattr(trained_model, 'feature_importances_'):
        importances = trained_model.feature_importances_
    elif hasattr(trained_model, 'coef_'):
        if trained_model.coef_.ndim > 1:
            importances = np.mean(np.abs(trained_model.coef_), axis=0)
        else:
            importances = np.abs(trained_model.coef_)
    else:
        print("Feature importance is not available for this model type.")
        importances = None

    if importances is not None:
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # --- NEW: Group by original column names ---
        # Add a new column to store the original feature name
        importance_df['original_feature'] = importance_df['feature'].apply(
            lambda x: x.split('_')[0] if len(x.split('_')) > 1 else x)

        # Group by the original feature and sum the importances
        grouped_importance = importance_df.groupby('original_feature')['importance'].sum().sort_values(
            ascending=False).reset_index()

        print("\nTop 10 Most Important Original Features:")
        print(grouped_importance.head(10).to_string(index=False))

        # Plot the top 10 most important original features
        plt.figure(figsize=(10, 6))
        plt.barh(grouped_importance['original_feature'].head(10), grouped_importance['importance'].head(10),
                 color='skyblue')
        plt.xlabel("Total Importance Score")
        plt.ylabel("Original Feature")
        plt.title("Top 10 Original Feature Importances")
        plt.gca().invert_yaxis()
        st.pyplot(plt.gcf())
        plt.clf()
        # --- END OF NEW SECTION ---

    # Save model file
    model_file = f"{model_name}_{best_name}.joblib"
    joblib.dump(best_model, model_file)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "algorithm": best_name,
        "dataset": file_path,
        "test_score": float(final_score),
        "problem_type": problem_type
    }
    with open(f"{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n✅ Model saved as {model_file}")
    print(f"✅ Metadata saved as {model_name}_metadata.json")

    return best_model, model_file, metadata