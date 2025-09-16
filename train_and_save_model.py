def train_and_save_model(file_path, model_name="MyCustomModel"):
    import pandas as pd
    import os
    import streamlit as st
    import joblib, json
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, \
        RandomizedSearchCV  # Import RandomizedSearchCV
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
    from sklearn.metrics import classification_report
    import numpy as np

    if hasattr(file_path, "read"):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read uploaded file as CSV: {e}")
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")
        df = pd.read_csv(file_path)
    if df.shape[1] < 2:
        raise ValueError("Dataset should have at least one feature and one target column")

    # Separate features and target
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    # Detect problem type
    if y.nunique() <= 30 or y.dtype == "object":
        problem_type = "classification"
        y_cat = y.astype('category')
        y = y_cat.cat.codes
        target_labels = dict(enumerate(y_cat.cat.categories))
        print("✅ Detected classification problem.")
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
            {'model__penalty': ['l1'], 'model__C': [0.1, 1, 10], 'model__solver': ['liblinear', 'saga']},
            {'model__penalty': ['l2'], 'model__C': [0.1, 1, 10], 'model__solver': ['lbfgs', 'liblinear', 'saga']}
        ]
        param_grid_rf = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [10, 20, None],
            'model__criterion': ['gini', 'entropy']
        }
        param_grid_ad = {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.1, 0.5, 1.0],
            'model__estimator__max_depth': [1, 3]
        }
        param_grid_xgb = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1]
        }
        models = {
            "LogisticRegression": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000))]), param_grid_lg,
                cv=5, scoring='accuracy'),
            "RandomForestClassifier": GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier())]), param_grid_rf, cv=5,
                scoring='accuracy'),
            "AdaBoostClassifier": GridSearchCV(Pipeline([("preprocessor", preprocessor), ("model", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=42), random_state=42))]), param_grid_ad, cv=5,
                                               scoring='accuracy'),
            "XGBoostClassifier": GridSearchCV(Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier())]),
                                              param_grid_xgb, cv=5, scoring='accuracy'),
            "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=4)
        }
        scoring = "f1_weighted"
    else:
        # --- MODIFICATION 1: Improved Custom Scorer ---
        # Added a small epsilon (1e-6) to the denominator to prevent division by zero.
        def Accuracy_Score(orig, pred):
            epsilon = 1e-6
            MAPE = np.mean(100 * (np.abs(orig - pred) / (orig + epsilon)))
            return (100 - MAPE)

        from sklearn.metrics import make_scorer
        custom_Scoring = make_scorer(Accuracy_Score, greater_is_better=True)

        # --- MODIFICATION 2: Reduced and Optimized Parameter Grids ---
        # These grids are much smaller and more efficient for a faster search.
        param_grid_ad = {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.1, 1.0],
            "model__loss": ["linear", "square"],
            "model__estimator__max_depth": [2, 3, 5]
        }
        param_grid_rf = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, 30],
            "model__max_features": ["sqrt", 0.5],  # "auto" can be slow, "sqrt" is a good default
        }
        param_grid_xgb = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.2]
        }

        # --- MODIFICATION 3: Using RandomizedSearchCV for Efficiency ---
        # RandomizedSearchCV samples from the grid instead of trying everything.
        # n_iter=20 means it will try 20 different combinations, which is much faster.
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomizedSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", RandomForestRegressor(criterion='squared_error'))]),
                param_distributions=param_grid_rf, n_iter=20, cv=5, scoring=custom_Scoring, random_state=42
            ),
            "AdaBoostRegressor": RandomizedSearchCV(
                Pipeline([("preprocessor", preprocessor), ("model", AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(random_state=42), random_state=42))]),
                param_distributions=param_grid_ad, n_iter=20, cv=5, scoring=custom_Scoring, random_state=42
            ),
            "XGBRegressor": RandomizedSearchCV(
                Pipeline([("preprocessor", preprocessor),
                          ("model", XGBRegressor(objective='reg:squarederror', booster='gbtree'))]),
                param_distributions=param_grid_xgb, n_iter=20, cv=5, scoring=custom_Scoring, random_state=42
            ),
            "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=8)
        }
        scoring = custom_Scoring

    st.markdown("### ✨ Training models with 5-fold cross-validation...")
    best_score, best_model, best_name = -float('inf'), None, None  # Use -inf for safety
    for name, model in models.items():
        st.write(f"  > Training **{name}**...")
        if isinstance(model, (GridSearchCV, RandomizedSearchCV)):
            model.fit(X_train, y_train)
            mean_score = model.best_score_
            print(
                f"  > {name}: {scoring if isinstance(scoring, str) else 'custom_score'}={mean_score:.4f} (Best params: {model.best_params_})")
            model = model.best_estimator_
        else:
            clf = Pipeline([("preprocessor", preprocessor), ("model", model)])
            # For non-gridsearch models in regression, ensure the scoring is callable
            cv_scoring = scoring if callable(scoring) else 'r2'
            scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=cv_scoring)
            mean_score = scores.mean()
            print(f"  > {name}: {scoring if isinstance(scoring, str) else 'custom_score'}={mean_score:.4f}")

        if mean_score > best_score:
            best_score, best_model, best_name = mean_score, model, name

    st.success("Model training complete!")
    st.markdown(f"### 🏆 Best Model: **{best_name}**")
    print(f"Name: {best_name}, Score: {best_score:.4f}")

    if not isinstance(best_model, Pipeline):
        best_model = Pipeline([("preprocessor", preprocessor), ("model", best_model)])

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    st.markdown("#### 📊 Evaluating on Test Set...")

    if problem_type == "classification":
        y_test_labels = np.array([target_labels.get(int(i), 'N/A') for i in y_test])
        y_pred_labels = np.array([target_labels.get(int(i), 'N/A') for i in y_pred])

        final_score = accuracy_score(y_test, y_pred)
        st.metric(label="Test Accuracy", value=f"{final_score:.4f}")

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=[v for k, v in sorted(target_labels.items())]))

        example_preds = pd.DataFrame({"Actual": y_test_labels[:5], "Predicted": y_pred_labels[:5]})
        st.write("### Example Predictions")
        st.table(example_preds)

        cm = confusion_matrix(y_test, y_pred)
        disp_labels = [target_labels[i] for i in sorted(target_labels.keys())]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
        fig, ax = plt.subplots()
        disp.plot(cmap="Blues", values_format="d", ax=ax)
        plt.title(f"Confusion Matrix for {best_name}")
        st.pyplot(fig)
        plt.clf()

    else:  # Regression
        final_mse = mean_squared_error(y_test, y_pred)
        final_r2 = r2_score(y_test, y_pred)
        final_accuracy = Accuracy_Score(y_test.values, y_pred)

        st.metric(label="Test Mean Squared Error (MSE)", value=f"{final_mse:.4f}")
        st.metric(label="Test R-squared (R²)", value=f"{final_r2:.4f}")
        st.metric(label="Test 'Accuracy' (100-MAPE)", value=f"{final_accuracy:.2f}%")

        example_preds = pd.DataFrame({"Actual": y_test[:10].values.round(2), "Predicted": y_pred[:10].round(2)})
        st.write("### Example Predictions")
        st.table(example_preds)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
        ax.set_title(f"Predicted vs. Actual Values for {best_name}")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.grid(True)
        st.pyplot(fig)
        plt.clf()

    # --- Feature Importance Analysis (unchanged, but now works with the best model from RandomizedSearchCV) ---
    print("\n🧐 Analyzing Feature Importance...")
    try:
        preprocessor = best_model.named_steps['preprocessor']
        trained_model = best_model.named_steps['model']

        cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
            categorical_features) if len(categorical_features) > 0 else []
        feature_names = list(numeric_features) + list(cat_features)

        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
        elif hasattr(trained_model, 'coef_'):
            importances = np.abs(trained_model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            importances = None
            st.warning("Feature importance is not available for this model type.")

        if importances is not None:
            importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
                by="importance", ascending=False)
            importance_df['original_feature'] = importance_df['feature'].apply(lambda x: x.split('_')[0])
            grouped_importance = importance_df.groupby('original_feature')['importance'].sum().sort_values(
                ascending=False).reset_index()

            st.write("### Top 10 Most Important Features")
            st.dataframe(grouped_importance.head(10))

            fig, ax = plt.subplots(figsize=(10, 6))
            top_10 = grouped_importance.head(10)
            ax.barh(top_10['original_feature'], top_10['importance'], color='skyblue')
            ax.set_xlabel("Total Importance Score")
            ax.set_ylabel("Original Feature")
            ax.set_title("Top 10 Original Feature Importances")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.clf()
    except Exception as e:
        st.error(f"Could not perform feature importance analysis: {e}")

    # Save model file
    model_file = f"{model_name}_{best_name}.joblib"
    joblib.dump(best_model, model_file)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "algorithm": best_name,
        "dataset_name": file_path.name if hasattr(file_path, 'name') else os.path.basename(file_path),
        "problem_type": problem_type,
        "target_labels": target_labels,
        "best_cv_score": float(best_score),
    }
    with open(f"{model_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    st.subheader("📑 Model Metadata")
    st.json(metadata)
    st.success(f"✅ Model training complete! Download links are now available.")

    # --- ✨ NEW: Save file contents to the app's memory (Session State) ✨ ---

    # Read the model file from disk into memory as bytes
    with open(model_file, "rb") as f:
        model_bytes = f.read()

    # Convert metadata dictionary to a JSON string in memory
    json_string = json.dumps(metadata, indent=4)

    # Store everything in st.session_state so it survives the page refresh
    st.session_state['training_complete'] = True
    st.session_state['model_bytes'] = model_bytes
    st.session_state['model_filename'] = model_file
    st.session_state['json_string'] = json_string
    st.session_state['json_filename'] = f"{model_name}_metadata.json"

    # Note: The download buttons are REMOVED from this file.

    return best_model, model_file, metadata

    return best_model, model_file, metadata