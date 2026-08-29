from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class BMIDecisionTreeImputer(BaseEstimator, TransformerMixin):
    """Predicts missing BMI values from age/gender using a small DecisionTreeRegressor.
    Fitted ONLY on whatever data is passed to .fit() (i.e. X_train, or each CV training
    fold during GridSearchCV), so no test-set leakage.

    Lives in its own module (rather than defined inline in a notebook) so that joblib/pickle
    can resolve it consistently by import path -- both when a notebook fits+exports a model
    that embeds this class (the SVM and Random Forest pipelines both do; ANN still ships it
    separately as ann_bmi_imputer.pkl since Keras models can't be sklearn Pipeline steps),
    and later when app.py unpickles that model. If this class were defined inline in a
    notebook cell instead, joblib would try to re-import it from the notebook's kernel module
    (e.g. '__main__'), which does not exist in the app.py process -- causing the exact
    "Can't get attribute 'BMIDecisionTreeImputer'" error.
    """

    def fit(self, X, y=None):
        X = X.copy()

        bmi_features = X[['age', 'gender']]
        bmi_target = X['bmi']

        # Only use rows with known BMI
        mask = bmi_target.notna()

        self.bmi_model = Pipeline([
            ('scale', StandardScaler()),
            ('tree', DecisionTreeRegressor(random_state=42))
        ])

        self.bmi_model.fit(
            bmi_features.loc[mask],
            bmi_target.loc[mask]
        )

        return self

    def transform(self, X):
        X = X.copy()

        missing_mask = X['bmi'].isna()

        if missing_mask.any():
            predicted_bmi = self.bmi_model.predict(
                X.loc[missing_mask, ['age', 'gender']]
            )
            X.loc[missing_mask, 'bmi'] = predicted_bmi

        return X