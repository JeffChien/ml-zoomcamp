from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class BasicPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.feature_names_in_ = columns
        self.feature_names_out = []

    def lower_case(self, df):
        # column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")

        # column values
        for c in df.dtypes[df.dtypes == "object"].index:
            df[c] = df[c].str.lower().str.replace(" ", "_")

        return df

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df = self.lower_case(df)
        df = df[self.feature_names_in_]
        df["age"] = df["year"].max() - df.year
        df["number_of_doors"] = df["number_of_doors"].fillna(0).astype("int")
        df["engine_fuel_type"] = df.engine_fuel_type.fillna("")
        df = df.fillna(0)

        # categories in market_category
        df_dummies = (
            df["market_category"].str.get_dummies(",").add_prefix("market_category=")
        )
        df = pd.concat([df.drop("market_category", axis=1), df_dummies], axis=1)
        self.feature_names_out = df.columns.to_numpy()
        return df

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out


class RecordizeDataframe(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df.to_dict(orient="records")
