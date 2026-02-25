from xgboost import XGBRegressor


def train_predictor(compressed_features, y):
    model = XGBRegressor()
    model.fit(compressed_features, y)
    return model