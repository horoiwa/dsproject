import pycaret



def by_caret(df, target):
    pass


class DecisionRidgeRegressor:

    def __init__(self):
        pass


if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    bos = load_boston()
    scaler = StandardScaler()
    X = pd.DataFrame(bos.data, columns=bos.feature_names)
    y = pd.DataFrame(bos.target, columns=["Price"])
    df = pd.concat([y, X], axis=1)
    by_crest(df, target="Price")
