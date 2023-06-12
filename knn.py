import joblib


class KNNClassifier:

    def __init__(self):
        import warnings
        warnings.filterwarnings('ignore')
        self.model = joblib.load('assets/knn1.pkl')

    def knnPredict(self, features):
        res = self.model.predict([features])
        return res
