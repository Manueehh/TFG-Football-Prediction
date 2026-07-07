from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
from tabpfn_client import TabPFNClassifier
from sklearn.utils.validation import check_X_y, check_array


class BalancedBaggingTabPFNBinary(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator,
        n_bags=30,
        positive_ratio = 0.5,
        positive_label = 'D',
        bag_size=512,
        class_ratios = None,
        random_state=42
    ):
        self.base_estimator = base_estimator
        self.n_bags = n_bags
        self.bag_size = bag_size
        self.class_ratios = class_ratios
        self.positive_ratio = positive_ratio
        self.positive_label = positive_label
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        rng = np.random.default_rng(self.random_state)

        pos_idx = np.where(y == self.positive_label)[0]
        neg_idx = np.where(y != self.positive_label)[0]

        if len(pos_idx) == 0:
            raise ValueError("No hay muestras positivas.")
        if len(neg_idx) == 0:
            raise ValueError("No hay muestras negativas.")

        if self.class_ratios is not None:
          pos_label = self.positive_label
          neg_label = [c for c in self.classes_ if c != pos_label[0]]
          n_pos = int(self.bag_size * self.class_ratios.get(pos_label, 0.5))
          n_neg = self.bag_size - n_pos
        else:
          n_pos = int(self.bag_size * self.positive_ratio)
          n_neg = self.bag_size - n_pos

        self.models_ = []

        for _ in range(self.n_bags):
            sampled_pos = rng.choice(pos_idx, size=n_pos, replace=True)
            sampled_neg = rng.choice(
                neg_idx,
                size=n_neg,
                replace=(n_neg > len(neg_idx))
            )

            bag_idx = np.concatenate([sampled_pos, sampled_neg])
            rng.shuffle(bag_idx)

            model = clone(self.base_estimator)
            model.fit(X[bag_idx], y[bag_idx])
            self.models_.append(model)

        return self

    def predict_proba(self, X):
        X = check_array(X)

        probas = np.array([
            model.predict_proba(X)
            for model in self.models_
        ])

        return probas.mean(axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]