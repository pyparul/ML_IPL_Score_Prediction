# write your import here
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek


class MyModel:

    def __init__(self) -> None:
        # Define Model here
        # For Example:
        # self.model = DummyRegressor(strategy = 'constant')
        self.model = self.get_dtc_model()

    def get_dtc_model(self):
        dtc_clf = DecisionTreeClassifier(
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=33,
            criterion='gini',
            random_state=0,
        )
        bagging_clf = BaggingClassifier(
            estimator=dtc_clf,
            n_estimators=100,
            random_state=0,
        )
        return dtc_clf

    def fit(self, training_data):
        # create training data
        # create dummy data now
        # dummy_training_data = [["MA Chidambaram Stadium", 1, "Mumbai Indians", "Kolkata Knight Riders",
        #                    "Quinton de Kock, Rohit Sharma, Suryakumar Yadav",
        #                    "Harbhajan Singh, Varun Chakravarthy, Shakib Al Hasan, Pat Cummins"],
        #                    ["MA Chidambaram Stadium", 2, "Kolkata Knight Riders", "Mumbai Indians",
        #                    "Harbhajan Singh, Varun Chakravarthy, Shakib Al Hasan, Pat Cummins",
        #                    "Quinton de Kock, Rohit Sharma, Suryakumar Yadav"]]
        dummy_training_labels = np.array([30, 30]).reshape(-1, 1)

        dummy_training_data = pd.DataFrame(data=training_data,
                                           columns=["venue", "innings", "batting_team",
                                                    "bowling_team", "batsmen", "bowlers"])

        # train the model
        self._model.fit(dummy_training_data, dummy_training_labels)

        return self

    def predict(self, test_data):
        x_test = test_data

        # compuate and return predictions
        return self._model.predict(x_test)
