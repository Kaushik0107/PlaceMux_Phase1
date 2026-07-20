from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

def train_model():

    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42
    )

    model = DummyClassifier(strategy="most_frequent")

    model.fit(X_train, y_train)

    return model, X_test, y_test