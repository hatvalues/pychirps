import data_preprocs.data_providers as dp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def test_model_harness():
    X_train, X_test, y_train, y_test = train_test_split(
        dp.cervicalb_pd.features, dp.cervicalb_pd.target, test_size=0.2, random_state=42
    )
    # train a random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    # assert the accuracy
    assert accuracy > 0.5
