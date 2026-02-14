import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
import os


def load_model_data(stored_model_data_file):
    try: 
        model_data = pd.read_csv(stored_model_data_file, header=None)
        features = model_data.iloc[:, :-1]
        labels = model_data.iloc[:, -1]
        return features, labels
    except FileNotFoundError:
        print(f"Error: The file '{stored_model_data_file}' was not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the model data: {e}")
        return None, None


def main():
    ab = input("Do you want to (a) train a new model or (b) load an existing model? ").strip().lower()

    if ab == "a":
        training_data_file = input("Enter the path to the training data: ")
        while not os.path.isfile(training_data_file):
            print(f"Error: The file '{training_data_file}' does not exist.")
            training_data_file = input("Enter the path to the training data: ")
        features, labels = load_model_data(training_data_file)

        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(features, labels)
        model_filename = input("Enter the filename to save the model: ")
        joblib.dump(model, model_filename)

        print(f"Model trained and saved as '{model_filename}'.")

    elif ab == "b":
        stored_model_data_file = input("Enter the path to the stored model data: ")
        while not os.path.isfile(stored_model_data_file):
            print(f"Error: The file '{stored_model_data_file}' does not exist.")
            stored_model_data_file = input("Enter the path to the stored model data: ")
        model = joblib.load(stored_model_data_file)
        
    test_data_file = input("Enter the path to the test data: ")
    while not os.path.isfile(test_data_file):
        print(f"Error: The file '{test_data_file}' does not exist.")
        test_data_file = input("Enter the path to the test data: ")
    testFeatures, testLabels = load_model_data(test_data_file)

    print("Testing the model...")

    predictions = model.predict(testFeatures)

    for index, row in testFeatures.iterrows():
        instance = ",".join(map(str, row.values))
        print(f"Instance: [{instance}] -> Classified as: {predictions[index]}")

    accuracy = model.score(testFeatures, testLabels)
    print("-----------------------------------------")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":    main()
