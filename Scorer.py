from class_and_functions import *


class Scorer():
    def __init__(self, df_path) -> None:
        print("\n > Initialized Scoring Pipeline.")
        self.df_path = df_path

    def orchestrator(self):
        print("\n > Initialized Scoring process.")
        test_data=pd.read_csv(self.df_path)
        print("\n > Test data read.")
        #sample_data=test_data.drop(['Id'],axis=1)
        print("\n > Loading model.")
        model = self.load_model()
        print("\n > Model loaded.")
        
        print("\n > Preprocessing.")
        sample_data = self.preprocessing(test_data)
        # print("\n > Loading vectorizer.")
        thr=self.load_threshold()
        print("\n > Making predictions.")
        predictions = self.predict(model=model, X=sample_data,thr=thr)
        test_data["Revenue"] = predictions
        print('\n------------------\nPredictions for Test Data\n------------------\n')
        #print(sample_data)
        filename = f'Revenue Predictions - {datetime.now().strftime("%Y-%m-%d %H-%M")}.csv'     
        with open(filename, 'a') as file:
            file.write("Index,Revenue\n")
            for i in range(test_data.shape[0]):
                text = str(test_data.index[i])
                predictions = str(test_data.Revenue[i])
                encoded_text = text.encode('utf-8', errors='ignore')
                encoded_predictions = predictions.encode('utf-8', errors='ignore')
                decoded_text = encoded_text.decode('utf-8')
                decoded_predictions = encoded_predictions.decode('utf-8')

                line = f"{decoded_text},{decoded_predictions}"
                file.write(line + "\n")
        print(f"Training results have been saved to {filename}")

    def load_model(self):
        print("\n > Loading BestModel from memory.")
        with open('Pickle/best_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model

    def load_threshold(self):
        with open('Data/Threshold.txt', 'rb') as file:
            print("\n > Loading Threshold from memory.")
            threshold = file.read()
            return float(threshold)
    
    def preprocessing(self, X_test):
        transformers=load_transformers('Pipelines')
            
        for trans in transformers:
            X_test=trans.transform(X_test)        
        
        print("\n > Data preprocessed.")
        return X_test
    
    def predict(self, model, X,thr=0.3):
        print("\n > Predicting...")
        predictions = model.predict_proba(X)[:, 1]
        predictions = (predictions > thr).astype(int)
        return predictions