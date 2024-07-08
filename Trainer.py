from class_and_functions import *
import datetime as dat

class Trainer():
    def __init__(self,df_path) -> None:
        print("\n > Initialized Training Pipeline.")
        self.df_path = df_path

    def orchestrator(self):
        try:
            print("\n > Starting Training process.")
            
            dataset = self.read_data(path=self.df_path)
            X_train, X_test, y_train, y_test = self.train_validation_split(dataset)
            X_train, X_val,y_train,y_val = self.preprocessing(X_train, X_test, y_train, y_test)
            
            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val
    
            self.train_models(X_train, X_val, y_train, y_val)
            self.evaluate_and_save_results()
            print("\n > Finished Training process succesfully")
        except Exception as e:
            print(f"\n > Exception during the training process: {e}")
            #traceback.print_exc()
    def read_data(self,path):
        print("\n > Reading data.")
        df=pd.read_csv(path)
        return df
    
    def train_validation_split(self, df):
        y = df.Revenue
        X = df.drop(['Revenue'], axis=1	)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
        splits = X_train, X_test, y_train, y_test
        print(f"\n > Performed an 80/20 split. Training set has {X_train.shape[0]} examples. Validation set has {X_test.shape[0]} examples.")
        return splits
        
    def preprocessing(self, X_train, X_test,y_train,y_test):
        try:
            print("\n > Saving test data and target to 'Data' folder.")
            X_test.to_csv('Data/test.csv',index=False)
            y_test.to_csv('Data/test_target.csv',index=False)
            
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=42)
            print("\n > Starting Preprocessing process.")
            transformers={
                        'Outlier_winsorize':OutlierTransformation(upper_bound=0.98,lower_bound=0.01),
                        'New_variables':FunctionTransformer(feature_engineering),
                        'Log_transformation':FunctionTransformer(Log),
                        'Standarize':Standarize(),
                        'Column_check':ColumnCheck(categorical_columns=['OperatingSystems', 'Browser', 'Region', 'TrafficType']),
                        'OneHotEncoding':OneHotEncoding(['Weekend','VisitorType']),
                        'FrequencyEncoding':CountFrequencyEncoder(encoding_method='frequency',variables=['OperatingSystems', 'Browser', 'TrafficType'],unseen='encode'),
                        'TargetEncdoing':TargetEncoder(cols=['Month','Region'])}
            
            for name, transformer in transformers.items():
                pipeline=CustomPipeline(transformer)
                pipeline.fit(X_train,y_train)
                X_train=pipeline.transform(X_train)
                pipeline.save('Pipelines',name)
                
            transformers=load_transformers('Pipelines')
            
            for trans in transformers:
                X_val=trans.transform(X_val)
            
            print("\n > Loaded test data to 'Data' folder.")
            print("\n > Finished Preprocessing process succesfully")
            return  X_train, X_val,y_train,y_val
        except Exception as e:
            print(f"\n > Exception during the training process: {e}")
        #traceback.print_exc()
    
    def train_models(self,X_train, X_val, y_train, y_val):
        self.model_instances = {}
        self.models = {'Models':[],'Accuracy':[],'Precision':[],'Recall':[],'F1':[],'AUC':[],'PRAUC':[],'Threshold':[]}
        try:
            print("\n > Starting Training models process.")
            models = {
                'LogisticRegression':[LogisticRegression(max_iter=1000),{'C': [0.01, 0.1, 1, 10, 100],'solver': ['liblinear', 'saga']}],
                'GradientBoosting':[GradientBoostingClassifier(),{'n_estimators': [100, 200],'learning_rate': [0.01, 0.1],'max_depth': [3, 5, 7]}],
                'RandomForest':[RandomForestClassifier(),{'n_estimators': [100], 'max_depth': [10],'min_samples_split': [2],'min_samples_leaf': [1]}],
                'XGBoost':[xgb.XGBClassifier(eval_metric='logloss',use_label_encoder=False,objective='binary:logistic'),{'n_estimators': [500],'max_depth': [10],'learning_rate': [0.01],'scale_pos_weight': [0.18307]}],
                'LightGBM':[lgb.LGBMClassifier(objective='binary',metric='binary_logloss'),{'boosting_type':['dart'],'learning_rate': [0.01],'n_estimators': [400],'max_depth': [10],}]
             }
            
            skf=StratifiedKFold(n_splits=5,shuffle=True)
            
            transformers_names = sorted(
                [f for f in os.listdir('Pipelines') if os.path.isfile(os.path.join('Pipelines', f)) and f.endswith('.pkl')],
                key=lambda x: int(x.split('_')[0])
            )
            signature= infer_signature(X_train, y_train)
            mlflow.set_tracking_uri('sqlite:///mlruns.db')
            mlflow.set_experiment('Model Training')
            for name, model in models.items():
                print(f"\n >Training {name}.")
                gscv = GridSearchCV(
                    model[0],
                    param_grid=model[1],
                    cv=skf,
                    n_jobs=-1
                )
                gscv.fit(X_train, y_train)

                best_estimator = gscv.best_estimator_
                y_pred = best_estimator.predict_proba(X_val)[:, 1]

                fpr, tpr, thresholds = roc_curve(y_val, y_pred)
                # get the best threshold
                J = tpr - fpr
                ix = argmax(J)
                best_thresh_auc = thresholds[ix]

                thresholds = np.arange(0, 1, 0.001)
                f1_scores = [f1_score(y_val, (y_pred > thr).astype(int)) for thr in thresholds]
                ix = argmax(f1_scores)
                best_thresh_f1 = thresholds[ix]
                print('Best Threshold=%f' % (best_thresh_f1))

                best_thresh = (best_thresh_auc + best_thresh_f1) / 2

                print('Best Threshold=%f' % (best_thresh))
                y_pred_bin = (y_pred > best_thresh).astype(int)
                print_metrics(y_val, y_pred_bin)
                fig1 = plot_confusion_matrix(y_val, y_pred_bin)
                fig2 = plot_roc_curve(y_val, y_pred_bin)
                if name=='LogisticRegression':
                    fig3 = plot_logistic_regression_feature_importances(best_estimator, X_train)
                else:
                    fig3 = plot_feature_importances(best_estimator, X_val)
                fig4 = plot_learning_curve(best_estimator, "Learning Curve", X_train, y_train, cv=skf, n_jobs=-1, scoring='roc_auc')

                visualizaciones = {fig1: 'Visualizaciones/Confusion_matrix.png', fig2: 'Visualizaciones/ROC_curve.png',fig3:'Visualizaciones/Feature_importances.png',fig4: 'Visualizaciones/Learning_curve.png'}

                accuracy=accuracy_score(y_val, y_pred_bin)
                precision=precision_score(y_val, y_pred_bin)
                recall=recall_score(y_val, y_pred_bin)
                f1=f1_score(y_val, y_pred_bin)
                auc=roc_auc_score(y_val, y_pred_bin)
                prauc=average_precision_score(y_val, y_pred_bin)
                
                metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC': auc, 'PRAUC': prauc, 'Threshold': best_thresh}
                
                self.models['Models'].append(name)
                self.models['Accuracy'].append(accuracy)
                self.models['Precision'].append(precision)
                self.models['Recall'].append(recall)
                self.models['F1'].append(f1)
                self.models['AUC'].append(auc)
                self.models['PRAUC'].append(prauc)
                self.models['Threshold'].append(best_thresh)
                self.model_instances.setdefault(name, best_estimator)
                with mlflow.start_run(run_name=name) as run:
                    
                    for i in range(len(transformers_names)):
                        mlflow.log_artifact(f"Pipelines/{transformers_names[i]}", "Pipelines")
                    mlflow.log_params(best_estimator.get_params())
                    for key, value in visualizaciones.items():
                        mlflow.log_figure(key, value)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("Training Info", f"{name} Model")
                    #signature = infer_signature(X_train, (best_estimator.predict_proba(X_train)[:, 1] > best_thresh).astype(int))
                    mlflow.sklearn.log_model(best_estimator, f"{name} Model", signature=signature)
                    mlflow.end_run()
                
            print("\n > Finished Training models process succesfully")
        except Exception as e:
            print(f"\n > Exception during the training process: {e}")
            #traceback.print_exc()
    
    def evaluate_and_save_results(self):
        print(f"\n > Evaluating and saving best model.")
        models_results=pd.DataFrame(self.models)
        # save results to txt
        filename = f'training_output - {date.today()}.csv'
        with open(filename, 'a') as file:
            file.write("Models,Accuracy,Precision,Recall,F1,AUC,PRAUC,Threshold,PonderatedScore\n")
            for i in range(models_results.shape[0]):
                pondered_score=(models_results.Accuracy[i]+models_results.Precision[i]+models_results.Recall[i]*1.25+models_results.F1[i]*1.5+models_results.AUC[i]*1.75+models_results.PRAUC[i]+models_results.Threshold[i]*0.5)/7
                line = str(models_results.Models[i]) + "," + str(models_results.Accuracy[i])+","+ str(models_results.Precision[i])+ "," + str(models_results.Recall[i]) + "," + str(models_results.F1[i]) + "," + str(models_results.AUC[i]) + "," + str(models_results.PRAUC[i])+ "," + str(models_results.Threshold[i])+ "," + str(pondered_score)
                file.write(line + "\n")
        print(f"Training results have been saved to {filename}")

        best_model,thr = self.get_best_model()
        self.save_model(best_model,thr)

    def get_best_model(self):
        # get best model based on the RMSE Metric
        models_results=pd.DataFrame(self.models)
        models_results['PonderatedScore']=(models_results['Accuracy']+models_results['Precision']+models_results['Recall']*1.3+models_results['F1']*1.5+models_results['AUC']*1.75+models_results['PRAUC']+models_results['Threshold']*.5)/7
        best_model_name=models_results.loc[models_results['PonderatedScore'].idxmax()]['Models']
        print(f"\n ----> 🧠 Best model is {best_model_name} 🧠 <----")
        return self.model_instances[best_model_name],models_results.loc[models_results['Models']==best_model_name,'Threshold'].iloc[0]

    def save_model(self, best,thr):
        # Dump model as pkl file
        with open('Data/Threshold.txt', 'wb') as file:
            print(f"\n > Saving Threshold to 'Data' folder.")
                                
            file.write(str(thr).encode())
            
        with open('Pickle/best_model.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'best_model.pkl'⬇️💾")
            pickle.dump(best, file)
