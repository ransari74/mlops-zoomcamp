if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
import pathlib
import pickle
import mlflow
@data_exporter
def export_data(data, *args, **kwargs):
    model,dv=data
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("home_work_03")

    with mlflow.start_run():

        mlflow.sklearn.log_model(model, artifact_path="models")
       
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    mlflow.end_run()


