import os

from mlflow.tracking import MlflowClient
import mlflow
import pickle

MLFLOW_ADDRESS = os.getenv("MLFLOW_ADDRESS", "http://127.0.0.1:5051")
MLFLOW_EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'exp_flow_2')
SERVE_LOCAL_FOLDER = os.getenv('SERVE_LOCAL_FOLDER', '/tmp/serve')

class ModelLoader():

    def load_model_from_mlflow(_self, run_id):
        mlflow.set_tracking_uri(MLFLOW_ADDRESS)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        client = MlflowClient()

        # mlflow_model_path = f'models:/best_model-exp_flow_2/Staging'
        mlflow_model_path = f'runs:/{run_id}/model'
        # mlflow_dv_path = f'mlflow-artifacts:/best_model-exp_flow_2/Staging'

        if not os.path.exists(SERVE_LOCAL_FOLDER): os.mkdir(SERVE_LOCAL_FOLDER)
        model = mlflow.xgboost.load_model(mlflow_model_path)
        client.download_artifacts(run_id, "preprocesor", SERVE_LOCAL_FOLDER)

        with open('/tmp/serve/preprocesor/preprocesor.b', 'rb') as f_in:
            dv = pickle.load(f_in)

        return model, dv
