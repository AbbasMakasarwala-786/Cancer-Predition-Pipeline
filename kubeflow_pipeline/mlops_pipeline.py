import kfp #to create kubeflow pipelines
from kfp import dsl
import kfp.compiler # help to define pipeline components dsl -data specific language


## Commonents of pipeline
def data_processing_op():
    return dsl.ContainerOp(
        name="Data Processing",
        image = "abbas31/coletrol-cancer-prediction-pipeline-app:latest",
        command=["python","src/data_processing.py"]
    )# COntainerOp- container operation

def model_training_op():
    return dsl.ContainerOp(
        name="model training",
        image = "abbas31/coletrol-cancer-prediction-pipeline-app:latest",
        command=["python","src/model_training.py"]
    )

### Pipeline starts here .

@dsl.pipeline(
    name="Mlops Pipeline",
    description="Colerotal cancer prediction pipeline"
)
def mlops_pipeline():
    data_processing =data_processing_op()
    model_training = model_training_op().after(data_processing) #to give kubeflow information that model_training to be done after data processing


### Run pipeline
if __name__ =="__main__":
    ## whenever this pipeline will execute it will generate a yaml file
    # upload this yaml file to kubeflow dashboard
    kfp.compiler.Compiler().compile(
        mlops_pipeline,"mlops_pipeline.yaml"
    )