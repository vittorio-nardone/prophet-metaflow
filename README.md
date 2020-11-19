# prophet-metaflow

An example of how to run Facebook Prophet in Metaflow. 

[Read full blog post here!!](https://www.vittorionardone.it/en/2020/11/19/time-series-forecasting-with-prophet-and-metaflow/)

**Minimum daily temperatures dataset is used**.
This dataset describes the minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia. The units are in degrees Celsius and there are 3650 observations. The source of the data is credited as the Australian Bureau of Meteorology.

## What is it?

ProphetFlow is a [Metaflow](https://metaflow.org/) flow using Facebook [Prophet](https://facebook.github.io/prophet/) to predict future values of a timeseries. 
A Prophet model is trained to predict next year minimum daily temperatures. [Hyperparameters tuning](https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning) is performed with cross-validation. 

```
Step start
    Init configuration
    => load

Step load
    Raw data is loaded and prepared
    => hyper_tuning

Step hyper_tuning
    Hyperparameters tuning
    => cross_validation

Step cross_validation
    Perform cross-validation on given hyperparameters
    => train

Step train
    Check cross-validation results and find best parameters.
    A new Prophet model is fitted.
    => end

Step end
    Last step, process is finished
```

## How to play with it

If you need, install required Python packages (Metaflow & FBProphet) as usual: `pip install -r requirements.txt`

Simply run training flow with: `python src/ProphetFlow.py run`

In `predict.ipynb` notebook it's explained how to use Prophet trained model to predict future values, accessing Metaflow runs artifacts.

![Forecast](https://github.com/vittorio-nardone/prophet-metaflow/blob/main/img/forecast.png "Forecast")
![Forecast components](https://github.com/vittorio-nardone/prophet-metaflow/blob/main/img/forecast_components.png "Forecast components")

A simple flow without hyperparameters tuning: `python src/ProphetSimpleFlow.py run`

## Run in AWS

To configure Metaflow to use AWS Cloud, you first need to create stack:

```
aws s3 cp aws/metaflow-cfn-template.yml s3://<your_bucket>
aws cloudformation create-stack --stack-name metaflow \
                                --template-url <url_of_template_in_your_bucket> \
                                --capabilities CAPABILITY_IAM
```

When Cloudformation stack is created, you must configure Metaflow.

`metaflow configure aws`

To run all steps in flow in AWS Batch: 

`python src/ProphetFlow.py run --with batch:image=vnardone/prophet-metaflow`

An example of how to run a single step in AWS Batch (using @batch decorator):

`python src/ProphetAWSStepFlow.py run` 
