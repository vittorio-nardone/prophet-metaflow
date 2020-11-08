import pandas as pd
import numpy as np

from metaflow import FlowSpec, step, Parameter

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
import itertools

class ProphetFlow(FlowSpec):
    """
    ProphetFlow use Facebook Prophet to predict future values of a
    timeseries.
    """       

    data_filename = Parameter('data-file',
                            default="data/daily-min-temperatures.txt",
                            help="Time series data file - csv file format")

    columns_mapping = Parameter('columns',
                            default={'Date':'ds','Temp':'y'},
                            help="Rename columns according to Prophet standards")

    @step
    def start(self):
        """
        Init configuration
        """        
        if self.data_filename is None:
            raise Exception("No filename provided.")        
        
        self.next(self.load)

    @step 
    def load(self):
        """
        Raw data is loaded and prepared
        """
        # Load csv in pandas dataframe
        self.df = pd.read_csv(self.data_filename) 

        # Rename columns to meet Prophet input dataframe standards
        self.df.rename(columns=self.columns_mapping, inplace=True)

        # Convert Date column to datetime64 dtype
        self.df['ds']= pd.to_datetime(self.df['ds'], infer_datetime_format=True)

        self.next(self.hyper_tuning)

    @step
    def hyper_tuning(self):
        """
        Hyperparameters tuning
        """
        # Tune hyperparameters of the model
        param_grid = {  
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        }

        # Generate all combinations of parameters
        self.all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

        # Use cross validation to evaluate all parameters
        self.next(self.cross_validation, foreach='all_params')

    @step
    def cross_validation(self):
        """
        Perform cross-validation on given hyperparameters
        """             
        # Fit model with given params
        m = Prophet(**self.input).fit(self.df)  
        # Perform cross-validation
        df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)

        # Store the RMSE
        self.rmses = df_p['rmse'].values[0]
        
        self.next(self.train)


    @step
    def train(self, inputs):
        """
        Check cross-validation results and find best parameters.
        A new Prophet model is fitted.
        """        
        # Merge artifacts
        self.merge_artifacts(inputs, exclude=['rmses'])

        # Get RMSEs from previous steps
        rmses = [input.rmses for input in inputs]

        # Find the best parameters
        self.best_params = self.all_params[np.argmin(rmses)]

        # Fit a new model using best params
        self.m = Prophet(**self.best_params)
        self.m.fit(self.df)

        self.next(self.end)

    @step
    def end(self):
        """
        Last step, process is finished
        """
        print("ProphetFlow is all done.")


if __name__ == '__main__':
    ProphetFlow()
