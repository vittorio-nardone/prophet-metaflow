import pandas as pd
import numpy as np
from io import StringIO

from metaflow import FlowSpec, step, Parameter, IncludeFile

from fbprophet import Prophet

class ProphetFlow(FlowSpec):
    """
    ProphetFlow use Facebook Prophet to predict future values of a
    timeseries.
    """       
    data_file = IncludeFile('datafile',
                            is_text=True,
                            help='Time series data file - csv file format',
                            default='data/daily-min-temperatures.txt')
    columns_mapping = Parameter('columns',
                            default={'Date':'ds','Temp':'y'},
                            help="Rename columns according to Prophet standards")

    @step 
    def start(self):
        """
        Raw data is loaded and prepared
        """
        # Load csv in pandas dataframe
        self.df = pd.read_csv(StringIO(self.data_file)) 

        # Rename columns to meet Prophet input dataframe standards
        self.df.rename(columns=self.columns_mapping, inplace=True)

        # Convert Date column to datetime64 dtype
        self.df['ds']= pd.to_datetime(self.df['ds'], infer_datetime_format=True)

        self.next(self.train)

    @step
    def train(self):
        """
        A new Prophet model is fitted.
        """        
        # Fit a new model using defaults
        self.m = Prophet()
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
