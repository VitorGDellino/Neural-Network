import pandas as pd

class PreProcessing:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataframe = pd.DataFrame(pd.read_csv(self.dataset_name, header=None))
        self.normalized_dataframe = None
    
    def normalize(self, ignore_first_column=False):
        if ignore_first_column:
            self.normalized_dataframe = (self.dataframe.iloc[:, 1:len(self.dataframe.columns)] - self.dataframe.iloc[:, 1:len(self.dataframe.columns)].min())/(self.dataframe.iloc[:, 1:len(self.dataframe.columns)].max() - self.dataframe.iloc[:, 1:len(self.dataframe.columns)].min())
            self.normalized_dataframe.insert(0, 0, self.dataframe[0])
        else:
            self.normalized_dataframe = (self.dataframe - self.dataframe.min())/(self.dataframe.max() - self.dataframe.min())

    def show(self):
        print("CSV")
        print(self.dataframe)

    def show_normalized(self):  
        print("CSV NORMALIZED")
        print(self.normalized_dataframe)
    