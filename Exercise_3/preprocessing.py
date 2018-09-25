import pandas as pd

class PreProcessing:
    
    def __init__(self, dataset_name, separator=','):
        self.dataset_name = dataset_name
        self.dataframe = pd.DataFrame(pd.read_csv(self.dataset_name, header=None, sep=separator))
        self.normalized_dataframe = None
    
    def normalize(self, ignore_first_column=False):
        """
        Normalizes columns between values of 0 and 1
        """
        if ignore_first_column:
            self.normalized_dataframe = (self.dataframe.iloc[:, 1:len(self.dataframe.columns)] - self.dataframe.iloc[:, 1:len(self.dataframe.columns)].min())/(self.dataframe.iloc[:, 1:len(self.dataframe.columns)].max() - self.dataframe.iloc[:, 1:len(self.dataframe.columns)].min())
            self.normalized_dataframe.insert(0, 0, self.dataframe[0])
        else:
            self.normalized_dataframe = (self.dataframe - self.dataframe.min())/(self.dataframe.max() - self.dataframe.min())

    def switch_first_last_column(self):
        """
        Switches position of first and last column. 
        Useful if the class attribute is given as the first column instead of the last.

        """
        cols = self.dataframe.columns.tolist()
        cols = cols[1:] + cols[0:1]
        self.dataframe = self.dataframe[cols]
        if(not self.normalized_dataframe.empty):
            self.normalized_dataframe = self.normalized_dataframe[cols] 

    def normalize_class(self):
        """
        Turns class attribute into various attributes for use in neural network
        Assumes class column is the last one in the dataset.
        Example:
                    x1  x2  x3
        class 1:    1   0   0
        class 2:    0   1   0
        class 3:    0   0   1
        """
         
        #define number of classes 
        classes = self.normalized_dataframe[self.normalized_dataframe.columns[-1]].unique()
        class_column = self.normalized_dataframe.columns[-1]
        temp = self.normalized_dataframe[class_column]

        
        length = len(self.normalized_dataframe.columns)

        #add extra columns
        #put 1 in column i if row is of class i 
        i=0
        for c in classes:
            self.normalized_dataframe.loc[self.normalized_dataframe[class_column]==c, length+i]=1
            self.normalized_dataframe.loc[self.normalized_dataframe[class_column]!=c, length+i]=0
            i+=1

        #removes original class column
        cols = self.normalized_dataframe.columns.tolist()
        cols = cols[:length-1] + cols[length:]
        self.normalized_dataframe = self.normalized_dataframe[cols]

        
            
       

    def show(self):
        """
        Print dataset
        """
        print("CSV")
        print(self.dataframe)

    def show_normalized(self):  
        """
        Print normalized dataset
        """
        print("CSV NORMALIZED")
        print(self.normalized_dataframe)
    