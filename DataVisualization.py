import matplotlib.pyplot as plt 
import numpy as mp
import matplotlib.colors as colors 
import matplotlib.cm as cmx

class DataVisualization:
    
    def __init__(self,data):
        self.data = data

    def scatterPlot(self,feature1,feature2):
        """
        [20 points]
        •	This method plots the relationship between 
            two features, feature1 and feature2 as a scatter plot. 
        •	If the reference for the data is set, this 
            method colors the scattered points using
            the underlying reference array.
        •	The color scheme should be set to “jet”.
        """
        
        Legend_Plot = plt.subplots()
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        
        reference_array = self.data.getReference()
        
        x = self.data.dataDict[feature1]
        y = self.data.dataDict[feature2]
        
        if reference_array == None:
            Legend_Plot[1].scatter(x, y, cmap='jet')
        else:
            num_sequence = self.data.dataDict[reference_array]
            Scatter_Plot = Legend_Plot[1].scatter(x, y, c= num_sequence, cmap = 'jet')
            Legend_Plot[1].legend(*Scatter_Plot.legend_elements(), loc= 'upper right')

        