# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:31:34 2024

@author: RASULEVLAB
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import os


X_file_path = r'/mmfs1/projects/bakhtiyor.rasulev/Rahil/Project_4/combinatorial_descriptors_header.csv'
output_path = r'/mmfs1/projects/bakhtiyor.rasulev/Rahil/Project_4'   
     
X = pd.read_csv(X_file_path, sep=',',  header=0)    
print("combinatorial ", X)  
print("combinatorial shape", X.shape) 
 
#Initialize the StandardScaler
scaler = StandardScaler()  
   
# Fit and transform the data
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled , columns= X.columns)
print("combinatorial_normalized ", X_scaled_df)  
print("combinatorial_normalized shape", X_scaled_df.shape)  

try:
    
  # Create the output directory if it doesn't exist                                                        
    os.makedirs(output_path, exist_ok = True)     
    file_name = 'combinatorial_normalized_latest_version.csv'
    file_path = os.path.join(output_path, file_name)
            
    X_scaled_df.to_csv(file_path, sep = ',', header =True, index = True ) 

    file_path_dict = {'normalized_combinatorial': file_path}
    print("CSV file written successfully.")
    print ("CSV file size is  ", os.path.getsize(file_path))
    print ("CSV file column number is  " , X_scaled_df.shape[1])
    print ("file_path_dictionary is  " , file_path_dict)

except Exception as e:
    print("Error occurred while writing matrices to CSV:", e)