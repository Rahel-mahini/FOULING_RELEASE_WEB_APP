from  combinatorixPy import initialize_dask_cluster, get_result
from dask.distributed import Client

if __name__ == '__main__': 
    
     # User-defined configuration
    config = {
        'n_workers': 1,
        'threads_per_worker': 128,
        'memory_limit': '500GB',
        'timeout': 300
    }
    
    
    # Initialize the Dask client with the provided config
    cluster = initialize_dask_cluster(config)
    client = Client(cluster) 
    
    
    descriptors_file_path = r'Dataset/descriptors.csv'
    concentrations_file_path = r'Dataset/components_fractions.csv'
    output_path = r''

        
    # Define thresholds for column variation, and correlation
    threshold_const = 0.01
    threshold_corr = 0.95
    batch_num = 100000
    
    
    result_path = get_result(descriptors_file_path, concentrations_file_path, output_path, threshold_const, threshold_corr, batch_num, client )  
    print  ("result path", result_path)
    
    client.close()
    cluster.close()