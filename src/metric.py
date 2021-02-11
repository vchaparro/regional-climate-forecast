"""
Metric for the regional forecast climate challenge
"""

import numpy as np
import sys

def climate_metric_function(dataframe_y_true, dataframe_y_pred):
    """
        Example of custom metric function.

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the learning database values.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_base_x = pd.read_csv(CSV_1_FILE_PATH, index_col=False, sep=',')
            5 columns :
            - DATASET: Define the dataset id. database stores several consistent dataset. Each dataset are independant
                       a dataset is composed by:
                     * 3072 temperature anomalies within the all world from 22 models (model id from 1 to 22) during 10 years
                     * 3072 temperature anomalies within the all world from the observation (model id = 0)during 10 years.
                     * the predicted 192 temparature anomalies within the all world for the 22 models
            - MODEL: model id (1-22) for models and (0) or the observation
            - TIME: id of the time (0-9) for the 10 year history and 10 for the predicted date.
            - POSITION: earth coordinate in healpix (nside=4 for prediction, nside=16 for history) the ordering is in nested
            - VALUE : the corresponding temperature anomalies.

        dataframe_y_true: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=False, sep=',')
            Note: y_true as (number of timestamp)x(192)x(22) values. Where 192 is the number of spatial pixels.
            22 is the distribution of possible solution for a given time and position.
            Dataframe containing the learning database values.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_base_x = pd.read_csv(CSV_1_FILE_PATH, index_col=False, sep=',')
            5 columns :
            - DATASET: Define the dataset id. database stores several consistent dataset. Each dataset are independant
                       a dataset is composed by:
                     * 3072 temperature anomalies within the all world from 22 models (model id from 1 to 22) during 10 years
                     * 3072 temperature anomalies within the all world from the observation (model id = 0)during 10 years.
                     * the predicted 192 temparature anomalies within the all world for the 22 models
            - MODEL: model id (1-22) for models and (0) or the observation
            - TIME: id of the time (0-9) for the 10 year history and 10 for the predicted date.
            - POSITION: earth coordinate in healpix (nside=4 for prediction, nside=16 for history) the ordering is in nested
            - VALUE : the corresponding temperature anomalies.

    Returns
        score: Float
            The metric evaluated with the two dataframes. 
    """

    # RETURN -1E30 if the dimension of the dataframe are not the proper one
    nline,ncol=dataframe_y_pred.shape
    if ncol!=4:
        return -1E30
    if nline%192!=0:
        return -1E30
    ndata=nline//(192)
    
    nline,ncol=dataframe_y_true.shape
    if ncol!=3:
        return -1E30
    if nline%192!=0:
        return -1E30

    df_pred_mean=np.zeros([ndata,192])
    df_pred_variance=np.zeros([ndata,192])
    df_true=np.zeros([ndata,192])

    for i in range(ndata):
        df_pred_mean[:,:]=np.array(dataframe_y_pred['MEAN']).reshape(ndata,192)
        df_pred_variance[:,:]=np.array(dataframe_y_pred['VARIANCE']).reshape(ndata,192)
        df_true[:,:]=np.array(dataframe_y_true['VALUE']).reshape(ndata,192)

    chi2 = (df_pred_mean-df_true)**2
    
    R2 = np.sum(chi2)/np.sum((df_true)**2)
        
    RELIABILITY = np.sqrt(np.mean(chi2/df_pred_variance))

    print(np.log(R2),np.log(RELIABILITY))
    score=-(np.log(R2)+abs(np.log(RELIABILITY)))
    
    return score


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_Y_TRUE = sys.argv[1]
    CSV_FILE_Y_PRED = sys.argv[2]
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    print(climate_metric_function(df_y_true, df_y_pred))
