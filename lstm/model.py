import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#df = pd.read_csv('/data/raw/ehr-transformer/csvs/aggregated-00.csv')
past = 1
step = 1
future = 1
sequence_length = int(past/step)

label_columns = {
    'ALBUMIN', 
    'ALKALINE_PHOSPHATASE', 
    'ALT', 
    'ANION_GAP', 
    'AST', 
    'BICARB', 
    'BUN', 
    'CALCIUM', 
    'CHLORIDE', 
    'CREATININE', 
    'GFR_AFRICAN_AMERICAN',
    'GFR_NON_AFRICAN_AMERICAN',
  #  'GLOBULIN',
    'GLUCOSE',
    'HEMATOCRIT',
    'HEMOGLOBIN',
    'MCH',
    'MCHC',
    'MCV',
    'MPV',
    'PLATELET_COUNT',
    'POTASSIUM',
    'RBC',
    'RDW-CV',
    'SODIUM',
    'TOTAL_BILIRUBIN',
    'TOTAL_PROTEIN',
    'WBC',
}
num_features = len(label_columns)

aggregated_df = pd.read_csv('aggregated-00.csv')
patients = aggregated_df['patient_id'].unique()
num_patients = len(patients)
data = []
label = []

for p in patients:
    patient_df = aggregated_df.loc[lambda df: df['patient_id'] == p]
    patient_df = patient_df.sort_values(by='time_elapsed_in_min')
    patient_data_df = patient_df[label_columns]
    patient_data_df = patient_data_df.replace(np.nan, -1)  
    #print(patient_data_df.head())
    np_data = patient_data_df.to_numpy()
    if np_data.shape[0] != sequence_length+1:
        for i in range(sequence_length, np_data.shape[0]):
            data.append(np_data[i-sequence_length:i])
            label.append(np_data[i])

data, label = np.array(data), np.array(label)
print(data.shape, label.shape)

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=num_features)
])

lstm_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
lstm_model.fit(data, label, epochs=45, batch_size=4)