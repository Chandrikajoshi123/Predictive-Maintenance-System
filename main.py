from tensorflow.keras.preprocessing.sequence import pad_sequences

# Convert to sequences per engine
def create_sequences(data, sequence_length=50):
    sequences = []
    labels = []
    for unit in data['unit_number'].unique():
        unit_data = data[data['unit_number'] == unit]
        for i in range(len(unit_data) - sequence_length):
            seq = unit_data.iloc[i:i+sequence_length][sensor_cols].values
            label = unit_data.iloc[i+sequence_length]['RUL']
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

X_seq, y_seq = create_sequences(train_data)