import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def preprocess_data(file_path='Lung Cancer.csv'):
    """Loads and preprocesses the lung cancer dataset."""
    df = pd.read_csv(file_path)
    
    
    df.drop_duplicates(inplace=True)
    le = LabelEncoder()
 
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
    
    return df

def build_and_train_model(X_train, y_train, X_test, y_test):
    """Builds, trains, and evaluates a neural network model."""
    
    
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid') 
    ])
    
   
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("\n--- Model Summary ---")
    model.summary()
    
   
    print("\n--- Training Model ---")
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1)
    
    
    print("\n--- Evaluating Model ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
   
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    return model, history

if __name__ == '__main__':
    data = preprocess_data()
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']
    
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
  
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    trained_model, training_history = build_and_train_model(X_train, y_train, X_test, y_test)