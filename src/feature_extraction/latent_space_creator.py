from tensorflow.keras.models import Model, load_model # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import cv2
import os
import re
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout  # type: ignore
from sklearn.model_selection import train_test_split

# 1. Load images
def get_images_and_filenames(path):
    image_list = []
    filename_list = []

    for filename in os.listdir(path):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img / 255.0
            image_list.append(img)
            filename_list.append(filename)
    
    # Extract numeric parts for sorting
    filename_list = [int(re.findall(r'\d+', s)[0]) for s in filename_list]
    X_train = np.array(image_list, dtype=np.float32)
    return X_train, filename_list

# 2. Build Autoencoder
def build_autoencoder(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.4, name="dropout_4")(x)

    # Decoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, output_img)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# 3. Train Autoencoder
def train_autoencoder(X_train):
    autoencoder = build_autoencoder()
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    autoencoder.fit(X_train, X_train, epochs=30, batch_size=32, shuffle=True, validation_data=(X_val, X_val))
    autoencoder.save("autoencoder.h5")
    return autoencoder

# 4. Create Encoder
def create_encoder(model_path):
    autoencoder = load_model(model_path)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dropout_4').output)
    return encoder

# 5. Extract Latent Spaces
def get_latent_spaces(data, encoder):
    latent_spaces = []
    for img in data:
        img = np.expand_dims(img, axis=0)
        latent_space = encoder.predict(img, verbose=0)
        flattened_latent_space = np.reshape(latent_space, (-1))
        latent_spaces.append(flattened_latent_space)
    return np.array(latent_spaces)

# 6. Scale and PCA
def scale_and_pca(latent_spaces, n_components=100):
    scaler = StandardScaler()
    latent_spaces_scaled = scaler.fit_transform(latent_spaces)
    pca = PCA(n_components=n_components)
    latent_spaces_pca = pca.fit_transform(latent_spaces_scaled)
    return latent_spaces_pca

# 7. Save to HDF5 with string conversion
def save_to_dataframe(file_names, latent_spaces_pca, output_path):
    df = pd.DataFrame({
        'path': file_names,
        'latent_space': [','.join(map(str, vec)) for vec in latent_spaces_pca]
    })
    df.to_hdf(output_path, key='df_items', mode='w', format='table')
    print(f"Saved to {output_path} successfully!")

# 8. Read HDF5 and decode
def load_dataframe(input_path):
    df = pd.read_hdf(input_path, key='df_items')
    df['latent_space'] = df['latent_space'].apply(lambda x: np.array(x.split(','), dtype=float))
    return df

# 9. Main pipeline
def main():
    IMAGES_PATH = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/images/segmented_images"
    MODEL_PATH = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/models/autoencoder.h5"
    OUTPUT_PATH = "C:/Users/ADMIN/Desktop/ITDSIU21099_HoangVanManh/Fashion-Marketing-Automation-Solutions/data/processed/latent_spaces.h5"
    
    print("🔄 Loading images...")
    X_train, file_names = get_images_and_filenames(IMAGES_PATH)
    
    if not os.path.exists(MODEL_PATH):
        print("🟢 Training Autoencoder...")
        train_autoencoder(X_train)
    else:
        print("✅ Model found, skipping training.")

    print("🟣 Creating encoder...")
    encoder = create_encoder(MODEL_PATH)

    print("⚙️ Extracting latent spaces...")
    latent_spaces = get_latent_spaces(X_train, encoder)

    print("📊 Scaling and applying PCA...")
    latent_spaces_pca = scale_and_pca(latent_spaces)

    print("💾 Saving results...")
    save_to_dataframe(file_names, latent_spaces_pca, OUTPUT_PATH)

    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
