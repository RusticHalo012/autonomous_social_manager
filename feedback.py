import numpy as np
import torch


def update_model(autoencoder, scaler, engagement_model,
                 compressed_features, y, new_data_point):

    X_new = scaler.transform([new_data_point[:-1]])
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

    with torch.no_grad():
        compressed = autoencoder.encoder(X_new_tensor).numpy()

    engagement_model.fit(
        np.vstack([compressed_features, compressed]),
        np.append(y, new_data_point[-1])
    )

    return engagement_model