import numpy as np
import pandas as pd

def generate_mock_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        "likes": np.random.randint(0, 1000, n),
        "comments": np.random.randint(0, 300, n),
        "shares": np.random.randint(0, 200, n),
        "hour": np.random.randint(0, 24, n),
        "content_type": np.random.randint(0, 5, n),
        "sentiment": np.random.uniform(-1, 1, n)
    })

    data["engagement"] = (
        0.4 * data["likes"] +
        0.3 * data["comments"] +
        0.3 * data["shares"]
    ) / 1000

    return data