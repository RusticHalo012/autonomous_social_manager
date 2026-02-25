def find_best_posting_time(content_vector, engagement_model):
    best_hour = 0
    best_score = -1

    for hour in range(24):
        modified_vector = content_vector.copy()
        modified_vector[0] = hour / 24.0
        score = engagement_model.predict([modified_vector])[0]

        if score > best_score:
            best_score = score
            best_hour = hour

    return best_hour, best_score