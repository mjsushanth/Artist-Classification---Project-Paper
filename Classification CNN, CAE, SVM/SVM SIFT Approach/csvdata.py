import pickle
import pandas as pd
import load_data
df = pd.read_csv('project_data/artists.csv')
# # print(df.head())    


# genres = []

# for genre in df['genre']:
#     if ',' in genre:
#         genres.extend(genre.split(','))
#     else:
#         genres.append(genre)


# genres = list(set(genres))


# # names = df['name']

# # names = [name.replace(' ', '_') for name in names]

# artists_list = load_data.load_artist_names_list()
# # print(artists_list)
# df['name'] = artists_list
# # print(df[['name', 'genre']])

# df_genres = pd.DataFrame(columns = ['name', 'genre'].extend(genres))
# df_genres['name'] = artists_list
# df_genres['genre'] = df['genre']

# for genre in genres:
#     df_genres[genre] = df_genres['genre'].apply(lambda x: 1 if genre in x else 0)

# # print(df_genres)

# artist_genre_map = {}

# for i in range(0,50):
#     artist_genre_map[i] = list(df_genres.loc[i][2:])


# print(artist_genre_map)
# pickle.dump(artist_genre_map, open('artist_genre_map.pkl', 'wb'))

# train_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))

# train_genres = []

# for label in train_labels:
#     train_genres.append(artist_genre_map[label])

# pickle.dump(train_genres, open('training_data/train_genres.pkl', 'wb'))

artist_genre_map = pickle.load(open('artist_genre_map.pkl', 'rb'))
test_genres = []

test_labels = pickle.load(open('testing_data/test_labels.pkl', 'rb'))
for label in test_labels:
    test_genres.append(artist_genre_map[label])

pickle.dump(test_genres, open('testing_data/test_genres.pkl', 'wb'))