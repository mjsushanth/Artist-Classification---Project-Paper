import pickle 
import pandas as pd
import load_data

# df = pd.read_csv('project_data/artists.csv')

# nationalities = []

# for nationality in df['nationality']:
#     if ',' in nationality:
#         nationalities.extend(nationality.split(','))
#     else:
#         nationalities.append(nationality)

# nationalities = list(set(nationalities))


# df['name'] = df['name'].apply(lambda x: x.replace(' ', '_'))


# df_nationalities = df[['name', 'nationality']]

# # df_nationalities = pd.DataFrame(columns = ['name', 'nationality'].extend(nationalities))

# print(df_nationalities['nationality'])


# for nationality in nationalities:
#     print(nationality)
#     df_nationalities[nationality] = df_nationalities['nationality'].apply(lambda x: 1 if nationality in x else 0)

# print(df_nationalities.head())

# df_nationalities.to_csv('project_data/artist_nationalities.csv')


artist_genre_map = pickle.load(open('artist_genre_map.pkl', 'rb'))
train_labels = pickle.load(open('training_data/train_labels.pkl', 'rb'))
test_labels = pickle.load(open('testing_data/test_labels.pkl', 'rb'))

train_nationalities = []
test_nationalities = []

for label in train_labels:
    train_nationalities.append(artist_genre_map[label])

for label in test_labels:
    test_nationalities.append(artist_genre_map[label])

pickle.dump(train_nationalities, open('training_data/train_nationalities.pkl', 'wb'))
pickle.dump(test_nationalities, open('testing_data/test_nationalities.pkl', 'wb'))

# artists_list = load_data.load_artist_names_list()

