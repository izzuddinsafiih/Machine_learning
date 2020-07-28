import pandas as pd
from apyori import apriori

df = pd.read_csv('xlxs/datasets_264386_555058_groceries - groceries.csv')
df = df.iloc[1:, 1:34].values
n_rows, n_columns = df.shape

transactions = []
for i in range(n_rows):
    transactions.append([str(df[i, j]) for j in range(n_columns)])
max_length = 7
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.3, min_lift=3, min_length=2,
                max_length=max_length)
results = list(rules)
item_tuple = [tuple(results[0]) for results in results]


def inspect(results):
    first = [items[0] for items in item_tuple]
    second = [items[1] for items in item_tuple]
    third = [items[2] if len(items) > 2 else 'nan' for items in item_tuple]
    fourth = [items[3] if len(items) > 3 else 'nan' for items in item_tuple]
    fifth = [items[4] if len(items) > 4 else 'nan' for items in item_tuple]
    sixth = [items[5] if len(items) > 5 else 'nan' for items in item_tuple]
    seventh = [items[6] if len(items) > 6 else 'nan' for items in item_tuple]
    support = [results[1] for results in results]
    confidence = [results[2][0][2] for results in results]
    lift = [results[2][0][3] for results in results]
    return list(zip(first, second, third, fourth, fifth, sixth, seventh, support, confidence, lift))


columns_names = [f'Item {n + 1}' for n in range(max_length)]
columns_names.append('Support')
columns_names.append('Confident')
columns_names.append('Lift')
resultInArray = pd.DataFrame(inspect(results), columns=columns_names)
# print(resultInArray)

resultInArray.nlargest(n=10, columns='Lift')
