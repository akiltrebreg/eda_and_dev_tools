import pandas as pd


ABALONE_DATASET = "https://raw.githubusercontent.com/aiedu-courses/stepik_eda_and_dev_tools/main/datasets/abalone.csv"
df = pd.read_csv(ABALONE_DATASET)

# для удобства работы с данными изменим названия колонок
df.columns = df.columns.str.lower().str.replace(' ', '_')

# создаем целевую переменную 'age' и удаляем колонку 'rings'
df['age'] = df['rings'] + 1.5
df = df.drop(columns=['rings'])

# заменяем пропуски медианными значениями параметров
median_diameter = df['diameter'].median()
median_whole_weight = df['whole_weight'].median()
median_shell_weight = df['shell_weight'].median()

df['diameter'].fillna(median_diameter, inplace=True)
df['whole_weight'].fillna(median_whole_weight, inplace=True)
df['shell_weight'].fillna(median_shell_weight, inplace=True)

# заменяем ошибочное обозначение женского пола морского ушка в данных
df['sex'] = df['sex'].replace({'f': 'F'})

df.to_csv('preprocessed_data.csv', index=False)
