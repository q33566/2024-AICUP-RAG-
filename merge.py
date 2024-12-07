import json
import pandas as pd

faq = pd.read_csv('faq.csv')
insurance = pd.read_csv('insurance.csv')
finance = pd.read_csv('finance_512.csv')

# print(faq)
# print(insurance)
# print(finance)

total = pd.concat([faq, insurance, finance])
total['retrieve'] = total['retrieve'].apply(str)


with open('answer.json', 'w') as f:
    json.dump({
        'answers': total.to_dict(orient='records')
    }, f)

