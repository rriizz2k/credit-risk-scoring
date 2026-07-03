from sklearn.datasets import fetch_openml
df = fetch_openml(name="credit-g", version=1, as_frame=True).frame
df.to_csv("german_credit.csv", index=False)