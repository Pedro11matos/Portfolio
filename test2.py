import pandas as pd
import matplotlib.pyplot as plt

listings = pd.read_csv('listings_processed.csv')

print(listings['calculated_host_listings_count'].describe())

plt.scatter(listings['calculated_host_listings_count'], listings['price'], alpha=0.4)
plt.show()

print(listings.columns)
