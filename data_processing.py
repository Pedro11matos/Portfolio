import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def normalize(col):
    maximum = max(col)
    minimum = min(col)

    for i in range(len(col)):
        col.iloc[i] = (col.iloc[i] - minimum) / (maximum - minimum)

    return col


listings = pd.read_csv('listings.csv')

listings = listings.drop(columns=['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'reviews_per_month', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'availability_30', 'availability_60', 'availability_90', 'first_review', 'last_review', 'has_availability', 'minimum_nights', 'maximum_nights', 'beds', 'property_type', 'neighbourhood_cleansed', 'host_has_profile_pic', 'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
                                  'host_verifications', 'neighbourhood', 'latitude', 'longitude', 'bathrooms', 'amenities', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'calendar_last_scraped', 'license', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms'])


listings = listings.dropna()

print(listings.head())

print(listings.info())

listings.host_identity_verified = listings.host_identity_verified.replace({
                                                                          't': 1, 'f': 0})

listings.instant_bookable = listings.instant_bookable.replace({'t': 1, 'f': 0})

listings.bathrooms_text = listings.bathrooms_text.str.strip(
    'baths shared S half- priv P H').replace('', 0)

listings.bathrooms_text = listings.bathrooms_text.apply(float)

bins = [-1, 0, 1.5, 2.5, 18]
labels = ['0', '1', '2', '3+']
listings.bathrooms_text = pd.cut(
    listings.bathrooms_text, bins=bins, labels=labels)

listings = pd.get_dummies(
    listings, columns=['neighbourhood_group_cleansed', 'room_type', 'bathrooms_text'])

listings.accommodates = normalize(listings.accommodates)
listings.bedrooms = normalize(listings.bedrooms)
listings.availability_365 = normalize(listings.availability_365)
listings.number_of_reviews = normalize(listings.number_of_reviews)
listings.review_scores_rating = normalize(listings.review_scores_rating)
listings['price'] = listings['price'].str.replace(',', '')
listings['price'] = listings['price'].str.replace('$', '')
listings['price'] = listings['price'].astype(float)
listings.calculated_host_listings_count = normalize(
    listings.calculated_host_listings_count)

# calculate z-scores for all numerical columns in the dataframe
z_scores = np.abs(stats.zscore(listings.select_dtypes(include=np.number)))

# filter out rows where any z-score exceeds a threshold value (e.g. 3)
threshold = 2
listings = listings[(z_scores < threshold).all(axis=1)]

listings.to_csv('listings_processed.csv', index=False)

print(listings.info())

# Separate the target variable from the independent variables
X = listings.drop('price', axis=1)
y = listings['price']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)

reg = LinearRegression(n_jobs=-1)

# Fit the linear regression model to the data
reg.fit(x_train, y_train)

# Print the coefficients of the regression model
print('Coefficients:', reg.coef_)

y_pred = reg.predict(x_test)

plt.scatter(y_test, y_pred)
plt.plot(range(1000), range(1000))

plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Actual Rent vs Predicted Rent")

plt.show()

print(reg.score(x_test, y_test))
