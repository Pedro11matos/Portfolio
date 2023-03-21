import pandas as pd


def normalize(col):
    maximum = max(col)
    minimum = min(col)

    for i in range(len(col)):
        col.iloc[i] = (col.iloc[i] - minimum) / (maximum - minimum)

    return col


listings = pd.read_csv('listings.csv')

listings = listings.drop(columns=['first_review', 'last_review', 'has_availability', 'minimum_nights', 'maximum_nights', 'beds', 'property_type', 'neighbourhood_cleansed', 'host_has_profile_pic', 'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
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
listings.availability_30 = normalize(listings.availability_30)
listings.availability_60 = normalize(listings.availability_60)
listings.availability_90 = normalize(listings.availability_90)
listings.availability_365 = normalize(listings.availability_365)
listings.number_of_reviews = normalize(listings.number_of_reviews)
listings.number_of_reviews_ltm = normalize(listings.number_of_reviews_ltm)
listings.number_of_reviews_l30d = normalize(listings.number_of_reviews_l30d)
listings.review_scores_rating = normalize(listings.review_scores_rating)
listings.review_scores_accuracy = normalize(listings.review_scores_accuracy)
listings.review_scores_cleanliness = normalize(
    listings.review_scores_cleanliness)
listings.review_scores_checkin = normalize(listings.review_scores_checkin)
listings.review_scores_communication = normalize(
    listings.review_scores_communication)
listings.review_scores_location = normalize(listings.review_scores_location)
listings.review_scores_value = normalize(listings.review_scores_value)
listings.reviews_per_month = normalize(listings.reviews_per_month)

listings.price = listings.price.str.strip('$')

listings.to_csv('listings_processed.csv', index=False)
