import pandas as pd

listings = pd.read_csv('listings.csv')

listings = listings.drop(columns=['neighbourhood_cleansed', 'host_has_profile_pic', 'id', 'listing_url', 'scrape_id', 'last_scraped', 'source', 'name', 'description', 'neighborhood_overview', 'picture_url', 'host_id', 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
                                  'host_verifications', 'neighbourhood', 'latitude', 'longitude', 'bathrooms', 'amenities', 'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm', 'calendar_updated', 'calendar_last_scraped', 'license', 'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms'])


listings = listings.dropna()

print(listings.head())

print(listings.info())

print(listings.property_type.value_counts(normalize=True))

listings.host_identity_verified = listings.host_identity_verified.replace({
                                                                          't': 1, 'f': 0})
listings = pd.get_dummies(listings, columns=['neighbourhood_group_cleansed'])
