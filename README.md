# Flight Booking Analysis
Introduction

This project aims to analyze a flight booking dataset obtained from the “Ease My Trip” website and conduct various statistical hypothesis tests to extract meaningful information. The 'Linear Regression' algorithm is used to train the dataset and predict the continuous target variable, ticket price. 'Ease My Trip' is an internet platform for booking flight tickets, used by potential passengers to purchase tickets. A thorough study of the data will aid in the discovery of valuable insights beneficial to passengers.

Research Questions

The study aims to answer the following research questions:

Does price vary with Airlines?

How is the price affected when tickets are bought just 1 or 2 days before departure?

Does the ticket price change based on the departure and arrival times?

How does the price change with changes in Source and Destination?

How does the ticket price vary between Economy and Business class?

Data Collection and Methodology

Data was extracted from the Ease My Trip website using the Octoparse scraping tool. The data was collected in two parts: one for economy class tickets and another for business class tickets, over 50 days (from February 11th to March 31st, 2022). The dataset consists of 300,261 distinct flight booking options.

# Dataset
The dataset contains information about flight booking options for travel between India's top 6 metro cities. It includes 300,261 datapoints and 11 features:

Airline: The name of the airline company (categorical, 6 unique airlines).

Flight: Information regarding the plane's flight code (categorical).

Source City: City from which the flight takes off (categorical, 6 unique cities).

Departure Time: Time of departure, grouped into 6 time bins (categorical).

Stops: Number of stops between source and destination cities (categorical, 3 distinct values).

Arrival Time: Time of arrival, grouped into 6 time bins (categorical).

Destination City: City where the flight lands (categorical, 6 unique cities).

Class: Seat class (categorical, 2 distinct values: Business and Economy).

Duration: Overall travel time between cities in hours (continuous).

Days Left: Calculated by subtracting the booking date from the trip date (derived feature).

Price: Ticket price (target variable).
