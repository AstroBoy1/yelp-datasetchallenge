# yelp-datasetchallenge
Title: Where do the Locals Eat? Discovering Hidden Delicacies in Hawaii Using Yelp Reviews
Project members: Crystal Boyce, Michael Omori
Data set: The Yelp Dataset Challenge consists of multiple datasets, including data and metadata on Yelp users and the reviews they make, the metadata and content of Yelp reviews, business profiles on Yelp, and other datasets connecting users to reviews of businesses. As one of the earliest contributors to the crowd sourcing review scene, Yelp has been building its product since 2004, but now faces competition from upstarts like FourSquare and web mega-Giants like Amazon, Facebook, and Google.
Preliminary questions: A question asked by many vacationers or out-of-towners is “where do the locals eat” or “what’s the place you don’t tell the tourists about.” This is not as simple a question as it might seem, as there is both the consideration of distance to said restaurant as well as the idea of the local secret: that place the locals know about but keep “hidden” so as to not allow it to change. Using the data provided by Yelp for the challenge we can begin to look at review networks, considering some of the following questions. This list is by no means exhaustive.
Do user accounts with a “local” affiliation review “local” restaurants more or less than out-of-town visitors? Local here can be defined geographically using zip codes; Oahu zip codes could be queried for all those beginning 968*.
If a restaurant gets a positive review by a “local” is it likely get other positive reviews by “locals”?
Are “local” reviews of restaurants substantively different than visitor reviews? Is one group more or less likely to review a restaurant positively? Can we perform a content analysis on the reviews to see if visitors remark on different restaurant features than locals?
How many positive reviews are required for a restaurant to “get noticed”? Can we use review timestamps to identify an increase in review activity?
Can we use GIS data, such as this dataset for Hawaii, to identify areas that have a high hotel density and compare the local vs. visitor review rates of restaurants in a particular zip code?
Is there such a thing as a “local” diner review for highly trafficked tourist areas like Waikiki and other similar heavily “touristed” areas in San Francisco, Chicago, Los Angeles, New York, Houston, Philadelphia, Phoenix, San Antonio, San Diego, Dallas, and San Jose? 
For restaurants which exist in low density review areas (to be determined after an examination of the data), are there differences in users which review said restaurants or in the content of the reviews? In other words, why might some more rural restaurants get more review traffic than others?

Deliverables
Project proposal (posted to slack) is due on Nov 1, 2018.
Jupyter Notebook (ipynb file) describing your work due on Dec 12, 2018.
Any additional scripts you have used outside of the Python/Jupyter environment
A 10-minute video presentation of your work (posted to youtube as unlisted video and submit the link to laulima)
In-class presentation of the video with live Q & A (Dec 4 & 5, 2018)
Project proposals must include
title
names of project members
brief description of the data set(s) analyzed
initial list of questions to investigate
You should organize your notebook so that the main narrative comes first with additional appendixes giving details. Think of the main narrative as a short 4-6 page article/paper that tells the “story”.
For those who have problems running the processing within Jupyter Notebook, you may run the heavy lifting python code outside of Jupyter Notebook, but you must write up your work in Jupyter Notebook for submission.
Do not submit the data set with your submission, but you need to submit any code/scripts needed to reproduce your analysis.

Technical Plan
Language: Python
Data format: JSON
Database: SQL / NOSQL
Machine Learning
Classification/regression
XGBoost, dense deep neural networks
Statistics
data distributions
summary statistics
probability mass functions and probability density functions
hypothesis testing
confidence intervals
conditional probability and Bayes theorem
Frequentist vs Bayesian statistics
singular value decomposition, linear discriminant analysis
Front end/visuals: Python libraries, along with jupyter notebook
Github for code tracking

Bibliography
"Understanding Hidden Memories of Recurrent Neural Networks" by Yao Ming, Shaozu Cao, Ruixiang Zhang, Zhen Li, Yuanzhe Chen, Yangqiu Song, and Huamin Qu from the Hong Kong University of Science and Technology.
"CORALS: Who are My Potential New Customers? Tapping into the Wisdom of Customers' Decisions" by Ruirui Li, Chelsea J-T Ju, Jyunyu Jiang, and Wei Wang from the Department of Computer Science of the University of California in Los Angeles.
"Clustered Model Adaption for Personalized Sentiment Analysis" by Lin Gong, Benjamin Haines, and Hongnin Wang from the Department of Computer Science of the University of Virginia.

Success Criteria
The questions may be difficult to answer, but we should at least shine light onto the questions and enhance people’s understandings.

Cost
In order to speed up computations, some work will be done using Google Cloud. Estimated costs will be $100.

https://www.yelp.com/dataset/documentation/main
