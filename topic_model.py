import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

texts = [
    "I love football and cricket.",
    "The government passed a new policy today.",
    "Python and machine learning are amazing.",
    "The match was exciting and full of energy.",
    "AI models improve with more data and training."
]

#cleaned = [re.sub(r"[^a-zA-Z]", " ", t).lower() for t in texts]
cleaned = [re.sub(r"[^a-zA-Z]", " ", t).lower().replace("ai", "artificial intelligence") for t in texts]


vect = CountVectorizer(stop_words='english')
X = vect.fit_transform(cleaned)

#lda = LatentDirichletAllocation(n_components=2, random_state=42).fit(X)
lda = LatentDirichletAllocation(n_components=5, random_state=7).fit(X)

feature_names = vect.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Feature branch version: Topic {idx}")

