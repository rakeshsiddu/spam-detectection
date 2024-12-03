import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from flask import Flask, render_template, request

# Step 1: Generate a synthetic dataset with ham and spam messages
ham_messages = [
    "Hello, how are you?",
    "Meeting at 3 PM today.",
    "The weather is nice today.",
    "Are we still on for the party tonight?",
    "Please confirm your appointment.",
    "Can I get the report by tomorrow?",
    "Let's catch up over the weekend.",
    "I will be home in 10 minutes."
]

spam_messages = [
    "You have won a $1000 gift card! Claim now!",
    "Congratulations! You have been selected for a free prize!",
    "URGENT: Your account has been compromised. Click here to secure it.",
    "Limited time offer! Buy 1 get 1 free, only today!",
    "Win a free iPhone! Just click here!",
    "You are the lucky winner of a free vacation package!",
    "Exclusive deal: Get a $500 discount, only for today!",
    "Hurry! Last chance to claim your free gift!"
]

data = []
for _ in range(5000):  # Number of "ham" messages
    data.append({"label": "ham", "text": random.choice(ham_messages)})

for _ in range(5000):  # Number of "spam" messages
    data.append({"label": "spam", "text": random.choice(spam_messages)})


df = pd.DataFrame(data)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)


df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),       
    ('tfidf', TfidfTransformer()),           
    ('classifier', MultinomialNB())          
])

model_pipeline.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def classify_email():
    result = None
    if request.method == 'POST':
        
        email_content = request.form['email_content']
        
        prediction = model_pipeline.predict([email_content])[0]
        
        result = "SPAM" if prediction == 1 else "HAM"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
