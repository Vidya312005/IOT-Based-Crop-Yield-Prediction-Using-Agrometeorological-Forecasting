from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv('FinalDataSet.csv')

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Prepare the model
X = data[['N', 'P', 'K', 'SHumidity', 'STempearture', 'AHumidity', 'ATempearture', 'PH', 'Rainfall']]
y = data['Label']
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Product data
products = {
    1: {
        "name": "Crop Monitoring Module",
        "image": "image3.jpg",
        "description": "This module helps in monitoring the crop conditions in real-time.",
        "specs": ["Real-time monitoring", "Weather resistant", "Easy installation"]
    },
    2: {
        "name": "Product 2",
        "image": "image4.jpg",
        "description": "Description of Product 2.",
        "specs": ["Spec 1", "Spec 2", "Spec 3"]
    },
    3: {
        "name": "Product 3",
        "image": "image5.jpeg",
        "description": "Description of Product 3.",
        "specs": ["Spec 1", "Spec 2", "Spec 3"]
    }
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        params = {
            'N': request.form['N'],
            'P': request.form['P'],
            'K': request.form['K'],
            'SHumidity': request.form['SHumidity'],
            'STempearture': request.form['STempearture'],
            'AHumidity': request.form['AHumidity'],
            'ATempearture': request.form['ATempearture'],
            'PH': request.form['PH'],
            'Rainfall': request.form['Rainfall']
        }
        # Convert form data to dataframe
        input_data = pd.DataFrame([params])
        # Predict
        prediction = model.predict(input_data)[0]
        return render_template('predict.html', prediction=prediction)
    return render_template('predict.html')

@app.route('/buy')
def buy():
    return render_template('buy.html')

@app.route('/product/<int:product_id>')
def product(product_id):
    product = products.get(product_id)
    if product:
        return render_template('product.html', 
                               product_name=product['name'], 
                               product_image=product['image'], 
                               product_description=product['description'], 
                               product_specs=product['specs'])
    return "Product not found", 404

if __name__ == '__main__':
    app.run(debug=True)
