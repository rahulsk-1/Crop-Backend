from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import pickle
import numpy as np
import math
from flask_cors import CORS
from geopy.geocoders import Nominatim
import json

app = Flask(__name__)
CORS(app)
geolocator=Nominatim(user_agent="my_crop_recommendation")


app.config["MONGO_URI"] = "mongodb+srv://kewalramanirahul15:Rahul123@cluster0.cqdo3.mongodb.net/crop?retryWrites=true&w=majority&appName=cluster0"
app.config["JWT_SECRET_KEY"] = "your_secret_key"

mongo = PyMongo(app) 
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

Jmodel = pickle.load(open('jmodel.pkl','rb'))
Wmodel = pickle.load(open('wmodel.pkl','rb'))
Cmodel = pickle.load(open('cmodel.pkl','rb'))
Smodel = pickle.load(open('smodel.pkl','rb'))
Bmodel = pickle.load(open('bmodel.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

db = mongo.db
users = db.users



@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if users.find_one({"username": username}):
        return jsonify({"error": "Username already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    users.insert_one({"username": username, "password": hashed_password})

    return jsonify({"message": "Signup successful"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    user = users.find_one({"username": username})
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(identity=username)
    return jsonify({"user":username,"token": access_token}), 200

@app.route("/api", methods=["GET"])
# @jwt_required()
def protected():
    return jsonify({"message": "Hello current_user"}), 200



@app.route("/crop_predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid request, JSON expected"}), 400

        N = data.get("nitrogen")
        P = data.get("phosphorus")
        K = data.get("potassium")
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        ph = data.get("ph")
        rain = data.get("rainfall")

        print([N,P,K,temperature,humidity,ph,rain])
        if None in [N, P, K, temperature, humidity, ph, rain]:
            return jsonify({"error": "Missing required input fields"}), 400

        features = np.array([[N, P, K, temperature, humidity, ph, rain]])

        prediction = model.predict(features)[0]  # Predicted class
        probabilities = model.predict_proba(features)[0]  # Probabilities for all classes

        # Map probabilities to class labels
        class_labels = model.classes_  # Get the class names
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
        prob_dict = {
            crop_dict[int(model.classes_[i])]: round(float(probabilities[i]) * 100, 4)
            for i in range(len(probabilities)) if probabilities[i] > 0
        }

        return jsonify({"prob_dict":prob_dict,"crop":crop_dict[int(prediction)]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    



def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in km

@app.route("/crop_count", methods=["POST"])
def crop_count():
    try:
        data = request.json
        address = data["address"]
        # lat = data.get("latitude")
        # lon = data.get("longitude")
        threshold = data.get("threshold")  # Distance in km
        print(address)

        if address is None or threshold is None:
            return jsonify({"error": "Missing required parameters"}), 400

        predictions = db.predictions.find({}, {"latitude": 1, "longitude": 1, "crop_name": 1, "_id": 0})

        crop_counts = {}
        location=geolocator.geocode(address)
        lat=location.latitude
        lon=location.longitude

        for record in predictions:
            record_lat = record["latitude"]
            record_lon = record["longitude"]
            crop_name = record["crop_name"]

            distance = haversine(lat, lon, record_lat, record_lon)

            if distance <= threshold:
                crop_counts[crop_name] = crop_counts.get(crop_name, 0) + 1

        return jsonify({"crop_counts": crop_counts,"address":location.address,"latitude":lat,"longitude":lon}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/add_details", methods=["POST"])
def add_detail():
    try:
        data = request.json
        crop_name = data.get("crop_name")
        username = data.get("username")
        lat = data.get("latitude")
        lon = data.get("longitude")

        if not (crop_name and username and lat and lon):
            return jsonify({"error": "Missing required parameters!"}), 400

        existing_record = mongo.db.predictions.find_one({
            "username": username, 
            "latitude": lat, 
            "longitude": lon,
            "crop_name":crop_name
        })

        if existing_record:
            return jsonify({"error": "Crop information already exists!"}), 400

        record = mongo.db.predictions.insert_one({
            "username": username, 
            "latitude": lat, 
            "longitude": lon, 
            "crop_name": crop_name
        })

        return jsonify({"message": "Crop details added successfully!", "inserted_id": str(record.inserted_id)}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/result', methods=['POST'])
def result():
    try:
        if request.method == 'POST':
            try:
                data = request.json
                commoditytype = data['commodityname']
                month = data['month']
                Year = data['year']
                NextYear = int(Year) + 1
                average_rain_fall = data['average_rain_fall']
            except Exception as e:
                return jsonify({"error": f"Invalid input data: {str(e)}"}), 400

            print(commoditytype)
            avgPriceyear, mspyear, mspnextyear, avgPriceNextyear = [], [], [], []
            months_labels = ["jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]
            monthcount = 1
            yearcount = 2021

            rainfall2024 = [1,2,3,1,8,673,1318,779,408,106,44,8]
            rainfall2023 = [90,7.2,29.9,41.4,67.6,148.6,315.9,162.7,190.3,50.8,9.3,8]
            x_count = 0

            cropimges = ['jowarlogo.webp','wheatlogo.avif','cottonlogo.jpg','sugarcanelogo.jpg','bajralogo.jpg']

            try:
                features = np.array([[month, Year, average_rain_fall]], dtype=object)
                transformed_features = preprocessor.transform(features)
            except Exception as e:
                return jsonify({"error": f"Feature transformation failed: {str(e)}"}), 500

            try:
                if commoditytype == "Jowar":
                    cropface = cropimges[0]
                    model = Jmodel
                    min_factor, max_factor = 1550, 2970
                elif commoditytype == "Wheat":
                    cropface = cropimges[1]
                    model = Wmodel
                    min_factor, max_factor = 1350, 2125
                elif commoditytype == "Cotton":
                    cropface = cropimges[2]
                    model = Cmodel
                    min_factor, max_factor = 3600, 6080
                elif commoditytype == "Sugarcane":
                    cropface = cropimges[3]
                    model = Smodel
                    min_factor, max_factor = 2250, 2775
                elif commoditytype == "Bajara":
                    cropface = cropimges[4]
                    model = Bmodel
                    min_factor, max_factor = 1175, 2350
                else:
                    return jsonify({"error": "Invalid commodity type"}), 400

                prediction = model.predict(transformed_features).reshape(1, -1)
                predicted_value = round(prediction[0][0], 3)
                min_value = round((predicted_value * min_factor) / 100, 2)
                max_value = round((predicted_value * max_factor) / 100, 2)
                avg_value = round((min_value + max_value) / 2, 2)

                for x in rainfall2023:
                    features = np.array([[monthcount, yearcount, x]], dtype=object)
                    transformed_features = preprocessor.transform(features)
                    prediction = model.predict(transformed_features).reshape(1, -1)
                    predicted_value = round(prediction[0][0], 3)
                    mspyear.append(round((predicted_value * max_factor) / 100, 2))
                    avgPriceyear.append(round((predicted_value * min_factor) / 100, 2))
                    monthcount += 1
                    x_count += 1

                for x in rainfall2024:
                    features = np.array([[monthcount, yearcount, x]], dtype=object)
                    transformed_features = preprocessor.transform(features)
                    prediction = model.predict(transformed_features).reshape(1, -1)
                    predicted_value = round(prediction[0][0], 3)
                    mspnextyear.append(round((predicted_value * max_factor) / 100, 2))
                    avgPriceNextyear.append(round((predicted_value * min_factor) / 100, 2))
                    x_count += 1

            except Exception as e:
                return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

            try:
                maxmspyear = max(mspyear)
                maxavgPriceyear = max(avgPriceyear)
                minmspyear = min(mspyear)
                minavgPriceyear = min(avgPriceyear)
                goldmonthindex = mspyear.index(maxmspyear) + 1
                silvermonthindex = mspyear.index(minmspyear) + 1
            except Exception as e:
                return jsonify({"error": f"Post-processing failed: {str(e)}"}), 500

            return jsonify({
                "prediction": predicted_value,
                "cropface": cropface,
                "min_value": min_value,
                "max_value": max_value,
                "avg_value": avg_value,
                "year": Year,
                "NextYear": NextYear,
                "month": month,
                "maxhigh": maxmspyear,
                "maxlow": maxavgPriceyear,
                "minhigh": minmspyear,
                "minlow": minavgPriceyear,
                "goldmonth": goldmonthindex,
                "silvermonth": silvermonthindex,
                "months_labels": months_labels,
                "mspyear": json.dumps(mspyear),
                "minPriceYear": json.dumps(avgPriceyear),
                "mspnextyear": json.dumps(mspnextyear),
                "minPriceNextYear": json.dumps(avgPriceNextyear)
            }), 200
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500



if __name__ == "__main__":
    app.run()
