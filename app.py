from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

model = pickle.load(open("customer_classifier.sav","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    inp = [float(x) for x in request.form.values()]
    print(inp)
    inp_ = [np.array(inp)]
    result = model.predict(inp_)
    return render_template("index.html", prediction_text="This customer belongs to cluster {}".format(result[0]))


if __name__ == "__main__":
    app.run(debug=True)
