import flask
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row
from pyngrok import ngrok
 
 
 # Running the flask app
app = Flask(__name__)

#load model
loaded_model_gbtr = PipelineModel.load(r'C:\Users\Manu\Documents\GitHub\Code-Collection\VS CODE\Xebia Summer Internship\gbtr_model')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    int_feature = [float(x) for x in request.form.values()]
    numerical_cols = ['Longitude',
                    'Latitude',
                    'Housing Median Age',
                    'Total Rooms',
                    'Households',
                    'Median Income']
    
    featureDict = dict(zip(numerical_cols, int_feature))

    row_df = Row(**featureDict)
    single_df = spark.createDataFrame([row_df])

    assembler = VectorAssembler(inputCols=numerical_cols, outputCol = "sc_features")
    pipeline = Pipeline(stages=[assembler])

    inputModel = pipeline.fit(single_df)
    transformed_single_df = inputModel.transform(single_df)

    prediction_single_row = loaded_model_gbtr.transform(transformed_single_df)
    predicted_value = prediction_single_row.select('prediction').collect()[0]['prediction']


    return render_template('index.html', prediction_text='The Housing price is {}'.format(predicted_value))

if __name__ == '__main__':
    #creating spark session
    spark = SparkSession \
        .builder \
        .appName('Housing Data') \
        .getOrCreate()

    app.run()
    # Run ngrok to expose the app
    public_url = ngrok.connect(port=5000)
    print(" * Running on", public_url)