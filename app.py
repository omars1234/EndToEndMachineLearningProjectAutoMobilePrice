from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os
from AutoMobilePriceRegression.pipeline.stage_05_prediction import PredictionPipeline



app=Flask(__name__)


@app.route('/',methods=['GET'])
def homepage():
    return render_template('index.html')



@app.route('/train',methods=["GET"])
def training():
    os.system("python main.py")
    return ("Training Successful")




@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            num_of_doors=str(request.form['num_of_doors'])
            body_style=str(request.form['body_style'])
            drive_wheels=str(request.form['drive_wheels'])
            engine_location=str(request.form['engine_location'])
            length=int(request.form['length'])
            width=int(request.form['width'])
            height=int(request.form['height'])
            curb_weight=int(request.form['curb_weight'])
            num_of_cylinders=str(request.form['num_of_cylinders'])
            engine_size=int(request.form['engine_size'])
            fuel_system=str(request.form['fuel_system'])
            peak_rpm=int(request.form['peak_rpm'])
            city_mpg=int(request.form['city_mpg'])
            highway_mpg=int(request.form['highway_mpg'])
      
         
            data = [num_of_doors,body_style,drive_wheels,engine_location,num_of_cylinders,fuel_system,length,width,height,
                    curb_weight,engine_size,peak_rpm,city_mpg,highway_mpg]
            
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')






if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)

