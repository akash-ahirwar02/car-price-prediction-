# -*- coding: utf-8 -*-


from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle


app=Flask(__name__)



mdc = pickle.load(open("Car_model2000.pkl", "rb"))

data =pd.read_csv('car_data_clean2.csv')

@app.route('/')
def index():
  
        
        return render_template('index2.html',)
    
@app.route('/predict',methods=['POST'])

def predict():
       
        company=request.form.get('company')
        model=request.form.get('Model')
        year=(request.form.get('year'))
        fuel =request.form.get('fuel')
        driven=(request.form.get('km driven'))
        
        output = mdc.predict(pd.DataFrame(columns=['name', 'year', 'km_driven', 'fuel', 'Company'],
                                              data=np.array([model,year,driven,fuel,company]).reshape(1,5)))
        
        
   

        output2 = output[0].round(0)
        return render_template('index2.html', prediction='You can Sell this Car in {} Rupees '.format(output2))

if __name__ == "__main__":
   
    app.run()



    
    
    
    
    
    
    
    
    
    
    
    
    
    