import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('log_pickle.pkl','rb'))
 
# creating function for prediction

def crop_predict(data1):
    
    data1 = np.array([[9,1,28,1300,130,3,7]])   
    input_data = data1.reshape(1,-1)
    prd1 = loaded_model.predict(input_data)
    
    
    crop_probability = trained_model.predict_proba([[State,Season,AreaHectars,AvgTemp,WindSpeed,Precipitation,Humidity,SoilType,N,P,K]])
    predicted_values = crop_probability.tolist()
    sorted_label = np.flip(np.argsort(predicted_values))[0]
    return [crop_name_dict.get(sorted_label[i]) for i in range(3)]
        
def main():
    st.title("Crop Prediction Web App")
    # Getting input data from user
    
    State= st.selectbox('Enter district' ,('Andhra Pradesh','Arunachal Pradesh','Assam','Bihar','Chattisgarh','Goa','Gujarat','Haryana',
                                           'Himachal Pradesh','Jammu & Kashmir','Jharkhand','Karnataka','Kerala','Madhya Pradesh','Maharashtra',
                                           'Manipur','Meghalaya','Mizoram','Nagaland','Odisha','Punjab','Rajasthan','Sikkim','Tamil Nadu','Telangana',
                                           'Uttar Pradesh','Uttarakhand','West Bengal','A & N Islands','D & N Haveli','Delhi'))
    Season= st.selectbox('Enter season',('Rabi','Kharif'))
    st.write('Selected season:' , Season)
    AvgTemp = st.text_input('AvgTemp')
    Rainfall= st.text_input('Rainfall')    
    Fertilizer= st.text_input('Fertilizer')
    PhValue= st.text_input('PhValue')
    Soiltype=st.selectbox('Enter Soiltype',('alluvial soil','clay soil','sandy soil','black soil','sandysoil or black soil','laterite soil','sandy soil or clay soil','red laterite soil'))
 
    
    Crops = " "
    if st.button("Crop"):
        Crops = crop_predict([State,Season,AvgTemp,Rainfall,Fertilizer,PhValue,Soiltype])
    st.success(Crops)
    
if __name__ == '__main__':
	main()
