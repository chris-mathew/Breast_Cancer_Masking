#Initialisation
import streamlit as st
import pandas as pd
from PIL import Image


#st.set_page_config(page_title='BI-RADS Score Determiner')

st.title('BI-RADS Score Determiner')
st.write("Please upload a mammogram image below to recieve a BI-RADS value.")
file = st.file_uploader('Upload Picture', type=['JPEG', 'PNG', 'DICOM'], accept_multiple_files=False)

categories = {
    'B-IRADS Category': ['A', 'B', 'C', 'D'],
    'Description': ['Almost Entirely Fat', 'Scattered Fibroglandular Densities', 'Heterogeneously Dense', 'Extremely Dense'],
    'Cancer Risk': ['Negligible risk Of Cancer', 'Low Risk of Cancer', 'Likely Cancer', 'High Risk of Cancer']
}

df = pd.DataFrame(categories)

# Convert the DataFrame to HTML
table_html = df.to_html(index=False)

# Add custom styles to the header
table_html = table_html.replace('<th>', '<th style="text-align:left; font-weight:bold;">')



if file is not None:
    image = Image.open(file)

    st.header('Mammogram Image:')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.header("BI-RADS Classification:")
    st.write("Using our machine learning model to determine BIR-RADS classification")

    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>1</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.header("Cancer Risk")
    st.write("Using BI-RADS Score to determine the level of cancer risk")

    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>No Cancer</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")

    st.header("BI-RADS Classifications Explained:")

    st.markdown(table_html, unsafe_allow_html=True)

  
    #st.markdown(
     #   """
      #  <div style='width: 100px; height: 100px; background-color: red;'></div>
      #  """, unsafe_allow_html=True
    #)



# Custom HTML and CSS for white background, dark blue headers, and black text

html_code = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: white; /* Set background color to white */
        height: 100vh;
        margin: 0; /* Remove default margin */
        display: flex;
        flex-direction: column;
        align-items: center;
        color: black ; /* Set text color to black */
    }

    [data-testid="stHeader"] {
        background-color: #001F3F; /* Set header background color to dark blue */
        color: black; /* Set header text color to black */
    }

    h1,h2{
    color: #001F3F ;
    }

    

    

</style>
"""

# Render the HTML
st.markdown(html_code, unsafe_allow_html=True)