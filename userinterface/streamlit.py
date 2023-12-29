import streamlit as st

# Custom HTML and CSS for linear gradient background

def test():
    st.markdown(
        """
        <div id="cookies-popup" style="
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 10px;
            background-color: #333;
            color: white;
            text-align: center;
        ">
            <p>This website uses cookies to ensure you get the best experience.</p>
            <button onclick="acceptCookies()">Accept Cookies</button>
        </div>
        <script>
            function acceptCookies() {
                // Remove the popup from the DOM
                var cookiesPopup = document.getElementById('cookies-popup');
                cookiesPopup.parentNode.removeChild(cookiesPopup);
                // Set a session state variable to remember that the user has accepted cookies
                Streamlit.setComponentValue(true, 'accepted_cookies');
            }
        </script>
        """,
        unsafe_allow_html=True,
    )

html_code = """
<style>

  
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #323232 0%, #3F3F3F 40%, #1C1C1C 150%), linear-gradient(to top, rgba(255,255,255,0.40) 0%, rgba(0,0,0,0.25) 200%);
        background-blend-mode: multiply;
        background-size: cover;
        height: 100vh;
        margin: 0; /* Remove default margin */
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: white; /* Set text color to contrast with the background */
    }
    
    [data-testid="stHeader"] {

	}
    
    
</style>
"""

# Render the HTML
#st.markdown(html_code, unsafe_allow_html=True)

# Streamlit content
st.title("Streamlit with Linear Gradient Background")
st.write("This is an example of Streamlit with a linear gradient background.")
