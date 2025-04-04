# Styling utilities
def get_custom_css():
    """
    Get custom CSS for the dashboard
    """
    return """
        <style>
            /* Main background and text colors */
            .main, .block-container {
                background-color: #111111;
                color: #f0f0f0;
                font-family: 'Roboto', sans-serif;
            }
            
            /* Sidebar styling */
            .sidebar .sidebar-content {
                background-color: #000000;
                color: #ffffff;
            }
            
            /* Headers styling */
            h1, h2, h3 {
                color: #ffffff;
                font-family: 'Roboto', sans-serif;
                font-weight: 300;
            }
            
            /* Card styling */
            div.css-1r6slb0.e1tzin5v2 {
                background-color: #1c1c1c;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 10px;
                margin: 10px 0px;
            }
            
            /* Button styling */
            .stButton>button {
                background-color: #ffffff;
                color: #000000;
                border-radius: 3px;
                border: none;
                font-weight: 500;
            }
            
            .stButton>button:hover {
                background-color: #cccccc;
                color: #000000;
            }
            
            /* Metric styling */
            div.css-12w0qpk.e1tzin5v1 {
                background-color: #1c1c1c;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 10px;
            }
            
            /* Plot background */
            .js-plotly-plot .plotly {
                background-color: #1c1c1c !important;
            }
            
            /* Tables */
            .dataframe {
                background-color: #1c1c1c;
                color: #f0f0f0;
            }
            
            /* Widget labels */
            .css-2trqyj {
                color: #f0f0f0;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 5px;
                height: 5px;
            }
            
            ::-webkit-scrollbar-track {
                background: #111111;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #333333;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #555555;
            }
            
            /* Remove Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """

def apply_custom_css():
    """
    Apply custom CSS to the Streamlit app
    """
    import streamlit as st
    
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def create_impact_badge(impact):
    """
    Create an HTML badge for impact level
    """
    color = "#ff4444" if impact == "HIGH" else "#ff9900" if impact == "MEDIUM" else "#ffdd00"
    
    html = f"""
        <div style="
            background-color: {color};
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            color: white;
            margin-top: 20px;
        ">
        <strong>{impact} IMPACT</strong>
        </div>
    """
    
    return html