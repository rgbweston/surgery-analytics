import streamlit as st
import pandas as pd
import os
import google.generativeai as genai

# Set up page
st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Exploration")

# Define your data structures here
DATA_CONFIG = {
    "filepath": "df_cleaned.csv",
    "dtypes": {
        "Year": "int64",
        "Operation_Age": "float64",
        "Sex": "bool",
        "Ethnic_Category": "object",
        "Expected_Length": "int64",
        "Anaesthetic_Type_Code": "object",
        "Theatre_Code": "object",
        "Proc_Code_1_Read": "object",
        "Intended_Management": "bool",
        "IMD_Score": "float64",
        "Actual_Length": "int64",
        "High_Volume_and_Low_Complexity_Category": "object",
        "High_Volume_and_Low_Complex_or_not": "bool",
        "weekday_name": "object",
        "day_working": "bool",
        "Season": "object",
        "Pseudo_Consultant": "category",
        "Pseudo_Surgeon": "category",
        "Pseudo_Anaesthetist": "category",
        "Previous_Operation_Length": "float64",
        "Previous_Proc_1": "object",
        "Heart_Condition": "bool",
        "Hypertension": "bool",
        "Obesity": "bool",
        "Diabetes": "bool",
        "Cancer": "bool",
        "Chronic_Kidney_Disease": "bool",
    },
    "description": """
    This dataset contains surgical operation records with patient demographics, 
    procedure details, and health conditions. It includes information about 
    operation duration, anaesthetic type, theatre details, and various 
    patient comorbidities.
    """,
}

# Initialize Gemini model
@st.cache_resource
def get_gemini_model():
    """Initialize Gemini model with API key"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        genai.configure(api_key=api_key)
        
        # Updated model names for current Gemini versions
        model_names = [
            'gemini-2.0-flash',      # Fast, efficient model (recommended)
            'gemini-2.5-flash',      # Latest flash model
            'gemini-flash-latest',   # Always points to latest flash
            'gemini-2.0-pro-exp',    # Pro experimental
            'gemini-2.5-pro',        # Latest pro model
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                # Quick test to ensure model works
                test_response = model.generate_content("Say 'test'")
                if test_response and test_response.text:
                    st.sidebar.success(f"✅ Using model: {model_name}")
                    return model
            except Exception as e:
                # Try next model
                continue
        
        # If no model works, list available models
        st.error("Could not initialize any Gemini model")
        st.info("Checking available models...")
        
        try:
            available = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available.append(m.name)
            if available:
                st.write("Available models:", available)
            else:
                st.error("No models available with your API key")
        except Exception as e:
            st.error(f"Could not list models: {str(e)}")
        
        st.stop()
        
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        if "API key not valid" in str(e):
            st.info("Please check that your API key is correct")
        st.stop()

# Load data
@st.cache_data
def load_data(config):
    try:
        df = pd.read_csv(config["filepath"])
        if config["dtypes"]:
            df = df.astype(config["dtypes"])
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {config['filepath']}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Generate visualization
def generate_visualization(df, config, user_request):
    """Use Google's Gemini API to generate visualization code"""
    model = get_gemini_model()
    
    # Prepare concise but comprehensive data context
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Get value counts for categorical columns (first 3)
    cat_examples = {}
    for col in categorical_cols[:3]:
        unique_vals = df[col].value_counts().head(3).index.tolist()
        cat_examples[col] = unique_vals
    
    data_context = f"""
You have a pandas DataFrame called 'df' with the following structure:

Shape: {df.shape[0]} rows, {df.shape[1]} columns

Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}
Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}

Sample statistics:
{df[numeric_cols[:5]].describe().round(2).to_string() if numeric_cols else "No numeric columns"}

Categorical examples: {cat_examples}

Data description: {config['description']}
"""

    prompt = f"""You are an expert Python data scientist. Generate visualization code for this request.

{data_context}

User request: {user_request}

CRITICAL Requirements:
1. The DataFrame 'df' is already loaded - do NOT read files or create sample data
2. Import matplotlib.pyplot as plt at the start
3. Create informative, professional visualizations
4. Include proper titles, labels, and formatting
5. Use figsize appropriate for the visualization
6. End with plt.show()
7. Handle missing values appropriately (use .dropna() where needed)
8. Output ONLY executable Python code - no explanations, no markdown, no comments

Generate the complete Python code:"""

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000,
            )
        )
        
        generated_code = response.text
        
        # Clean up the response - remove any markdown or explanations
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1]
            if "```" in generated_code:
                generated_code = generated_code.split("```")[0]
        
        # Remove any non-code lines (explanations)
        lines = generated_code.strip().split('\n')
        code_lines = []
        for line in lines:
            # Skip obvious non-code lines
            if line.strip() and not line.strip().startswith('#'):
                if any(keyword in line for keyword in ['import', 'plt', 'df', 'fig', 'ax', '=', '(', ')', '[', ']', '.', 'for', 'if', 'else']):
                    code_lines.append(line)
            elif line.strip().startswith('#'):
                code_lines.append(line)  # Keep comments
            elif not line.strip():
                code_lines.append(line)  # Keep empty lines for formatting
        
        generated_code = '\n'.join(code_lines)
        
        # Ensure matplotlib is imported
        if "plt" in generated_code and "import matplotlib" not in generated_code:
            generated_code = "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\n" + generated_code
            
        return generated_code.strip()
        
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        if "quota" in str(e).lower():
            st.info("API quota exceeded. Please wait a moment and try again.")
        elif "API key" in str(e):
            st.info("Please check that your API key is valid")
        return None

# Execute code safely
def execute_code(df, generated_code):
    """Execute the generated code with proper error handling"""
    try:
        # Import required libraries
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create execution environment
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
        }
        
        # Execute the generated code
        exec(generated_code, exec_globals)
        
        # Display the plot in Streamlit
        fig = plt.gcf()
        if fig.get_axes():  # Check if plot was created
            st.pyplot(fig)
            return True
        else:
            st.warning("No plot was generated. The code may need adjustment.")
            return False
            
    except Exception as e:
        st.error(f"Error executing code: {str(e)}")
        
        # Show the problematic code for debugging
        with st.expander("🔍 View problematic code"):
            st.code(generated_code, language='python')
            
        # Provide helpful error context
        if "KeyError" in str(e):
            st.info(f"Column not found. Available columns: {', '.join(df.columns[:10])}...")
        elif "ValueError" in str(e):
            st.info("There might be an issue with data types or values. Check if numeric operations are being performed on non-numeric data.")
        elif "TypeError" in str(e):
            st.info("There might be a type mismatch. Check if the operations are appropriate for the data types.")
            
        return False

# Main app
def main():
    # Load data
    df_explore = load_data(DATA_CONFIG)
    
    if df_explore is not None:
        # Sidebar with data information
        with st.sidebar:
            st.header("📊 Data Summary")
            st.write(f"**Rows:** {len(df_explore):,}")
            st.write(f"**Columns:** {len(df_explore.columns)}")
            
            st.subheader("Column Information")
            
            # Group columns by type
            numeric_cols = df_explore.select_dtypes(include=['float64', 'int64']).columns
            bool_cols = df_explore.select_dtypes(include=['bool']).columns
            cat_cols = df_explore.select_dtypes(include=['object', 'category']).columns
            
            if len(numeric_cols) > 0:
                st.write("**Numeric columns:**")
                for col in numeric_cols:
                    st.write(f"• {col}")
                    
            if len(bool_cols) > 0:
                st.write("**Boolean columns:**")
                for col in bool_cols:
                    st.write(f"• {col}")
                    
            if len(cat_cols) > 0:
                st.write("**Categorical columns:**")
                for col in cat_cols:
                    st.write(f"• {col}")
        
        # Main content area
        st.markdown("### 🎯 Explore and analyze the surgical operations dataset")
        
        # Example queries in columns
        st.markdown("#### Example queries you can try:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Distributions & Trends:**
            - Show distribution of Operation_Age
            - Create histogram of Actual_Length
            - Plot IMD_Score distribution
            
            **Categorical Analysis:**
            - Bar chart of procedures by Season
            - Show Anaesthetic_Type_Code frequencies
            - Theatre usage by weekday
            """)
            
        with col2:
            st.markdown("""
            **Comparisons & Relationships:**
            - Average Actual_Length by Season
            - Compare Expected vs Actual Length
            - Correlation heatmap of numeric columns
            
            **Complex Analysis:**
            - Show operation age by health conditions
            - Analyze delays (Actual - Expected Length)
            - Weekend vs weekday operations
            """)
        
        # Input area
        st.markdown("---")
        user_request = st.text_area(
            "🔍 **What would you like to explore?**",
            height=100,
            placeholder="Enter your data exploration request... (e.g., 'Show the relationship between age and operation length')"
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            generate_button = st.button("🚀 Generate Visualization", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.rerun()
        
        if generate_button:
            if not user_request.strip():
                st.warning("⚠️ Please enter a visualization request")
            else:
                with st.spinner("🤖 Generating visualization code..."):
                    generated_code = generate_visualization(df_explore, DATA_CONFIG, user_request)
                    
                    if generated_code:
                        # Show generated code
                        with st.expander("📝 Generated Code", expanded=False):
                            st.code(generated_code, language="python")
                        
                        # Execute and display visualization
                        st.markdown("### 📈 Visualization Result")
                        if execute_code(df_explore, generated_code):
                            st.success("✅ Visualization generated successfully!")
                        else:
                            st.info("💡 Try rephrasing your request or check the error message above")
                    else:
                        st.error("❌ Failed to generate code. Please check your API key and try again.")
                        st.info("Make sure you have set the GEMINI_API_KEY environment variable")
    else:
        st.error("❌ Unable to load data. Please check that 'df_cleaned.csv' exists in the current directory.")

if __name__ == "__main__":
    main()