import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai

# Set up page
st.set_page_config(page_title="Surgical Analytics Dashboard", layout="wide")

# Define your data structures here
DATA_CONFIG = {
    "filepath": "df_cleaned_filtered.csv",  # Filtered to exclude 2010-2011 (data quality issues)
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

SHAP_DIR = "shap_data_enhanced"  # Using filtered data (2012+, excludes poor quality 2010-2011 data)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def get_gemini_model():
    """Initialize Gemini model with API key"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        genai.configure(api_key=api_key)
        
        model_names = [
            'gemini-2.0-flash',
            'gemini-2.5-flash',
            'gemini-flash-latest',
            'gemini-2.0-pro-exp',
            'gemini-2.5-pro',
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                test_response = model.generate_content("Say 'test'")
                if test_response and test_response.text:
                    return model
            except Exception as e:
                continue
        
        st.error("Could not initialize any Gemini model")
        st.stop()
        
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        st.stop()

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

@st.cache_data
def load_shap_global():
    """Load global SHAP data from enhanced format"""
    try:
        # Load enhanced SHAP data format
        shap_reg = np.load(f"{SHAP_DIR}/duration_shap.npy")
        shap_clf = np.load(f"{SHAP_DIR}/overrun_shap.npy")
        X_test = pd.read_csv(f"{SHAP_DIR}/X_test.csv", index_col=0)
        X_test_enc = pd.read_csv(f"{SHAP_DIR}/X_test_encoded.csv", index_col=0)

        # Load predictions
        predicted_duration = np.load(f"{SHAP_DIR}/predicted_duration.npy")
        predicted_overrun_prob = np.load(f"{SHAP_DIR}/predicted_overrun_prob.npy")
        actual_duration = np.load(f"{SHAP_DIR}/actual_duration.npy")
        actual_overrun = np.load(f"{SHAP_DIR}/actual_overrun.npy")

        # Create predictions dataframe
        predictions = pd.DataFrame({
            'Pred_Duration': predicted_duration,
            'Overrun_Prob': predicted_overrun_prob,
            'Actual_Length': actual_duration,
            'Overrun_Actual': actual_overrun
        })

        # Load metadata
        with open(f"{SHAP_DIR}/metadata.json", "r") as f:
            metadata_json = json.load(f)

        feature_names = metadata_json["feature_names"]
        expected_values = {
            'regression': metadata_json["expected_duration"],
            'classification': metadata_json["expected_overrun_prob"]
        }

        # Calculate metrics
        from sklearn.metrics import roc_auc_score
        metadata = {
            'reg_mae': np.abs(predicted_duration - actual_duration).mean(),
            'reg_r2': 1 - np.sum((actual_duration - predicted_duration)**2) / np.sum((actual_duration - actual_duration.mean())**2),
            'clf_accuracy': ((predicted_overrun_prob > 0.5).astype(int) == actual_overrun).mean(),
            'clf_auc': roc_auc_score(actual_overrun, predicted_overrun_prob)
        }

        return {
            "shap_reg": shap_reg,
            "shap_clf": shap_clf,
            "X_test": X_test,
            "X_test_enc": X_test_enc,
            "predictions": predictions,
            "feature_names": feature_names,
            "expected_values": expected_values,
            "metadata": metadata
        }
    except Exception as e:
        st.error(f"Error loading global SHAP data: {str(e)}")
        st.exception(e)
        return None

@st.cache_data
def load_shap_procedure(proc_code):
    """Load procedure-specific SHAP data"""
    try:
        proc_dir = f"{SHAP_DIR}/procedures/{proc_code}"
        
        shap_reg = np.load(f"{proc_dir}/shap_reg.npy")
        shap_clf = np.load(f"{proc_dir}/shap_clf.npy")
        X_test_enc = pd.read_csv(f"{proc_dir}/X_test_enc.csv")
        predictions = pd.read_csv(f"{proc_dir}/predictions.csv")
        
        with open(f"{proc_dir}/expected_values.pkl", "rb") as f:
            expected_values = pickle.load(f)
        
        # Load feature names from global (same for all)
        with open(f"{SHAP_DIR}/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        return {
            "shap_reg": shap_reg,
            "shap_clf": shap_clf,
            "X_test_enc": X_test_enc,
            "predictions": predictions,
            "feature_names": feature_names,
            "expected_values": expected_values
        }
    except Exception as e:
        st.error(f"Error loading procedure SHAP data: {str(e)}")
        return None

@st.cache_data
def get_available_procedures():
    """Get list of procedures with SHAP analysis"""
    try:
        with open(f"{SHAP_DIR}/procedures/metadata.json", "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.warning(f"Could not load procedure metadata: {str(e)}")
        return {}

def plot_shap_summary(shap_values, X_test_enc, feature_names, title, top_n=20):
    """Create SHAP summary bar plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Get top N features
    top_indices = np.argsort(mean_shap)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_shap[top_indices]
    
    # Create horizontal bar plot
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
    ax.barh(range(len(top_features)), top_values, color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Mean |SHAP value|', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_shap_scatter(shap_values, X_test_enc, feature_names, feature_name):
    """Create SHAP scatter plot for a specific feature"""
    if feature_name not in feature_names:
        st.error(f"Feature '{feature_name}' not found")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_idx = feature_names.index(feature_name)
    feature_shap = shap_values[:, feature_idx]
    feature_values = X_test_enc.iloc[:, feature_idx].values
    
    scatter = ax.scatter(feature_values, feature_shap, 
                        c=feature_values, cmap='viridis', 
                        alpha=0.6, s=20)
    
    ax.set_xlabel(f'{feature_name} (encoded value)', fontsize=12)
    ax.set_ylabel('SHAP value', fontsize=12)
    ax.set_title(f'SHAP Impact of {feature_name}', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label=feature_name)
    plt.tight_layout()
    return fig

def plot_shap_dependence(shap_values, X_test_enc, feature_names, feature1, feature2=None):
    """Create SHAP dependence plot"""
    if feature1 not in feature_names:
        st.error(f"Feature '{feature1}' not found")
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature1_idx = feature_names.index(feature1)
    feature1_shap = shap_values[:, feature1_idx]
    feature1_values = X_test_enc.iloc[:, feature1_idx].values
    
    if feature2 and feature2 in feature_names:
        feature2_idx = feature_names.index(feature2)
        color_values = X_test_enc.iloc[:, feature2_idx].values
        scatter = ax.scatter(feature1_values, feature1_shap, 
                           c=color_values, cmap='coolwarm', 
                           alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label=feature2)
    else:
        ax.scatter(feature1_values, feature1_shap, 
                  c='steelblue', alpha=0.6, s=20)
    
    ax.set_xlabel(f'{feature1} (encoded value)', fontsize=12)
    ax.set_ylabel('SHAP value', fontsize=12)
    ax.set_title(f'SHAP Dependence: {feature1}', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_model_performance(predictions, metadata, model_type):
    """Plot model performance metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if model_type == "regression":
        # Actual vs Predicted
        ax = axes[0]
        ax.scatter(predictions['Actual_Length'], predictions['Pred_Duration'], 
                  alpha=0.5, s=10)
        
        min_val = min(predictions['Actual_Length'].min(), predictions['Pred_Duration'].min())
        max_val = max(predictions['Actual_Length'].max(), predictions['Pred_Duration'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        ax.set_xlabel('Actual Duration (min)', fontsize=12)
        ax.set_ylabel('Predicted Duration (min)', fontsize=12)
        ax.set_title('Actual vs Predicted Duration', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Residuals
        ax = axes[1]
        residuals = predictions['Actual_Length'] - predictions['Pred_Duration']
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add metrics text
        mae = metadata.get('reg_mae', 'N/A')
        r2 = metadata.get('reg_r2', 'N/A')
        fig.text(0.5, 0.02, f'MAE: {mae:.2f} min  |  R²: {r2:.4f}', 
                ha='center', fontsize=12, fontweight='bold')
        
    else:  # classification
        # Probability distribution
        ax = axes[0]
        ax.hist(predictions[predictions['Overrun_Actual'] == 0]['Overrun_Prob'], 
               bins=30, alpha=0.6, label='No Overrun', edgecolor='black')
        ax.hist(predictions[predictions['Overrun_Actual'] == 1]['Overrun_Prob'], 
               bins=30, alpha=0.6, label='Overrun', edgecolor='black')
        ax.set_xlabel('Predicted Overrun Probability', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Overrun Probability Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Confusion matrix at 0.5 threshold
        ax = axes[1]
        threshold = 0.5
        pred_binary = (predictions['Overrun_Prob'] > threshold).astype(int)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(predictions['Overrun_Actual'], pred_binary)
        
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Overrun', 'Overrun'])
        ax.set_yticklabels(['No Overrun', 'Overrun'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix (threshold=0.5)', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                             color="white" if cm[i, j] > cm.max()/2 else "black",
                             fontsize=16, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        # Add metrics text
        accuracy = metadata.get('clf_accuracy', 'N/A')
        auc = metadata.get('clf_auc', 'N/A')
        fig.text(0.5, 0.02, f'Accuracy: {accuracy:.4f}  |  AUC: {auc:.4f}', 
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

# ============================================================================
# DATA EXPLORATION TAB
# ============================================================================

def generate_visualization(df, config, user_request):
    """Use Google's Gemini API to generate visualization code"""
    model = get_gemini_model()
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
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
        
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1]
            if "```" in generated_code:
                generated_code = generated_code.split("```")[0]
        
        lines = generated_code.strip().split('\n')
        code_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                if any(keyword in line for keyword in ['import', 'plt', 'df', 'fig', 'ax', '=', '(', ')', '[', ']', '.', 'for', 'if', 'else']):
                    code_lines.append(line)
            elif line.strip().startswith('#'):
                code_lines.append(line)
            elif not line.strip():
                code_lines.append(line)
        
        generated_code = '\n'.join(code_lines)
        
        if "plt" in generated_code and "import matplotlib" not in generated_code:
            generated_code = "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\n" + generated_code
            
        return generated_code.strip()
        
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return None

def execute_code(df, generated_code):
    """Execute the generated code with proper error handling"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        
        plt.clf()
        plt.close('all')
        
        exec_globals = {
            'df': df,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
        }
        
        exec(generated_code, exec_globals)
        
        fig = plt.gcf()
        if fig.get_axes():
            st.pyplot(fig)
            return True
        else:
            st.warning("No plot was generated. The code may need adjustment.")
            return False
            
    except Exception as e:
        st.error(f"Error executing code: {str(e)}")
        
        with st.expander("🔍 View problematic code"):
            st.code(generated_code, language='python')
            
        if "KeyError" in str(e):
            st.info(f"Column not found. Available columns: {', '.join(df.columns[:10])}...")
        elif "ValueError" in str(e):
            st.info("There might be an issue with data types or values.")
        elif "TypeError" in str(e):
            st.info("There might be a type mismatch.")
            
        return False

def data_exploration_tab():
    """Data exploration tab content"""
    st.title("📊 Data Exploration")
    
    df_explore = load_data(DATA_CONFIG)
    
    if df_explore is not None:
        with st.sidebar:
            st.header("📊 Data Summary")
            st.write(f"**Rows:** {len(df_explore):,}")
            st.write(f"**Columns:** {len(df_explore.columns)}")
            
            st.subheader("Column Information")
            
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
        
        st.markdown("### 🎯 Explore and analyze the surgical operations dataset")
        
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
        
        st.markdown("---")
        user_request = st.text_area(
            "🔍 **What would you like to explore?**",
            height=100,
            placeholder="Enter your data exploration request..."
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
                        with st.expander("📝 Generated Code", expanded=False):
                            st.code(generated_code, language="python")
                        
                        st.markdown("### 📈 Visualization Result")
                        if execute_code(df_explore, generated_code):
                            st.success("✅ Visualization generated successfully!")
                        else:
                            st.info("💡 Try rephrasing your request")
                    else:
                        st.error("❌ Failed to generate code")
    else:
        st.error("❌ Unable to load data")

# ============================================================================
# SHAP ANALYSIS TAB
# ============================================================================

def shap_analysis_tab():
    """SHAP analysis tab content"""
    st.title("🔍 SHAP Analysis")
    
    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** values show how much each feature contributes 
    to the model's predictions. Positive SHAP values increase the prediction, negative values decrease it.
    """)
    
    # Check if SHAP data exists
    if not os.path.exists(SHAP_DIR):
        st.error(f"❌ SHAP data directory not found: {SHAP_DIR}")
        st.info("Please run the SHAP analysis script first to generate the data.")
        return
    
    # Sidebar: Select analysis scope
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        analysis_scope = st.radio(
            "Select Analysis Scope:",
            ["Global (All Surgeries)", "Procedure-Specific"],
            help="Choose whether to view SHAP analysis for all surgeries or a specific procedure"
        )
        
        selected_procedure = None
        if analysis_scope == "Procedure-Specific":
            procedures_meta = get_available_procedures()
            
            if procedures_meta:
                # Create a nice display for procedures
                proc_options = []
                for proc_code, meta in procedures_meta.items():
                    n_cases = meta.get('n_cases', 'N/A')
                    proc_options.append(f"{proc_code} (n={n_cases})")
                
                selected_display = st.selectbox(
                    "Select Procedure:",
                    proc_options,
                    help="Choose a specific procedure to analyze"
                )
                
                # Extract procedure code from display string
                selected_procedure = selected_display.split(" (n=")[0]
            else:
                st.warning("No procedure-specific data available")
                analysis_scope = "Global (All Surgeries)"
        
        st.markdown("---")
        
        model_type = st.radio(
            "Model Type:",
            ["Regression (Duration)", "Classification (Overrun)"],
            help="Choose which model's SHAP values to analyze"
        )
        
        st.markdown("---")
        
        visualization_type = st.selectbox(
            "Visualization Type:",
            ["Feature Importance", "Model Performance", "Feature Detail", "Dependence Plot"],
            help="Choose how to visualize SHAP values"
        )
    
    # Load appropriate SHAP data
    if analysis_scope == "Global (All Surgeries)":
        st.subheader("🌍 Global Analysis (All Surgeries)")
        shap_data = load_shap_global()
    else:
        st.subheader(f"🎯 Procedure-Specific Analysis: {selected_procedure}")
        shap_data = load_shap_procedure(selected_procedure)
    
    if shap_data is None:
        st.error("Failed to load SHAP data")
        return
    
    # Select appropriate SHAP values based on model type
    if model_type == "Regression (Duration)":
        shap_values = shap_data['shap_reg']
        predictions = shap_data['predictions']
        expected_value = shap_data['expected_values']['regression']
        model_label = "Duration Prediction"
    else:
        shap_values = shap_data['shap_clf']
        predictions = shap_data['predictions']
        expected_value = shap_data['expected_values']['classification']
        model_label = "Overrun Prediction"
    
    X_test_enc = shap_data['X_test_enc']
    feature_names = shap_data['feature_names']
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Samples", f"{len(X_test_enc):,}")
    
    with col2:
        st.metric("Features", len(feature_names))
    
    if model_type == "Regression (Duration)":
        with col3:
            mae = shap_data.get('metadata', {}).get('reg_mae', 'N/A')
            st.metric("MAE", f"{mae:.2f} min" if isinstance(mae, (int, float)) else mae)
        with col4:
            r2 = shap_data.get('metadata', {}).get('reg_r2', 'N/A')
            st.metric("R²", f"{r2:.4f}" if isinstance(r2, (int, float)) else r2)
    else:
        with col3:
            acc = shap_data.get('metadata', {}).get('clf_accuracy', 'N/A')
            st.metric("Accuracy", f"{acc:.4f}" if isinstance(acc, (int, float)) else acc)
        with col4:
            auc = shap_data.get('metadata', {}).get('clf_auc', 'N/A')
            st.metric("AUC", f"{auc:.4f}" if isinstance(auc, (int, float)) else auc)
    
    st.markdown("---")
    
    # Visualizations
    if visualization_type == "Feature Importance":
        st.subheader(f"📊 Feature Importance - {model_label}")
        
        top_n = st.slider("Number of top features to display:", 5, 30, 20, 5)
        
        fig = plot_shap_summary(shap_values, X_test_enc, feature_names, 
                               f"Top {top_n} Features by Mean |SHAP|", top_n)
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - Features at the top have the highest average impact on predictions
        - The bar length shows the magnitude of impact (mean absolute SHAP value)
        - This shows which features the model relies on most
        """)
        
    elif visualization_type == "Model Performance":
        st.subheader(f"📈 Model Performance - {model_label}")
        
        metadata = shap_data.get('metadata', {})
        model_key = 'regression' if model_type == "Regression (Duration)" else 'classification'
        
        fig = plot_model_performance(predictions, metadata, model_key)
        st.pyplot(fig)
        
        if model_type == "Regression (Duration)":
            st.markdown("""
            **Interpretation:**
            - **Left plot**: Points closer to the red line indicate better predictions
            - **Right plot**: Residuals centered around 0 indicate unbiased predictions
            - MAE (Mean Absolute Error) shows average prediction error in minutes
            - R² shows how well the model explains variance (closer to 1 is better)
            """)
        else:
            st.markdown("""
            **Interpretation:**
            - **Left plot**: Shows how well the model separates overrun vs non-overrun cases
            - **Right plot**: Confusion matrix at 0.5 threshold
            - Accuracy shows overall correct predictions
            - AUC (Area Under Curve) measures classification quality (closer to 1 is better)
            """)
    
    elif visualization_type == "Feature Detail":
        st.subheader(f"🔬 Feature Detail Analysis - {model_label}")
        
        # Let user select a feature
        selected_feature = st.selectbox(
            "Select a feature to analyze:",
            feature_names,
            help="Choose a feature to see its SHAP value distribution"
        )
        
        if selected_feature:
            fig = plot_shap_scatter(shap_values, X_test_enc, feature_names, selected_feature)
            if fig:
                st.pyplot(fig)
                
                st.markdown("""
                **Interpretation:**
                - **X-axis**: Encoded feature values (higher usually means more of that feature)
                - **Y-axis**: SHAP value (impact on prediction)
                - **Color**: Feature value intensity
                - Points above 0 increase the prediction, points below decrease it
                """)
                
                # Show feature statistics
                feature_idx = feature_names.index(selected_feature)
                feature_shap = shap_values[:, feature_idx]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean SHAP", f"{np.mean(feature_shap):.4f}")
                with col2:
                    st.metric("Mean |SHAP|", f"{np.mean(np.abs(feature_shap)):.4f}")
                with col3:
                    st.metric("Max |SHAP|", f"{np.max(np.abs(feature_shap)):.4f}")
    
    elif visualization_type == "Dependence Plot":
        st.subheader(f"🔗 Feature Dependence Analysis - {model_label}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            main_feature = st.selectbox(
                "Main feature:",
                feature_names,
                help="Feature to analyze on X-axis"
            )
        
        with col2:
            interaction_feature = st.selectbox(
                "Interaction feature (optional):",
                ["None"] + feature_names,
                help="Feature to color points by (shows interactions)"
            )
        
        if main_feature:
            interact_feat = None if interaction_feature == "None" else interaction_feature
            
            fig = plot_shap_dependence(shap_values, X_test_enc, feature_names, 
                                      main_feature, interact_feat)
            if fig:
                st.pyplot(fig)
                
                st.markdown("""
                **Interpretation:**
                - Shows how SHAP values change with feature values
                - If colored by interaction feature, color patterns reveal interactions
                - Non-linear patterns indicate complex relationships
                - Horizontal spread at same X value shows interaction effects
                """)
    
    # Additional insights section
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    # Calculate top positive and negative features
    mean_shap = shap_values.mean(axis=0)
    top_positive_idx = np.argsort(mean_shap)[-5:][::-1]
    top_negative_idx = np.argsort(mean_shap)[:5]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔺 Top Features Increasing Predictions:**")
        for idx in top_positive_idx:
            feature = feature_names[idx]
            value = mean_shap[idx]
            st.write(f"• **{feature}**: +{value:.4f}")
    
    with col2:
        st.markdown("**🔻 Top Features Decreasing Predictions:**")
        for idx in top_negative_idx:
            feature = feature_names[idx]
            value = mean_shap[idx]
            st.write(f"• **{feature}**: {value:.4f}")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        csv_shap = shap_df.to_csv(index=False)
        st.download_button(
            label="📥 Download SHAP Values (CSV)",
            data=csv_shap,
            file_name=f"shap_values_{analysis_scope.replace(' ', '_')}_{model_type.split()[0].lower()}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download predictions
        csv_pred = predictions.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv_pred,
            file_name=f"predictions_{analysis_scope.replace(' ', '_')}_{model_type.split()[0].lower()}.csv",
            mime="text/csv"
        )

# ============================================================================
# INSIGHT FEED TAB
# ============================================================================

def insight_feed_tab():
    """Automated insight discovery tab"""
    st.title("💡 Insight Feed")

    st.markdown("""
    **Automated Insight Discovery** - The system analyzes SHAP patterns to find:
    - 🔥 **Smoking Guns**: Hidden overrun risk factors
    - 🔀 **Model Divergence**: Where duration vs overrun models disagree
    - 📈 **Temporal Drift**: How patterns change over time
    - ⚠️ **Anomalies**: Unusual prediction patterns
    """)

    # Load enhanced SHAP data for insights
    try:
        from shap_enhanced import EnhancedSHAPData
        from insight_engine import InsightEngine

        with st.spinner("Loading SHAP data and generating insights..."):
            enhanced_data = EnhancedSHAPData.load(SHAP_DIR)
            engine = InsightEngine(enhanced_data)

            # Get insights
            max_insights = st.sidebar.slider("Number of insights to display:", 5, 30, 15, 5)
            insights = engine.generate_all_insights(max_insights=max_insights)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Insights", len(insights))
            with col2:
                smoking_guns = len([i for i in insights if i.category == 'smoking_gun'])
                st.metric("🔥 Smoking Guns", smoking_guns)
            with col3:
                drift_count = len([i for i in insights if i.category == 'drift'])
                st.metric("📈 Drift Detected", drift_count)
            with col4:
                anomalies = len([i for i in insights if i.category == 'anomaly'])
                st.metric("⚠️ Anomalies", anomalies)

            st.markdown("---")

            # Filter by category
            category_filter = st.sidebar.multiselect(
                "Filter by category:",
                ["smoking_gun", "divergence", "drift", "anomaly"],
                default=["smoking_gun", "divergence", "drift", "anomaly"]
            )

            filtered_insights = [i for i in insights if i.category in category_filter]

            # Display insights
            for i, insight in enumerate(filtered_insights, 1):
                # Category badge color
                badge_colors = {
                    "smoking_gun": "🔥",
                    "divergence": "🔀",
                    "drift": "📈",
                    "anomaly": "⚠️"
                }

                badge = badge_colors.get(insight.category, "ℹ️")

                with st.expander(f"{badge} {insight.title} (Importance: {insight.importance:.2f})", expanded=(i <= 5)):
                    st.markdown(f"**Description:** {insight.description}")

                    if insight.recommendation:
                        st.info(f"💡 **Recommendation:** {insight.recommendation}")

                    # Show evidence details
                    if insight.evidence:
                        with st.expander("📊 Evidence Details"):
                            st.json(insight.evidence)

            if len(filtered_insights) == 0:
                st.info("No insights match the selected filters.")

    except ImportError as e:
        st.error(f"Could not load insight engine: {str(e)}")
        st.info("Make sure shap_enhanced.py and insight_engine.py are in the same directory.")
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        st.exception(e)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Add a clear header before tabs
    st.markdown("---")
    st.markdown("## 📑 Navigation")
    st.markdown("Select a tab below to explore different features:")

    # Create tabs with more explicit styling
    tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "🔍 SHAP Analysis", "💡 Insight Feed"])

    with tab1:
        data_exploration_tab()

    with tab2:
        shap_analysis_tab()

    with tab3:
        insight_feed_tab()

if __name__ == "__main__":
    main()