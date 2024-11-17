import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import requests
from io import StringIO
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Inventory Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure Anthropic API
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

def get_claude_response(prompt, api_key=None):
    """
    Get response from Claude AI with better error handling
    """
    headers = {
        "x-api-key": api_key or ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1500,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Request Error: {str(e)}")


def load_data_from_text(file):
    """
    Load data from a text file and convert it to a pandas DataFrame
    """
    try:
        # Read the content of the text file
        content = file.read().decode('utf-8')
        
        # Convert the content to a CSV-like format using StringIO
        csv_data = StringIO(content)
        
        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv(csv_data)
        
        # Process the DataFrame
        return process_dataframe(df)
    except Exception as e:
        st.error(f"Error loading data from text file: {str(e)}")
        return None

def process_dataframe(df):
    """
    Process and clean the DataFrame
    """
    try:
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Convert date columns
        date_columns = ['date', 'last_restock_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert numeric columns
        numeric_columns = ['quantity_in_stock', 'reorder_level', 'unit_price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def generate_analysis_prompt(question, df_info):
    """
    Generate prompt for Claude AI analysis with column name guidance
    """
    return f"""You are an inventory analyst. Provide direct answers and use exact column names from the data.

Data Information:
{df_info}

Question: {question}

Rules for your response:
1. Provide actual numerical results and specific findings
2. Use exact column names in visualization data:
   - Use 'category' (not 'Category')
   - Use 'product_name' (not 'Product Name')
   - Use 'quantity_in_stock' (not 'Quantity')
   - Use 'unit_price' (not 'Price')
   - Use 'supplier_name' (not 'Supplier')
   - Use 'storage_location' (not 'Location')
   - Use 'inventory_value' for price * quantity calculations

Format your response as a JSON with these exact keys:
{{
    "answer": "Give direct results here with actual numbers and findings. Include currency symbols for monetary values.",
    "visualization_needed": true/false,
    "visualization_type": "bar/line/scatter/pie",
    "visualization_data": {{"x": "exact_column_name", "y": "exact_column_name"}},
    "visualization_title": "Clear title for the visualization"
}}

Example correct column usage:
- "x": "category"
- "y": "quantity_in_stock"
- "x": "supplier_name"
- "y": "inventory_value"
"""

def parse_claude_response(response_text):
    """
    Safely parse Claude's response and ensure it's valid JSON with direct results
    """
    try:
        # Clean the response text
        cleaned_text = response_text.replace('\n', ' ').replace('\r', '')
        
        # Extract JSON part if there's any other text
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = cleaned_text[start_idx:end_idx]
        else:
            raise ValueError("No JSON found in response")

        # Parse JSON
        analysis = json.loads(json_str)
        
        # Format numerical values in the answer if present
        answer = analysis['answer']
        # Look for numbers followed by decimal places
        import re
        answer = re.sub(r'(\d+)\.(\d{2})(?![0-9])', lambda m: f"{float(m.group()):,.2f}", answer)
        # Look for large numbers without decimals
        answer = re.sub(r'(?<![\d.])\d+(?![\d.])', lambda m: f"{int(m.group()):,}", answer)
        
        analysis['answer'] = answer
        
        return analysis
    except Exception as e:
        raise Exception(f"Error parsing response: {str(e)}")

def create_visualization(viz_info, df):
    """
    Create visualization with case-insensitive column matching
    """
    try:
        # Create a copy of the dataframe for visualization
        plot_df = df.copy()
        
        # Get column mappings (lowercase to actual)
        column_map = {col.lower(): col for col in df.columns}
        
        # Handle x column - case insensitive
        x_column = viz_info["visualization_data"].get("x", "").lower()
        if x_column not in column_map:
            st.warning(f"Column '{x_column}' not found. Available columns: {', '.join(df.columns)}")
            return None
        x_column = column_map[x_column]
        
        # Handle y column(s) - case insensitive
        y_column = viz_info["visualization_data"].get("y")
        if isinstance(y_column, list):
            y_columns = [column_map.get(col.lower()) for col in y_column if column_map.get(col.lower())]
        else:
            y_column = y_column.lower()
            if y_column not in column_map:
                st.warning(f"Column '{y_column}' not found. Available columns: {', '.join(df.columns)}")
                return None
            y_columns = [column_map[y_column]]
        
        # Pre-process data if needed
        if 'inventory_value' not in plot_df.columns:
            plot_df['inventory_value'] = plot_df['quantity_in_stock'] * plot_df['unit_price']
            column_map['inventory_value'] = 'inventory_value'

        # Create appropriate visualization
        if viz_info["visualization_type"].lower() == "bar":
            if len(y_columns) > 1:
                fig = go.Figure()
                for y_col in y_columns:
                    fig.add_trace(
                        go.Bar(
                            x=plot_df[x_column],
                            y=plot_df[y_col],
                            name=y_col
                        )
                    )
            else:
                # For single y column, create grouped data if needed
                if len(plot_df) > 10:  # If too many rows, aggregate
                    grouped_df = plot_df.groupby(x_column)[y_columns[0]].sum().reset_index()
                    fig = px.bar(
                        grouped_df,
                        x=x_column,
                        y=y_columns[0],
                        title=viz_info["visualization_title"]
                    )
                else:
                    fig = px.bar(
                        plot_df,
                        x=x_column,
                        y=y_columns[0],
                        title=viz_info["visualization_title"]
                    )

        elif viz_info["visualization_type"].lower() == "line":
            if len(y_columns) > 1:
                fig = go.Figure()
                for y_col in y_columns:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df[x_column],
                            y=plot_df[y_col],
                            name=y_col,
                            mode='lines+markers'
                        )
                    )
            else:
                fig = px.line(
                    plot_df,
                    x=x_column,
                    y=y_columns[0],
                    title=viz_info["visualization_title"]
                )

        elif viz_info["visualization_type"].lower() == "scatter":
            if len(y_columns) > 1:
                fig = go.Figure()
                for y_col in y_columns:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df[x_column],
                            y=plot_df[y_col],
                            name=y_col,
                            mode='markers'
                        )
                    )
            else:
                fig = px.scatter(
                    plot_df,
                    x=x_column,
                    y=y_columns[0],
                    title=viz_info["visualization_title"]
                )

        elif viz_info["visualization_type"].lower() == "pie":
            if len(plot_df) > 10:  # If too many rows, aggregate
                grouped_df = plot_df.groupby(x_column)[y_columns[0]].sum().reset_index()
                fig = px.pie(
                    grouped_df,
                    names=x_column,
                    values=y_columns[0],
                    title=viz_info["visualization_title"]
                )
            else:
                fig = px.pie(
                    plot_df,
                    names=x_column,
                    values=y_columns[0],
                    title=viz_info["visualization_title"]
                )

        else:
            st.warning(f"Unsupported visualization type: {viz_info['visualization_type']}")
            return None

        # Update layout with proper formatting
        fig.update_layout(
            title=viz_info["visualization_title"],
            xaxis_title=x_column,
            yaxis_title=y_columns[0] if len(y_columns) == 1 else "Value",
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add formatting for values
        if any(col.lower() in ['unit_price', 'inventory_value'] for col in y_columns):
            fig.update_layout(
                yaxis=dict(
                    tickprefix="$",
                    tickformat=",.",
                )
            )
        else:
            fig.update_layout(
                yaxis=dict(
                    tickformat=",",
                    separatethousands=True
                )
            )

        return fig

    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.write("Visualization info:", viz_info)  # Debug info
        st.write("Available columns:", df.columns.tolist())  # Debug info
        return None

def main():
    """
    Main application function containing all UI elements and logic
    """
    # Page title and layout
    st.title("📦 Inventory Analysis Dashboard")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("Settings")
        
        # API Key input
        api_key = st.text_input(
            "Anthropic API Key:",
            type="password",
            help="Enter your API key here or set it in .env file",
            key="api_key"
        )
        
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload inventory data (TXT/CSV)",
            type=["txt", "csv"],
            help="Upload the inventory data file",
            key="file_uploader"
        )
    
    # Initialize session state for dataframe
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Handle file upload and data loading
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            try:
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                if file_type == 'txt':
                    st.session_state.df = load_data_from_text(uploaded_file)
                else:
                    st.session_state.df = pd.read_csv(uploaded_file)
                    st.session_state.df = process_dataframe(st.session_state.df)
                
                if st.session_state.df is not None:
                    st.sidebar.success("✅ Data loaded successfully!")
                    
                    # Display data info in sidebar
                    st.sidebar.write("### Dataset Information")
                    st.sidebar.write(f"Total Records: {len(st.session_state.df):,}")
                    st.sidebar.write(f"Categories: {st.session_state.df['category'].nunique()}")
                    st.sidebar.write(f"Unique Products: {st.session_state.df['product_id'].nunique()}")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.session_state.df = None
    
    # Main Content - Only show if data is loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # 1. Quick Statistics Section
        st.header("📊 Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_stock_value = (df['quantity_in_stock'] * df['unit_price']).sum()
        low_stock_count = len(df[df['quantity_in_stock'] <= df['reorder_level']])
        
        with col1:
            st.metric(
                "Total Products",
                f"{df['product_id'].nunique():,}",
                help="Number of unique products in inventory"
            )
        with col2:
            st.metric(
                "Total Stock Value",
                f"${total_stock_value:,.2f}",
                help="Total value of current inventory"
            )
        with col3:
            st.metric(
                "Low Stock Items",
                f"{low_stock_count:,}",
                help="Items below reorder level"
            )
        with col4:
            st.metric(
                "Average Unit Price",
                f"${df['unit_price'].mean():,.2f}",
                help="Average price per unit across all products"
            )
        
        # 2. Data Overview Section
        st.header("📋 Data Overview")
        with st.expander("View Raw Data", expanded=False):
            # Add a search box for filtering
            search = st.text_input("Search products:", key="data_search")
            
            # Filter dataframe based on search
            if search:
                filtered_df = df[df['product_name'].str.contains(search, case=False) |
                               df['category'].str.contains(search, case=False) |
                               df['supplier_name'].str.contains(search, case=False)]
            else:
                filtered_df = df
            
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
        
        # 3. Analysis Options Section
        st.header("🔍 Analysis Options")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3 = st.tabs(["Inventory Analysis", "Supplier Analysis", "Location Analysis"])
        
        with tab1:
            analysis_type = st.selectbox(
                "Select Analysis Type",
                [
                    "Stock by Category",
                    "Price Distribution",
                    "Low Stock Items",
                    "Stock Value Analysis"
                ],
                key="inventory_analysis"
            )
            
            if analysis_type == "Stock by Category":
                fig = px.bar(
                    df.groupby('category')['quantity_in_stock'].sum().reset_index(),
                    x='category',
                    y='quantity_in_stock',
                    title="Total Stock by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == "Price Distribution":
                fig = px.box(
                    df,
                    x='category',
                    y='unit_price',
                    title="Price Distribution by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_type == "Low Stock Items":
                low_stock = df[df['quantity_in_stock'] <= df['reorder_level']]
                st.write("### Items Below Reorder Level")
                st.dataframe(
                    low_stock[['product_name', 'quantity_in_stock', 'reorder_level', 'supplier_name']],
                    use_container_width=True,
                    hide_index=True
                )
                
            elif analysis_type == "Stock Value Analysis":
                df['stock_value'] = df['quantity_in_stock'] * df['unit_price']
                fig = px.treemap(
                    df,
                    path=['category', 'product_name'],
                    values='stock_value',
                    title="Stock Value Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            supplier_metric = st.selectbox(
                "Select Supplier Metric",
                ["Value by Supplier", "Items per Supplier", "Supplier Restock Analysis"],
                key="supplier_analysis"
            )
            
            if supplier_metric == "Value by Supplier":
                df['inventory_value'] = df['quantity_in_stock'] * df['unit_price']
                fig = px.pie(
                    df.groupby('supplier_name')['inventory_value'].sum().reset_index(),
                    values='inventory_value',
                    names='supplier_name',
                    title="Inventory Value by Supplier"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif supplier_metric == "Items per Supplier":
                fig = px.bar(
                    df.groupby(['supplier_name', 'category'])['product_id'].count().reset_index(),
                    x='supplier_name',
                    y='product_id',
                    color='category',
                    title="Items per Supplier by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif supplier_metric == "Supplier Restock Analysis":
                st.write("### Recent Restocks by Supplier")
                recent_restocks = df.sort_values('last_restock_date', ascending=False)
                st.dataframe(
                    recent_restocks[['supplier_name', 'product_name', 'last_restock_date', 'quantity_in_stock']],
                    use_container_width=True,
                    hide_index=True
                )
        
        with tab3:
            location_analysis = st.selectbox(
                "Select Location Analysis",
                ["Storage Utilization", "Category Distribution", "Value by Location"],
                key="location_analysis"
            )
            
            if location_analysis == "Storage Utilization":
                fig = px.treemap(
                    df,
                    path=['storage_location', 'category'],
                    values='quantity_in_stock',
                    title="Storage Location Utilization"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif location_analysis == "Category Distribution":
                location_cat_dist = pd.crosstab(df['storage_location'], df['category'])
                st.write("### Category Distribution by Location")
                st.dataframe(location_cat_dist, use_container_width=True)
                
            elif location_analysis == "Value by Location":
                df['location_value'] = df['quantity_in_stock'] * df['unit_price']
                fig = px.bar(
                    df.groupby('storage_location')['location_value'].sum().reset_index(),
                    x='storage_location',
                    y='location_value',
                    title="Inventory Value by Location"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 4. Custom Analysis Section
        st.header("🤖 Custom Analysis")
        
        # Example questions
        with st.expander("View Example Questions", expanded=False):
            st.markdown("""
            Try these specific questions:
            - What is the total value of Electronics inventory?
            - Show me the top 5 most valuable products in stock
            - Which supplier has the highest total inventory value?
            - What is the total stock value in Warehouse A-12?
            - How many products are below reorder level in each category?
            """)
        
        # Question input and analysis
        question = st.text_area(
            "Ask a specific question about the inventory:",
            height=100,
            key="analysis_question",
            help="Ask for specific values, counts, or rankings"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("Analyze", type="primary", key="analyze_button")
        with col2:
            clear_button = st.button("Clear", key="clear_button")
        
        if clear_button:
            st.session_state.analysis_question = ""
            st.experimental_rerun()
        
        if question and analyze_button:
            with st.spinner("Analyzing..."):
                try:
                    # Prepare concise data info for prompt
                    df_info = f"""
                    Available metrics: quantity_in_stock, unit_price, reorder_level
                    Categories: {', '.join(df['category'].unique())}
                    Suppliers: {', '.join(df['supplier_name'].unique())}
                    Locations: {', '.join(df['storage_location'].unique())}
                    Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
                    
                    Current totals:
                    - Total inventory value: ${(df['quantity_in_stock'] * df['unit_price']).sum():,.2f}
                    - Total items in stock: {df['quantity_in_stock'].sum():,}
                    - Total unique products: {df['product_id'].nunique():,}
                    """
                    
                    # Get analysis from Claude
                    prompt = generate_analysis_prompt(question, df_info)
                    response = get_claude_response(prompt, api_key)
                    
                    # Parse response safely
                    analysis = parse_claude_response(response["content"][0]["text"])
                    
                    # Display results in a clean format
                    st.write("### Results")
                    st.info(analysis["answer"])
                    
                    # Create and display visualization if needed
                    if analysis["visualization_needed"]:
                        fig = create_visualization(analysis, df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add download button for the visualization
                            buf = io.BytesIO()
                            fig.write_image(buf, format="png")
                            st.download_button(
                                label="Download Visualization",
                                data=buf.getvalue(),
                                file_name="visualization.png",
                                mime="image/png"
                            )
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    if "API Request Error" in str(e):
                        st.warning("Please check your API key and internet connection.")
        
    else:
        # Show this when no data is loaded
        st.info("👆 Please upload your inventory data file to begin analysis.")
        
        # Sample data format
        st.write("### Expected Data Format")
        st.write("Your CSV/TXT file should contain the following columns:")
        st.code("""
date,product_id,product_name,category,quantity_in_stock,reorder_level,unit_price,supplier_name,last_restock_date,storage_location
2024-01-01,INV001,Laptop Dell XPS 13,Electronics,45,20,1299.99,TechSupply Corp,2023-12-25,Warehouse A-12
        """)
    
    # Footer
    st.write("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 10px;'>
        Made with ❤️ using Streamlit and Claude AI
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
