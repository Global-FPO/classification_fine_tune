from flask import Flask, request, render_template
import pandas as pd
from gen import vendor_gen
 
app = Flask(__name__)
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request", 400
 
    file = request.files['file']
 
    if file.filename == '':
        return "No selected file", 400
 
    try:
        # Read the uploaded file into a DataFrame
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return "Unsupported file type. Please upload a CSV or Excel file.", 400
        print(df.columns.to_list())
        try:
            print('serching..')
            if 'Descriptions' in df.columns.to_list():
                print('found')
                
                predicted_df=vendor_gen(df)
                #pred=pd.read_excel(predicted_df)
                print(predicted_df)
                print(predicted_df.isnull().sum())

                #print(predicted_df)
                 
        except Exception as e:
            raise e
        else:
        # Convert DataFrame to HTML table
            table_html = predicted_df.to_html(classes='table table-bordered', index=False)
     
            # Render data.html with the table
            return render_template('data.html', table=table_html)
 
    except Exception as e:
        return f"Error processing the file: {e}", 500
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
 