from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
import os
import pandas as pd
from gen import vendor_gen

app = Flask(__name__)
api = Api(app)
CORS(app)

# Upload directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class UploadPDF(Resource):
    def post(self):
        files = request.files.getlist('file')
        if not files:
            return jsonify({"error": "No file provided"}), 400

        uploaded_files = []
        predicted_data = None  # Variable to store the predicted dataframe

        for file in files:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print(f"File uploaded: {file_path}")
            
            try:
                # Read the uploaded file into a DataFrame
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.filename.endswith('.xlsx'):
                    df = pd.read_excel(file)
                else:
                    return "Unsupported file type. Please upload a CSV or Excel file.", 400
                print(df.columns.to_list())
                
                if 'Descriptions' in df.columns.to_list():
                    print('Found Descriptions column')
                    predicted_df = vendor_gen(df)  # Process with your vendor_gen function
                    print(predicted_df)
                    print(predicted_df.isnull().sum())

                    # Convert predicted_df to JSON format
                    predicted_data = predicted_df.to_json(orient="split")
                else:
                    return jsonify({"error": '"Descriptions" column not found in your file.'}), 400

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return jsonify({"message": "Files uploaded successfully", "files": uploaded_files, "predicted_data": predicted_data})
    
    

api.add_resource(UploadPDF, "/upload_pdf")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002, threaded=True)
