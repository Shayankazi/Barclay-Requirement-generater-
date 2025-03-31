import requests
import os
from pathlib import Path

def process_srs_file(file_path):
    """Process the SRS file and create Jira issues, then export to Excel"""
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found!")
        return
    
    # Upload and process SRS
    print("Uploading and processing SRS file...")
    files = {'file': open(file_path, 'rb')}
    response = requests.post('http://localhost:8000/upload-srs/', files=files)
    
    if response.status_code == 200:
        print("SRS processed successfully!")
        result = response.json()
        print(f"\nCreated {len(result.get('created_issues', []))} issues:")
        for issue in result.get('created_issues', []):
            print(f"- {issue.get('key')}: {issue.get('summary')}")
        
        # Generate Excel document
        print("\nGenerating Excel document...")
        try:
            doc_response = requests.post(
                'http://localhost:8000/generate-documents', 
                json={
                    "format": "excel",
                    "include_metadata": True,
                    "template_type": "simple"
                },
                stream=True  # Stream the response
            )
            
            if doc_response.status_code == 200:
                # Save the file
                filename = 'requirements.xlsx'
                with open(filename, 'wb') as f:
                    for chunk in doc_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"\nExcel file '{filename}' has been created successfully!")
                print(f"You can find it at: {os.path.abspath(filename)}")
            else:
                print(f"Error generating Excel: {doc_response.status_code}")
                print(doc_response.text)
        except Exception as e:
            print(f"Error creating Excel file: {str(e)}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    srs_file = "system_srs.txt"
    process_srs_file(srs_file)
