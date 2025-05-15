"# AI_Days_Backend" 

To Set Up and Run the Backend Application
Open a command prompt in your IDE
1. create a Virtual Environment : 
        python -m venv venv

2. Activate the virtual Environment:
        venv\Scripts\activate 

3. Install required dependencies which are there in the requirements.txt:
        pip install -r requirements.txt

4. Create a .env file in the project root. You can use ".env.example" as reference for the required variables

5. API Key Encoding(One-Time Step)
    i. To securely encode your OpenAI API key:
        run-> python encode_key.py
    ii. Enter you API key
    iii. Paste the output Encoded_API_key into your .env file like this
        OPENAI_API_KEY_ENCODED=ENCODED_KEY_HERE

6. Once Everything is set up run the backend server:
        python run.py






         