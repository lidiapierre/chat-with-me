# Chat-With-Me

Create your own personal bot with the Chat-With-Me Streamlit application.


## Getting Started

Clone this repository and install the required packages:

   ```bash
   git clone https://github.com/yourusername/chat-with-me.git
   cd chat-with-me
   pip install -r requirements.txt
   ```

## Ingesting data
1. Create a new Pinecone index, default name `chatwithme`
2. Create a `.env` file with your API keys, see `example.env` as an example
3. Add your documents in the source repository
4. Edit your details the `config.py` file
5. (Optional) Change the `profile.png` picture
6. Run `ingest.py`

## Running the app
Run `streamlit run app.py`, have fun!