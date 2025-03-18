# Import Dataset from kaggle
import kagglehub
# Import Vector Database and Model
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd

# Variables to export
df=pd.DataFrame
model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
qdrant_client = QdrantClient(":memory:")

# Load the model
def initialize_arxiv_data(df, model_encoder, qdrant_client, sample_size):

    # Download latest version of dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("Cornell-University/arxiv")
    print("Dataset downloaded successfully into " + path)

    # Read into Pandas DataFrame
    print("Loading data into pandas dataframe...")
    df = pd.read_json(path+'/arxiv-metadata-oai-snapshot.json', lines=True)
    print("Data loaded into pandas dataframe successfully")
    print(df.head())

    # Preprocess Data
    # Filter out NAN category
    print("Filtering data...")
    df = df[df['categories'].notna()]
    # Filter out withdrawn papers
    df = df[df['abstract'].notna()]
    df = df[~df['abstract'].str.contains('withdrawn', case=False, regex=False)]

    # Create vector database client and model client
    # Create collection to store
    print("Creating arxiv collection...")
    qdrant_client.recreate_collection(
        collection_name = "arxiv",
        vectors_config = models.VectorParams(
            # Note that for our model, this vector size is 384
            size=model_encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )
    if qdrant_client.collection_exists(collection_name="arxiv"):
        print("arxiv collection created successfully")
    # Insert quadrant data into collection
    # Note to self: qdrant_client.upsert is used for bulk upload of points, while qdrant_client.insert is used for single point upload
    # Before uploading the vectorized data, we load the dataframe into a key value pair dictionary to enable convenient iteration.

    # Convert the WHOLE dataframe to a dictionary
    # data_in_dict = df.to_dict(orient='records')
    # Convert sample of the dataframe to a dictionary
    data_in_dict = df.head(sample_size).to_dict(orient='records')
    print(data_in_dict)


    print("Uploading data...")
    qdrant_client.upload_points(
        collection_name = "arxiv",
        points=[
            models.PointStruct(
                id = index,
                payload=doc,
                vector = model_encoder.encode(doc["abstract"]),
            ) for index, doc in enumerate(data_in_dict)
        ]
    )
    print("Data uploaded successfully")
    model_loaded=True
    return path, df, model_encoder, qdrant_client, model_loaded

#Refactored for backend
def search_arxiv_papers(model_encoder, qdrant_client, user_prompt, limit_search_to):#-> List of Dicts
    query_prompt = "You are an AI agent searching for arXiv papers based on the following instructions: " + user_prompt
    print("Query Prompt: " + query_prompt)
    # Search based on user prompt
    hits = qdrant_client.search(
        collection_name = "arxiv",
        query_vector=model_encoder.encode(query_prompt),
        limit=limit_search_to
    )

    # Print Queried Outputs
    for hit in hits:
        print("TITLE: " + hit.payload["title"] + '\n\n', 
              "arXivID: " + hit.payload["id"] + '\n\n', 
              "AUTHORS: " + hit.payload["authors"] + '\n\n', 
              "ABSTRACT: " + hit.payload["abstract"][0:100] + '...' + '\n\n', 
              '============break==========\n')
    
    # Return search results
    search_results = [hit.payload for hit in hits]
    print(search_results)
    return search_results
